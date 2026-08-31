"""``precompute-embeddings`` must embed what it was asked to, and store all of it.

Two families of test live here, both about the bookkeeping around the
embedding rather than the embedding itself, which is stubbed out.

**Every document must reach the LMDB.** The command embeds each document,
submits the compression to a thread pool, and flushes *completed* jobs to the
writer only when the in-flight backlog grows past ``MAX_BACKLOG``. That flush is
what keeps the backlog under the threshold, so when the rows run out there are
always jobs still in flight and never enough of them to trip the threshold again
— which is why the drain after the row loop cannot be guarded by the same
condition. The failure this prevents is silent (the command reports ``Done.``
either way), and at the extreme — a dataset shorter than ``MAX_BACKLOG`` —
nothing at all gets written. Each stub embedding is filled with its own pubmed
id, so a key/value mix-up in the drain fails these too.

**Every flag must do what it says.** ``--batch_size``, ``--max_length`` and
``--force-regenerate`` were all once accepted and then ignored, so a run
silently used the embedder's own defaults and re-embedded documents it already
held. The tests below assert against what the embedder was actually *called*
with, since a flag that never reaches it leaves the stored output unchanged and
so cannot be caught by inspecting the LMDB alone.

**A document the corpus has no text for is not a document.** All three
precompute commands read the corpus through ``document_text`` so they describe
it the same way, and the other two skip the empty document. Embedding it stores
a 0-row matrix under a pubmed id the training splits drop, and that key then
reads as already embedded on every resume; under ``-f``, a document that has
since lost its text keeps its stale entry instead of losing it.

**A store that stopped early must not look like a finished one.** The LMDB is
opened with a fixed ``map_size``, and a pass needing more than that used to
commit the prefix it had and report ``Done.`` like any other run. The resume
path then read every truncated-in key as already embedded, so a rerun skipped
straight to the same wall. The family here pins the two halves of that: the
reservation is large enough for the corpus and adjustable, and running out of it
ends the run. It also pins the case where the flag names no reservation at all:
LMDB reads a ``map_size`` of zero as the size the store already has, so the run
used to embed its way to the GPU-shaped end of a 1 MiB default nobody asked for,
and refuses a negative one only once ``lmdb.open`` is reached.

**The store must say what wrote it.** The command takes the base model on
its command line and stores one matrix per pubmed id; two encoders of the same
hidden size produce matrices of the same shape, so a store built with one and
read under another is caught by nothing downstream. The pass records the
model, window and stride it used, and refuses to add to a store recording
anything else.

**A dead writer must not become a hang.** The writer thread is the only
consumer of the queue the embedding loop puts into, and that queue is bounded.
A failure the writer has no branch for — a disk error, a corrupt page, an
`lmdb.Error` out of `commit` — used to end the thread without telling anyone,
and the producer then filled the queue and waited on a consumer that no longer
existed: two live progress bars, no error, forever. The last test here runs the
command with a deadline, because the regression it guards is a run that never
returns rather than one that fails.
"""

import mmap
import pathlib
import re
import threading
import types
from collections.abc import Callable
from typing import Any

import lmdb
import numpy as np
import pytest
import torch
import tqdm
import transformers
from d3text import utils
from d3text.cli import precompute_embeddings
from d3text.embeddings_store import (
    StoreProvenance,
    bytes_to_tensor,
    read_provenance,
    tensor_to_bytes,
)

_EMBEDDING_SHAPE = (2, 4)
_CONTEXT_WINDOW = 512

# A reservation past any 64-bit address space (2**62 bytes), so `lmdb.open`'s
# mmap fails on every host rather than only on a small one.
_UNMAPPABLE_GIB = float(2**32)

# What `transformers` reports for a tokenizer whose config declares no limit —
# which is true of the default base model, michiyasunaga/BioLinkBERT-base.
_NO_LIMIT_DECLARED = 1000000000000000019884624838656


class _RecordingEmbedder:
    """Stands in for `utils.embed_document`, recording how it was called.

    Each embedding is stamped with its own pubmed id, so a key/value mix-up
    between the row loop, the compression pool and the writer fails too. The
    store narrows to bf16, so the stamp comes back rounded above 256 and the
    ids used here have to stay distinct through that — `_stamp` is where that
    is enforced.
    `stream_rows` feeds `main` the abstract and fulltext joined by a newline,
    so the id `_write_dataset` writes into both is the document's first token.

    The `embedder` fixture's other two stubs record into `loaded_tokenizers`
    and `loaded_base_models` here, so a test can assert that a run ended
    before either of them was loaded.
    """

    def __init__(self, fill: float | None = None) -> None:
        self.calls: list[types.SimpleNamespace] = []
        self.loaded_tokenizers: list[str] = []
        self.loaded_base_models: list[str] = []
        self._fill = fill

    def __call__(self, doc: str, **kwargs: object) -> torch.Tensor:
        if not doc:
            # What the real embedder answers for a document with no text: one
            # window of `[CLS][SEP]`, both sliced away as specials. Recorded
            # rather than refused, so a caller that embeds the empty document
            # fails on the assertion about it and not inside this stub.
            self.calls.append(types.SimpleNamespace(pubmed_id=None, **kwargs))
            return torch.empty((0, _EMBEDDING_SHAPE[1]))

        pubmed_id = int(doc.split()[0])
        self.calls.append(types.SimpleNamespace(pubmed_id=pubmed_id, **kwargs))
        fill = pubmed_id if self._fill is None else self._fill
        return torch.full(_EMBEDDING_SHAPE, float(fill))

    @property
    def embedded_ids(self) -> list[int | None]:
        return [call.pubmed_id for call in self.calls]


_FAKE_CONFIG = transformers.BertConfig(
    max_position_embeddings=_CONTEXT_WINDOW,
    name_or_path="fake-base-model",
)


class _FakeBaseModel:
    """Stands in for the frozen transformer, which `_RecordingEmbedder` never
    calls. `main` moves it to a device and puts it in eval mode."""

    config = _FAKE_CONFIG

    def to(self, _device: torch.device) -> "_FakeBaseModel":
        return self

    def eval(self) -> "_FakeBaseModel":
        return self


@pytest.fixture
def embedder(monkeypatch: pytest.MonkeyPatch) -> _RecordingEmbedder:
    """Run `main` with no network, no config, no tokenizer, no transformer.

    The stub tokenizer carries the sentinel `model_max_length` that the real
    default base model reports, so any attempt to derive the window size from
    the tokenizer shows up in the recorded `max_len`. The tokenizer and the
    base model record, because "before anything is loaded" is a claim about
    the two a run normally makes no other trace of. The config stub does not:
    reading a config.json is what a run is allowed to do before it commits to
    the weights, so a call to it is not the thing being counted.
    """
    recorder = _RecordingEmbedder()

    def load_fast_tokenizer(base_model: str) -> types.SimpleNamespace:
        recorder.loaded_tokenizers.append(base_model)
        return types.SimpleNamespace(model_max_length=_NO_LIMIT_DECLARED)

    def config_from_pretrained(
        *_args: object, **_kwargs: object
    ) -> transformers.PretrainedConfig:
        return _FAKE_CONFIG

    def from_pretrained(*args: object, **_kwargs: object) -> _FakeBaseModel:
        recorder.loaded_base_models.append(str(args[0]))
        return _FakeBaseModel()

    monkeypatch.setattr(utils, "load_fast_tokenizer", load_fast_tokenizer)
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", config_from_pretrained
    )
    monkeypatch.setattr(
        transformers.AutoModel, "from_pretrained", from_pretrained
    )
    monkeypatch.setattr(utils, "embed_document", recorder)
    # `main` setdefaults this; keep the mutation out of the wider test session.
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return recorder


def _write_dataset(path: pathlib.Path, pubmed_ids: list[int]) -> pathlib.Path:
    """Write the csv layout `stream_rows` expects: a leading unnamed index
    column (dropped by name), then pubmed_id/abstract/fulltext."""
    rows = "\n".join(
        f"{row},{pubmed_id},{pubmed_id} abstract,{pubmed_id} fulltext"
        for row, pubmed_id in enumerate(pubmed_ids)
    )
    path.write_text(f",pubmed_id,abstract,fulltext\n{rows}\n")
    return path


def _stored_embeddings(output_path: pathlib.Path) -> dict[bytes, np.ndarray]:
    """Read the store through its own codec, which is the only thing that
    knows the byte layout. Reaching for `blosc2` directly does not merely read
    the wrong thing — `unpack_array` **segfaults** on a blob it did not write,
    taking the whole session with it rather than failing one test."""
    env = lmdb.open(str(output_path), readonly=True, lock=False)
    try:
        with env.begin() as txn:
            # The provenance record shares the keyspace and is not a document,
            # and `bytes_to_tensor` would refuse it as a foreign blob.
            return {
                key: bytes_to_tensor(value).float().numpy()
                for key, value in txn.cursor().iternext()
                if key.decode().isdigit()
            }
    finally:
        env.close()


def _run(
    monkeypatch: pytest.MonkeyPatch,
    output_path: pathlib.Path,
    datasets: list[pathlib.Path],
    *flags: str,
) -> dict[bytes, np.ndarray]:
    monkeypatch.setattr(
        "sys.argv",
        [
            "precompute-embeddings",
            "base-model",
            str(output_path),
            *(str(dataset) for dataset in datasets),
            *flags,
        ],
    )
    precompute_embeddings.main()
    return _stored_embeddings(output_path)


def _stamp(pubmed_id: int) -> float:
    """The stamp as the store can hold it.

    bf16 carries 8 significant bits, so an id above 256 does not survive the
    round trip: 801 and 802 both come back as 800. That is harmless for
    activations and fatal for a mix-up detector, hence the distinctness
    assertion in `_assert_holds_embeddings_for`."""
    return torch.tensor(float(pubmed_id)).to(torch.bfloat16).float().item()


def _assert_holds_embeddings_for(
    stored: dict[bytes, np.ndarray], pubmed_ids: list[int]
) -> None:
    stamps = [_stamp(pubmed_id) for pubmed_id in pubmed_ids]
    assert len(set(stamps)) == len(stamps), (
        f"{pubmed_ids} do not stay distinct as bf16 stamps ({stamps}), so a "
        f"key/value mix-up would pass this assertion unnoticed"
    )

    assert sorted(stored) == sorted(str(p).encode() for p in pubmed_ids)
    for pubmed_id, stamp in zip(pubmed_ids, stamps):
        embedding = stored[str(pubmed_id).encode()]
        assert embedding.shape == _EMBEDDING_SHAPE
        assert (
            embedding == stamp
        ).all(), f"{pubmed_id} was stored under the wrong key"


@pytest.mark.usefixtures("embedder")
def test_writes_every_document_of_a_dataset_shorter_than_the_backlog(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The extreme case: with fewer rows than `MAX_BACKLOG` (at least 8), the
    in-loop flush never fires even once, so the drain is the only thing that
    writes anything."""
    pubmed_ids = [101, 102, 103]
    assert len(pubmed_ids) < precompute_embeddings.MAX_BACKLOG

    stored = _run(
        monkeypatch,
        tmp_path / "embeddings.lmdb",
        [_write_dataset(tmp_path / "small.csv", pubmed_ids)],
    )

    _assert_holds_embeddings_for(stored, pubmed_ids)


@pytest.mark.usefixtures("embedder")
def test_writes_the_documents_left_in_flight_when_the_rows_run_out(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The general case: the in-loop flush fires, but the rows run out with a
    tail of jobs still in flight. `MAX_BACKLOG` is pinned to a small value so
    the flush is exercised regardless of the box's core count."""
    monkeypatch.setattr(precompute_embeddings, "MAX_BACKLOG", 2)
    pubmed_ids = [201, 202, 203, 204, 205]

    stored = _run(
        monkeypatch,
        tmp_path / "embeddings.lmdb",
        [_write_dataset(tmp_path / "tail.csv", pubmed_ids)],
    )

    _assert_holds_embeddings_for(stored, pubmed_ids)


@pytest.mark.usefixtures("embedder")
def test_writes_each_dataset_in_full(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Datasets must not depend on each other to get flushed. When the backlog
    was shared across them, one dataset's leftovers were written only once a
    *later* dataset pushed the backlog over the threshold, and the final
    dataset's tail was lost for good."""
    first = [301, 302, 303]
    second = [401, 402, 403]

    stored = _run(
        monkeypatch,
        tmp_path / "embeddings.lmdb",
        [
            _write_dataset(tmp_path / "first.csv", first),
            _write_dataset(tmp_path / "second.csv", second),
        ],
    )

    _assert_holds_embeddings_for(stored, first + second)


def test_batch_size_and_window_reach_the_embedder_as_given(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    embedder: _RecordingEmbedder,
) -> None:
    """Both flags were accepted and dropped on the floor, so a run used
    `embed_document`'s own defaults no matter what was asked for."""
    _run(
        monkeypatch,
        tmp_path / "embeddings.lmdb",
        [_write_dataset(tmp_path / "flags.csv", [501, 502])],
        "--batch_size",
        "3",
        "--max_length",
        "128",
    )

    assert embedder.embedded_ids == [501, 502]
    for call in embedder.calls:
        assert call.batch_size == 3
        assert call.max_len == 128


def test_window_defaults_to_the_model_context_not_the_tokenizer_sentinel(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    embedder: _RecordingEmbedder,
) -> None:
    """With no `--max_length`, the window is the base model's context.

    The obvious source — `tokenizer.model_max_length` — is a ~1e30 sentinel for
    the default base model, and `split_and_tokenize` pads *to* the window, so
    deriving it from the tokenizer asks for an impossible tensor.
    """
    _run(
        monkeypatch,
        tmp_path / "embeddings.lmdb",
        [_write_dataset(tmp_path / "default.csv", [601])],
    )

    (call,) = embedder.calls
    assert call.max_len == _CONTEXT_WINDOW
    assert call.batch_size == 50


def test_a_window_past_the_model_context_is_rejected(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    embedder: _RecordingEmbedder,
) -> None:
    """A window longer than the position-embedding table indexes past it. The
    command must say so, rather than dying inside the base model's forward."""
    with pytest.raises(ValueError, match=f"between 1 and {_CONTEXT_WINDOW}"):
        _run(
            monkeypatch,
            tmp_path / "embeddings.lmdb",
            [_write_dataset(tmp_path / "toolong.csv", [701])],
            "--max_length",
            str(_CONTEXT_WINDOW + 1),
        )

    assert embedder.calls == []


def test_documents_already_in_the_lmdb_are_not_re_embedded(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    embedder: _RecordingEmbedder,
) -> None:
    """Re-running over a dataset must resume, not redo. Embedding is the
    expensive half of this command; the LMDB already holds the answer."""
    output_path = tmp_path / "embeddings.lmdb"
    dataset = _write_dataset(tmp_path / "resume.csv", [801, 803])

    _run(monkeypatch, output_path, [dataset])
    assert embedder.embedded_ids == [801, 803]

    embedder.calls.clear()
    stored = _run(monkeypatch, output_path, [dataset])

    assert embedder.embedded_ids == []
    _assert_holds_embeddings_for(stored, [801, 803])


def test_force_regenerate_re_embeds_documents_already_in_the_lmdb(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    embedder: _RecordingEmbedder,
) -> None:
    """`-f` is the escape hatch from the skip above: it must re-embed, and
    overwrite the stored value rather than recompute and discard it."""
    output_path = tmp_path / "embeddings.lmdb"
    dataset = _write_dataset(tmp_path / "regen.csv", [901])

    _run(monkeypatch, output_path, [dataset])

    # A different fill proves the stored value was rewritten, not left behind.
    regenerated = _RecordingEmbedder(fill=7.0)
    monkeypatch.setattr(utils, "embed_document", regenerated)
    stored = _run(monkeypatch, output_path, [dataset], "-f")

    assert regenerated.embedded_ids == [901]
    assert (stored[b"901"] == 7.0).all()


def _write_dataset_with_empty_text(
    path: pathlib.Path, pubmed_ids: list[int]
) -> pathlib.Path:
    """The same layout as `_write_dataset`, with both text columns empty.

    Polars reads an empty cell as null, which `document_text` resolves to the
    empty string — the corpus's way of saying a document has neither an
    abstract nor a fulltext."""
    rows = "\n".join(
        f"{row},{pubmed_id},," for row, pubmed_id in enumerate(pubmed_ids)
    )
    path.write_text(f",pubmed_id,abstract,fulltext\n{rows}\n")
    return path


def test_a_document_with_no_text_is_not_embedded_or_stored(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    embedder: _RecordingEmbedder,
) -> None:
    """A row the corpus has no text for is described the same way by all three
    precompute commands: warned about and left out. Embedding it instead costs
    a forward pass and stores a 0-row matrix for a document the training splits
    drop, and that key then reads as already done on every resume."""
    output_path = tmp_path / "embeddings.lmdb"
    dataset = _write_dataset_with_empty_text(tmp_path / "empty.csv", [811])

    stored = _run(monkeypatch, output_path, [dataset])

    assert stored == {}
    assert embedder.embedded_ids == []


def test_force_regenerate_deletes_a_document_that_lost_its_text(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    embedder: _RecordingEmbedder,
) -> None:
    """`-f` makes the store agree with the corpus, in both directions: a
    document whose text is gone loses its stale entry rather than being
    re-embedded into one."""
    output_path = tmp_path / "embeddings.lmdb"
    dataset = _write_dataset(tmp_path / "lost.csv", [821, 823])
    _run(monkeypatch, output_path, [dataset])
    assert sorted(_stored_embeddings(output_path)) == [b"821", b"823"]

    embedder.calls.clear()
    _write_dataset_with_empty_text(dataset, [821])
    stored = _run(monkeypatch, output_path, [dataset], "-f")

    assert b"821" not in stored
    assert embedder.embedded_ids == []


class _BulkyEmbedder:
    """An embedding the store cannot shrink away.

    The exhaustion test needs a known number of bytes to reach the LMDB, and
    the codec turns `_RecordingEmbedder`'s constant matrix into a few hundred
    of them however large it is. Noise is what makes the size predictable: bf16
    randn compresses by about 1.4x and nothing more.
    """

    _COLUMNS = 768

    def __init__(self, rows: int) -> None:
        self._rows = rows
        self.embedded_ids: list[int] = []

    def __call__(self, doc: str, **_kwargs: object) -> torch.Tensor:
        self.embedded_ids.append(int(doc.split()[0]))
        generator = torch.Generator().manual_seed(len(self.embedded_ids))
        return torch.randn(self._rows, self._COLUMNS, generator=generator)


def _recorded_lmdb_open(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """The keyword arguments `main` opens the writable environment with.

    Reading `map_size` back off the store does not answer this: a read-only
    reopen brings its own reservation, so the only place the requested one is
    visible is the call itself.
    """
    opened: dict[str, object] = {}
    real_open = lmdb.open

    def recording_open(path: str, **kwargs: object) -> lmdb.Environment:
        opened.update(kwargs)
        return real_open(path, **kwargs)

    monkeypatch.setattr(precompute_embeddings.lmdb, "open", recording_open)
    return opened


@pytest.mark.usefixtures("embedder")
def test_the_reserved_map_size_covers_the_corpus_and_follows_the_flag(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole corpus measures 100.8 GiB through this store's codec, so a
    default below that runs out near the end of a full pass — after the GPU
    time is already spent. The reservation is virtual on Linux, so the headroom
    is free."""
    opened = _recorded_lmdb_open(monkeypatch)
    dataset = _write_dataset(tmp_path / "budget.csv", [1001])

    _run(monkeypatch, tmp_path / "default.lmdb", [dataset])
    assert opened["map_size"] >= 128 * 1024**3

    _run(monkeypatch, tmp_path / "flagged.lmdb", [dataset], "--map_size", "2")
    assert opened["map_size"] == 2 * 1024**3


@pytest.mark.parametrize(
    "map_size",
    ["0", "-1", "1e-12"],
    ids=["zero", "negative", "under-a-byte"],
)
def test_a_map_size_reserving_nothing_is_rejected_before_any_embedding(
    map_size: str,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    embedder: _RecordingEmbedder,
) -> None:
    """A reservation of zero bytes is not a small budget, it is no budget.

    `--max_length` beside it has always been checked; this one was not, and
    the complaint LMDB does have comes too late or not at all (see the
    characterizations below). The command must say so before it loads
    anything, because the alternative is hours of GPU time ending at the very
    first write.
    """
    output_path = tmp_path / "nomap.lmdb"

    # As the message renders it, which is the float argparse parsed.
    named = re.escape(str(float(map_size)))
    with pytest.raises(ValueError, match=f"--map_size .*got {named}"):
        _run(
            monkeypatch,
            output_path,
            [_write_dataset(tmp_path / "nomap.csv", [1201])],
            "--map_size",
            map_size,
        )

    assert embedder.calls == []
    assert embedder.loaded_tokenizers == []
    assert embedder.loaded_base_models == []
    assert not output_path.exists()


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--batch_size", "0"),
        ("--batch_size", "-1"),
        ("--stream_batch", "0"),
        ("--stream_batch", "-1"),
        ("--commit_every", "0"),
        ("--commit_every", "-1"),
    ],
    ids=[
        "batch_size-zero",
        "batch_size-negative",
        "stream_batch-zero",
        "stream_batch-negative",
        "commit_every-zero",
        "commit_every-negative",
    ],
)
def test_a_non_positive_batch_flag_is_rejected_before_any_embedding(
    flag: str,
    value: str,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    embedder: _RecordingEmbedder,
) -> None:
    """`--commit_every <= 0` is the dangerous one of the three: it makes
    `n_since >= commit_every` true on every write, so the writer commits once
    per document instead of once per batch, and the run still reports
    `Done.` with every document stored — a throughput cliff with no error
    anywhere. `--stream_batch` is worse in one specific way: a negative step
    makes `range(0, total, batch_size)` yield nothing, so the command loads
    the base model, iterates zero rows, writes zero documents and still
    reports `Done.`, looking exactly like a resume-safe empty pass.
    `--batch_size <= 0` is the one `embed_document` was already going to
    reject, but only after the weights were on the device. All three must be
    refused before any of that loading happens.
    """
    output_path = tmp_path / "nonpositive.lmdb"

    with pytest.raises(ValueError, match=f"{flag[2:]} .*got {value}"):
        _run(
            monkeypatch,
            output_path,
            [_write_dataset(tmp_path / "nonpositive.csv", [1301])],
            flag,
            value,
        )

    assert embedder.calls == []
    assert embedder.loaded_tokenizers == []
    assert embedder.loaded_base_models == []
    assert not output_path.exists()


def test_a_map_size_lmdb_rejects_is_refused_before_anything_loads(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    embedder: _RecordingEmbedder,
) -> None:
    """The reservations `map_size_bytes` cannot judge are LMDB's to refuse,
    and it has to get the chance while refusing is still cheap.

    A map past the address space is one such: the arithmetic is fine, and only
    the mmap can say it is not. That call used to sit below the base-model
    load, so the answer arrived with the weights already on the device.

    `lmdb.MemoryError` is the library's own class and not the builtin one it
    shadows, so a bare `MemoryError` here would match nothing.
    """
    with pytest.raises(lmdb.MemoryError):
        _run(
            monkeypatch,
            tmp_path / "unmappable.lmdb",
            [_write_dataset(tmp_path / "unmappable.csv", [1601])],
            "--map_size",
            str(_UNMAPPABLE_GIB),
        )

    assert embedder.loaded_tokenizers == []
    assert embedder.loaded_base_models == []


@pytest.mark.parametrize("map_size", [0.0, -1.0, 1e-12])
def test_the_map_size_rejection_describes_the_lmdb_it_stands_in_for(
    map_size: float,
) -> None:
    """The refusal explains itself, and the explanation has to hold for every
    value it refuses. It once said LMDB "reads a non-positive map_size as the
    size the store already has": true of zero, false of a negative value,
    which `lmdb.open` raises `OverflowError` on. The characterizations below
    are what this is asserting against.

    It then said `lmdb.open` refuses a negative one "only once the base model
    is loaded", which stopped being true when that call moved above the load.
    """
    with pytest.raises(ValueError) as raised:
        precompute_embeddings.map_size_bytes(map_size)

    message = str(raised.value)
    assert "non-positive" not in message
    assert "base model" not in message
    assert "1 MiB" in message
    assert "negative" in message


def test_lmdb_reads_a_map_size_of_zero_as_the_store_default(
    tmp_path: pathlib.Path,
) -> None:
    """Why the check above has to exist at all.

    Zero is not an error to LMDB — it means "keep whatever this store already
    has", which for a new one is its own 1 MiB default. Pinning the fact costs
    one empty environment, and if a future LMDB ever raised instead, this is
    the test that says the validator's reasoning has changed rather than
    leaving it standing on a claim nothing checks.
    """
    env = lmdb.open(str(tmp_path / "fresh.lmdb"), map_size=0)
    try:
        assert env.info()["map_size"] == 1024**2
    finally:
        env.close()


def test_lmdb_refuses_a_negative_map_size(tmp_path: pathlib.Path) -> None:
    """The other half of the same premise, and it does not behave like zero:
    the reservation reaches C as an unsigned size, so `lmdb.open` refuses it
    outright. That it refuses is not why the validator exists: the
    `OverflowError` names neither the flag nor the value that produced it, and
    the command that has both must be the one to say so."""
    with pytest.raises(OverflowError):
        lmdb.open(str(tmp_path / "negative.lmdb"), map_size=-1)


def test_lmdb_rounds_a_map_size_up_to_whole_pages(
    tmp_path: pathlib.Path,
) -> None:
    """Why the validator draws its line at one byte and claims nothing more.

    A one-byte reservation is not honoured literally: LMDB rounds it up to
    whole pages and reports two of them, which still holds no document. The
    floor is an arithmetic boundary, and a store that outgrows its map is what
    catches everything above it.

    The page is the host's, not a constant: 8192 here and 32768 on a
    16 KiB-page host, where a literal would fail with nothing wrong."""
    env = lmdb.open(str(tmp_path / "onebyte.lmdb"), map_size=1)
    try:
        assert env.info()["map_size"] == 2 * mmap.PAGESIZE
        with pytest.raises(lmdb.MapFullError), env.begin(write=True) as txn:
            txn.put(b"1301", b"0123456789")
    finally:
        env.close()


@pytest.mark.usefixtures("embedder")
def test_a_map_size_too_small_for_the_dataset_ends_the_run(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Running out of `map_size` used to be indistinguishable from finishing:
    the writer committed its prefix, the command logged `Done.`, and the next
    run read the missing documents as already embedded."""
    bulky = _BulkyEmbedder(rows=512)
    monkeypatch.setattr(utils, "embed_document", bulky)
    output_path = tmp_path / "toosmall.lmdb"
    dataset = _write_dataset(tmp_path / "bulky.csv", [1101, 1102, 1103])

    with pytest.raises(RuntimeError) as raised:
        _run(
            monkeypatch,
            output_path,
            [dataset],
            # 1.5 MiB against three embeddings of roughly half a MiB each,
            # committed one at a time so that two of them are already on disk
            # when the third runs out.
            "--map_size",
            str(1.5 / 1024),
            "--commit_every",
            "1",
        )

    message = str(raised.value)
    assert "map_size" in message
    named = [p for p in bulky.embedded_ids if str(p) in message]
    assert len(named) == 1, message

    # What is left is a prefix no reader can tell from a finished store, which
    # is what makes reporting success on it dangerous: the resume path reads
    # every key it holds as done and stops asking about the rest.
    stored = _stored_embeddings(output_path)
    assert stored
    assert str(named[0]).encode() not in stored


def test_a_map_size_too_small_for_one_document_is_refused_before_anything_loads(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    embedder: _RecordingEmbedder,
) -> None:
    """A `map_size` can clear `lmdb.open` and still be hopeless: LMDB accepts
    any reservation it can mmap, however small, and rounds it up to whole
    pages rather than raising. That used to surface only at the first real
    `put`, after the tokenizer and the base model had already loaded — this
    one is refused before either does, the same way an unmappable or
    non-positive `--map_size` already was.

    The fake config's window (512) and default hidden size (768) put one
    window's worth of bf16 activations at 768 KiB uncompressed; the map here
    is a quarter of that, well past what `record_provenance`'s own small
    write would have caught on its own.
    """
    output_path = tmp_path / "toosmall_for_any_document.lmdb"
    dataset = _write_dataset(tmp_path / "toosmall.csv", [1401])

    with pytest.raises(ValueError, match="map_size"):
        _run(
            monkeypatch,
            output_path,
            [dataset],
            "--map_size",
            str(0.25 / 1024),
        )

    assert embedder.calls == []
    assert embedder.loaded_tokenizers == []
    assert embedder.loaded_base_models == []


# Long enough that the embedding loop fills the queue it hands the writer and
# has to wait for room, which is where a dead writer turns into a hang.
_LONGER_THAN_THE_QUEUE = list(range(2001, 2201))


class _WriterDied(RuntimeError):
    """Injected: a failure inside the writer's loop it has no branch for."""


class _FailingProgressBar(tqdm.tqdm):  # type: ignore[type-arg]
    """A `Written` bar that fails the first time the writer counts a document.

    Subclassing the real bar rather than standing in for one keeps the writer's
    own contract intact, so what is being injected is a failure *inside* its
    loop and nothing else. The bar is only a convenient site: the writer has no
    branch for anything there except `MapFullError`.
    """

    error: BaseException

    def update(self, n: float | None = 1) -> bool | None:
        super().update(n)
        raise self.error


def _writer_dies_mid_document(
    monkeypatch: pytest.MonkeyPatch, _output_path: pathlib.Path
) -> type[BaseException]:
    """Kill the writer with a live transaction to unwind."""
    error = _WriterDied("the writer died holding an open transaction")
    real_bar = tqdm.tqdm

    def bar(**kwargs: Any) -> Any:
        if str(kwargs.get("desc", "")).strip() != "Written":
            return real_bar(**kwargs)
        failing = _FailingProgressBar(**kwargs)
        failing.error = error
        return failing

    monkeypatch.setattr(
        precompute_embeddings, "tqdm", types.SimpleNamespace(tqdm=bar)
    )
    return _WriterDied


def _writer_cannot_open_its_transaction(
    monkeypatch: pytest.MonkeyPatch, output_path: pathlib.Path
) -> type[BaseException]:
    """Kill the writer before it has a transaction at all.

    A store on a read-only mount is the everyday version of this. It matters
    separately because the writer's first act is to open a write transaction,
    which used to sit outside its own error handling entirely.
    """
    real_open = lmdb.open
    real_open(str(output_path)).close()

    def readonly_open(path: str, **kwargs: Any) -> lmdb.Environment:
        kwargs.setdefault("readonly", True)
        kwargs.setdefault("lock", False)
        return real_open(path, **kwargs)

    monkeypatch.setattr(precompute_embeddings.lmdb, "open", readonly_open)
    return lmdb.ReadonlyError


def _run_with_deadline(
    monkeypatch: pytest.MonkeyPatch,
    output_path: pathlib.Path,
    datasets: list[pathlib.Path],
    *flags: str,
    seconds: float = 30.0,
) -> BaseException | None:
    """Run `main` off the test thread and return whatever it raised.

    Calling `main` directly would turn the regression this guards — a command
    that never returns — into a pytest run that never returns, which is not a
    failing test. The deadline is what makes it one.
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "precompute-embeddings",
            "base-model",
            str(output_path),
            *(str(dataset) for dataset in datasets),
            *flags,
        ],
    )
    raised: list[BaseException] = []

    def run() -> None:
        try:
            precompute_embeddings.main()
        except BaseException as exc:
            raised.append(exc)

    runner = threading.Thread(target=run, daemon=True)
    runner.start()
    runner.join(timeout=seconds)
    assert not runner.is_alive(), (
        f"the command was still running after {seconds}s: the writer thread "
        f"is gone and the embedding loop is still waiting to hand it work"
    )
    return raised[0] if raised else None


@pytest.mark.parametrize(
    "inject",
    [_writer_dies_mid_document, _writer_cannot_open_its_transaction],
    ids=["mid-document", "before-the-first-transaction"],
)
@pytest.mark.usefixtures("embedder")
def test_a_writer_that_dies_ends_the_run_instead_of_hanging(
    inject: Callable[[pytest.MonkeyPatch, pathlib.Path], type[BaseException]],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The writer is the only consumer of a bounded queue, so a thread that
    ends without saying so leaves the embedding loop waiting on a consumer that
    no longer exists: hours of GPU time, two live progress bars and no error.
    Only `MapFullError` was ever announced; everything else was a hang, and a
    hang is worse than a crash because nothing on the other end can tell.
    """
    monkeypatch.setattr(precompute_embeddings, "MAX_BACKLOG", 2)
    output_path = tmp_path / "dying.lmdb"
    expected = inject(monkeypatch, output_path)
    dataset = _write_dataset(tmp_path / "dying.csv", _LONGER_THAN_THE_QUEUE)

    raised = _run_with_deadline(monkeypatch, output_path, [dataset])

    assert isinstance(raised, expected), (
        f"the writer's failure has to reach the caller rather than be "
        f"swallowed into a `Done.`; got {raised!r}"
    )


def test_the_lmdb_is_closed_when_main_raises_mid_dataset(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    embedder: _RecordingEmbedder,
) -> None:
    """`env.close()` used to sit only on the success path, so anything that
    raised out of the dataset loop — `embed_document` OOMing, a corrupt row —
    left the environment open with its lock file held. That is cheap for a
    one-shot CLI, since process exit cleans it up, but the test suite (and any
    future retry wrapper) calls `main` more than once in a process, so the
    leak is checked directly here rather than through process exit.
    """
    output_path = tmp_path / "leaky.lmdb"
    opened: list[lmdb.Environment] = []
    real_open = lmdb.open

    def recording_open(path: str, **kwargs: Any) -> lmdb.Environment:
        env = real_open(path, **kwargs)
        opened.append(env)
        return env

    monkeypatch.setattr(precompute_embeddings.lmdb, "open", recording_open)

    class _EmbedBlewUp(RuntimeError):
        pass

    def failing_embed(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise _EmbedBlewUp("embed_document blew up")

    monkeypatch.setattr(utils, "embed_document", failing_embed)

    with pytest.raises(_EmbedBlewUp):
        _run(
            monkeypatch,
            output_path,
            [_write_dataset(tmp_path / "boom.csv", [1701])],
        )

    (env,) = opened
    with pytest.raises(lmdb.Error):
        env.stat()


def _provenance(output_path: pathlib.Path) -> StoreProvenance | None:
    env = lmdb.open(str(output_path), readonly=True, lock=False)
    try:
        return read_provenance(env)
    finally:
        env.close()


@pytest.mark.usefixtures("embedder")
def test_the_store_records_the_model_window_and_stride_that_wrote_it(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Everything a reader needs to tell this store from another one. The blob
    header carries rows and columns, which are equal between encoders of the
    same hidden size, so without this the only mistake the geometry cannot
    catch is also the only one nothing else catches."""
    output_path = tmp_path / "embeddings.lmdb"

    _run(
        monkeypatch,
        output_path,
        [_write_dataset(tmp_path / "stamp.csv", [1301])],
        "--max_length",
        "128",
    )

    assert _provenance(output_path) == StoreProvenance(
        base_model="base-model",
        max_length=128,
        stride=precompute_embeddings.STRIDE,
    )


def test_adding_to_a_store_another_model_wrote_is_refused(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    embedder: _RecordingEmbedder,
) -> None:
    """A resume onto somebody else's store is how the two representation
    spaces get into one LMDB in the first place, and it is the last moment
    anything can tell them apart: once written, the matrices are the same
    shape and carry no mark.

    The refusal is also free: the store says who wrote it, so the run that
    cannot join it ends before the model it would have joined it with is
    loaded."""
    output_path = tmp_path / "embeddings.lmdb"
    dataset = _write_dataset(tmp_path / "mixed.csv", [1401])
    _run(monkeypatch, output_path, [dataset])

    embedder.calls.clear()
    embedder.loaded_tokenizers.clear()
    embedder.loaded_base_models.clear()
    monkeypatch.setattr(
        "sys.argv",
        [
            "precompute-embeddings",
            "another-base-model",
            str(output_path),
            str(_write_dataset(tmp_path / "second.csv", [1402])),
        ],
    )
    with pytest.raises(ValueError, match="was written by base-model"):
        precompute_embeddings.main()

    assert embedder.calls == []
    assert embedder.loaded_tokenizers == []
    assert embedder.loaded_base_models == []
    assert sorted(_stored_embeddings(output_path)) == [b"1401"]


def test_adding_to_a_store_that_names_no_model_is_refused(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    embedder: _RecordingEmbedder,
) -> None:
    """A store from before the record existed cannot be shown to hold this
    run's activations, and a resume that assumed it did would produce exactly
    the mixture the record exists to prevent."""
    output_path = tmp_path / "unstamped.lmdb"
    with lmdb.open(str(output_path), map_size=2**20) as env:
        with env.begin(write=True) as transaction:
            transaction.put(b"1501", tensor_to_bytes(torch.rand(2, 4)))

    with pytest.raises(ValueError, match="does not record which model"):
        _run(
            monkeypatch,
            output_path,
            [_write_dataset(tmp_path / "unstamped.csv", [1502])],
        )

    assert embedder.calls == []
    assert embedder.loaded_tokenizers == []
    assert embedder.loaded_base_models == []


@pytest.mark.usefixtures("embedder")
def test_a_resume_by_the_model_that_wrote_the_store_carries_on(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The check must not cost the run it is meant to protect: the same model
    over the same store is the resume path this command is built around."""
    output_path = tmp_path / "embeddings.lmdb"
    _run(monkeypatch, output_path, [_write_dataset(tmp_path / "a.csv", [161])])

    stored = _run(
        monkeypatch,
        output_path,
        [_write_dataset(tmp_path / "b.csv", [162])],
    )

    _assert_holds_embeddings_for(stored, [161, 162])
