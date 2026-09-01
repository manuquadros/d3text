"""`precompute-embeddings` embeds what it was asked to, and stores it all.

Everything here is about the bookkeeping around the embedding, which is stubbed
out: that every document reaches the LMDB, that every flag reaches the
embedder, that a run which stopped early cannot look finished, that the store
records what wrote it, and that a dead writer ends the run instead of hanging
on a bounded queue.
"""

import mmap
import pathlib
import queue
import re
import shutil
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
    between the row loop, the compression pool and the writer fails too.
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

    The tokenizer and base model stubs record that they were loaded, so a test
    can assert a run ended before either. The config stub does not: reading a
    config.json is what a run may do before it commits to the weights.
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

    bf16 carries 8 significant bits, so 801 and 802 both come back as 800 —
    harmless for activations and fatal for a mix-up detector.
    """
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

    `tokenizer.model_max_length` is a ~1e30 sentinel for the default base
    model, and the tokenizer pads *to* the window.
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

    The codec turns a constant matrix into a few hundred bytes however large it
    is; bf16 noise compresses by about 1.4x and nothing more.
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

    A read-only reopen brings its own reservation, so the requested one is
    visible only in the call itself.
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

    The complaint LMDB does have comes too late or not at all, and the
    alternative is hours of GPU time ending at the very first write.
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
    """All three count flags are refused before anything loads.

    A negative `--stream_batch` iterates zero rows and still reports `Done.`;
    `--commit_every <= 0` commits once per document, a throughput cliff with no
    error; `--batch_size <= 0` would only fail once the weights were on the
    device.
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
    """A map past the address space is LMDB's to refuse, and cheaply.

    `lmdb.MemoryError` is the library's own class, not the builtin it shadows,
    so a bare `MemoryError` would match nothing.
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
    """The refusal's explanation holds for every value it refuses.

    It once claimed LMDB reads any non-positive `map_size` as the store's
    current size, which is true of zero and false of a negative value.
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
    """Zero means "keep whatever this store has", which for a new one is 1 MiB.

    Pinned so that a future LMDB raising instead reads as the validator's
    reasoning having changed, rather than standing on a claim nothing checks.
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

    The page size is the host's — 8192 here, 32768 on a 16 KiB-page host — so a
    literal would fail with nothing wrong.
    """
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
    """A map can clear `lmdb.open` and still be hopeless.

    The probe is one window of bf16 activations, well past what the provenance
    record's own small write would have caught.
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

    Subclassing the real bar keeps the writer's contract intact, so what is
    injected is a failure *inside* its loop and nothing else.
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

    A store on a read-only mount is the everyday version: the writer's first
    act is to open a write transaction, which used to sit outside its error
    handling.
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

    Calling it directly would turn the regression this guards — a command that
    never returns — into a pytest run that never returns.
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
    """A writer ending silently leaves the producer waiting forever.

    Only `MapFullError` was ever announced; everything else was a hang, which
    is worse than a crash because nothing on the other end can tell.
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


# One value, fixed rather than random, so the page layout the search below
# lands on is the same on every run.
_FILLER = b"\xa5" * 180


def _commits(path: pathlib.Path, documents: int) -> bool:
    shutil.rmtree(path, ignore_errors=True)
    env = lmdb.open(str(path), map_size=1 << 20)
    try:
        with env.begin(write=True) as txn:
            for document in range(documents):
                txn.put(f"{document:06d}".encode(), _FILLER)
    except lmdb.MapFullError:
        return False
    finally:
        env.close()
    return True


def _store_with_no_room_left(path: pathlib.Path) -> lmdb.Environment:
    """A real LMDB filled to its last page, where even a delete runs out.

    A store built by writing until a `put` fails is no use: the pages its
    earlier commits freed are exactly what the delete then spends.
    """
    low, high = 1, 1 << 15
    while low < high:
        middle = (low + high + 1) // 2
        if _commits(path, middle):
            low = middle
        else:
            high = middle - 1
    assert _commits(path, low), "no transaction at all fits in the map"
    return lmdb.open(str(path), map_size=1 << 20)


class _SnapshotBar(tqdm.tqdm):  # type: ignore[type-arg]
    """A `Written` bar that reads the store's committed snapshot per document.

    `commit_every` leaves no trace in the finished store, so mid-run is the
    only moment the writer's accounting can be observed.
    """

    env: lmdb.Environment
    watched: bytes
    seen: list[bool]

    def update(self, n: float | None = 1) -> bool | None:
        with self.env.begin() as txn:
            self.seen.append(txn.get(self.watched) is not None)
        return super().update(n)


def _snapshot_bar(env: lmdb.Environment, watched: bytes) -> _SnapshotBar:
    bar = _SnapshotBar(disable=True)
    bar.env = env
    bar.watched = watched
    bar.seen = []
    return bar


def _drain(
    env: lmdb.Environment,
    items: list[tuple[bytes, bytes | None]],
    commit_every: int,
    pbar: tqdm.tqdm | None = None,
) -> precompute_embeddings.WriterState:
    """Run the writer over `items` here, and return what it recorded.

    Everything it needs in order to stop is in the queue before it starts, so
    these assertions need not race a thread.
    """
    in_q: queue.Queue[tuple[bytes, bytes | None]] = queue.Queue()
    for item in items:
        in_q.put(item)
    stop_evt = threading.Event()
    stop_evt.set()
    state = precompute_embeddings.WriterState()
    precompute_embeddings.writer_thread(
        env,
        in_q,
        stop_evt,
        commit_every,
        tqdm.tqdm(disable=True) if pbar is None else pbar,
        state,
    )
    return state


def test_a_map_full_on_a_delete_does_not_report_a_write(
    tmp_path: pathlib.Path,
) -> None:
    """The queue carries deletes as well as puts, and a delete has to grow the
    map the same way a put does, so it reaches the map-full branch just as
    readily. Reporting it as a write names an operation the run never
    attempted, in the one message someone reads while diagnosing a store that
    would not grow."""
    env = _store_with_no_room_left(tmp_path / "brim.lmdb")
    try:
        state = _drain(env, [(b"000000", None)], commit_every=100)
    finally:
        env.close()

    assert isinstance(state.failure, precompute_embeddings.StoreFullError)
    message = str(state.failure)
    assert "000000" in message
    assert "writing document" not in message, message
    assert "deleting" in message, message


def test_a_map_full_on_a_put_still_reports_a_write(
    tmp_path: pathlib.Path,
) -> None:
    env = _store_with_no_room_left(tmp_path / "brim.lmdb")
    try:
        state = _drain(env, [(b"000000", _FILLER)], commit_every=100)
    finally:
        env.close()

    assert "writing document 000000" in str(state.failure)


def test_a_delete_of_a_key_the_store_lacks_does_not_close_the_batch(
    tmp_path: pathlib.Path,
) -> None:
    """An item that wrote nothing must not spend one of `commit_every`'s slots.

    `-f` drops stale entries without reading the store first, so most of those
    deletes remove nothing at all.
    """
    env = lmdb.open(str(tmp_path / "absent.lmdb"), map_size=1 << 20)
    bar = _snapshot_bar(env, watched=b"4101")
    try:
        state = _drain(
            env,
            [
                (b"4101", _FILLER),
                (b"4102", None),
                (b"4103", None),
                (b"4104", _FILLER),
            ],
            commit_every=3,
            pbar=bar,
        )
    finally:
        env.close()

    assert state.failure is None
    assert bar.seen == [False, False], (
        f"nothing between the two stored documents changed the transaction, "
        f"so the batch of three should not have closed; got {bar.seen}"
    )


def test_a_delete_that_removes_a_key_closes_the_batch(
    tmp_path: pathlib.Path,
) -> None:
    """The other half, and the guard against over-correcting: a delete that
    actually removes something is work the transaction is holding, so it has
    to keep counting."""
    env = lmdb.open(str(tmp_path / "present.lmdb"), map_size=1 << 20)
    with env.begin(write=True) as txn:
        txn.put(b"4202", _FILLER)
    bar = _snapshot_bar(env, watched=b"4201")
    try:
        state = _drain(
            env,
            [(b"4201", _FILLER), (b"4202", None), (b"4203", _FILLER)],
            commit_every=2,
            pbar=bar,
        )
    finally:
        env.close()

    assert state.failure is None
    assert bar.seen == [False, True], (
        f"the delete removed a key, closing the batch of two and committing "
        f"the document before it; got {bar.seen}"
    )


def test_the_lmdb_is_closed_when_main_raises_mid_dataset(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    embedder: _RecordingEmbedder,
) -> None:
    """Anything raising out of the dataset loop still closes the environment.

    Cheap to leak for a one-shot CLI, but the suite calls `main` more than once
    in a process, so the lock file is checked directly rather than through
    exit.
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
    """A resume onto another model's store is the last chance to tell.

    Once written, the matrices are the same shape and carry no mark. The
    refusal is free: the store says who wrote it, so the run ends before the
    model loads.
    """
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
