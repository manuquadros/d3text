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

**A store that stopped early must not look like a finished one.** The LMDB is
opened with a fixed ``map_size``, and a pass needing more than that used to
commit the prefix it had and report ``Done.`` like any other run. The resume
path then read every truncated-in key as already embedded, so a rerun skipped
straight to the same wall. The last family here pins the two halves of that: the
reservation is large enough for the corpus and adjustable, and running out of it
ends the run.
"""

import pathlib
import types

import lmdb
import numpy as np
import pytest
import torch
import transformers
from d3text import utils
from d3text.cli import precompute_embeddings
from d3text.embeddings_store import bytes_to_tensor

_EMBEDDING_SHAPE = (2, 4)
_CONTEXT_WINDOW = 512

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
    """

    def __init__(self, fill: float | None = None) -> None:
        self.calls: list[types.SimpleNamespace] = []
        self._fill = fill

    def __call__(self, doc: str, **kwargs: object) -> torch.Tensor:
        pubmed_id = int(doc.split()[0])
        self.calls.append(types.SimpleNamespace(pubmed_id=pubmed_id, **kwargs))
        fill = pubmed_id if self._fill is None else self._fill
        return torch.full(_EMBEDDING_SHAPE, float(fill))

    @property
    def embedded_ids(self) -> list[int]:
        return [call.pubmed_id for call in self.calls]


class _FakeBaseModel:
    """Stands in for the frozen transformer, which `_RecordingEmbedder` never
    calls. `main` moves it to a device, puts it in eval mode, and reads the
    context window off its config."""

    config = transformers.BertConfig(
        max_position_embeddings=_CONTEXT_WINDOW,
        name_or_path="fake-base-model",
    )

    def to(self, _device: torch.device) -> "_FakeBaseModel":
        return self

    def eval(self) -> "_FakeBaseModel":
        return self


@pytest.fixture
def embedder(monkeypatch: pytest.MonkeyPatch) -> _RecordingEmbedder:
    """Run `main` with no network, no tokenizer, and no transformer.

    The stub tokenizer carries the sentinel `model_max_length` that the real
    default base model reports, so any attempt to derive the window size from
    the tokenizer shows up in the recorded `max_len`.
    """
    recorder = _RecordingEmbedder()
    monkeypatch.setattr(
        utils,
        "load_fast_tokenizer",
        lambda _base_model: types.SimpleNamespace(
            model_max_length=_NO_LIMIT_DECLARED
        ),
    )
    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        lambda *_args, **_kwargs: _FakeBaseModel(),
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
            return {
                key: bytes_to_tensor(value).float().numpy()
                for key, value in txn.cursor().iternext()
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
        assert (embedding == stamp).all(), (
            f"{pubmed_id} was stored under the wrong key"
        )


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
