"""Every document handed to ``precompute-embeddings`` must reach the LMDB.

The command embeds each document, submits the compression to a thread pool, and
flushes *completed* jobs to the writer only when the in-flight backlog grows
past ``MAX_BACKLOG``. That flush is what keeps the backlog under the threshold,
so when the rows run out there are always jobs still in flight and never enough
of them to trip the threshold again — which is why the drain after the row loop
cannot be guarded by the same condition. These tests pin the drain: the failure
it prevents is silent (the command reports ``Done.`` either way), and at the
extreme — a dataset shorter than ``MAX_BACKLOG`` — nothing at all gets written.

The embedding itself is stubbed out: what is under test is the bookkeeping
between the row loop, the compression pool, and the writer thread, not the
transformer. Each stub embedding is filled with its own pubmed id, so the tests
also catch a key/value mix-up in the drain.
"""

import pathlib

import blosc2
import lmdb
import numpy as np
import pytest
import torch
import transformers
from d3text import utils
from d3text.cli import precompute_embeddings

_EMBEDDING_SHAPE = (2, 4)


def _fake_embed_document(doc: str, **_kwargs: object) -> torch.Tensor:
    """Return an embedding stamped with the document's pubmed id.

    `stream_rows` feeds `main` the abstract and fulltext joined by a newline,
    so the id written into both by `_write_dataset` is the first token.
    """
    pubmed_id = int(doc.split()[0])
    return torch.full(_EMBEDDING_SHAPE, float(pubmed_id))


class _FakeBaseModel:
    """Stands in for the frozen transformer, which `_fake_embed_document`
    never calls. `main` only moves it to a device and puts it in eval mode."""

    def to(self, _device: torch.device) -> "_FakeBaseModel":
        return self

    def eval(self) -> "_FakeBaseModel":
        return self


@pytest.fixture
def offline_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run `main` with no network, no tokenizer, and no transformer."""
    monkeypatch.setattr(utils, "load_fast_tokenizer", lambda _base_model: None)
    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        lambda *_args, **_kwargs: _FakeBaseModel(),
    )
    monkeypatch.setattr(utils, "embed_document", _fake_embed_document)
    # `main` setdefaults this; keep the mutation out of the wider test session.
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


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
    env = lmdb.open(str(output_path), readonly=True, lock=False)
    try:
        with env.begin() as txn:
            return {
                key: blosc2.unpack_array(value)
                for key, value in txn.cursor().iternext()
            }
    finally:
        env.close()


def _run(
    monkeypatch: pytest.MonkeyPatch,
    output_path: pathlib.Path,
    datasets: list[pathlib.Path],
) -> dict[bytes, np.ndarray]:
    monkeypatch.setattr(
        "sys.argv",
        [
            "precompute-embeddings",
            "base-model",
            str(output_path),
            *(str(dataset) for dataset in datasets),
        ],
    )
    precompute_embeddings.main()
    return _stored_embeddings(output_path)


def _assert_holds_embeddings_for(
    stored: dict[bytes, np.ndarray], pubmed_ids: list[int]
) -> None:
    assert sorted(stored) == sorted(str(p).encode() for p in pubmed_ids)
    for pubmed_id in pubmed_ids:
        embedding = stored[str(pubmed_id).encode()]
        assert embedding.shape == _EMBEDDING_SHAPE
        assert (embedding == pubmed_id).all(), (
            f"{pubmed_id} was stored under the wrong key"
        )


@pytest.mark.usefixtures("offline_pipeline")
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


@pytest.mark.usefixtures("offline_pipeline")
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


@pytest.mark.usefixtures("offline_pipeline")
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
