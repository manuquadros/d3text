"""`LayerBoundaryStore` reading back what `precompute-embeddings` wrote.

Mirrors `test_embeddings_store_reader.py`; the windowed codec's round trip
is pinned in `test_embeddings_store.py`.
"""

import pathlib
import struct

import lmdb
import pytest
import torch
from d3text import embeddings_store
from d3text.embeddings_store import (
    MAX_SUB_DATABASES,
    LayerBoundaryStore,
    ProvenanceError,
    StoreProvenance,
    boundary_name,
    bytes_to_windowed_tensor,
    windowed_tensor_to_bytes,
    write_provenance,
)

BASE_MODEL = "michiyasunaga/BioLinkBERT-base"
UNFROZEN = 6
MAX_LENGTH = 512
PROVENANCE = StoreProvenance(
    base_model=BASE_MODEL, max_length=MAX_LENGTH, stride=20
)


def _write_store(
    path: pathlib.Path,
    provenance: StoreProvenance | None = PROVENANCE,
    documents: dict[int, torch.Tensor] | None = None,
    unfrozen: int = UNFROZEN,
) -> pathlib.Path:
    """An LMDB stamped with `provenance`, holding windowed `documents` in
    the sub-database of the boundary `unfrozen` names."""
    env = lmdb.open(str(path), map_size=8 * 1024**2, max_dbs=MAX_SUB_DATABASES)
    if provenance is not None:
        write_provenance(env, provenance)
    db = env.open_db(boundary_name(unfrozen).encode())
    with env.begin(write=True, db=db) as transaction:
        for pubmed_id, tensor in (documents or {}).items():
            transaction.put(
                str(pubmed_id).encode(), windowed_tensor_to_bytes(tensor)
            )
    env.close()

    return path


@pytest.fixture
def store_path(tmp_path):
    """An LMDB holding one 3-window document under pubmed id 100."""
    return _write_store(
        tmp_path / "layer-boundary",
        documents={100: torch.rand(3, 12, 8)},
    )


def test_an_env_holding_only_another_boundary_is_refused(tmp_path):
    """The failure this store exists for on top of `EmbeddingsStore`: a
    prefix cached at another boundary is a valid tensor of the right shape
    for the wrong layer, so nothing downstream would fail loudly if the
    boundary were not what selects the rows."""
    path = _write_store(
        tmp_path / "other-boundary",
        documents={100: torch.rand(3, 12, 8)},
        unfrozen=3,
    )

    with pytest.raises(lmdb.NotFoundError):
        LayerBoundaryStore(
            path, BASE_MODEL, unfrozen_top_layers=UNFROZEN, max_length=512
        )


def test_a_store_that_does_not_say_who_wrote_it_is_refused(tmp_path):
    """An unstamped store is not a store known to be right; it is a store
    nothing attributes to anything."""
    path = _write_store(
        tmp_path / "unstamped",
        provenance=None,
        documents={100: torch.rand(3, 12, 8)},
    )

    with pytest.raises(ProvenanceError, match="does not record which model"):
        LayerBoundaryStore(
            path,
            BASE_MODEL,
            unfrozen_top_layers=UNFROZEN,
            max_length=MAX_LENGTH,
        )


def test_a_store_stamped_at_another_window_is_refused(tmp_path):
    """`get`'s window-count check cannot catch this: the count agrees
    whatever width each window was embedded at. The requested window is
    neither the stamped one nor `MAX_LENGTH`, so the refusal must come from
    the argument, not a hardcoded constant.
    """
    path = _write_store(
        tmp_path / "other-window",
        provenance=StoreProvenance(
            base_model=BASE_MODEL, max_length=128, stride=20
        ),
        documents={100: torch.rand(1, 128, 8)},
    )

    with pytest.raises(ProvenanceError, match="window 128"):
        LayerBoundaryStore(
            path, BASE_MODEL, unfrozen_top_layers=UNFROZEN, max_length=256
        )


def test_a_window_count_that_disagrees_with_the_encodings_is_refused(
    store_path,
):
    """The stored document has 3 windows; a document whose encodings imply
    4 was built from different text than the store, and must be run live
    rather than handed the wrong prefix."""
    store = LayerBoundaryStore(store_path, BASE_MODEL, UNFROZEN, MAX_LENGTH)

    assert store.get(100, expected_windows=4) is None
    assert (store.hits, store.mismatches) == (0, 1)


def test_the_window_mismatch_is_warned_about_once(store_path, caplog):
    """Once, not once per document, for the same reason `EmbeddingsStore`
    limits its own mismatch warning to one line."""
    store = LayerBoundaryStore(store_path, BASE_MODEL, UNFROZEN, MAX_LENGTH)

    with caplog.at_level("WARNING"):
        for _ in range(3):
            store.get(100, expected_windows=4)

    warnings = [
        record for record in caplog.records if record.levelname == "WARNING"
    ]
    assert len(warnings) == 1
    assert store.mismatches == 3


def test_the_env_is_closed_when_provenance_error_is_raised_in_init(
    tmp_path, monkeypatch
):
    """`__init__` opens the environment before it knows the store is
    attributable; a `ProvenanceError` must not leak that handle. A closed
    `lmdb.Environment` raises on any further use, so that is the probe."""
    path = _write_store(
        tmp_path / "unstamped",
        provenance=None,
        documents={100: torch.rand(3, 12, 8)},
    )
    opened: dict[str, lmdb.Environment] = {}
    real_open = lmdb.open

    def spy_open(*args: object, **kwargs: object) -> lmdb.Environment:
        env = real_open(*args, **kwargs)
        opened["env"] = env
        return env

    monkeypatch.setattr(embeddings_store.lmdb, "open", spy_open)

    with pytest.raises(ProvenanceError):
        LayerBoundaryStore(
            path,
            BASE_MODEL,
            unfrozen_top_layers=UNFROZEN,
            max_length=MAX_LENGTH,
        )

    with pytest.raises(lmdb.Error):
        opened["env"].stat()


def test_bytes_to_windowed_tensor_refuses_the_wrong_magic():
    """A blob with the `D3WL` header's exact length and a valid compressed
    body, but someone else's magic. Without the check, `_unpack` reads the
    header and shape as written and `_decompress` succeeds on the
    untouched body, so this would decode silently into the tensor it was
    written as rather than raise; the magic is what turns that into an
    error instead."""
    packed = windowed_tensor_to_bytes(torch.rand(3, 4, 5))
    wrong_magic = b"XXXX" + packed[4:]

    with pytest.raises(ValueError, match="not a layer-boundary store blob"):
        bytes_to_windowed_tensor(wrong_magic)


def test_a_future_windowed_format_version_is_refused():
    packed = windowed_tensor_to_bytes(torch.ones(2, 3, 4))
    bumped = struct.pack("<4sB", b"D3WL", 2) + packed[5:]

    with pytest.raises(ValueError, match="version 2 is not readable"):
        bytes_to_windowed_tensor(bumped)


def test_the_store_keeps_readahead_on(store_path):
    """As for `EmbeddingsStore`: a `get` reads one multi-megabyte run of
    windows whole, which `MADV_RANDOM` would fault in a page at a time."""
    store = LayerBoundaryStore(store_path, BASE_MODEL, UNFROZEN, MAX_LENGTH)

    assert store.env.flags()["readahead"]


def test_a_created_store_reads_back_what_was_put_into_it(tmp_path):
    """`create` stamps before anything is written, so the store a training
    run builds is one the next run, and `precompute-embeddings`, will
    attribute."""
    prefix = torch.rand(3, 12, 8)
    path = tmp_path / "a" / "layer-boundary"
    store = LayerBoundaryStore.create(path, PROVENANCE, UNFROZEN)
    store.put(100, prefix)
    store.close()

    reopened = LayerBoundaryStore(path, BASE_MODEL, UNFROZEN, MAX_LENGTH)
    stored = reopened.get(100, expected_windows=3)

    assert not reopened.writable
    assert stored is not None
    assert torch.equal(stored, prefix.bfloat16())


def test_a_read_only_store_refuses_a_put(store_path):
    store = LayerBoundaryStore(store_path, BASE_MODEL, UNFROZEN, MAX_LENGTH)
    with pytest.raises(RuntimeError, match="read-only"):
        store.put(1, torch.rand(1, 12, 8))


def test_a_failed_write_stops_the_writing_not_the_run(tmp_path, caplog):
    """A full disk or map must cost the run its cache, not its training."""
    store = LayerBoundaryStore.create(
        tmp_path / "layer-boundary", PROVENANCE, UNFROZEN
    )
    store.env.set_mapsize(64 * 1024)

    store.put(100, torch.rand(4, 512, 64))

    assert not store.writable
    assert store.written == 0
    assert "stops growing" in caplog.text
    store.close()
