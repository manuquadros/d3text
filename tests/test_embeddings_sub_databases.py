"""One LMDB env per base model: the aggregated rows and every layer boundary
are named sub-databases of it, stamped once at the env level.

A boundary is named by its unfrozen top-layer count, so two boundaries share
one env and one provenance record; an env in the older one-cut-per-env layout
is refused rather than read, since its rows sit where no sub-database is.
"""

import json

import lmdb
import pytest
import torch
from d3text.embeddings_store import (
    EmbeddingsStore,
    LayerBoundaryStore,
    ProvenanceError,
    StoreProvenance,
    tensor_to_bytes,
    windowed_tensor_to_bytes,
)

BASE_MODEL = "michiyasunaga/BioLinkBERT-base"
MAX_LENGTH = 512
PROVENANCE = StoreProvenance(
    base_model=BASE_MODEL, max_length=MAX_LENGTH, stride=20
)


def test_two_boundaries_share_one_env_and_each_reads_its_own(tmp_path):
    """A prefix read at the wrong boundary is a valid tensor of the right
    shape, so nothing downstream would notice; the sub-database the store
    opens is the only thing keeping the two apart."""
    path = tmp_path / "embeddings"
    one = LayerBoundaryStore.create(path, PROVENANCE, unfrozen_top_layers=1)
    three = LayerBoundaryStore.create(path, PROVENANCE, unfrozen_top_layers=3)
    one.put(100, torch.full((2, 4, 8), 1.0))
    three.put(100, torch.full((2, 4, 8), 3.0))
    one.close()
    three.close()

    for unfrozen in (1, 3):
        store = LayerBoundaryStore(
            path, BASE_MODEL, unfrozen_top_layers=unfrozen, max_length=512
        )
        stored = store.get(100, expected_windows=2)
        store.close()
        assert stored is not None
        assert torch.all(stored == unfrozen)


def test_a_boundary_is_added_to_an_env_that_already_holds_others(tmp_path):
    """Adding a depth later is one more sub-database, not a new store: the
    aggregated rows and the first boundary survive it unchanged."""
    path = tmp_path / "embeddings"
    aggregated = EmbeddingsStore.create(path, PROVENANCE)
    aggregated.put(100, torch.full((12, 8), 5.0))
    aggregated.close()
    first = LayerBoundaryStore.create(path, PROVENANCE, unfrozen_top_layers=2)
    first.put(100, torch.full((1, 4, 8), 2.0))
    first.close()

    added = LayerBoundaryStore.create(path, PROVENANCE, unfrozen_top_layers=4)
    added.put(100, torch.full((1, 4, 8), 4.0))
    added.close()

    reader = EmbeddingsStore(path, BASE_MODEL, MAX_LENGTH)
    boundary = LayerBoundaryStore(path, BASE_MODEL, 2, MAX_LENGTH)
    try:
        row = reader.get(100, expected_tokens=12)
        prefix = boundary.get(100, expected_windows=1)
    finally:
        reader.close()
        boundary.close()
    assert row is not None and torch.all(row == 5.0)
    assert prefix is not None and torch.all(prefix == 2.0)


def test_a_boundary_the_env_does_not_hold_is_not_opened_read_only(tmp_path):
    """Read-only, an absent boundary is a miss for every document, and the
    caller must be told so rather than handed an empty reader."""
    path = tmp_path / "embeddings"
    LayerBoundaryStore.create(path, PROVENANCE, unfrozen_top_layers=2).close()

    with pytest.raises(lmdb.NotFoundError):
        LayerBoundaryStore(path, BASE_MODEL, 3, MAX_LENGTH)


def test_an_aggregated_env_in_the_old_layout_is_refused(tmp_path):
    """The old layout kept its rows in the main database under a format-1
    stamp; read as the new one it would look like an env with no aggregated
    rows at all, so it is refused and named as needing a rebuild."""
    path = tmp_path / "old"
    record = {"format": 1, "base_model": BASE_MODEL}
    record |= {"max_length": MAX_LENGTH, "stride": 20}
    with lmdb.open(str(path), map_size=2**20) as env:
        with env.begin(write=True) as transaction:
            transaction.put(b"\x00provenance", json.dumps(record).encode())
            transaction.put(b"100", tensor_to_bytes(torch.rand(12, 8)))

    with pytest.raises(ProvenanceError, match="[Rr]ebuild"):
        EmbeddingsStore(path, BASE_MODEL, MAX_LENGTH)


def test_a_layer_boundary_env_in_the_old_layout_is_refused(tmp_path):
    """The old layer-boundary env carries only its own stamp, which records
    the frozen count rather than the unfrozen one a sub-database is named
    by; it is refused, not silently read as unstamped."""
    path = tmp_path / "old-layer"
    record = {"format": 1, "base_model": BASE_MODEL, "frozen_layers": 8}
    record |= {"max_length": MAX_LENGTH, "stride": 20}
    with lmdb.open(str(path), map_size=2**20) as env:
        with env.begin(write=True) as transaction:
            transaction.put(
                b"\x00layer_provenance", json.dumps(record).encode()
            )
            transaction.put(
                b"100", windowed_tensor_to_bytes(torch.rand(1, 4, 8))
            )

    with pytest.raises(ProvenanceError, match="[Rr]ebuild"):
        LayerBoundaryStore(path, BASE_MODEL, 4, MAX_LENGTH)
