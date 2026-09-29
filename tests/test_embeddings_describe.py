"""`describe` reports each database of an env from its frame headers alone."""

import json

import lmdb
import torch
from d3text.embeddings_store import (
    AGGREGATED,
    MAIN_DATABASE,
    MAX_SUB_DATABASES,
    EmbeddingsStore,
    LayerBoundaryStore,
    StoreProvenance,
    boundary_name,
    describe,
    tensor_to_bytes,
)

PROVENANCE = StoreProvenance(
    base_model="michiyasunaga/BioLinkBERT-base", max_length=512, stride=20
)


def test_each_sub_database_is_counted_on_its_own(tmp_path):
    """Two sub-databases share one env; a count read off the env as a whole
    would blur them together."""
    path = tmp_path / "embeddings"
    aggregated = EmbeddingsStore.create(path, PROVENANCE)
    for pubmed_id in (1, 2, 3):
        aggregated.put(pubmed_id, torch.zeros(12, 8))
    aggregated.close()
    boundary = LayerBoundaryStore.create(path, PROVENANCE, 2)
    boundary.put(1, torch.zeros(1, 4, 8))
    boundary.close()

    description = describe(path)

    assert description.provenance == PROVENANCE
    assert description.provenance_error is None
    counts = {info.name: info.documents for info in description.databases}
    assert counts == {AGGREGATED: 3, boundary_name(2): 1}
    for info in description.databases:
        assert info.compressed_frames == info.documents
        assert info.raw_frames == info.unknown_frames == 0


def test_frames_written_at_level_zero_are_reported_raw(tmp_path):
    """`--no_compress` still writes a blosc2 frame, so only the frame's
    memcpy flag tells it apart from a compressed one. Zeros, because they
    would shrink at any other level."""
    path = tmp_path / "embeddings"
    EmbeddingsStore.create(path, PROVENANCE).close()
    with lmdb.open(str(path), max_dbs=MAX_SUB_DATABASES) as env:
        db = env.open_db(AGGREGATED.encode())
        with env.begin(write=True, db=db) as transaction:
            blob = tensor_to_bytes(torch.zeros(12, 8), compress=False)
            transaction.put(b"1", blob)
            transaction.put(b"2", b"not a blob")

    (info,) = describe(path).databases

    assert (info.raw_frames, info.compressed_frames) == (1, 0)
    assert info.unknown_frames == 1
    assert info.decompressed_bytes == 12 * 8 * 2


def test_an_env_in_the_old_layout_is_described_not_refused(tmp_path):
    """Opening a store refuses the old layout; describing it is how one
    finds out what a rebuild would replace."""
    path = tmp_path / "old"
    record = {"format": 1} | {
        "base_model": PROVENANCE.base_model,
        "max_length": 512,
        "stride": 20,
    }
    with lmdb.open(str(path), map_size=2**20) as env:
        with env.begin(write=True) as transaction:
            transaction.put(b"\x00provenance", json.dumps(record).encode())
            transaction.put(b"100", tensor_to_bytes(torch.rand(12, 8)))

    description = describe(path)

    assert description.provenance is None
    assert "format 1" in (description.provenance_error or "")
    (info,) = description.databases
    assert (info.name, info.documents) == (MAIN_DATABASE, 1)
