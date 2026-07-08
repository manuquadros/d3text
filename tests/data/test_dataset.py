"""Fixture-backed tests for BrendaDataset and its sampler.

These use the `tiny_brenda` HDF5 fixture (see conftest.py) rather than the
~300 MB BRENDA files, so they run fast and offline.
"""

import pytest
import torch

from d3text.data.data import LengthLimitedRandomSampler

_INT_KEYS = {"id", "sequence", "entities", "relations", "classes"}


def test_getitem_int_omits_doc_id(tiny_brenda):
    item = tiny_brenda.present[0]
    assert set(item) == _INT_KEYS
    assert "doc_id" not in item
    assert item["id"] == 10
    assert item["sequence"]["input_ids"].shape[0] == 2  # chunks for pmid 10


def test_getitems_list_includes_doc_id_as_batch_position(tiny_brenda):
    items = tiny_brenda.present[[0, 1]]
    assert len(items) == 2
    assert "doc_id" in items[0]
    # doc_id repeats the batch position once per HDF5 chunk (not the pmid).
    assert items[0]["doc_id"].tolist() == [0, 0]  # pmid 10 -> 2 chunks
    assert items[1]["doc_id"].tolist() == [1, 1, 1, 1, 1]  # pmid 20 -> 5 chunks
    assert items[0]["doc_id"].dtype == torch.uint8


@pytest.mark.xfail(
    reason="__getitem__ returns different keys for int vs list indexing "
    "(list adds doc_id); codebase_review §2.9",
    strict=True,
)
def test_getitem_schema_consistent_across_index_types(tiny_brenda):
    assert set(tiny_brenda.present[0]) == set(tiny_brenda.present[[0]][0])


def test_getitems_raises_for_pmid_absent_from_hdf5(tiny_brenda):
    # The DataFrame lists pmid 40 (row 3) but the HDF5 file has no such group.
    # _getitems only guards the empty-data (TypeError) path, so a truly missing
    # group surfaces as a KeyError rather than being silently skipped.
    with pytest.raises(KeyError):
        tiny_brenda.full[[0, 1, 2, 3]]


def test_length_limited_sampler_filters_by_chunk_count(tiny_brenda):
    sampler = LengthLimitedRandomSampler(tiny_brenda.present, max_length=3)
    yielded = set(sampler)
    # chunk counts are [2, 5, 1]; only indices 0 and 2 are strictly < 3.
    assert yielded <= {0, 2}
    assert 1 not in yielded  # pmid 20 has 5 chunks -> always excluded
