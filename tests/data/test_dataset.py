"""Fixture-backed tests for BrendaDataset and its sampler.

These use the `tiny_brenda` HDF5 fixture (see conftest.py) rather than the
~300 MB BRENDA files, so they run fast and offline.
"""

import h5py
import pytest
import torch

from d3text.data.data import LengthLimitedRandomSampler

_ITEM_KEYS = {"id", "sequence", "entities", "relations", "classes", "doc_id"}


def test_getitem_int_returns_single_document_with_full_schema(tiny_brenda):
    item = tiny_brenda.present[0]
    assert set(item) == _ITEM_KEYS
    assert item["id"] == 10
    assert item["sequence"]["input_ids"].shape[0] == 2  # chunks for pmid 10
    # A lone int is batch position 0, repeated once per chunk.
    assert item["doc_id"].tolist() == [0, 0]


def test_getitem_int_raises_for_pmid_absent_from_hdf5(tiny_brenda):
    # Single-int access can't skip a row the way a batch does, so a pmid absent
    # from the HDF5 file surfaces as a KeyError rather than a silent None.
    with pytest.raises(KeyError):
        tiny_brenda.full[3]  # pmid 40, missing from the fixture


def test_getitems_list_includes_doc_id_as_batch_position(tiny_brenda):
    items = tiny_brenda.present[[0, 1]]
    assert len(items) == 2
    assert "doc_id" in items[0]
    # doc_id repeats the batch position once per HDF5 chunk (not the pmid).
    assert items[0]["doc_id"].tolist() == [0, 0]  # pmid 10 -> 2 chunks
    assert items[1]["doc_id"].tolist() == [1, 1, 1, 1, 1]  # pmid 20 -> 5 chunks
    assert items[0]["doc_id"].dtype == torch.uint8


def test_getitem_schema_consistent_across_index_types(tiny_brenda):
    assert set(tiny_brenda.present[0]) == set(tiny_brenda.present[[0]][0])


def test_getitems_skips_pmid_absent_from_hdf5(tiny_brenda):
    # The DataFrame lists pmid 40 (row 3) but the HDF5 file has no such group.
    # _getitems catches the KeyError and skips the row rather than aborting the
    # whole batch; the three present pmids come back.
    items = tiny_brenda.full[[0, 1, 2, 3]]
    assert len(items) == 3
    assert [item["id"] for item in items] == [10, 20, 30]
    # doc_id is the (contiguous) batch position of the surviving rows.
    assert [item["doc_id"][0].item() for item in items] == [0, 1, 2]


def test_length_limited_sampler_filters_by_chunk_count(tiny_brenda):
    sampler = LengthLimitedRandomSampler(tiny_brenda.present, max_length=3)
    yielded = set(sampler)
    # chunk counts are [2, 5, 1]; only indices 0 and 2 are strictly < 3.
    assert yielded <= {0, 2}
    assert 1 not in yielded  # pmid 20 has 5 chunks -> always excluded


def test_sampler_opens_no_file_while_iterating(tiny_brenda, monkeypatch):
    # The lengths are read once, when the sampler is built; iterating must be
    # pure arithmetic over that mapping. Reading them per index instead means
    # an epoch pulls every document off disk twice.
    sampler = LengthLimitedRandomSampler(tiny_brenda.present, max_length=3)

    def forbidden(*args, **kwargs):
        raise AssertionError("the sampler read the HDF5 file while iterating")

    monkeypatch.setattr(h5py, "File", forbidden)

    # Sampling without replacement walks a permutation of every index, so the
    # accepted set is exact, not a subset.
    assert set(sampler) == {0, 2}


def test_sampler_lengths_match_the_documents(tiny_brenda):
    assert tiny_brenda.present.sequence_lengths == dict(
        enumerate(tiny_brenda.chunks)
    )


def test_sampler_raises_for_pmid_absent_from_hdf5(tiny_brenda):
    # The missing row is skipped when the lengths are read, so the KeyError
    # still surfaces where it used to: at the index, mid-iteration.
    sampler = LengthLimitedRandomSampler(tiny_brenda.full, max_length=3)
    with pytest.raises(KeyError):
        set(sampler)
