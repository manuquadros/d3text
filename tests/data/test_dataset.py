"""Fixture-backed tests for BrendaDataset and its sampler.

These use the `tiny_brenda` HDF5 fixture (see conftest.py) rather than the
~300 MB BRENDA files, so they run fast and offline.
"""

import os
import pickle

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


def _count_h5_opens(monkeypatch) -> list[str]:
    """Record every `h5py.File` open, and keep them working."""
    real_file = h5py.File
    opened: list[str] = []

    def counting(name, *args, **kwargs):
        opened.append(str(name))
        return real_file(name, *args, **kwargs)

    monkeypatch.setattr(h5py, "File", counting)
    return opened


def test_getitems_opens_the_hdf5_file_once_across_batches(
    tiny_brenda, monkeypatch
):
    # The file used to be reopened for every fetched batch, so an epoch paid
    # one open per batch instead of one per process.
    dataset = tiny_brenda.present
    opened = _count_h5_opens(monkeypatch)

    batches = [dataset[[0, 1]], dataset[[2]], dataset[[0, 2]]]

    assert len(opened) == 1
    assert [item["id"] for batch in batches for item in batch] == [
        10,
        20,
        30,
        10,
        30,
    ]


def test_cached_handle_returns_the_same_items_as_a_fresh_open(tiny_brenda):
    # A reused handle must not drift: the second read of a batch has to match
    # the first, byte for byte, or which tokens train changes silently.
    dataset = tiny_brenda.present
    first = dataset[[0, 1, 2]]
    dataset.close()
    second = dataset[[0, 1, 2]]

    assert [item["id"] for item in first] == [item["id"] for item in second]
    for before, after in zip(first, second):
        assert before["doc_id"].tolist() == after["doc_id"].tolist()
        assert set(before["sequence"]) == set(after["sequence"])
        for key, value in before["sequence"].items():
            assert (value == after["sequence"][key]).all()


def test_getitems_still_skips_pmid_absent_from_hdf5_on_the_cached_handle(
    tiny_brenda,
):
    # The per-pmid KeyError guard has to survive the shared handle: a missing
    # row is dropped, and the handle stays usable for the batch after it.
    dataset = tiny_brenda.full
    items = dataset[[0, 3, 2]]
    assert [item["id"] for item in items] == [10, 30]
    # doc_id is the batch position of the surviving rows, gaps closed up.
    assert [item["doc_id"][0].item() for item in items] == [0, 2]
    assert [item["id"] for item in dataset[[1]]] == [20]


def test_dataset_pickles_after_reading_and_reads_again(tiny_brenda):
    # `DataLoader` pickles the dataset to reach a worker under the `spawn`
    # start method; an `h5py.File` on the instance is unpicklable, which would
    # break `num_workers > 0` outright.
    dataset = tiny_brenda.present
    dataset[[0]]

    revived = pickle.loads(pickle.dumps(dataset))

    assert revived._h5_handle is None
    assert [item["id"] for item in revived[[0, 1]]] == [10, 20]


def test_handle_is_reopened_rather_than_shared_across_a_fork(
    tiny_brenda, monkeypatch
):
    # An HDF5 handle inherited across a fork shares the parent's file offset;
    # reading through it returns wrong bytes rather than raising, so a worker
    # must open its own instead of using the one it inherited.
    dataset = tiny_brenda.present
    dataset[[0]]
    parent_handle = dataset._h5_handle

    opened = _count_h5_opens(monkeypatch)
    monkeypatch.setattr(os, "getpid", lambda: os.getppid())

    items = dataset[[1]]

    assert len(opened) == 1
    assert dataset._h5_handle is not parent_handle
    assert [item["id"] for item in items] == [20]
