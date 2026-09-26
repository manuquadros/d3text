"""Fixture-backed tests for BrendaDataset and its sampler.

These use the `tiny_brenda` HDF5 fixture (see conftest.py) rather than the
~300 MB BRENDA files, so they run fast and offline.
"""

import os
import pickle

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from d3text.data.data import LengthLimitedRandomSampler
from d3text.encodings_store import EncodingsProvenance, write_provenance

_ITEM_KEYS = {"id", "sequence", "relations", "classes", "doc_id"}


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


def test_getitem_schema_consistent_across_index_types(tiny_brenda):
    assert set(tiny_brenda.present[0]) == set(tiny_brenda.present[[0]][0])


def test_getitems_skips_pmid_absent_from_hdf5(tiny_brenda):
    # The DataFrame lists pmid 40 (row 3) but the HDF5 file has no such group.
    # `__getitems__` catches the KeyError and skips the row rather than
    # aborting the whole batch; the three present pmids come back.
    items = tiny_brenda.full[[0, 1, 2, 3]]
    assert len(items) == 3
    assert [item["id"] for item in items] == [10, 20, 30]
    # doc_id is the (contiguous) batch position of the surviving rows.
    assert [item["doc_id"][0].item() for item in items] == [0, 1, 2]


def test_getitems_renumbers_doc_id_after_dropping_a_middle_row(tiny_brenda):
    # pmid 40 (row 3) is requested between two present rows. doc_id must be
    # contiguous over the *returned* items (0, 1), not the requested
    # positions (0, 2) the old numbering left behind.
    items = tiny_brenda.full[[0, 3, 2]]
    assert [item["id"] for item in items] == [10, 30]
    assert [item["doc_id"][0].item() for item in items] == [0, 1]


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


def test_sampler_skips_pmid_absent_from_hdf5(tiny_brenda):
    # Row 3 is in the frame but not in the file, so it has no length. Indexing
    # the mapping for it used to raise mid-iteration and end the run on a stale
    # artifact; it is now dropped, as `__getitems__` drops it.
    sampler = LengthLimitedRandomSampler(tiny_brenda.full, max_length=1000)
    assert sorted(sampler) == [0, 1, 2]


def test_sampler_still_filters_over_an_uncovered_frame(tiny_brenda):
    # The skip must not turn the sampler into a pass-through: chunk counts are
    # [2, 5, 1], so a bound of 3 still excludes the 5-chunk document.
    sampler = LengthLimitedRandomSampler(tiny_brenda.full, max_length=3)
    assert set(sampler) == {0, 2}


def _count_h5_opens(monkeypatch) -> list[str]:
    """Record every `h5py.File` open, and keep them working.

    A subclass rather than a bare wrapping function: `BrendaDataset.__init__`
    annotates a local as `h5py.File | None`, evaluated at call time since the
    module carries no `from __future__ import annotations`, and a plain
    function has no `__or__` to satisfy that union.
    """
    real_file = h5py.File
    opened: list[str] = []

    class CountingFile(real_file):
        def __init__(self, name, *args, **kwargs):
            opened.append(str(name))
            super().__init__(name, *args, **kwargs)

    monkeypatch.setattr(h5py, "File", CountingFile)
    return opened


def test_getitems_opens_the_hdf5_file_once_across_batches(
    tiny_hdf5, tiny_dataframe, monkeypatch
):
    # The file used to be reopened for every fetched batch, so an epoch paid
    # one open per batch instead of one per process. Two opens are expected
    # here: `__init__`'s own transient open for the empty-document drop and
    # length walk (closed before it returns), then one persistent handle that
    # every batch fetch below reuses.
    from d3text.data.data import BrendaDataset

    opened = _count_h5_opens(monkeypatch)
    dataset = BrendaDataset(tiny_dataframe.iloc[:3].copy(), encodings=tiny_hdf5)

    batches = [dataset[[0, 1]], dataset[[2]], dataset[[0, 2]]]

    assert len(opened) == 2
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
    assert [item["doc_id"][0].item() for item in items] == [0, 1]
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


# --------------------------------------------------------------------------- #
# a document whose encoding holds no token                                     #
# --------------------------------------------------------------------------- #
def _blank_middle_document(tmp_path):
    """Encodings and frame for three documents, the middle one text-free.

    Pmid 20 is the shape the corpus really holds: one window whose attention
    mask covers `[CLS]` and `[SEP]` and nothing else. The frame carries a
    non-`RangeIndex`, as the corpus splits do — they are boolean-filtered
    without resetting — and one distinct label vector per row, so a drop that
    is not positional shows up as the wrong labels rather than as nothing.
    """
    path = tmp_path / "blank.hdf5"
    real_tokens = {"10": 22, "20": 2, "30": 14}
    with h5py.File(path, "w") as f:
        for pmid, real in real_tokens.items():
            group = f.create_group(pmid)
            mask = np.zeros((1, 32), dtype=np.int64)
            mask[0, :real] = 1
            group.create_dataset(
                "input_ids", data=np.zeros((1, 32), dtype=np.int64)
            )
            group.create_dataset("attention_mask", data=mask)

    frame = pd.DataFrame(
        {
            "pubmed_id": [10, 20, 30],
            "relations": pd.Series([[], [], []]),
            "classes": [
                np.array([1, 0], dtype=np.float32),
                np.array([0, 1], dtype=np.float32),
                np.array([1, 1], dtype=np.float32),
            ],
        },
        index=[5, 9, 13],
    )
    return path, frame


def test_a_document_with_no_tokens_is_dropped_from_the_split(tmp_path):
    """One corpus row is JATS markup wrapping newlines, which strips to a
    truthy string of indentation and was encoded without complaint. It reaches
    the model as zero tokens, where the four poolings return `-inf` (a
    confidently correct negative), `NaN` into the loss, or raise.

    It is dropped from the split, so no sampler can draw it and the rows around
    it keep their own labels.
    """
    from d3text.data.data import BrendaDataset

    path, frame = _blank_middle_document(tmp_path)

    dataset = BrendaDataset(frame, encodings=path)

    assert len(dataset) == 2
    assert [item["id"] for item in dataset[[0, 1]]] == [10, 30]
    assert [item["classes"].tolist() for item in dataset[[0, 1]]] == [
        [1.0, 0.0],
        [1.0, 1.0],
    ]
    # Keyed by row position, so a mapping still naming three rows means the
    # split was not filtered, only the fetch.
    assert dataset.sequence_lengths == {0: 1, 1: 1}


def test_no_batch_hands_the_pooling_a_document_of_zero_tokens(tmp_path):
    """`evaluate` loads with `batch_size=1`, and `BatchSampler(drop_last=False)`
    makes a lone-document batch reachable in training too, so the text-free
    document used to arrive at the pooling alone and unpadded.

    What the model pools is what `aggregate_embeddings` returns, so that is
    what this counts: every batch must be non-empty and every document in it
    must keep at least one token once `[CLS]` and `[SEP]` come off.
    """
    from d3text.data.data import BrendaDataset, get_batch_loader
    from d3text.utils import aggregate_embeddings

    path, frame = _blank_middle_document(tmp_path)
    loader = get_batch_loader(BrendaDataset(frame, encodings=path), 1)

    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert batch
        for item in batch:
            mask = item["sequence"]["attention_mask"]
            mask = mask.reshape(-1, mask.shape[-1])
            tokens = aggregate_embeddings(
                torch.zeros(mask.shape[0], mask.shape[1], 2), mask
            )
            assert tokens.shape[0] > 0


# --------------------------------------------------------------------------- #
# Encodings provenance                                                        #
# --------------------------------------------------------------------------- #
def _stamped_hdf5(tmp_path, provenance):
    path = tmp_path / "stamped.hdf5"
    with h5py.File(path, "w") as f:
        write_provenance(f, provenance)
        group = f.create_group("10")
        group.create_dataset("input_ids", data=np.zeros((1, 8), dtype=np.int64))
        group.create_dataset(
            "attention_mask", data=np.ones((1, 8), dtype=np.int64)
        )
    return path


def _one_row_frame():
    return pd.DataFrame(
        {
            "pubmed_id": [10],
            "relations": pd.Series([[]]),
            "classes": [np.array([1, 0], dtype=np.float32)],
        }
    )


def test_a_store_tokenized_by_another_model_is_refused(tmp_path):
    from d3text.data.data import BrendaDataset

    path = _stamped_hdf5(
        tmp_path,
        EncodingsProvenance(
            base_model="other-model", max_length=512, stride=20
        ),
    )

    with pytest.raises(ValueError, match="was tokenized by other-model"):
        BrendaDataset(_one_row_frame(), encodings=path, base_model="this-model")


def test_a_store_tokenized_by_this_model_is_read(tmp_path):
    from d3text.data.data import BrendaDataset

    path = _stamped_hdf5(
        tmp_path,
        EncodingsProvenance(base_model="this-model", max_length=512, stride=20),
    )

    dataset = BrendaDataset(
        _one_row_frame(), encodings=path, base_model="this-model"
    )
    assert len(dataset) == 1


def test_a_store_tokenized_at_another_stride_is_refused(tmp_path):
    """The stride is not recoverable from the stored ids, and
    `aggregate_embeddings` merges every window at its own. A store built at
    another one is stitched at the wrong offset at each seam — plausible
    shapes, wrong tokens — so the stamp is what has to catch it."""
    from d3text.data.data import BrendaDataset

    path = _stamped_hdf5(
        tmp_path,
        EncodingsProvenance(base_model="this-model", max_length=512, stride=30),
    )

    with pytest.raises(ValueError, match="stride of 30"):
        BrendaDataset(_one_row_frame(), encodings=path, base_model="this-model")


def test_the_accepted_stride_is_the_one_the_aggregation_merges_at(tmp_path):
    """A store stamped with the stride the aggregation actually takes is read,
    whatever that constant is — so the check tracks the geometry rather than
    pinning a literal beside a second copy of it."""
    from d3text.data.data import BrendaDataset
    from d3text.utils import WINDOW_STRIDE

    path = _stamped_hdf5(
        tmp_path,
        EncodingsProvenance(
            base_model="this-model", max_length=512, stride=WINDOW_STRIDE
        ),
    )

    dataset = BrendaDataset(
        _one_row_frame(), encodings=path, base_model="this-model"
    )
    assert len(dataset) == 1


def test_a_store_at_a_shorter_window_is_read(tmp_path):
    """`max_length` is the one stamped field the aggregation never consults:
    windows are stitched off the attention mask, so a shorter window still
    reconstructs the document token-for-token and is no reason to refuse."""
    from d3text.data.data import BrendaDataset

    path = _stamped_hdf5(
        tmp_path,
        EncodingsProvenance(base_model="this-model", max_length=256, stride=20),
    )

    dataset = BrendaDataset(
        _one_row_frame(), encodings=path, base_model="this-model"
    )
    assert len(dataset) == 1


def test_an_unstamped_store_is_still_read_under_a_named_base_model(tmp_path):
    """Every encodings file written before the stamp existed carries no
    geometry at all. Refusing those would make the stride check cost the whole
    corpus a rebuild, so an unstamped store is read at the assumed stride."""
    from d3text.data.data import BrendaDataset

    path = tmp_path / "unstamped.hdf5"
    with h5py.File(path, "w") as f:
        group = f.create_group("10")
        group.create_dataset("input_ids", data=np.zeros((1, 8), dtype=np.int64))
        group.create_dataset(
            "attention_mask", data=np.ones((1, 8), dtype=np.int64)
        )

    dataset = BrendaDataset(
        _one_row_frame(), encodings=path, base_model="this-model"
    )
    assert len(dataset) == 1


def test_an_unstamped_store_is_read_with_no_base_model_given(
    tiny_brenda,
):
    """The default before this existed: nothing is checked and every
    existing caller keeps working unchanged."""
    assert len(tiny_brenda.present) == 3


def test_a_group_left_without_ids_yields_no_length(tmp_path):
    """A pass killed between `create_group` and the `create_dataset` after it
    leaves a keyed group holding no ids, and a resume skips a key already
    present, so it stays. The row is left in the split and simply has no
    length, as a pmid absent from the file has none — the same tolerance
    `encodings_store.content_digest` reads the store under, since both ask
    one predicate now."""
    from d3text.data.data import BrendaDataset

    path = tmp_path / "partial.hdf5"
    with h5py.File(path, "w") as f:
        group = f.create_group("10")
        group.create_dataset("input_ids", data=np.zeros((1, 8), dtype=np.int64))
        group.create_dataset(
            "attention_mask", data=np.ones((1, 8), dtype=np.int64)
        )
        f.create_group("20")

    frame = pd.DataFrame(
        {
            "pubmed_id": [10, 20],
            "relations": pd.Series([[], []]),
            "classes": [np.array([1, 0], dtype=np.float32)] * 2,
        }
    )

    dataset = BrendaDataset(frame, encodings=path)

    assert len(dataset) == 2
    assert dataset.sequence_lengths == {0: 1}


def test_a_group_with_attention_mask_but_no_input_ids_is_dropped_not_raised(
    tmp_path,
):
    """A group carrying `attention_mask` but not `input_ids` is truthy under
    `group.keys()` — the third spelling of "this group holds no ids" that
    `__getitems__` used to test instead of asking `stored_ids` like the other
    two readers. A strong attention mask keeps `_drop_empty_documents` from
    removing the row first, so it reaches `__getitems__` still present in the
    split; the fetch must drop it as a row, not raise `KeyError` reaching for
    `input_ids` on a dict that never had it."""
    from d3text.data.data import BrendaDataset

    path = tmp_path / "reversed_partial.hdf5"
    with h5py.File(path, "w") as f:
        group = f.create_group("10")
        group.create_dataset(
            "attention_mask", data=np.ones((1, 8), dtype=np.int64)
        )

    frame = pd.DataFrame(
        {
            "pubmed_id": [10],
            "relations": pd.Series([[]]),
            "classes": [np.array([1, 0], dtype=np.float32)],
        }
    )

    dataset = BrendaDataset(frame, encodings=path)

    assert len(dataset) == 1  # the row survives _drop_empty_documents
    assert dataset[[0]] == []


# --------------------------------------------------------------------------- #
# one walk for the empty-document drop and the length mapping                 #
# --------------------------------------------------------------------------- #
def test_drop_and_lengths_share_one_hdf5_open_and_agree_on_the_result(
    tmp_path, monkeypatch
):
    """The empty-document drop and the length walk used to each open the file
    and re-walk the split on their own, doing the same
    `f.get(str(pubmed_id))` per row twice over. This pins the merged single
    walk: one open for the pass, and the same surviving row set and length
    mapping the two separate passes used to produce by hand."""
    from d3text.data.data import BrendaDataset

    path = tmp_path / "merged.hdf5"
    with h5py.File(path, "w") as f:
        group = f.create_group("10")
        group.create_dataset("input_ids", data=np.zeros((3, 8), dtype=np.int64))
        group.create_dataset(
            "attention_mask", data=np.ones((3, 8), dtype=np.int64)
        )

        group = f.create_group("20")  # blank: one window, CLS+SEP only
        group.create_dataset("input_ids", data=np.zeros((1, 8), dtype=np.int64))
        blank_mask = np.zeros((1, 8), dtype=np.int64)
        blank_mask[0, :2] = 1
        group.create_dataset("attention_mask", data=blank_mask)

        group = f.create_group("30")
        group.create_dataset("input_ids", data=np.zeros((1, 8), dtype=np.int64))
        group.create_dataset(
            "attention_mask", data=np.ones((1, 8), dtype=np.int64)
        )
        # pmid 40 is deliberately absent from the file.

    frame = pd.DataFrame(
        {
            "pubmed_id": [10, 20, 30, 40],
            "relations": pd.Series([[], [], [], []]),
            "classes": [np.array([1, 0], dtype=np.float32)] * 4,
        }
    )

    opened = _count_h5_opens(monkeypatch)

    dataset = BrendaDataset(frame, encodings=path)

    # pmid 20 (blank) is dropped; pmid 40 (absent) is kept, as a row with no
    # encoding always is.
    assert dataset.data["pubmed_id"].tolist() == [10, 30, 40]
    # New positions 0, 1, 2 -> pmid 10 (3 chunks), 30 (1 chunk), 40 (no
    # length: absent from the file). Checked *before* the open count: reading
    # `sequence_lengths` must not be what pays for a second open.
    assert dataset.sequence_lengths == {0: 3, 1: 1}
    assert opened == [str(path)]


# --------------------------------------------------------------------------- #
# a torn group from an interrupted precompute pass                            #
# --------------------------------------------------------------------------- #
def test_an_ids_only_group_is_skipped_not_served_with_a_missing_mask(
    tmp_path,
):
    """A pass killed between `create_group` and the `attention_mask` write
    leaves a group holding only `input_ids`. `stored_ids` alone used to
    accept it, so it reached `__getitems__` and was served without a mask —
    the model's own `item["sequence"]["attention_mask"]` then raised
    `KeyError`, hours into an epoch rather than at construction. It must
    instead be treated the same as a pmid the file holds no group for."""
    from d3text.data.data import BrendaDataset

    path = tmp_path / "ids_only.hdf5"
    with h5py.File(path, "w") as f:
        group = f.create_group("10")
        group.create_dataset("input_ids", data=np.zeros((2, 8), dtype=np.int64))

        group = f.create_group("20")
        group.create_dataset("input_ids", data=np.zeros((1, 8), dtype=np.int64))
        group.create_dataset(
            "attention_mask", data=np.ones((1, 8), dtype=np.int64)
        )

    frame = pd.DataFrame(
        {
            "pubmed_id": [10, 20],
            "relations": pd.Series([[], []]),
            "classes": [np.array([1, 0], dtype=np.float32)] * 2,
        }
    )

    dataset = BrendaDataset(frame, encodings=path)

    assert len(dataset) == 2  # kept in the split, as an absent pmid would be
    assert dataset.sequence_lengths == {1: 1}
    assert [item["id"] for item in dataset[[0, 1]]] == [20]


def test_a_zero_mask_multi_window_group_is_skipped_not_served_as_empty(
    tmp_path,
):
    """A pass killed after `create_dataset("attention_mask", ...)` but before
    it is filled leaves the array at h5py's own zero fill. The single-window
    whitespace check in `_drop_empty_documents` never looks at a multi-window
    mask, so this used to reach the model as a real document of zero tokens
    across every window, which the poolings mis-score, NaN on, or refuse."""
    from d3text.data.data import BrendaDataset

    path = tmp_path / "zero_mask.hdf5"
    with h5py.File(path, "w") as f:
        group = f.create_group("10")
        group.create_dataset("input_ids", data=np.zeros((3, 8), dtype=np.int64))
        group.create_dataset(
            "attention_mask", data=np.zeros((3, 8), dtype=np.int64)
        )

        group = f.create_group("20")
        group.create_dataset("input_ids", data=np.zeros((1, 8), dtype=np.int64))
        group.create_dataset(
            "attention_mask", data=np.ones((1, 8), dtype=np.int64)
        )

    frame = pd.DataFrame(
        {
            "pubmed_id": [10, 20],
            "relations": pd.Series([[], []]),
            "classes": [np.array([1, 0], dtype=np.float32)] * 2,
        }
    )

    dataset = BrendaDataset(frame, encodings=path)

    assert len(dataset) == 2  # kept in the split, as an absent pmid would be
    assert dataset.sequence_lengths == {1: 1}
    assert [item["id"] for item in dataset[[0, 1]]] == [20]


# --------------------------------------------------------------------------- #
# a whole configured source missing from the store                            #
# --------------------------------------------------------------------------- #
def test_a_source_wholly_missing_from_the_store_is_refused(tmp_path):
    """A `--limit` subset can shrink a configured source, such as the
    enzyme-negative pool, down to a handful of rows. If the store predates
    that source, every one of those rows is absent from it — the signature
    of a corpus file the store was never built over, not the ordinary
    per-document gap `__getitems__` already tolerates. Construction must
    refuse and name the source, not fall through to the per-row skip."""
    from d3text.data.data import BrendaDataset

    path = tmp_path / "gap.hdf5"
    with h5py.File(path, "w") as f:
        group = f.create_group("10")
        group.create_dataset("input_ids", data=np.zeros((1, 8), dtype=np.int64))
        group.create_dataset(
            "attention_mask", data=np.ones((1, 8), dtype=np.int64)
        )
        # pmid 20, the enzyme_negative source's only row, is never written.

    frame = pd.DataFrame(
        {
            "pubmed_id": [10, 20],
            "relations": pd.Series([[], []]),
            "classes": [np.array([1, 0], dtype=np.float32)] * 2,
            "source": ["training", "enzyme_negative"],
        }
    )

    with pytest.raises(ValueError, match="enzyme_negative"):
        BrendaDataset(frame, encodings=path)


def test_a_scattered_miss_within_a_source_still_constructs(tmp_path):
    """One row of a source missing from the store, with a sibling row of the
    same source present, is the ordinary scattered gap — not every row of
    that source, so it must not refuse."""
    from d3text.data.data import BrendaDataset

    path = tmp_path / "scattered.hdf5"
    with h5py.File(path, "w") as f:
        for pmid in ("10", "20"):
            group = f.create_group(pmid)
            group.create_dataset(
                "input_ids", data=np.zeros((1, 8), dtype=np.int64)
            )
            group.create_dataset(
                "attention_mask", data=np.ones((1, 8), dtype=np.int64)
            )
        # pmid 30, one of two enzyme_negative rows, is absent.

    frame = pd.DataFrame(
        {
            "pubmed_id": [10, 20, 30],
            "relations": pd.Series([[], [], []]),
            "classes": [np.array([1, 0], dtype=np.float32)] * 3,
            "source": ["training", "enzyme_negative", "enzyme_negative"],
        }
    )

    dataset = BrendaDataset(frame, encodings=path)

    assert len(dataset) == 3


# --------------------------------------------------------------------------- #
# positional indexing over a shuffled, non-RangeIndex split                    #
# --------------------------------------------------------------------------- #
def test_getitems_reads_by_row_position_not_by_index_label(tmp_path):
    """`__getitems__` reads `pubmed_id`/`relations`/`classes` from arrays
    materialised in `__init__`, in place of `.iloc[ix]`. The materialisation
    must preserve `.iloc`'s row-position semantics exactly: the corpus splits
    carry a shuffled, non-`RangeIndex` (boolean-filtered without a reset, per
    `datasets/brenda.py`), so label-based indexing at position `ix` would
    silently fetch a *different* row than `.iloc[ix]` did, matching one
    document's `id`/`relations`/`classes` against another's HDF5 sequence.
    """
    from d3text.data.data import BrendaDataset

    path = tmp_path / "shuffled.hdf5"
    with h5py.File(path, "w") as f:
        for pmid, n_chunks in (("10", 1), ("20", 2), ("30", 3)):
            group = f.create_group(pmid)
            group.create_dataset(
                "input_ids", data=np.zeros((n_chunks, 8), dtype=np.int64)
            )
            group.create_dataset(
                "attention_mask", data=np.ones((n_chunks, 8), dtype=np.int64)
            )

    relations = [{("bac1", "enz1"): 0}, {}, {("bac2", "enz2"): 1}]
    classes = [
        np.array([1, 0], dtype=np.float32),
        np.array([0, 1], dtype=np.float32),
        np.array([1, 1], dtype=np.float32),
    ]
    # Index labels [2, 0, 1]: label-based access at position 0 would land on
    # the row labelled 0 (pmid 20, position 1), not the row actually at
    # position 0 (pmid 10) — the exact mismatch this test must catch.
    # Built from plain lists, not `Series`: a `Series` column would itself get
    # realigned onto the declared index at construction time (a second,
    # unrelated reordering hazard), which would defeat the point of this test.
    frame = pd.DataFrame(
        {
            "pubmed_id": [10, 20, 30],
            "relations": relations,
            "classes": classes,
        },
        index=[2, 0, 1],
    )

    dataset = BrendaDataset(frame, encodings=path)
    items = dataset[[0, 1, 2]]

    assert [item["id"] for item in items] == [10, 20, 30]
    assert [type(item["id"]) for item in items] == [
        type(v) for v in frame["pubmed_id"].to_numpy()
    ]
    assert [item["relations"] for item in items] == relations
    assert [item["classes"].tolist() for item in items] == [
        c.tolist() for c in classes
    ]
    assert [item["classes"].dtype for item in items] == [np.float32] * 3
    # The HDF5-backed sequence must line up with the same document.
    assert [item["sequence"]["input_ids"].shape[0] for item in items] == [
        1,
        2,
        3,
    ]
