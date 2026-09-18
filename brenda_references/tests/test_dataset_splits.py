"""Regression tests for `GMESampler.dataset_splits` share validation."""

import logging

import pandas as pd
import pytest
from brenda_references.sampling import GMESampler
from gme.gme import GreedyMaximumEntropySampler

_SPLIT_DTYPES = [
    ("pubmed_id", "int32"),
    ("subject", "float64"),
    ("object", "float64"),
]


def _make_sampler(n: int, relationless: int = 0) -> GMESampler:
    """Build a `GMESampler` around a synthetic, deterministic pool.

    Bypasses `__init__` (and its document-parsing constructor path) so the
    sampler can be pointed at a small, controlled `pubmed_id`/subject/object
    frame instead of real BRENDA records. `relationless` adds documents that,
    like real ones without relations, count toward the split sizes but never
    enter the pool.
    """
    subjects = [f"bac_{i % 5}" for i in range(n)]
    objects = [f"enz_{i % 7}" for i in range(n)]
    frame = pd.DataFrame(
        {
            "pubmed_id": range(n),
            "predicate": "HasEnzyme",
            "subject": subjects,
            "object": objects,
        }
    ).astype(
        dtype={
            "pubmed_id": "int32",
            "predicate": "category",
            "subject": "string",
            "object": "string",
        }
    )

    sampler = GMESampler.__new__(GMESampler)
    sampler.on_columns = ["subject", "object"]
    sampler.item_column = "pubmed_id"
    sampler._sampler = GreedyMaximumEntropySampler(
        selector="dutopia", binarised=False
    )
    sampler._data = list(range(n + relationless))
    sampler._sampling_df = frame
    return sampler


def _dtypes(frame: pd.DataFrame) -> list[tuple[str, str]]:
    return list(frame.dtypes.astype(str).items())


def test_dataset_splits_rejects_shares_summing_past_one(monkeypatch) -> None:
    """`training + validation > 1` must be rejected before any sampling runs.

    The old code let a negative test share reach `sample()`, which shrinks
    the pool on every call; asserting only `pytest.raises(ValueError)`
    would also pass against the old code by accident, since exhausting the
    pool this way happens to crash inside pandas with an unrelated
    `ValueError`. Patching `sample()` to fail loudly if it is ever called
    is what actually pins "rejected before sampling starts".
    """
    sampler = _make_sampler(30)

    def _fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("sample() must not run for invalid shares")

    monkeypatch.setattr(sampler, "sample", _fail_if_called)

    with pytest.raises(ValueError):
        sampler.dataset_splits(training=0.8, validation=0.5)


def test_dataset_splits_returns_disjoint_pubmed_id_sets() -> None:
    """Valid shares still produce three non-empty, pairwise-disjoint splits."""
    sampler = _make_sampler(30)

    splits = sampler.dataset_splits(training=0.7, validation=0.15)
    ids = {name: set(frame["pubmed_id"]) for name, frame in splits.items()}

    assert all(ids.values())
    assert ids["training"] & ids["validation"] == set()
    assert ids["training"] & ids["test"] == set()
    assert ids["validation"] & ids["test"] == set()


def test_dataset_splits_allows_shares_summing_to_exactly_one() -> None:
    """A zero test share yields an empty test split typed like the others.

    The first two draws drain the pool, and handing gme that empty pool
    crashed in pandas with an unrelated string-versus-float64 merge error.
    """
    sampler = _make_sampler(30)

    splits = sampler.dataset_splits(training=0.8, validation=0.2)

    assert splits["test"].empty
    assert all(_dtypes(frame) == _SPLIT_DTYPES for frame in splits.values())
    drawn = set(splits["training"]["pubmed_id"])
    drawn |= set(splits["validation"]["pubmed_id"])
    assert drawn == set(range(30))


def test_dataset_splits_allows_an_empty_validation_split() -> None:
    """A zero-size draw from a non-empty pool yields a typed, empty split.

    gme answers a zero-size draw with an all-`float64` frame, whose missing
    last row the per-split summary then failed to index.
    """
    sampler = _make_sampler(30)

    splits = sampler.dataset_splits(training=0.85, validation=0.0)

    assert splits["validation"].empty
    assert not splits["test"].empty
    assert all(_dtypes(frame) == _SPLIT_DTYPES for frame in splits.values())


def test_dataset_splits_handles_a_pool_no_bigger_than_the_confidence_window() -> (
    None
):
    """A pool of <= 20 documents must not crash the confidence approximation.

    `1 - 20/pool_size` is <= 0 for any `pool_size <= 20`, and `math.log` of
    that raises `ValueError: math domain error`. Such a small pool has no
    approximation to make anyway, so `get_sample` should fall back to gme's
    exact (`approx=0`) computation instead of crashing.
    """
    sampler = _make_sampler(10)

    splits = sampler.dataset_splits(training=0.7, validation=0.15)

    ids = {name: set(frame["pubmed_id"]) for name, frame in splits.items()}
    assert ids["training"] | ids["validation"] | ids["test"] == set(range(10))
    assert all(_dtypes(frame) == _SPLIT_DTYPES for frame in splits.values())


def test_dataset_splits_sizes_off_the_pool_not_every_document(caplog) -> None:
    """Split sizes must scale with the pool, not `len(self._data)`.

    22 of 30 documents carry a relation and enter the pool; 8 do not.
    Sizing off `len(self._data)` (30) instead of the pool (22) drew
    training at the wrong size first and starved validation, then test,
    until test came back empty. Fixed, this same input gives test its
    intended ~15% share of the 22-document pool, not of all 30.
    """
    sampler = _make_sampler(22, relationless=8)

    splits = sampler.dataset_splits(training=0.7, validation=0.15)

    pool_size = 22
    assert not splits["test"].empty
    assert len(splits["test"]) == round(pool_size * 0.15)
    assert len(splits["validation"]) == round(pool_size * 0.15)
    assert len(splits["training"]) == pool_size - len(splits["test"]) - len(
        splits["validation"]
    )
    assert all(_dtypes(frame) == _SPLIT_DTYPES for frame in splits.values())
    drawn = (
        set(splits["training"]["pubmed_id"])
        | set(splits["validation"]["pubmed_id"])
        | set(splits["test"]["pubmed_id"])
    )
    assert drawn == set(range(pool_size))  # only pooled ids, never relationless

    # The pool is now exactly drained; `sample()` must still warn rather
    # than crash if asked for more (the earlier drained-pool fix).
    assert sampler._sampling_df.empty
    with caplog.at_level(logging.WARNING, logger="brenda_references.sampling"):
        extra = sampler.sample(1)

    assert extra.empty
    assert [
        record.levelno
        for record in caplog.records
        if record.name == "brenda_references.sampling"
    ] == [logging.WARNING]
