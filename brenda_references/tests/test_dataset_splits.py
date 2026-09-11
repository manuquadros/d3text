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


def test_dataset_splits_warns_when_the_pool_runs_dry(caplog) -> None:
    """A pool drained before the test draw gives an empty split and a warning.

    Split sizes count documents without relations, which never enter the
    pool, so at the default shares the test draw can find it empty. That
    crashed the same way as a zero test share, and a short split must not
    pass silently.
    """
    sampler = _make_sampler(22, relationless=8)

    with caplog.at_level(logging.WARNING, logger="brenda_references.sampling"):
        splits = sampler.dataset_splits(training=0.7, validation=0.15)

    assert splits["test"].empty
    assert all(_dtypes(frame) == _SPLIT_DTYPES for frame in splits.values())
    assert [
        record.levelno
        for record in caplog.records
        if record.name == "brenda_references.sampling"
    ] == [logging.WARNING]
