"""Regression tests for `GMESampler.dataset_splits` share validation."""

import pandas as pd
import pytest
from brenda_references.sampling import GMESampler
from gme.gme import GreedyMaximumEntropySampler


def _make_sampler(n: int) -> GMESampler:
    """Build a `GMESampler` around a synthetic, deterministic pool.

    Bypasses `__init__` (and its document-parsing constructor path) so the
    sampler can be pointed at a small, controlled `pubmed_id`/subject/object
    frame instead of real BRENDA records.
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
    sampler._data = list(range(n))
    sampler._sampling_df = frame
    return sampler


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
