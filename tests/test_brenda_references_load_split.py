"""What `load_split`'s `limit` counts, and what it scales with it.

A short run is meant to be a scaled-down run, not a differently-composed
one: the noise pools are appended after the real documents, so a `limit`
that does not shrink them leaves a slice that is mostly synthetic and
converges on nothing the whole corpus would. These pin a
`brenda_references` function from d3text's own gated suite on purpose, as
`test_noise_splits.py` does, and stand the split's CSV and both noise pools
up small so they need none of the BRENDA data files.
"""

import pandas as pd
import pytest
from brenda_references import brenda_references
from brenda_references import data_paths
from brenda_references.brenda_references import load_split

# Real rows are numbered from here, well past the stub pools' ids, so a
# loaded frame can be split into real and synthetic by `pubmed_id` alone.
REAL_ID_BASE = 10_000

TEXTLESS = 10
USABLE = 100


@pytest.fixture
def tiny_split(tmp_path, monkeypatch):
    """A training CSV of `USABLE` usable rows behind `TEXTLESS` textless ones.

    The textless rows come first so that truncating before dropping them
    yields a different count from dropping them first.
    """
    rows = TEXTLESS + USABLE
    frame = pd.DataFrame(
        {
            "pubmed_id": range(REAL_ID_BASE, REAL_ID_BASE + rows),
            "abstract": ["an abstract"] * rows,
            "fulltext": [None] * TEXTLESS + ["a full text"] * USABLE,
            "bacteria": ["{}"] * rows,
            "other_organisms": ["{}"] * rows,
            "strains": ["[]"] * rows,
            "enzymes": ["[]"] * rows,
            "relations": ["{}"] * rows,
        }
    )
    frame.to_csv(tmp_path / "training_data.csv")
    monkeypatch.setattr(data_paths, "DATA_DIR", tmp_path)

    pool = pd.DataFrame({"pubmed_id": range(1000), "abstract": [""] * 1000})
    monkeypatch.setattr(
        brenda_references, "psycholinguistics_data", lambda: pool
    )
    monkeypatch.setattr(brenda_references, "enzyme_negative_data", lambda: pool)
    return frame


def _real_and_synthetic(split: pd.DataFrame) -> tuple[int, int]:
    real = split["pubmed_id"] >= REAL_ID_BASE
    return int(real.sum()), int((~real).sum())


def test_a_negative_limit_is_refused_rather_than_emptying_the_split():
    """The check has to fire before the split's CSV is even opened, so this
    needs no fixture at all."""
    with pytest.raises(ValueError, match="non-negative"):
        load_split("training", limit=-1)


def test_the_limit_counts_documents_that_carry_text(tiny_split):
    """`limit` is the number of documents the run trains on, so the rows
    `dropna` discards must not be spent out of its budget."""
    split = load_split("training", limit=25)

    real, _ = _real_and_synthetic(split)
    assert real == 25


def test_the_noise_counts_shrink_with_the_limit(tiny_split):
    """Noise is appended after the truncation, so an unscaled count is a
    slice whose composition has nothing to do with the corpus's. A quarter
    of the usable split draws a quarter of each pool: 10 of 40 and 5 of 20.
    """
    split = load_split("training", noise=40, enzyme_noise=20, limit=25)

    assert _real_and_synthetic(split) == (25, 15)


def test_an_absent_limit_draws_every_noise_document(tiny_split):
    """The scaling applies to a truncated split alone — a whole one still
    gets the noise its caller asked for."""
    split = load_split("training", noise=40, enzyme_noise=20)

    assert _real_and_synthetic(split) == (USABLE, 60)


def test_a_limit_past_the_split_does_not_inflate_the_noise(tiny_split):
    """The fraction is capped at 1: asking for more documents than the split
    holds is not a request for more noise than the pools were asked for."""
    split = load_split("training", noise=40, enzyme_noise=20, limit=10_000)

    assert _real_and_synthetic(split) == (USABLE, 60)
