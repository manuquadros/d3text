"""What `load_split`'s `limit` counts, and what it scales with it.

A short run should be a scaled-down run, not a differently-composed one: the
noise pools are appended after the real documents, so a `limit` that left them
unscaled would train on a mostly synthetic slice.
"""

import logging

import pandas as pd
import pytest
from brenda_references import brenda_references
from brenda_references import data_paths
from brenda_references.brenda_references import load_split
from pandera.errors import SchemaErrors

# Real rows are numbered from here, well past the stub pools' ids, so a
# loaded frame can be split into real and synthetic by `pubmed_id` alone.
REAL_ID_BASE = 10_000

TEXTLESS = 10
USABLE = 100


@pytest.fixture
def tiny_split(tmp_path, monkeypatch):
    """A training CSV of `USABLE` usable rows behind `TEXTLESS` textless ones.

    The textless rows come first so that truncating before dropping them
    yields a different count from dropping them first. The first usable row
    is repeated at the end, as each shipped split repeats a `pubmed_id`, so
    loading takes the path that merges duplicates; the merge leaves the
    counts unchanged.
    """
    rows = TEXTLESS + USABLE
    frame = pd.DataFrame(
        {
            "pubmed_id": range(REAL_ID_BASE, REAL_ID_BASE + rows),
            "pmc_id": range(5000000, 5000000 + rows),
            "year": [2024] * rows,
            "volume": ["10 Suppl 1"] * rows,
            "pmc_open": [True] * rows,
            "abstract": ["an abstract"] * rows,
            "fulltext": [None] * TEXTLESS + ["a full text"] * USABLE,
            "bacteria": ["{}"] * rows,
            "other_organisms": ["{}"] * rows,
            "strains": ["[]"] * rows,
            "enzymes": ["[]"] * rows,
            "entity_spans": ["[]"] * rows,
            "relations": ["{}"] * rows,
            "path": [None] * rows,
        }
    )
    frame = pd.concat([frame, frame.iloc[[TEXTLESS]]], ignore_index=True)
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


def test_every_row_carries_its_source(tiny_split):
    """`BrendaDataset` groups by this column to refuse a store missing a
    whole configured source; a row that survived the `pd.concat` without
    one would silently fall outside that check."""
    split = load_split("training", noise=40, enzyme_noise=20, limit=25)

    real = split["pubmed_id"] >= REAL_ID_BASE
    assert set(split.loc[real, "source"]) == {"training"}

    counts = split["source"].value_counts()
    assert counts["training"] == 25
    assert counts["psycholinguistics"] == 10
    assert counts["enzyme_negative"] == 5


def test_load_split_logs_each_pools_size_before_the_concat(tiny_split, caplog):
    """One INFO line per split, with each pool's count taken from the three
    frames that feed the `pd.concat` — not the merged result's `source`
    counts, which is what lets a pool a `--limit` scales down to zero still
    show up as 0 instead of vanishing with no trace, as it does here: 1 of
    100 usable rows rounds both noise pools down to nothing.
    """
    with caplog.at_level(
        logging.INFO, logger="brenda_references.brenda_references"
    ):
        load_split("training", noise=40, enzyme_noise=20, limit=1)

    records = [
        record
        for record in caplog.records
        if record.name == "brenda_references.brenda_references"
    ]
    assert len(records) == 1
    assert records[0].getMessage() == (
        "split=training real=1 psycholinguistics=0 enzyme_negative=0"
    )


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("enzymes", "[1, 'x']"),
        ("enzymes", "[1,"),
        ("relations", "[]"),
        ("pubmed_id", 0),
        ("year", "2024a"),
        ("bacteria", "{'abc': 'Escherichia coli'}"),
        ("bacteria", "{1: 'Escherichia coli'}"),
        ("relations", "{'HasEnzyme': [{'subject': 1}]}"),
        ("relations", "{'HasEnzyme': [{'subject': True, 'object': 2}]}"),
    ],
)
def test_a_malformed_row_is_refused_before_preprocessing(
    tiny_split, tmp_path, column, value
):
    """A cell off the split's shape raises the schema's error, not a
    `literal_eval`/`int` crash or a silent load. The cell sits on a row whose
    `pubmed_id` repeats, so merging duplicates would parse it first."""
    frame = tiny_split.astype({column: object})
    frame.loc[TEXTLESS, column] = value
    frame.to_csv(tmp_path / "training_data.csv")

    with pytest.raises(SchemaErrors):
        load_split("training")


def test_a_relation_naming_no_row_entity_still_loads(tiny_split, tmp_path):
    """Such a pair is left for `preprocess_relations` to drop, so the
    schema must not refuse the row it sits in."""
    frame = tiny_split.copy()
    frame.loc[TEXTLESS, "relations"] = (
        "{'HasEnzyme': [{'subject': 7, 'object': 8}]}"
    )
    frame.to_csv(tmp_path / "training_data.csv")

    real, _ = _real_and_synthetic(load_split("training"))
    assert real == USABLE
