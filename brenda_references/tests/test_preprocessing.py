import ast
import logging

import pandas as pd
import pytest
from apiadapters.ncbi.parser import is_scanned
from brenda_references import brenda_references as br
from brenda_references.brenda_references import (
    merge_duplicate_documents,
    preprocess_labels,
)
from brenda_references.data_paths import split_path


def test_is_scanned():
    xml = """<jats:body xmlns:jats=\"https://jats.nlm.nih.gov/ns/archiving/1.3/\">\n    <jats:supplementary-material content-type=\"scanned-pages\" position=\"float\">\n      <jats:graphic xmlns:xlink=\"http://www.w3.org/1999/xlink\" position=\"float\" xlink:href=\"brjcancer00184-0039.tif\" xlink:role=\"969\" xlink:title=\"scanned-page\"></jats:graphic>\n      <jats:graphic xmlns:xlink=\"http://www.w3.org/1999/xlink\" position=\"float\" xlink:href=\"brjcancer00184-0040.tif\" xlink:role=\"970\" xlink:title=\"scanned-page\"></jats:graphic>\n      <jats:graphic xmlns:xlink=\"http://www.w3.org/1999/xlink\" position=\"float\" xlink:href=\"brjcancer00184-0041.tif\" xlink:role=\"971\" xlink:title=\"scanned-page\"></jats:graphic>\n      <jats:graphic xmlns:xlink=\"http://www.w3.org/1999/xlink\" position=\"float\" xlink:href=\"brjcancer00184-0042.tif\" xlink:role=\"972\" xlink:title=\"scanned-page\"></jats:graphic>\n      <jats:graphic xmlns:xlink=\"http://www.w3.org/1999/xlink\" position=\"float\" xlink:href=\"brjcancer00184-0043.tif\" xlink:role=\"973\" xlink:title=\"scanned-page\"></jats:graphic>\n      <jats:graphic xmlns:xlink=\"http://www.w3.org/1999/xlink\" position=\"float\" xlink:href=\"brjcancer00184-0044.tif\" xlink:role=\"974\" xlink:title=\"scanned-page\"></jats:graphic>\n      <jats:graphic xmlns:xlink=\"http://www.w3.org/1999/xlink\" position=\"float\" xlink:href=\"brjcancer00184-0045.tif\" xlink:role=\"975\" xlink:title=\"scanned-page\"></jats:graphic>\n      <jats:graphic xmlns:xlink=\"http://www.w3.org/1999/xlink\" position=\"float\" xlink:href=\"brjcancer00184-0046.tif\" xlink:role=\"976\" xlink:title=\"scanned-page\"></jats:graphic>\n      <jats:graphic xmlns:xlink=\"http://www.w3.org/1999/xlink\" position=\"float\" xlink:href=\"brjcancer00184-0047.tif\" xlink:role=\"977\" xlink:title=\"scanned-page\"></jats:graphic>\n    </jats:supplementary-material>\n  </jats:body>"""

    assert is_scanned(xml) is True


def test_none_fill_spells_pairs_the_way_the_typed_keys_are_spelled() -> None:
    """The `none` fill must key a pair the way a typed key of it would be.

    The fill walks the entity columns (bacteria, enzymes, strains,
    other_organisms) while the typed keys are sorted, and the two orders
    disagree for a (strain, other_organism) pair. A fill key spelled in
    column order would miss a typed key of the same pair, leaving the
    document holding that pair twice under two different labels.
    """
    frame = pd.DataFrame(
        {
            "bacteria": ["{}"],
            "enzymes": ["[5]"],
            "strains": ["[3]"],
            "other_organisms": ["{7: 'Vibrio sp.'}"],
            "relations": ["{'HasEnzyme': [{'subject': 3, 'object': 5}]}"],
        }
    )

    processed = preprocess_labels(frame)
    pairs = processed["relations"].iloc[0][0]

    assert processed["entities"].iloc[0] == ["enz5", "str3", "oth7"]
    assert set(pairs) == {
        ("enz5", "str3"),
        ("enz5", "oth7"),
        ("oth7", "str3"),
    }
    assert pairs[("enz5", "str3")].tolist() == [1.0, 0.0, 0.0]


def test_hasspecies_object_is_keyed_by_the_column_holding_it() -> None:
    """A `HasSpecies` object filed under `other_organisms` gets an `oth`
    key, not the `bac` key every object used to get regardless of which
    column actually held it.
    """
    frame = pd.DataFrame(
        {
            "bacteria": ["{}"],
            "enzymes": ["[]"],
            "strains": ["[3]"],
            "other_organisms": ["{7: 'Vibrio sp.'}"],
            "relations": ["{'HasSpecies': [{'subject': 3, 'object': 7}]}"],
        }
    )

    processed = preprocess_labels(frame)
    pairs = processed["relations"].iloc[0][0]

    assert set(pairs) == {("oth7", "str3")}
    assert pairs[("oth7", "str3")].tolist() == [0.0, 1.0, 0.0]


def test_hasspecies_pair_with_object_in_no_column_is_dropped_and_reported(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An object naming no entity column is dropped, and the drop is
    reported once per `preprocess_labels` call at WARNING: that prints at
    `d3text.logs.configure`'s default level (`DEFAULT_LEVEL =
    logging.INFO`) and still prints under `D3TEXT_LOG_LEVEL=WARNING`.
    """
    frame = pd.DataFrame(
        {
            "bacteria": ["{}"],
            "enzymes": ["[]"],
            "strains": ["[3]"],
            "other_organisms": ["{}"],
            "relations": ["{'HasSpecies': [{'subject': 3, 'object': 99}]}"],
        }
    )

    with caplog.at_level(
        logging.WARNING, logger="brenda_references.brenda_references"
    ):
        processed = preprocess_labels(frame)

    pairs = processed["relations"].iloc[0][0]
    assert pairs == {}
    assert any(
        record.levelno == logging.WARNING
        and "dropped 1 HasSpecies pair" in record.getMessage()
        for record in caplog.records
    )


def test_hasspecies_pair_whose_subject_is_not_a_strain_is_dropped() -> None:
    """A subject absent from `strains` never becomes a `str` key, the same
    membership check `HasEnzyme` already applies to its own subject.
    """
    frame = pd.DataFrame(
        {
            "bacteria": ["{7: 'Vibrio sp.'}"],
            "enzymes": ["[]"],
            "strains": ["[]"],
            "other_organisms": ["{}"],
            "relations": ["{'HasSpecies': [{'subject': 3, 'object': 7}]}"],
        }
    )

    processed = preprocess_labels(frame)
    pairs = processed["relations"].iloc[0][0]

    assert pairs == {}


def test_typed_relation_keys_only_name_row_entities() -> None:
    """Every typed key's two arguments are members of the row's `entities`,
    dropped `HasSpecies` pairs included: neither a subject outside
    `strains` nor an object outside `bacteria`/`other_organisms` may
    surface as a key naming an entity the row never declared.
    """
    frame = pd.DataFrame(
        {
            "bacteria": ["{7: 'Vibrio sp.'}"],
            "enzymes": ["[]"],
            "strains": ["[3]"],
            "other_organisms": ["{}"],
            "relations": [
                "{'HasSpecies': ["
                "{'subject': 3, 'object': 7}, "
                "{'subject': 3, 'object': 99}, "
                "{'subject': 5, 'object': 7}"
                "]}"
            ],
        }
    )

    processed = preprocess_labels(frame)
    row = processed.iloc[0]
    entities = set(row["entities"])

    for arg1, arg2 in row["relations"][0]:
        assert arg1 in entities
        assert arg2 in entities


def test_merge_duplicate_documents_is_a_noop_without_duplicates() -> None:
    """A frame with unique `pubmed_id`s comes back unchanged."""
    frame = pd.DataFrame(
        {
            "pubmed_id": [1, 2],
            "path": ["a.pdf", None],
            "enzymes": ["[1]", "[2]"],
            "strains": ["[]", "[]"],
            "entity_spans": ["[]", "[]"],
            "bacteria": ["{}", "{}"],
            "other_organisms": ["{}", "{}"],
            "relations": ["{}", "{}"],
        }
    )

    assert merge_duplicate_documents(frame) is frame


def test_merge_duplicate_documents_unions_gold_sets() -> None:
    """BRENDA's one-reference-per-enzyme rows collapse into one document.

    Mirrors the real pmid 23419073 (train) and 25401070 (validation) cases:
    the same paper curated once per enzyme, each row holding only that
    enzyme's slice of the gold set and one row missing `path`.
    """
    frame = pd.DataFrame(
        {
            "pubmed_id": [23419073, 23419073],
            "path": [None, "apis.pdf"],
            "enzymes": ["[44265]", "[67057]"],
            "strains": ["[]", "[]"],
            "entity_spans": ["[]", "[]"],
            "bacteria": ["{}", "{}"],
            "other_organisms": ["{}", "{}"],
            "relations": [
                "{'HasEnzyme': [{'subject': 14052, 'object': 44265}]}",
                "{'HasEnzyme': [{'subject': 14052, 'object': 67057}]}",
            ],
        }
    )

    merged = merge_duplicate_documents(frame)

    assert len(merged) == 1
    row = merged.iloc[0]
    assert row["path"] == "apis.pdf"
    assert sorted(ast.literal_eval(row["enzymes"])) == [44265, 67057]
    assert ast.literal_eval(row["relations"]) == {
        "HasEnzyme": [
            {"subject": 14052, "object": 44265},
            {"subject": 14052, "object": 67057},
        ]
    }


@pytest.mark.integration
@pytest.mark.parametrize("split", ["training", "validation", "test"])
def test_splits_have_no_duplicate_pubmed_id_after_merge(split: str) -> None:
    """The checked-in split CSVs no longer collide on `pubmed_id`.

    Regression for the two validation IDs (25401070, 32717805) a 256-row
    sample once found reading each other's gold masks out of the
    pubmed-id-keyed token-label store.
    """
    df = pd.read_csv(split_path(split), index_col=0)
    assert df["pubmed_id"].duplicated().any(), (
        f"{split}_data.csv has no duplicate pubmed_id left to merge; "
        "this test no longer exercises the fix"
    )

    merged = merge_duplicate_documents(df)

    assert not merged["pubmed_id"].duplicated().any()
    if split == "validation":
        for pmid in (25401070, 32717805):
            assert (merged["pubmed_id"] == pmid).sum() == 1


def test_noise_documents_skips_pool_load_when_noise_is_zero(
    monkeypatch,
) -> None:
    """`noise_documents(split, noise=0)` must not touch the noise pool.

    `_pool_block` already guards on `noise <= 0`, but the guard ran too
    late: `psycholinguistics_data()` was evaluated as the argument before
    `_pool_block` was ever entered, so a noise=0 caller still paid for (and
    could still crash on) loading the pool file.
    """

    def _fail_if_called() -> pd.DataFrame:
        raise AssertionError("psycholinguistics_data must not be called")

    monkeypatch.setattr(br, "psycholinguistics_data", _fail_if_called)

    assert br.noise_documents("training", 0).empty


def test_enzyme_negative_documents_skips_pool_load_when_noise_is_zero(
    monkeypatch,
) -> None:
    """Same guard, other pool: `enzyme_negative_documents` at noise=0.

    Regression for `load_split(..., enzyme_noise=0)` — the default, so
    every plain `training_data()`/`validation_data()`/`test_data()` call —
    raising `FileNotFoundError` reading `enzyme_negative_pool.json`, a file
    this checkout doesn't carry, even though noise=0 never needed it.
    """

    def _fail_if_called() -> pd.DataFrame:
        raise AssertionError("enzyme_negative_data must not be called")

    monkeypatch.setattr(br, "enzyme_negative_data", _fail_if_called)

    assert br.enzyme_negative_documents("training", 0).empty
