import ast

import pandas as pd
import pytest
from apiadapters.ncbi.parser import is_scanned
from brenda_references.brenda_references import (
    DATA_DIR,
    merge_duplicate_documents,
    preprocess_labels,
)


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
    df = pd.read_csv(DATA_DIR / f"{split}_data.csv", index_col=0)
    assert df["pubmed_id"].duplicated().any(), (
        f"{split}_data.csv has no duplicate pubmed_id left to merge; "
        "this test no longer exercises the fix"
    )

    merged = merge_duplicate_documents(df)

    assert not merged["pubmed_id"].duplicated().any()
    if split == "validation":
        for pmid in (25401070, 32717805):
            assert (merged["pubmed_id"] == pmid).sum() == 1
