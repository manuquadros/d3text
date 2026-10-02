"""Pin the organism-class invariant on the migrated document database.

Every organism placement in the document db and splits must agree with NCBI
lineage: a name stored under `bacteria` resolves to a taxid descending from
`BACTERIA_TAX_ID`, one under `other_organisms` to a taxid that does not.

The invariant covers only names that `ncbitax.resolve_any_tax_id` resolves;
names decided by decomposition or the fallback name list are out of scope.
"""

from __future__ import annotations

import ast
import csv
import pathlib
import sys

import pytest
from taxonomy import ncbitax

from brenda_references import data_paths
from brenda_references.db import BACTERIA_TAX_ID
from brenda_references.docdb import BrendaDocDB

CLASSES = ("bacteria", "other_organisms")


def _collect_organisms_from_docdb() -> set[tuple[str, str]]:
    """Every `(name, stored_class)` in the document db; skip if absent."""
    placements: set[tuple[str, str]] = set()
    try:
        with BrendaDocDB() as db:
            for doc in db.documents:
                for stored_class in CLASSES:
                    for name in doc.get(stored_class, {}).values():
                        placements.add((name, stored_class))
    except FileNotFoundError:
        pytest.skip("Document database not found")
    return placements


def _collect_organisms_from_split(split: str) -> set[tuple[str, str]]:
    """Every `(name, stored_class)` in one split CSV; skip if absent."""
    path = data_paths.split_path(split)
    if not path.exists():
        pytest.skip(f"Split file not found: {path}")

    placements: set[tuple[str, str]] = set()
    csv.field_size_limit(sys.maxsize)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            for stored_class in CLASSES:
                for name in ast.literal_eval(row[stored_class]).values():
                    placements.add((name, stored_class))
    return placements


def _check_organisms(
    placements: set[tuple[str, str]], source_name: str
) -> None:
    """Fail naming every placement whose class disagrees with NCBI lineage."""
    tax_ids = {
        name: ncbitax.resolve_any_tax_id(name)
        for name in {name for name, _ in placements}
    }
    under_bacteria = {
        name: ncbitax.is_descendant(tax_id, BACTERIA_TAX_ID)
        for name, tax_id in tax_ids.items()
        if tax_id is not None
    }
    wrong = sorted(
        (name, stored_class)
        for name, stored_class in placements
        if name in under_bacteria
        and under_bacteria[name] != (stored_class == "bacteria")
    )
    if wrong:
        sample = "".join(
            f"\n  {name!r} stored as {stored_class!r}, taxid {tax_ids[name]}"
            for name, stored_class in wrong[:3]
        )
        more = f"\n  ... and {len(wrong) - 3} more" if len(wrong) > 3 else ""
        pytest.fail(
            f"{source_name}: {len(wrong)} placement(s) in the wrong class:"
            + sample
            + more
        )


@pytest.mark.integration
def test_docdb_organisms_classified_by_ncbi_lineage() -> None:
    """Every organism in the document database is classified correctly.

    An archaea name would fail here if the document db still held the old
    (pre-migration) name-list classification.
    """
    _check_organisms(_collect_organisms_from_docdb(), "documents.json")


@pytest.mark.integration
@pytest.mark.parametrize("split", ["training", "validation", "test"])
def test_split_organisms_classified_by_ncbi_lineage(split: str) -> None:
    """Every organism in a split CSV is classified correctly.

    The invariant is per-split since each split can drift independently
    if not regenerated.
    """
    _check_organisms(_collect_organisms_from_split(split), f"{split} split")


def test_a_wrong_placement_is_not_masked_by_a_right_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A name filed in both classes is checked in each, not only the last."""
    archaeon = "Methanocaldococcus jannaschii"
    monkeypatch.setattr(ncbitax, "resolve_any_tax_id", {archaeon: 2190}.get)
    monkeypatch.setattr(ncbitax, "is_descendant", lambda tax_id, root: False)
    split = tmp_path / "split.csv"
    split.write_text(
        "bacteria,other_organisms\n"
        f"\"{{1: '{archaeon}'}}\",{{}}\n"
        f"{{}},\"{{2: '{archaeon}'}}\"\n"
    )
    monkeypatch.setattr(data_paths, "split_path", lambda name: split)

    with pytest.raises(pytest.fail.Exception, match=archaeon):
        _check_organisms(_collect_organisms_from_split("test"), "split")


@pytest.mark.integration
@pytest.mark.parametrize("split", ["training", "validation", "test"])
def test_split_species_name_an_organism_of_their_row(split: str) -> None:
    """Every `HasSpecies` object of a row is in its organism columns.

    `preprocess_relations` drops a pair whose object is in neither column,
    so the species such a document is about becomes a negative.
    """
    path = data_paths.split_path(split)
    if not path.exists():
        pytest.skip(f"Split file not found: {path}")

    lost: list[tuple[str, int]] = []
    csv.field_size_limit(sys.maxsize)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            organisms = {
                int(ident)
                for stored_class in CLASSES
                for ident in ast.literal_eval(row[stored_class])
            }
            lost += [
                (row["pubmed_id"], pair["object"])
                for pair in ast.literal_eval(row["relations"]).get(
                    "HasSpecies", []
                )
                if pair["object"] not in organisms
            ]
    assert not lost, (
        f"{split}: {len(lost)} HasSpecies object(s) in no organism column, "
        f"e.g. (pubmed_id, object) {lost[:3]}"
    )
