"""`is_bacteria` classifies by NCBI lineage, not by the bacteria name list."""

import pytest
from taxonomy import ncbitax

from brenda_references import db
from brenda_references.db import is_bacteria

BACTERIA_ROOT = 2
ARCHAEON, PHYLUM, STRAIN_SPECIES, HUMAN = 2285, 1224, 562, 9606
TAXIDS = {
    "Aeropyrum pernix": ARCHAEON,
    "Pseudomonadota": PHYLUM,
    "Escherichia coli": STRAIN_SPECIES,
    "Homo sapiens": HUMAN,
}
UNDER_BACTERIA = {PHYLUM, STRAIN_SPECIES}


@pytest.fixture(autouse=True)
def fake_ncbitax(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ncbitax, "resolve_any_tax_id", TAXIDS.get)
    monkeypatch.setattr(
        ncbitax,
        "is_descendant",
        lambda tax_id, ancestor: ancestor == BACTERIA_ROOT
        and tax_id in UNDER_BACTERIA,
    )
    monkeypatch.setattr(
        ncbitax,
        "decompose_name",
        lambda q: ncbitax.DecomposedName(
            species="Escherichia coli" if q.startswith("Escherichia") else None,
            strain=None,
        ),
    )


def test_archaeon_in_the_name_list_is_not_bacteria() -> None:
    assert "Aeropyrum pernix" in db.bacteria
    assert not is_bacteria("Aeropyrum pernix")


def test_taxon_above_species_is_bacteria() -> None:
    assert is_bacteria("Pseudomonadota")


def test_eukaryote_is_not_bacteria() -> None:
    assert not is_bacteria("Homo sapiens")


def test_strain_name_resolves_through_its_species() -> None:
    assert is_bacteria("Escherichia coli K-12 MG1655")


def test_unresolved_name_falls_back_to_the_list() -> None:
    listed = next(n for n in sorted(db.bacteria) if n and n not in TAXIDS)
    assert is_bacteria(listed)


def test_classification_says_whether_lineage_decided() -> None:
    """The migration report trusts a lineage decision over a name-list one."""
    listed = next(n for n in sorted(db.bacteria) if n and n not in TAXIDS)
    assert db.classify_organism("Aeropyrum pernix") == (False, True)
    assert db.classify_organism("Escherichia coli K-12 MG1655") == (True, True)
    assert db.classify_organism(listed) == (True, False)
