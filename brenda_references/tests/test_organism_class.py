"""`is_bacteria` classifies by NCBI lineage, not by the bacteria name list."""

import pytest
from taxonomy import ncbitax

from brenda_references import db
from brenda_references.db import is_bacteria

BACTERIA_ROOT = 2
ARCHAEON, PHYLUM, STRAIN_SPECIES, HUMAN = 2285, 1224, 562, 9606
STREPTOMYCES_TAXID, CANDIDA_TAXID, UNIDENTIFIED = 1883, 5475, 32644
TAXIDS = {
    "Aeropyrum pernix": ARCHAEON,
    "Pseudomonadota": PHYLUM,
    "Escherichia coli": STRAIN_SPECIES,
    "Homo sapiens": HUMAN,
    "Streptomyces": STREPTOMYCES_TAXID,
    "Candida": CANDIDA_TAXID,
    "unidentified": UNIDENTIFIED,
}
UNDER_BACTERIA = {PHYLUM, STRAIN_SPECIES, STREPTOMYCES_TAXID}


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


def test_genus_fallback_bacterial_sp_strain() -> None:
    """Unresolved 'Streptomyces sp. NCIM' resolves via genus to bacteria."""
    result = db.classify_organism("Streptomyces sp. NCIM")
    assert result == (True, True)


def test_genus_fallback_eukaryote_sp_strain() -> None:
    """Unresolved 'Candida sp. X' resolves via genus, eukaryote not bacteria."""
    result = db.classify_organism("Candida sp. X")
    assert result == (False, True)


def test_genus_excluded_if_virus_word() -> None:
    """A phage named after its host genus is not classified by that genus."""
    result = db.classify_organism("Streptomyces phage X")
    assert result.by_lineage is False


def test_lowercase_first_token_is_not_taken_as_a_genus() -> None:
    """'unidentified' resolves in NCBI but is no genus, so the list decides."""
    result = db.classify_organism("unidentified bacterium X")
    assert result.by_lineage is False
