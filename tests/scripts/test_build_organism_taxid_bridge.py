"""The taxid bridge resolves the corpus's names as written.

`other_organism_forms` also yields a genus abbreviation, because that is what
running text says; an index of NCBI's names is not running text. An
abbreviation NCBI lists under an unrelated taxon would make an entity whose
binomial resolved cleanly look contested, and the builder drops a contested
entity — so the expansion could only ever cost the table rows.
"""

import importlib.util
import pathlib

import polars as pl
import pytest
from taxonomy.ncbitax import ncbitax

from d3text.identifier_bridge import BridgeRow

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts/build_organism_taxid_bridge.py"
)


def _load_builder():
    """The builder script as a module, without putting `scripts/` on the path.

    Every name under `scripts/` is a top-level one, so importing by path keeps
    the whole directory from shadowing installed packages for the rest of the
    session.
    """
    spec = importlib.util.spec_from_file_location(_SCRIPT.stem, _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = _load_builder()

SCATOPHAGUS_ARGUS = 75038
SARGUS = 1000000
"""A taxon an index may list `S. argus` against, standing in for a real one."""

BACILLUS_SUBTILIS = 1423
BACILLUS_SUBTILIS_SUBTILIS = 135461
LACTOCOCCUS_LACTIS_LACTIS = 1360
HAFNIA_ALVEI = 569
HAFNIA_PARALVEI = 1082364
"""The taxid an all-division index answers `Hafnia alvei` with, standing in
for a name the two indexes disagree on."""

SUBTILIS_LPSN = 776360
"""LPSN's identifier for the subspecies, standing in for a real one."""

RETIRED = 2254
"""A taxid NCBI has merged away, standing in for one the cache still holds."""


def _split_csv(path: pathlib.Path, columns: list[dict[str, str]]):
    """A split csv carrying nothing but the inline other-organism column."""
    pl.DataFrame(
        {"other_organisms": [repr(column) for column in columns]},
        schema={"other_organisms": pl.Utf8},
    ).write_csv(path)
    return path


@pytest.fixture
def contested_abbreviation(monkeypatch: pytest.MonkeyPatch):
    """An index answering for a binomial and, differently, its abbreviation."""
    index = {
        ncbitax.normalize("Scatophagus argus"): (
            "Scatophagus argus",
            SCATOPHAGUS_ARGUS,
        ),
        ncbitax.normalize("S. argus"): ("Sargus", SARGUS),
    }
    monkeypatch.setattr(builder, "all_division_name_index", lambda: index)
    return index


def test_an_abbreviation_another_taxon_carries_keeps_the_row(
    tmp_path: pathlib.Path, contested_abbreviation
) -> None:
    """The entity is not contested; only the spelling the linker needs is."""
    path = _split_csv(tmp_path / "split.csv", [{"15528": "Scatophagus argus"}])

    rows, population = builder.other_organism_rows([str(path)], "oth")

    assert population == 1
    assert rows == [
        BridgeRow("oth15528", str(SCATOPHAGUS_ARGUS), "inline_name")
    ]


def test_two_verbatim_names_disagreeing_still_drop_the_entity(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An identifier two entities carry is gold for neither, and stays so.

    The names here are both the corpus's own, so the disagreement is about the
    entity rather than about an abbreviation nobody wrote.
    """
    index = {
        ncbitax.normalize("Nocardia erythropolis"): (
            "Nocardia erythropolis",
            1655,
        ),
        ncbitax.normalize("Nocardia rhodochrous"): (
            "Nocardia rhodochrous",
            1829,
        ),
    }
    monkeypatch.setattr(builder, "all_division_name_index", lambda: index)
    path = _split_csv(
        tmp_path / "split.csv",
        [
            {"978": "Nocardia erythropolis"},
            {"978": "Nocardia rhodochrous"},
        ],
    )

    rows, population = builder.other_organism_rows([str(path)], "oth")

    assert population == 1
    assert rows == []


@pytest.fixture
def mute_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bacteria-division resolver that answers nothing at all."""
    monkeypatch.setattr(ncbitax, "resolve_tax_id", lambda name: None)


@pytest.fixture
def subtilis_index() -> ncbitax.NameIndex:
    """An all-division index holding the trinomial and nothing else."""
    return {
        ncbitax.normalize("Lactococcus lactis lactis"): (
            "Lactococcus lactis subsp. lactis",
            LACTOCOCCUS_LACTIS_LACTIS,
        )
    }


def test_the_all_division_index_answers_where_resolve_tax_id_is_mute(
    mute_resolver: None, subtilis_index: ncbitax.NameIndex
) -> None:
    """Subspecies trinomials are the population the bacteria indexes miss.

    They are in the dump under every other index, so the muteness is a filter
    rather than an absence and the row it costs is recoverable.
    """
    row = builder.taxid_row(
        "bac1",
        {"organism": "Lactococcus lactis lactis"},
        subtilis_index,
        {},
    )

    assert row == BridgeRow(
        "bac1", str(LACTOCOCCUS_LACTIS_LACTIS), "organism_all_divisions"
    )


def test_a_synonym_reaches_the_all_division_index_too(
    mute_resolver: None, subtilis_index: ncbitax.NameIndex
) -> None:
    """The fallback runs the whole name list, not the organism alone."""
    row = builder.taxid_row(
        "bac1",
        {
            "organism": "Nothing NCBI holds",
            "synonyms": ["Lactococcus lactis lactis"],
        },
        subtilis_index,
        {},
    )

    assert row == BridgeRow(
        "bac1", str(LACTOCOCCUS_LACTIS_LACTIS), "synonym_all_divisions"
    )


def test_the_bacteria_division_answer_wins_over_the_all_division_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fallback is a fallback: it must not re-answer a resolved name.

    The two indexes disagree on a handful of names, and the rows already
    measured are the bacteria division's answers.
    """
    monkeypatch.setattr(ncbitax, "resolve_tax_id", lambda name: HAFNIA_ALVEI)
    index = {
        ncbitax.normalize("Hafnia alvei"): ("Hafnia alvei", HAFNIA_PARALVEI)
    }

    row = builder.taxid_row("bac1", {"organism": "Hafnia alvei"}, index, {})

    assert row == BridgeRow("bac1", str(HAFNIA_ALVEI), "organism")


def test_the_identifier_join_beats_a_name_resolving_to_the_species(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BRENDA's synonym for a subspecies is the binomial above it.

    So the name route answers, and answers with the parent — which is why the
    join is preferred outright rather than consulted where names are mute.
    """
    monkeypatch.setattr(
        ncbitax, "resolve_tax_id", lambda name: BACILLUS_SUBTILIS
    )

    row = builder.taxid_row(
        "bac3736",
        {
            "organism": "Bacillus subtilis subtilis",
            "synonyms": ["Bacillus subtilis"],
            "lpsn_id": SUBTILIS_LPSN,
        },
        {},
        {SUBTILIS_LPSN: BACILLUS_SUBTILIS_SUBTILIS},
    )

    assert row == BridgeRow(
        "bac3736", str(BACILLUS_SUBTILIS_SUBTILIS), "lpsn_id"
    )


def test_an_lpsn_id_no_strain_pairs_falls_through_to_the_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Most bacteria carry an identifier the cached taxa never mention."""
    monkeypatch.setattr(
        ncbitax, "resolve_tax_id", lambda name: BACILLUS_SUBTILIS
    )

    row = builder.taxid_row(
        "bac1",
        {"organism": "Bacillus subtilis", "lpsn_id": SUBTILIS_LPSN},
        {},
        {},
    )

    assert row == BridgeRow("bac1", str(BACILLUS_SUBTILIS), "organism")


def test_the_bacteria_half_consults_the_join_before_the_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The preference is the builder's, not something a caller assembles."""
    monkeypatch.setattr(
        ncbitax, "resolve_tax_id", lambda name: BACILLUS_SUBTILIS
    )
    monkeypatch.setattr(builder, "all_division_name_index", dict)
    monkeypatch.setattr(builder, "merged_taxids", dict)
    tables = {
        "bacteria": {
            "3736": {
                "organism": "Bacillus subtilis subtilis",
                "synonyms": ["Bacillus subtilis"],
                "lpsn_id": SUBTILIS_LPSN,
            }
        },
        "strains": {
            "1": {
                "taxon": {
                    "lpsn": SUBTILIS_LPSN,
                    "ncbi": BACILLUS_SUBTILIS_SUBTILIS,
                }
            }
        },
    }

    rows, population = builder.bacteria_rows(tables, "bac")

    assert population == 1
    assert rows == [
        BridgeRow("bac3736", str(BACILLUS_SUBTILIS_SUBTILIS), "lpsn_id")
    ]


def test_the_join_reads_the_lpsn_and_ncbi_a_strain_carries_together() -> None:
    """The pairing is identifier to identifier, with no name compared."""
    strains = {
        "1": {
            "taxon": {
                "name": "Bacillus subtilis subsp. subtilis",
                "lpsn": SUBTILIS_LPSN,
                "ncbi": BACILLUS_SUBTILIS_SUBTILIS,
            }
        },
        "2": {"taxon": None},
        "3": {},
    }

    assert builder.lpsn_taxids(strains, {}) == {
        SUBTILIS_LPSN: BACILLUS_SUBTILIS_SUBTILIS
    }


def test_a_retired_taxid_is_forwarded_before_the_pairing_is_checked() -> None:
    """The cached taxa predate the dump, so some name taxids NCBI merged away.

    Unforwarded, the two strains here read as two taxa and the identifier is
    dropped — losing the row to the very staleness the merge table repairs.
    """
    strains = {
        "1": {"taxon": {"lpsn": SUBTILIS_LPSN, "ncbi": RETIRED}},
        "2": {
            "taxon": {
                "lpsn": SUBTILIS_LPSN,
                "ncbi": BACILLUS_SUBTILIS_SUBTILIS,
            }
        },
    }

    paired = builder.lpsn_taxids(strains, {RETIRED: BACILLUS_SUBTILIS_SUBTILIS})

    assert paired == {SUBTILIS_LPSN: BACILLUS_SUBTILIS_SUBTILIS}


def test_an_lpsn_id_naming_two_taxa_pairs_with_neither() -> None:
    """One identifier over two live taxids leaves the entity in doubt.

    Dropping it is what `inline_name_row` does with two disagreeing names,
    and for the same reason: a row picked by dict order is a gold nobody can
    check.
    """
    strains = {
        "1": {"taxon": {"lpsn": SUBTILIS_LPSN, "ncbi": 358}},
        "2": {"taxon": {"lpsn": SUBTILIS_LPSN, "ncbi": 362}},
    }

    assert builder.lpsn_taxids(strains, {}) == {}
