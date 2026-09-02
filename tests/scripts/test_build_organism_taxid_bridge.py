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
