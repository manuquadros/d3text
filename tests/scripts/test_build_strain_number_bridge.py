"""A deposit recorded only in `designations` must still reach the bridge.

`surface_forms.strain_forms` indexes `cultures` and `designations` alike, so
a strain record naming its deposit only in `designations` — with a *different*
strain record naming the same deposit through `cultures` — must produce a row
for each: the bridge otherwise names one candidate while the surface-form
index reaches two, and a correct span scores strict-wrong.
"""

import importlib.util
import json
import pathlib

from d3text.identifier_bridge import BridgeRow

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts/build_strain_number_bridge.py"
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


def _dump(path: pathlib.Path, strains: dict[str, dict]) -> str:
    path.write_text(json.dumps({"strains": strains}), encoding="utf8")
    return str(path)


def test_a_designation_only_deposit_reaches_the_bridge(
    tmp_path: pathlib.Path,
) -> None:
    """The ticket's worked example: ATCC 4698 named on two strain records."""
    path = _dump(
        tmp_path / "documents.json",
        {
            "12677": {"designations": ["ATCC4698"], "cultures": []},
            "13088": {
                "designations": [],
                "cultures": [{"strain_number": "ATCC 4698"}],
            },
        },
    )

    rows, curated, deposits = builder.strain_rows(path, "str")

    assert curated == 2
    assert deposits == 1
    assert rows == [
        BridgeRow("str12677", "ATCC 4698", builder.DESIGNATION),
        BridgeRow("str13088", "ATCC 4698", builder.CULTURE_NUMBER),
    ]


def test_a_designation_that_is_not_an_accession_is_dropped(
    tmp_path: pathlib.Path,
) -> None:
    """A designation like `K-12` names a strain in a paper, not a deposit."""
    path = _dump(
        tmp_path / "documents.json",
        {"1": {"designations": ["K-12"], "cultures": []}},
    )

    rows, curated, deposits = builder.strain_rows(path, "str")

    assert curated == 1
    assert deposits == 0
    assert rows == []


def test_the_same_strain_naming_a_deposit_in_both_fields_stays_one_entity(
    tmp_path: pathlib.Path,
) -> None:
    """A record's own designation and culture entry agreeing name one strain.

    Not the ticket's collision case — that is two *different* strain
    records — so nothing here should read as ambiguity between two entities.
    """
    path = _dump(
        tmp_path / "documents.json",
        {
            "1": {
                "designations": ["ATCC4698"],
                "cultures": [{"strain_number": "ATCC 4698"}],
            }
        },
    )

    rows, _curated, _deposits = builder.strain_rows(path, "str")

    assert rows == [BridgeRow("str1", "ATCC 4698", builder.CULTURE_NUMBER)]
