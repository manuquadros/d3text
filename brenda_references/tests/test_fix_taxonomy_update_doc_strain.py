"""`update_doc_strain` stores StrainInfo's answer as a `d3types.Strain`."""

import os
import subprocess
import sys
from collections.abc import Collection, Mapping
from pathlib import Path

import pytest
from apiadapters.straininfo import StrainRecord

from brenda_references.docdb import BrendaDocDB
from scripts import fix_taxonomy

BREF = Path(__file__).resolve().parents[1]
REPO = BREF.parent

RECORD: StrainRecord = {
    "id": 7,
    "relation": {
        "designation": ["NCIB 8"],
        "culture": [{"id": 3, "strain_number": "DSM 579"}],
    },
}


class _ResolvingAdapter:
    """Resolves every key to `RECORD`, as StrainInfo would on a hit."""

    def __enter__(self) -> "_ResolvingAdapter":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def retrieve_strain_models(
        self, designations: Mapping[int, Collection[str]]
    ) -> dict[int, StrainRecord]:
        return dict.fromkeys(designations, RECORD)


def _stored_strain(strainname: str) -> dict:
    with BrendaDocDB(storage="memory") as testdb:
        testdb.documents.insert({"strains": []})
        doc = testdb.documents.get(doc_id=1)
        strainid = fix_taxonomy.update_doc_strain(testdb, doc, strainname)
        assert testdb.documents.get(doc_id=1)["strains"] == [strainid]
        return dict(testdb.strains.get(doc_id=strainid))


@pytest.mark.usefixtures("fake_strain_network")
def test_an_unresolved_name_is_stored_as_an_id_less_placeholder() -> None:
    """StrainInfo leaves an unresolved key out of its answer.

    The name is still stored, as the `id: None` row `fix_missing_strains.py`
    retries later.
    """
    assert _stored_strain("XYZ 1") == {
        "id": None,
        "doi": None,
        "merged": None,
        "bacdive": None,
        "taxon": None,
        "cultures": [],
        "designations": ["XYZ 1"],
    }


def test_a_resolved_name_is_stored_as_its_converted_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fix_taxonomy, "StrainInfoAdapter", _ResolvingAdapter)

    stored = _stored_strain("DSM 579")

    assert stored["id"] == 7
    assert stored["designations"] == ["NCIB 8"]
    assert stored["cultures"] == [{"siid": 3, "strain_number": "DSM 579"}]


def test_fix_taxonomy_type_checks(tmp_path: Path) -> None:
    """mypy finds no error in `scripts/fix_taxonomy.py`.

    Neither root gate reaches `brenda_references/`, so this is what keeps
    `update_doc_strain`'s adapter call typed. `MYPYPATH` puts this tree's
    package first and keeps any caller-supplied entries after it.
    """
    pytest.importorskip("mypy")
    mypypath = [str(BREF / "src"), os.environ.get("MYPYPATH", "")]
    env = dict(os.environ, MYPYPATH=os.pathsep.join(filter(None, mypypath)))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--config-file",
            str(REPO / "pyproject.toml"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--no-color-output",
            str(BREF / "scripts" / "fix_taxonomy.py"),
        ],
        cwd=BREF,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
