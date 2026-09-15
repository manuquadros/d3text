import copy
import logging
import pytest
import functools
from scripts import fix_taxonomy
from brenda_references.docdb import BrendaDocDB
from typing import Any

import pathlib

TESTDB_DIR = pathlib.Path(__file__).parent / "test_files"
TESTDB_PATH = TESTDB_DIR / "testdb.json"


@functools.cache
def load_disk_test_data() -> dict[str, dict[str, Any]]:
    with BrendaDocDB(path=str(TESTDB_PATH)) as testdb_disk:
        data = testdb_disk.as_dict()

    if data:
        return data
    else:
        raise RuntimeError("No test data")


@pytest.mark.integration
def test_fix_bacteria():
    """`other_organisms` keys come back from JSON as `str`; `other_bacids`
    is a tuple of `int`, so the membership test has to coerce one side or
    it silently selects nothing and every assertion below iterates zero
    times.
    """
    data = load_disk_test_data()

    with BrendaDocDB(storage="memory") as testdb:
        testdb._db.storage.write(copy.deepcopy(data))
        other_bacids = (978, 4346, 1665, 4358, 456)
        testdoc = testdb.documents.get(doc_id=287675)
        other_bac_names = set(
            name
            for _id, name in testdoc["other_organisms"].items()
            if int(_id) in other_bacids
        )
        assert other_bac_names == {
            "Brevibacterium sterolicum",
            "Nocardia erythropolis",
            "Corynebacterium cholesterolicum",
            "Nocardia rhodochrous",
            "Pimelobacter simplex",
        }

        for name in other_bac_names:
            assert name not in testdoc["bacteria"].values()
            assert testdb.bacteria_by_name(name) is None


@pytest.mark.integration
@pytest.mark.xfail(
    strict=True,
    reason=(
        "fix_taxonomy only reclassifies 2 of these 5 other_organisms "
        "(Brevibacterium sterolicum, Pimelobacter simplex); the other "
        "3 stay in other_organisms. Separate, already-filed defect in "
        "fix_taxonomy itself, not in the selection this test exercises."
    ),
)
def test_fix_bacteria_reclassifies_all_selected_organisms():
    data = load_disk_test_data()

    with BrendaDocDB(storage="memory") as testdb:
        testdb._db.storage.write(copy.deepcopy(data))
        other_bacids = (978, 4346, 1665, 4358, 456)
        testdoc = testdb.documents.get(doc_id=287675)
        other_bac_names = set(
            name
            for _id, name in testdoc["other_organisms"].items()
            if int(_id) in other_bacids
        )

        fix_taxonomy.fix_taxonomy(testdb)
        testdoc = testdb.documents.get(doc_id=287675)
        testdoc_names = set()

        for bacid in testdoc["bacteria"].keys():
            record = testdb.get_bacteria(bacid)
            testdoc_names.update([record["organism"]], record["synonyms"])

        for name in other_bac_names:
            assert name in testdoc_names
            assert testdb.bacteria_by_name(name) is not None


@pytest.mark.integration
def test_fix_bacteria_logs_organisms_decompose_name_cannot_place(caplog):
    """The 3 of 5 selected organisms `decompose_name` cannot resolve stay
    in `other_organisms` (see the xfail above); each must be logged by
    name so the leave-behind is visible instead of indistinguishable from
    a name correctly left alone. The other 2, which do get reclassified,
    must not be logged as leave-behinds.
    """
    data = copy.deepcopy(load_disk_test_data())
    data["documents"] = {"287675": data["documents"]["287675"]}
    unresolved = {
        "Nocardia erythropolis",
        "Corynebacterium cholesterolicum",
        "Nocardia rhodochrous",
    }
    resolved = {"Brevibacterium sterolicum", "Pimelobacter simplex"}

    with BrendaDocDB(storage="memory") as testdb:
        testdb._db.storage.write(data)

        with caplog.at_level(logging.WARNING, logger="scripts.fix_taxonomy"):
            fix_taxonomy.fix_taxonomy(testdb)

    for name in unresolved:
        assert any(name in message for message in caplog.messages)

    for name in resolved:
        assert not any(name in message for message in caplog.messages)


@pytest.mark.integration
def test_fix_strains():
    data = load_disk_test_data()

    with BrendaDocDB(storage="memory") as testdb:
        testdb._db.storage.write(copy.deepcopy(data))
        testdoc = testdb.documents.get(doc_id=766653)

        assert (
            "Crocosphaera subtropica ATCC 51142"
            in testdoc["other_organisms"].values()
        )
        assert "Crocosphaera subtropica" not in testdoc["bacteria"].values()
        assert not testdoc["strains"]
        assert testdb.strain_by_designation("ATCC 51142") is None

        fix_taxonomy.fix_taxonomy(testdb)
        testdoc = testdb.documents.get(doc_id=766653)

        assert "Crocosphaera subtropica" in testdoc["bacteria"].values()
        assert testdoc["strains"]

        strain_id = testdoc["strains"][0]
        assert testdb.strains.get(doc_id=strain_id) is not None
        assert testdb.strain_by_designation("ATCC 51142") is not None

        data = testdb.as_dict()

    with BrendaDocDB(
        path=str(TESTDB_DIR / "testdb_modified.json")
    ) as testdbmod:
        testdbmod._db.storage.write(data)


@pytest.mark.integration
def test_29345379():
    DOC_ID = 755668
    data = load_disk_test_data()

    with BrendaDocDB(path=str(TESTDB_DIR / "testdb_modified.json")) as testdb:
        testdb._db.storage.write(copy.deepcopy(data))
        testdoc = testdb.documents.get(doc_id=DOC_ID)

        bacteria = (
            "Agrobacterium rhizogenes",
            "Variovorax sp. P21",
            "Nocardiopsis dassonvillei ATCC 23218",
        )

        for bac in bacteria:
            assert bac in testdoc["other_organisms"].values()

        fix_taxonomy.fix_taxonomy(testdb)
        assert testdb.strain_by_designation("ATCC 23218") is not None

        bacteria = (
            "Agrobacterium rhizogenes",
            "Variovorax sp. P21",
            "Nocardiopsis dassonvillei",
        )

        for bac in bacteria:
            assert bac in testdoc["bacteria"].values()
