import copy
import pytest
import functools
from scripts import fix_taxonomy
from brenda_references.docdb import BrendaDocDB
from tinydb.storages import MemoryStorage
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


def test_fix_bacteria():
    data = load_disk_test_data()

    with BrendaDocDB(storage="memory") as testdb:
        testdb._db.storage.write(copy.deepcopy(data))
        other_bacids = (978, 4346, 1665, 4358, 456)
        testdoc = testdb.documents.get(doc_id=287675)
        other_bac_names = set(
            name
            for _id, name in testdoc["other_organisms"].items()
            if _id in other_bacids
        )

        for name in other_bac_names:
            assert name not in testdoc["bacteria"].values()
            assert testdb.bacteria_by_name(name) is None

        fix_taxonomy.fix_taxonomy(testdb)
        testdoc = testdb.documents.get(doc_id=287675)
        testdoc_names = set()

        for bacid in testdoc["bacteria"].keys():
            record = testdb.get_bacteria(bacid)
            testdoc_names.update([record["organism"]], record["synonyms"])

        for name in other_bac_names:
            assert name in testdoc_names
            assert testdb.bacteria_by_name(name) is not None


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
        test_doc = testdb.documents.get(doc_id=DOC_ID)
        assert testdb.strain_by_designation("ATCC 23218") is not None

        bacteria = (
            "Agrobacterium rhizogenes",
            "Variovorax sp. P21",
            "Nocardiopsis dassonvillei",
        )

        for bac in bacteria:
            assert bac in testdoc["bacteria"].values()
