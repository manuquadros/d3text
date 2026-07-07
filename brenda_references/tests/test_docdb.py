from brenda_references.docdb import BrendaDocDB

import pathlib

TESTDB_PATH = pathlib.Path(__file__).parent / "test_files/testdb.json"


def test_strain_search():
    with BrendaDocDB(path=TESTDB_PATH) as docdb:
        assert docdb.strain_by_designation("ATCC 51142") is None

        designations = ("Schizosaccharomyces pombe", "ATCC 201872", "GK1")
        for name in designations:
            assert docdb.strain_by_designation(name).doc_id == 289


def test_bacteria_search():
    with BrendaDocDB(path=TESTDB_PATH) as docdb:
        assert docdb.bacteria_by_name("Crocosphaera subtropica") is None

        for name in ("Streptomyces septatus", "Streptomyces griseocarneus"):
            assert docdb.bacteria_by_name(name).doc_id == 6027
