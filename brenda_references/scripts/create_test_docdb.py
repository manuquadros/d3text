from brenda_references.docdb import BrendaDocDB
import pathlib

TEST_DIR = pathlib.Path(__file__).parent.parent / "tests"

if __name__ == "__main__":
    with (
        BrendaDocDB() as maindb,
        BrendaDocDB(
            path=str(TEST_DIR / "test_files/testdb.json"), create=True
        ) as testdb,
    ):
        sample_ids = (287675, 766653, 755668)

        for doc_id in sample_ids:
            sample = maindb.get_reference(doc_id)
            assert sample is not None, f"reference {doc_id} not found"
            testdb.insert(table="documents", record=sample)

            for tblname in (
                "enzymes",
                "bacteria",
                "strains",
                "other_organisms",
            ):
                for organism in sample.get(tblname, []):
                    record = maindb.get_record(
                        table=tblname, doc_id=int(organism)
                    )
                    if record is not None:
                        testdb.insert(table=tblname, record=record)
