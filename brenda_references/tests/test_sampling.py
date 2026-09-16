import json
import pathlib
from brenda_references import relation_records

TEST_FILES = pathlib.Path(__file__).parent / "test_files"

with (TEST_FILES / "3170582.json").open() as test_file:
    test_doc = json.load(test_file)


def test_relation_records() -> None:
    assert relation_records(test_doc) == [
        {
            "pubmed_id": "3170582",
            "predicate": "HasEnzyme",
            "subject": "oos_6500",
            "object": "enz_3494",
        }
    ]
