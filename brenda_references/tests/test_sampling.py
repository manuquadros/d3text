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


def test_hasenzyme_subject_matches_int_strain_or_bacterium() -> None:
    """A strain/bacterium entity gets the same prefix whether it appears
    as a `HasSpecies` operand or as a `HasEnzyme` subject, whether
    `strains`/`bacteria` hold ints (in-memory) or the JSON-round-tripped
    string keys `bacteria` normally has.
    """
    doc = {
        "pubmed_id": "1",
        "bacteria": {42: "some species"},
        "strains": [289],
        "relations": {
            "HasSpecies": [{"subject": 289, "object": 42}],
            "HasEnzyme": [
                {"subject": 289, "object": 1},
                {"subject": 42, "object": 2},
            ],
        },
    }

    assert relation_records(doc) == [
        {
            "pubmed_id": "1",
            "predicate": "HasSpecies",
            "subject": "str_289",
            "object": "bac_42",
        },
        {
            "pubmed_id": "1",
            "predicate": "HasEnzyme",
            "subject": "str_289",
            "object": "enz_1",
        },
        {
            "pubmed_id": "1",
            "predicate": "HasEnzyme",
            "subject": "bac_42",
            "object": "enz_2",
        },
    ]
