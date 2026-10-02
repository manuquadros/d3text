"""StrainInfo records convert to the `strains` table's stored shape."""

from apiadapters.straininfo import StrainRecord

from brenda_references.straininfo import strain_record_to_d3types


def test_a_full_record_dumps_to_the_stored_strain_shape() -> None:
    """The relation lists flatten into `cultures` and `designations`.

    The dump is what every `strains` row holds, so a field lost or renamed
    here changes the table.
    """
    record: StrainRecord = {
        "id": 42,
        "doi": "10.60712/SI-ID42.1",
        "merged": [41],
        "bacdive": 99,
        "taxon": {"name": "Bacillus subtilis", "lpsn": 111, "ncbi": 1423},
        "relation": {
            "designation": ["NCIB 3610", "168"],
            "culture": [
                {"id": 10, "strain_number": "DSM 10", "origin": "x"},
            ],
        },
    }

    assert strain_record_to_d3types(record).model_dump() == {
        "id": 42,
        "doi": "10.60712/SI-ID42.1",
        "merged": [41],
        "bacdive": 99,
        "taxon": {"name": "Bacillus subtilis", "lpsn": 111, "ncbi": 1423},
        "cultures": [{"siid": 10, "strain_number": "DSM 10"}],
        "designations": ["168", "NCIB 3610"],
    }


def test_a_bare_record_dumps_with_empty_relations() -> None:
    record: StrainRecord = {"id": 1, "relation": {}}

    assert strain_record_to_d3types(record).model_dump() == {
        "id": 1,
        "doi": None,
        "merged": None,
        "bacdive": None,
        "taxon": None,
        "cultures": [],
        "designations": [],
    }
