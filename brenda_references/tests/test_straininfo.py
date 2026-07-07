from apiadapters.straininfo import normalize_strain_names


def test_parse_standard() -> None:
    standard_names = ("Delft L 40", "STAFF 1027", "DSMZ 2213")
    assert normalize_strain_names(standard_names) == set(standard_names)


def test_parse_non_standard() -> None:
    names = (
        "544 / ATCC 23448",
        "MNYC/BZ/M379",
        "NRRL B771",
        "NRRLB 15444r",
        "Gasser AM64T",
    )
    assert normalize_strain_names(names) == (
        set(names)
        | {
            "544",
            "ATCC 23448",
            "MNYC",
            "BZ",
            "M379",
            "NRRL B-771",
            "NRRL B-15444r",
            "Gasser AM64",
        }
    )
