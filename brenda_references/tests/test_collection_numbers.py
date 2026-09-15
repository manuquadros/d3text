from brenda_references.collection_numbers import is_collection_number


def test_admits_real_collection_numbers() -> None:
    for designation in ("ATCC 9202", "DSM 4252", "NCTC8511", "CBS 111.30"):
        assert is_collection_number(designation), designation


def test_rejects_generic_designations() -> None:
    for designation in ("A2", "F1", "UC", "P-24", "D273", "25433", "CHO-K1"):
        assert not is_collection_number(designation), designation


def test_rejects_a_number_the_grammar_only_partly_reads() -> None:
    """A compound suffix must not be truncated into a false accession."""
    assert not is_collection_number("ATCC BAA-245")
    assert not is_collection_number("LMG 16656QC1/01")
