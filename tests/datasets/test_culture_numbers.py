"""The culture-number grammar, and the comma that means two things at once.

`DSM 22,228` is one deposit number and `ATCC 35984, 35983` is two, so neither
stopping at the comma nor stripping every comma is right. Both failures are
silent and both fabricate an identifier that BRENDA can answer for: `DSM 22`
is a strain BRENDA holds while `DSM 22228` is not, so a matcher that truncates
does not come up empty, it comes up confidently wrong — and it is the *gold*
it is wrong about.

The rest pins the closed acronym list, which is the only thing separating a
deposit number from the strain designations that outnumber them: `PAO1`,
`ST 131` and `IP 32953` are shaped exactly like accessions and are not.
"""

import pytest
from d3text.datasets import culture_numbers
from d3text.identifier_bridge import ExternalMention


def _canonical(text: str) -> list[str]:
    return [accession.canonical for accession in culture_numbers.find(text)]


def _mention(surface: str, start: int = 0) -> ExternalMention:
    return ExternalMention(
        document="19135",
        start=start,
        end=start + len(surface),
        surface=surface,
        external_id=None,
    )


@pytest.mark.parametrize(
    ("surface", "canonical"),
    [
        ("Orbus hercynius DSM 22,228", "DSM 22228"),
        ("ATCC 11,859", "ATCC 11859"),
        ("S. aureus MSSA ATCC 25,923", "ATCC 25923"),
        ("E. coli ATCC 47,004", "ATCC 47004"),
        ("KCTC 92,072", "KCTC 92072"),
        ("ATCC 29,122", "ATCC 29122"),
        ("CCUG 29243; 3,578", "CCUG 29243"),
    ],
)
def test_a_thousands_separator_is_part_of_the_number(
    surface: str, canonical: str
) -> None:
    """Every spelling in the corpus that writes a deposit number with a
    thousands separator. Reading only as far as the comma leaves a prefix that
    is itself a valid accession, so nothing downstream disagrees."""
    assert _canonical(surface) == [canonical]


@pytest.mark.parametrize(
    ("surface", "canonical"),
    [
        ("ATCC 35984, 35983", "ATCC 35984"),
        ("NBRC 15308, 100", "NBRC 15308"),
    ],
)
def test_a_list_separator_survives_the_normalization(
    surface: str, canonical: str
) -> None:
    """The other half, and why a blanket comma strip is not the fix: these
    commas separate two deposits, and gluing them invents a number no
    collection issued. `NBRC 15308, 100` is why the rule turns on the space —
    its second item is three digits long, exactly like a thousands group."""
    assert _canonical(surface) == [canonical]
    assert "".join(surface.split(", ")) not in _canonical(surface)


def test_a_truncated_number_would_name_a_real_strain() -> None:
    """What the rule is worth, stated once: the truncation succeeds. BRENDA
    holds a `DSM 22` and no `DSM 22228`, so a gold built by stopping at the
    comma is a strain, not an empty result."""
    assert culture_numbers.normalize("DSM 22,228") == "DSM 22228"
    assert culture_numbers.normalize("ATCC 35984, 35983") == (
        "ATCC 35984, 35983"
    )


@pytest.mark.parametrize(
    "surface", ["PAO1", "ST 131", "IP 32953", "K-12", "DSMZ-26127", "T20"]
)
def test_a_designation_shaped_like_an_accession_is_not_one(
    surface: str,
) -> None:
    """The acronym is the whole of the difference. `DSMZ` is the institute
    rather than the collection, and admitting it by prefix would read the
    number after any capitals at all."""
    assert _canonical(surface) == []


@pytest.mark.parametrize(
    ("surface", "canonical"),
    [
        ("Staphylococcus aureus ATCC 6538", "ATCC 6538"),
        ("NCTC13129", "NCTC 13129"),
        ("Staphylococcus epidermidis ATCC14990", "ATCC 14990"),
        ("R. solanacearum LMG 2299", "LMG 2299"),
        ("ATCC BAA-1360", "ATCC BAA-1360"),
        ("CGMCC 1.6105", "CGMCC 1.6105"),
        ("VKM Ac-108", "VKM AC-108"),
    ],
)
def test_an_accession_is_spelled_the_one_way_both_sides_join_on(
    surface: str, canonical: str
) -> None:
    """The corpus writes the separator in and BRENDA leaves it out, or the
    other way round, so the canonical form is what the join is on."""
    assert _canonical(surface) == [canonical]


@pytest.mark.parametrize("number", ["CCUG 12534 C", "IMI 034912ii", "P-24"])
def test_a_number_the_grammar_only_partly_reads_is_not_paired(
    number: str,
) -> None:
    """`parse` is whole-string where `find` is not, and the asymmetry is the
    point: the part of `CCUG 12534 C` that parses names a different deposit,
    so a bridge row built from it would be a wrong pairing rather than a
    missing one."""
    assert culture_numbers.parse(number) is None


def test_a_span_carrying_no_accession_keeps_a_none_identifier() -> None:
    """It stays in the coverage denominator: the score is about strain spans,
    not about the strain spans that happen to name a deposit."""
    stamped = culture_numbers.assign([_mention("Escherichia coli K-12")])

    assert [mention.external_id for mention in stamped] == [None]


def test_a_span_carrying_two_accessions_is_stamped_with_both() -> None:
    """Emitted once per accession, so the scorer counts the span as gold-side
    ambiguity rather than this function picking one of them."""
    stamped = culture_numbers.assign([_mention("ATCC 6538 (= DSM 799)")])

    assert [mention.external_id for mention in stamped] == [
        "ATCC 6538",
        "DSM 799",
    ]
    assert {mention.start for mention in stamped} == {0}


def test_every_acronym_the_grammar_admits_is_a_collection() -> None:
    """The list is closed on purpose, and a stray entry in it would silently
    widen the population every coverage number is a share of."""
    assert "ATCC" in culture_numbers.COLLECTIONS
    assert "DSMZ" not in culture_numbers.COLLECTIONS
    assert all(name.isupper() for name in culture_numbers.COLLECTIONS)
