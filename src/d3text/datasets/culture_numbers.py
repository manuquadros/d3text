"""Culture-collection accessions: the strain identifiers running text spells.

A strain deposited in a public collection is named by the collection's acronym
and a deposit number — `ATCC 6538`, `DSM 22228` — and BRENDA's `cultures` table
records that string verbatim, so a span carrying one reaches a strain with no
name compared anywhere. The acronyms are a closed list because a pattern that
takes any capitals-then-digits reads `PAO1`, `IP 32953` and `ST 131` as
accessions. See the evaluation page of the documentation.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass, replace

from d3text.identifier_bridge import ExternalMention
from d3text.surface_forms import THOUSANDS

COLLECTIONS = frozenset(
    {
        "ACM",
        "AS",
        "ATCC",
        "BCC",
        "BCRC",
        "CBMAI",
        "CBS",
        "CCAC",
        "CCAP",
        "CCM",
        "CCMM",
        "CCMP",
        "CCRC",
        "CCT",
        "CCUG",
        "CDBB",
        "CECT",
        "CFBP",
        "CGMCC",
        "CIP",
        "CLIB",
        "CNCTC",
        "CRBIP",
        "DBVPG",
        "DSM",
        "FGSC",
        "FRR",
        "HAMBI",
        "HUT",
        "IAM",
        "ICMP",
        "IFO",
        "IHEM",
        "IMET",
        "IMI",
        "JCM",
        "KACC",
        "KCTC",
        "LMD",
        "LMG",
        "MUCL",
        "MUM",
        "NBIMCC",
        "NBRC",
        "NCAIM",
        "NCCB",
        "NCDO",
        "NCFB",
        "NCIB",
        "NCIM",
        "NCIMB",
        "NCMB",
        "NCPF",
        "NCPPB",
        "NCTC",
        "NCYC",
        "NIES",
        "NRRL",
        "PCC",
        "PDDCC",
        "RCC",
        "SAG",
        "TBRC",
        "TISTR",
        "UAMH",
        "UTEX",
        "VKM",
        "VTT",
    }
)
"""Acronyms of the culture collections BRENDA's deposits are held in.

Closed, and matched case-sensitively: `AS` is a collection and also two
ordinary letters, and the difference an accession has from a strain designation
is the acronym, not the shape.
"""

_BODY = r"(?:[A-Za-z]{1,3}[-.])?\d+(?:[./-]\d+)*[A-Za-z]?"

_ACCESSION = re.compile(
    r"(?<![A-Za-z0-9])"
    r"("
    + "|".join(sorted(COLLECTIONS, key=lambda name: (-len(name), name)))
    + r")(?![A-Za-z])[ -]{0,2}"
    r"(" + _BODY + r")"
    r"(?![A-Za-z0-9])"
)


@dataclass(frozen=True, slots=True)
class Accession:
    """A deposit number, as the text spells it and as both sides join on it.

    `written` keeps the spelling because whether the surface-form index holds
    that spelling is the question the strain evaluation is really asking;
    `canonical` is what the two sides of the join agree on.
    """

    written: str
    canonical: str


def normalize(text: str) -> str:
    """`text` with the thousands separators of deposit numbers removed.

    :param text: a span's surface form, or a culture number as BRENDA holds it.
    :return: the same text with `DSM 22,228` spelled `DSM 22228`, and every
        comma that separates two numbers left where it is.
    """
    return THOUSANDS.sub("", text)


def find(text: str) -> list[Accession]:
    """Every culture-collection accession `text` carries, in order.

    :param text: a span's surface form, which is a full designation more often
        than a bare accession — `Staphylococcus aureus ATCC 6538`.
    :return: the accessions found, empty if it carries none.
    """
    return [
        Accession(
            written=match.group(0),
            canonical=f"{match.group(1)} {match.group(2).upper()}",
        )
        for match in _ACCESSION.finditer(normalize(text))
    ]


def parse(number: str) -> Accession | None:
    """`number` read as a whole accession, or None if it is not one.

    Whole-string, unlike `find`: a BRENDA culture number the grammar only
    partly covers — `CCUG 12534 C`, `IMI 034912ii` — is dropped rather than
    truncated to the part that parses, since the truncation would name a
    different deposit.

    :param number: a culture number as BRENDA's `cultures` table holds it.
    :return: the accession, or None if the string is not one.
    """
    match = _ACCESSION.fullmatch(normalize(number.strip()))
    if match is None:
        return None
    return Accession(
        written=match.group(0),
        canonical=f"{match.group(1)} {match.group(2).upper()}",
    )


def assign(mentions: Iterable[ExternalMention]) -> list[ExternalMention]:
    """Stamp each span with the accessions its surface form carries.

    A span carrying two is emitted once per accession, so the scorer counts it
    as gold-side ambiguity rather than the caller picking one; a span carrying
    none keeps a `None` identifier and stays in the coverage denominator.

    :param mentions: the corpus's spans, which carry no identifier.
    :return: the spans with their accessions, longer than `mentions` wherever
        a surface form carries more than one.
    """
    stamped: list[ExternalMention] = []
    for mention in mentions:
        found = {accession.canonical for accession in find(mention.surface)}
        if not found:
            stamped.append(replace(mention, external_id=None))
            continue
        stamped.extend(
            replace(mention, external_id=canonical)
            for canonical in sorted(found)
        )
    return stamped


__all__ = [
    "COLLECTIONS",
    "Accession",
    "assign",
    "find",
    "normalize",
    "parse",
]
