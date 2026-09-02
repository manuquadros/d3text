"""Culture-collection accessions: the strain identifiers running text spells.

A strain deposited in a public collection is named by the collection's acronym
and a deposit number — `ATCC 6538`, `DSM 22228` — and BRENDA's `cultures` table
records that string verbatim, so a span carrying one reaches a strain with no
name compared anywhere. The grammar itself is `surface_forms.ACCESSION`, since
the index keys on the same shape; this module is what a span and a culture
number are read through. See the evaluation page of the documentation.
"""

from collections.abc import Iterable
from dataclasses import dataclass, replace

from d3text.identifier_bridge import ExternalMention
from d3text.surface_forms import ACCESSION, COLLECTIONS, THOUSANDS


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
        for match in ACCESSION.finditer(normalize(text))
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
    match = ACCESSION.fullmatch(normalize(number.strip()))
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
