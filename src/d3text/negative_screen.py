"""Whether a document is a true negative for one entity type.

Screened with the same surface-form index that builds the positives, and
under two readings whose difference is the measurement. Deliberately a leaf,
like `d3text.corpus`. See the data page of the documentation.
"""

import dataclasses
import re
from collections.abc import Sequence

from d3text.constraints import NonNegative
from d3text.surface_forms import (
    BRENDA_PREFIXES,
    SYMBOL_MAX_LENGTH,
    SurfaceFormIndex,
    form_key,
    form_words,
    has_letter,
)
from d3text.token_labels import MAX_MENTION_GAP, _EPITHET, find_mentions

ENZYME_PREFIX = BRENDA_PREFIXES["enzymes"]
"""The ID prefix the screen looks for by default.

Read off the schema for the reason `BRENDA_PREFIXES` is: a prefix that
disagrees with the corpus's spelling matches nothing, and every document then
screens as a negative.
"""


@dataclasses.dataclass(frozen=True, slots=True)
class Matches:
    """The forms of one entity type a document offers, in three kinds.

    Kept apart because the screens disagree about which count. `descriptive`
    is what `is_descriptive` holds of, `symbolic` the rest of the exact hits.
    `fuzzy` never asserts a type; an ambiguous comma-joined exact hit is
    folded into it on the same footing.
    """

    descriptive: tuple[str, ...] = ()
    symbolic: tuple[str, ...] = ()
    fuzzy: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True)
class Screen:
    """Which of a document's matches disqualify it as a negative.

    `symbols_disqualify` is the whole measurement, so a parameter. On, any
    exact match rejects; off (the default), only a descriptive name does,
    since the literal reading cannot certify a negative. The default misses
    documents whose only enzyme is a short form: `renin`, `NADH`, `PEP-CK`.
    """

    symbols_disqualify: bool = False
    fuzzy_disqualifies: bool = False

    def rejects(self, matches: Matches) -> tuple[str, ...]:
        """The forms that disqualify a document under this screen.

        :param matches: what the index found in the document.
        :return: the disqualifying forms, empty where the document survives.
        """
        return (
            matches.descriptive
            + (matches.symbolic if self.symbols_disqualify else ())
            + (matches.fuzzy if self.fuzzy_disqualifies else ())
        )

    def accepts(self, matches: Matches) -> bool:
        """Whether the document is a negative under this screen.

        :param matches: what the index found in the document.
        :return: whether it survives.
        """
        return not self.rejects(matches)


DESCRIPTIVE = Screen()
"""Reject on a descriptive name alone."""

LITERAL = Screen(symbols_disqualify=True)
"""Reject on any exact match, symbol or name."""


_GENUS = re.compile(r"[A-Z][a-z]*")
"""A genus, cut to its initial (`E`) or spelled out in full (`Mus`)."""

_ORGANISM_DESIGNATION = re.compile(_EPITHET.pattern + r"\d*")
"""A species epithet, or a strain/phage designation cut from the same cloth.

Widens `token_labels._EPITHET`'s letters-only shape with trailing digits: an
epithet (`coli`) and a designation that carries a number (`phi6`, `ce56`,
`aeh1`) differ only in that suffix.
"""

_SPECIES_PLACEHOLDER = frozenset({"sp", "spp"})
"""The un-named-species abbreviation, alone among these words carrying a
third word after it — a strain number the placeholder itself does not name.
"""


def _is_abbreviated_binomial(words: Sequence[str]) -> bool:
    """Whether `words` name a genus and a species, in one shape or another.

    Read off `form_words`, so `E. coli` and `Mus sp.` need no separate
    handling. Only a `sp.`/`spp.` placeholder may carry one more word, a
    strain number (`B. sp. A3`).

    :param words: a form's words, as `form_words` splits it.
    :return: whether the shape is a genus followed by a species or a
        species placeholder.
    """
    if len(words) < 2 or not _GENUS.fullmatch(words[0]):
        return False
    epithet = words[1]
    if epithet.lower() in _SPECIES_PLACEHOLDER:
        return len(words) <= 3
    return (
        len(words) == 2 and _ORGANISM_DESIGNATION.fullmatch(epithet) is not None
    )


def is_descriptive(form: str) -> bool:
    """Whether `form` names an entity in words rather than in a symbol.

    Judged by its words joined, since the index reads a registered `PP-1`
    across a statistic's `PP = 1`: short joined forms are symbols unless they
    name an organism. Past that, case decides only for a single word; a form
    with no letter names nothing.

    :param form: a matched span, as the document writes it.
    :return: whether it is a descriptive name.
    """
    if not has_letter(form):
        return False
    words = form_words(form)
    if len("".join(words)) <= SYMBOL_MAX_LENGTH:
        return len(words) > 1 and _is_abbreviated_binomial(words)
    return len(words) > 1 or not any(
        character.isupper() for character in form[1:]
    )


def _reads_descriptive(surface: str, index: SurfaceFormIndex) -> bool:
    """Whether `surface` counts as descriptive given how `index` matched it.

    A span matched only through a case-folded key must not read as symbolic
    on casing alone. Only casing is neutralised: the lowercased span still
    goes through `is_descriptive`'s length and binomial rules.

    :param surface: a matched span, as the document writes it.
    :param index: the index `surface` was matched against.
    :return: whether the mention counts as a descriptive name.
    """
    if is_descriptive(surface):
        return True
    key = form_key(form_words(surface)).lower()
    return key in index.folded and is_descriptive(surface.lower())


def matched_forms(
    text: str,
    index: SurfaceFormIndex,
    prefix: str = ENZYME_PREFIX,
    max_gap: NonNegative = MAX_MENTION_GAP,
) -> Matches:
    """Every span of `text` the index could read as an entity of one type.

    A mention counts when *any* of its candidate entities wears `prefix`: an
    ambiguous form that could be an enzyme is exactly what a document claiming
    to name no enzyme must not contain.

    :param text: the document text, as `d3text.corpus.document_text` builds
        it.
    :param index: the surface forms to match, guarded as the labeller's are.
    :param prefix: the entity-ID prefix of the type being screened for.
    :param max_gap: characters allowed between two words of one mention.
    :return: the matched spans, as written, in the three kinds a screen reads
        separately.
    """
    found: dict[str, list[str]] = {
        "descriptive": [],
        "symbolic": [],
        "fuzzy": [],
    }
    for mention in find_mentions(text, index, max_gap):
        if not any(
            entity_id.startswith(prefix) for entity_id in mention.entity_ids
        ):
            continue
        surface = text[mention.start : mention.end]
        if mention.fuzzy or mention.ambiguous:
            # This screen reads an ambiguous mention on the same footing as a
            # fuzzy one: neither may disqualify a negative under the
            # descriptive default, only under `fuzzy_disqualifies`.
            kind = "fuzzy"
        else:
            kind = (
                "descriptive"
                if _reads_descriptive(surface, index)
                else "symbolic"
            )
        found[kind].append(surface)
    return Matches(
        descriptive=tuple(found["descriptive"]),
        symbolic=tuple(found["symbolic"]),
        fuzzy=tuple(found["fuzzy"]),
    )


__all__ = [
    "DESCRIPTIVE",
    "ENZYME_PREFIX",
    "LITERAL",
    "Matches",
    "Screen",
    "is_descriptive",
    "matched_forms",
]
