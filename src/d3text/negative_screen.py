"""Whether a document is a true negative for one entity type.

A negative the loss can be consistent with is one the *labelling* index calls
empty, not one a topic filter calls off-topic, so a candidate is screened here
with the same surface-form index that builds the positives. Two readings of
those matches are kept apart, because the difference between them is the
measurement: a corpus that names no entity of the type by construction is a
control, and a screen that rejects most of it is measuring something other
than what it claims to.

Deliberately a leaf, like `d3text.corpus`: screening a corpus must not cost
the BRENDA data layer. See the data page of the documentation, which carries
the measured yields and the corpora they were measured on.
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

    Kept apart rather than summed because they are not the same evidence and
    the screens below disagree about which of them count. `descriptive` is
    the matches `is_descriptive` holds of, against `symbolic` — the acronyms,
    short forms and bare number sequences where the index is at its least
    reliable. `fuzzy` is a near-miss to a known form, which this screen never
    reads as asserting a type; an ambiguous mention -- an exact hit whose
    comma-joined span could equally be a sentence-context collision -- is
    folded into this same bucket, on the same non-asserting footing.
    """

    descriptive: tuple[str, ...] = ()
    symbolic: tuple[str, ...] = ()
    fuzzy: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True)
class Screen:
    """Which of a document's matches disqualify it as a negative.

    `symbols_disqualify` is the whole measurement, so it is a parameter and
    not a decision taken here. On, a document is rejected on any exact match,
    which is the literal reading of "zero matches from the enzyme index"; off
    — the default — only a descriptive name rejects it, because the literal
    reading rejects most of a corpus that names no enzyme by construction and
    so cannot certify a negative. The default is not free either: it ignores
    single-word acronyms and short forms, and any multi-word form whose words
    joined still fit within `SYMBOL_MAX_LENGTH`, unless they name an organism,
    so a document whose only enzyme is `renin`, `NADH` or `LasI` passes it, and
    so does a short hyphenated form like `PEP-CK` — but a longer one like
    `NADP-ME` still rejects it.
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

    Tokenized rather than matched against the raw span, so the dot of an
    abbreviated genus (`E. coli`) and the plain space of a full one (`Mus
    sp.`) need no separate handling — `form_words` has already dropped
    both. A bare `sp.`/`spp.` placeholder may still carry one more word, a
    strain or phage number (`B. sp. A3`); a named species allows no such
    tail.

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

    Judged by its words joined, because the index reads a registered `PP-1`
    across the `PP = 1` of a statistic: however the text spaces it, a form no
    longer than `SYMBOL_MAX_LENGTH` joined is a symbol unless it names an
    organism — a genus and a species, abbreviated or not, `sp.`/`spp.`
    placeholder included. Past that, case decides only for a single word,
    and a form holding no letter names nothing.

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

    A span reached only through a case-folded key is one the index already
    treats as non-symbol-like, so a document written in capitals must not
    read as symbolic on casing alone. Only the casing is neutralised here:
    `is_descriptive` still runs on the lowercased span, so its joined-length
    and binomial rules keep rejecting a short folded form (a registered
    `hsp-70` read across a statistic's `HSP = 70`) as a symbol.

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
