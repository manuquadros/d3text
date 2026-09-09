"""Whether a document is a true negative for one entity type.

A negative the loss can be consistent with is one the *labelling* index calls
empty, not one a topic filter calls off-topic, so a candidate is screened here
with the same surface-form index that builds the positives. Two readings of
those matches are reported side by side, because the difference between them
is the measurement: a corpus that names no entity of the type by construction
is a control, and a screen that rejects most of it is measuring something
other than what it claims to.

The matched forms are reported with their frequencies for the same reason. A
rate cannot say whether what it counted names an entity, and it is the form
table that says whether a rejection was an entity mention, an acronym the
index also registers, or a decimal sequence read as a bare designation.

Deliberately a leaf, like `d3text.corpus`: screening a corpus must not cost
the BRENDA data layer. See the data page of the documentation, which carries
the measured yields and the corpora they were measured on.
"""

import collections
import dataclasses
import itertools
import pathlib
import statistics
from collections.abc import Iterable, Iterator, Mapping, Sequence

from tqdm import tqdm

from d3text import corpus
from d3text.constraints import NonNegative, Positive
from d3text.surface_forms import (
    BRENDA_PREFIXES,
    SYMBOL_MAX_LENGTH,
    SurfaceFormIndex,
    form_words,
)
from d3text.token_labels import MAX_MENTION_GAP, find_mentions

ENZYME_PREFIX = BRENDA_PREFIXES["enzymes"]
"""The ID prefix the screen looks for by default.

Read off the schema for the reason `BRENDA_PREFIXES` is: a prefix that
disagrees with the corpus's spelling matches nothing, and every document then
screens as a negative.
"""

STREAM_BATCH = 1000
"""Rows per slice of the corpus pass."""


@dataclasses.dataclass(frozen=True, slots=True)
class Matches:
    """The forms of one entity type a document offers, in three kinds.

    Kept apart rather than summed because they are not the same evidence and
    the screens below disagree about which of them count. `descriptive` is
    the matches `is_descriptive` holds of, against `symbolic` — the acronyms,
    short forms and bare number sequences where the index is at its least
    reliable. `fuzzy` is a near-miss to a known form, which `Mention` may
    withhold a type on but never assert one from.
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
    every acronym and every short form, so a document whose only enzyme is
    `renin`, `NADH` or `LasI` passes it.
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

    @property
    def label(self) -> str:
        """A column heading naming which reading of the matches this is."""
        return ("literal" if self.symbols_disqualify else "descriptive") + (
            "+fuzzy" if self.fuzzy_disqualifies else ""
        )

    def describe(self) -> str:
        """What the label stands for, spelled out for a report."""
        return "; ".join(
            (
                "every exact match rejects"
                if self.symbols_disqualify
                else "only a descriptive name rejects",
                "near-misses reject"
                if self.fuzzy_disqualifies
                else "near-misses ignored",
            )
        )


DESCRIPTIVE = Screen()
"""Reject on a descriptive name alone."""

LITERAL = Screen(symbols_disqualify=True)
"""Reject on any exact match, symbol or name."""


def is_descriptive(form: str) -> bool:
    """Whether `form` names an entity in words rather than in a symbol.

    Not `surface_forms.is_symbol_like`, which answers whether case is
    load-bearing for a form and so reads `RNA polymerase`, `ATP synthase` and
    `cytochrome P450 monooxygenase` as symbols: case decides nothing for a
    name whose words already collide with no English word, so a form of more
    than one word is descriptive whatever its case. A form carrying no letter
    at all is symbolic for a different reason — BRENDA registers bare number
    sequences as strain designations, so that is `find_mentions` reading a
    section number or a confidence interval as one, which names nothing.

    :param form: a matched span, as the document writes it.
    :return: whether it is a descriptive name.
    """
    if not any(character.isalpha() for character in form):
        return False
    if len(form_words(form)) > 1:
        return True
    return len(form) > SYMBOL_MAX_LENGTH and not any(
        character.isupper() for character in form[1:]
    )


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
        if mention.fuzzy:
            kind = "fuzzy"
        else:
            kind = "descriptive" if is_descriptive(surface) else "symbolic"
        found[kind].append(surface)
    return Matches(
        descriptive=tuple(found["descriptive"]),
        symbolic=tuple(found["symbolic"]),
        fuzzy=tuple(found["fuzzy"]),
    )


@dataclasses.dataclass(slots=True)
class Survey:
    """The yield of one screen over a candidate pool, and what it matched.

    `screened_values` and the screened lengths are carried beside the
    survivors' because a survivor rate per journal, or a survivor's length
    against the pool's, is the only form in which the survivors' bias is
    readable — and that bias is the failure mode the pool exists to avoid. A
    screen that admits only the shortest documents has selected a genre, which
    is a free negative in a new costume.
    """

    screen: Screen = DESCRIPTIVE
    prefix: str = ENZYME_PREFIX
    documents: int = 0
    negatives: int = 0
    match_counts: collections.Counter[int] = dataclasses.field(
        default_factory=collections.Counter
    )
    descriptive_forms: collections.Counter[str] = dataclasses.field(
        default_factory=collections.Counter
    )
    symbolic_forms: collections.Counter[str] = dataclasses.field(
        default_factory=collections.Counter
    )
    fuzzy_forms: collections.Counter[str] = dataclasses.field(
        default_factory=collections.Counter
    )
    lengths: list[int] = dataclasses.field(default_factory=list)
    negative_lengths: list[int] = dataclasses.field(default_factory=list)
    screened_values: dict[str, collections.Counter[str]] = dataclasses.field(
        default_factory=dict
    )
    negative_values: dict[str, collections.Counter[str]] = dataclasses.field(
        default_factory=dict
    )

    def record(
        self,
        matches: Matches,
        text_length: int = 0,
        metadata: Mapping[str, str] | None = None,
    ) -> bool:
        """Tally one screened document.

        :param matches: what the index found in it.
        :param text_length: its length in characters.
        :param metadata: its descriptive columns, if the corpus carries any.
        :return: whether the document survived the screen.
        """
        survived = self.screen.accepts(matches)
        self.documents += 1
        self.negatives += survived
        self.match_counts[len(self.screen.rejects(matches))] += 1
        self.descriptive_forms.update(matches.descriptive)
        self.symbolic_forms.update(matches.symbolic)
        self.fuzzy_forms.update(matches.fuzzy)
        self.lengths.append(text_length)
        if survived:
            self.negative_lengths.append(text_length)
        for column, value in (metadata or {}).items():
            _tally(self.screened_values, column, value)
            if survived:
                _tally(self.negative_values, column, value)
        return survived

    @property
    def negative_rate(self) -> float:
        """The share of screened documents that named no entity of the type."""
        return self.negatives / self.documents if self.documents else 0.0

    def summary(self, forms: Positive = 15, values: Positive = 10) -> str:
        """The yield as prose, with the match mass it was computed from.

        :param forms: how many matched surface forms to list per table.
        :param values: how many metadata values to list per column.
        :return: the report.
        """
        lines = [
            f"{self.documents} documents screened for {self.prefix!r} "
            f"mentions, {self.screen.label} ({self.screen.describe()}).",
            f"{self.negatives} survive ({self.negative_rate:.1%}); median "
            f"{_median(self.negative_lengths):.0f} characters against "
            f"{_median(self.lengths):.0f} over the whole sample.",
            f"disqualifying matches per document: "
            f"{_histogram(self.match_counts)}",
            "descriptive matches: "
            + _frequencies(self.descriptive_forms, forms),
            "symbol-like matches: " + _frequencies(self.symbolic_forms, forms),
            "near-misses: " + _frequencies(self.fuzzy_forms, forms),
        ]
        for column, screened in self.screened_values.items():
            survivors = self.negative_values.get(column, collections.Counter())
            lines.append(f"{column}, survivors of screened:")
            lines.extend(
                f"  {value:40.40s} {survivors[value]:5d} / {count}"
                for value, count in screened.most_common(values)
            )
        return "\n".join(lines)


def comparison(surveys: Mapping[str, Sequence[Survey]]) -> str:
    """Several corpora's yields, one row each, one column per screen.

    A yield on its own is uninterpretable: it took a pool that is negative by
    construction and one that is positive by construction, screened beside the
    candidates, to show that the first reading of the index was measuring
    which documents avoid common acronyms.

    :param surveys: corpus name -> its surveys, one per screen, in a
        consistent order.
    :return: the table.
    :raises ValueError: if the corpora were not screened alike, which would
        put two different screens in one column.
    """
    labels = {
        tuple(survey.screen.label for survey in row) for row in surveys.values()
    }
    if len(labels) > 1:
        raise ValueError(f"the corpora were screened differently: {labels}")

    width = max((len(name) for name in surveys), default=0)
    header = "  ".join(
        [f"{'corpus':{width}.{width}s}", "documents"]
        + [f"{label:>17.17s}" for label in next(iter(labels), ())]
    )
    rows = [
        "  ".join(
            [
                f"{name:{width}.{width}s}",
                f"{row[0].documents:9d}" if row else "",
            ]
            + [
                f"{survey.negatives:9d} ({survey.negative_rate:5.1%})"
                for survey in row
            ]
        )
        for name, row in surveys.items()
    ]
    return "\n".join([header, *rows])


def _tally(
    columns: dict[str, collections.Counter[str]], column: str, value: str
) -> None:
    columns.setdefault(column, collections.Counter())[value] += 1


def _median(lengths: Sequence[int]) -> float:
    return float(statistics.median(lengths)) if lengths else 0.0


def _histogram(counts: collections.Counter[int], cap: Positive = 5) -> str:
    """`counts` as `matches: documents`, everything from `cap` in one bucket."""
    bucketed: collections.Counter[int] = collections.Counter()
    for matches, documents in counts.items():
        bucketed[min(matches, cap)] += documents
    return ", ".join(
        f"{matches}{'+' if matches == cap else ''}: {documents}"
        for matches, documents in sorted(bucketed.items())
    )


def _frequencies(forms: collections.Counter[str], limit: Positive) -> str:
    """The commonest `forms`, as `form (count)`."""
    if not forms:
        return "none"
    return ", ".join(
        f"{form} ({count})" for form, count in forms.most_common(limit)
    )


def survey_corpus(
    path: pathlib.Path,
    index: SurfaceFormIndex,
    screens: Sequence[Screen] = (DESCRIPTIVE, LITERAL),
    prefix: str = ENZYME_PREFIX,
    metadata_columns: Sequence[str] = (),
    limit: Positive | None = None,
    batch_size: Positive = STREAM_BATCH,
) -> tuple[Survey, ...]:
    """Screen every document of a corpus file, once per screen, in one pass.

    Reads the csv splits and the line-delimited PMC dumps alike, since both go
    through `d3text.corpus`; a candidate pool written in the noise pool's shape
    therefore needs no conversion. Several screens share the single expensive
    step, which is the matching — they disagree about which of the same
    matches count, not about what the matches are.

    :param path: the corpus file to screen.
    :param index: the surface forms to match.
    :param screens: the readings to tally, one survey each.
    :param prefix: the entity-ID prefix of the type being screened for.
    :param metadata_columns: descriptive columns to characterise the survivors
        by, where the file carries them.
    :param limit: screen at most this many documents.
    :param batch_size: rows per slice.
    :return: one survey per screen, in the order they were given.
    """
    metadata = dict(corpus.stream_metadata(path, batch_size, metadata_columns))
    total, rows = corpus.stream_rows(path, batch_size)
    surveys = tuple(Survey(screen=screen, prefix=prefix) for screen in screens)
    for pubmed_id, text in _limited(rows, total, limit):
        matches = matched_forms(text, index, prefix)
        for survey in surveys:
            survey.record(matches, len(text), metadata.get(pubmed_id))
    return surveys


def _limited(
    rows: Iterable[tuple[corpus.PubmedId, str]], total: int, limit: int | None
) -> Iterator[tuple[corpus.PubmedId, str]]:
    """`rows` behind a bar, cut to `limit`.

    `disable=None` rather than a flag: the bar is for a pass over a corpus at a
    terminal and must not write a line per document into a redirected log.

    Yielded rather than returned: a `tqdm` is `Sized`, so beartype deep-checks
    a returned one by pulling an item off it, and that item is gone from the
    underlying stream — one document silently missing from every pass.
    """
    counted = rows if limit is None else itertools.islice(rows, limit)
    yield from tqdm(
        counted,
        total=total if limit is None else min(total, limit),
        unit="doc",
        disable=None,
    )


__all__ = [
    "DESCRIPTIVE",
    "ENZYME_PREFIX",
    "LITERAL",
    "Matches",
    "Screen",
    "Survey",
    "comparison",
    "is_descriptive",
    "matched_forms",
    "survey_corpus",
]
