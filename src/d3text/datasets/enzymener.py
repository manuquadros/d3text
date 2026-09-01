"""enzymeNER: PMC sentences whose enzyme mentions are marked but not named.

The unit is the sentence, not the document, and the offsets are **half-open**
— the opposite convention from S800, so the `+ 1` that corpus needs is a
one-character error here. Three of the 2,274 rows address neither reading and
are dropped rather than costing the corpus, which is why the count of them is
carried on the loaded corpus. See the evaluation page of the documentation.
"""

import os
import pathlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from d3text.identifier_bridge import ExternalMention

SENTENCES = "GoldSet.txt"
"""The sentence table, relative to the corpus root."""

ANNOTATIONS = "GoldSetAnnot.txt"
"""The annotation table, relative to the corpus root."""

ENCODING = "utf-8-sig"
"""Both tables open with a byte-order mark."""

MISPLACED_LIMIT = 0.01
"""Share of misplaced rows above which the corpus is refused, not repaired."""

_SENTENCE_COLUMNS = 3
_ANNOTATION_COLUMNS = 5
_REPORTED = 5


@dataclass(frozen=True, slots=True)
class EnzymeNER:
    """The corpus: its sentences, its annotated spans, and its bad rows.

    `mentions` carries half-open offsets into the matching `texts` entry and
    every one of them addresses its own `surface`; `misplaced` holds the rows
    that did not, so a corpus quietly rotting one row at a time still shows up
    as a number. Neither carries an identifier — the corpus assigns none.
    """

    texts: Mapping[str, str]
    articles: Mapping[str, str]
    mentions: tuple[ExternalMention, ...]
    misplaced: tuple[ExternalMention, ...]


def sentence_id(article: str, sentence: str) -> str:
    """`PMC1233920`, `M01009` -> `PMC1233920:M01009`, the span's document.

    :param article: the PMC identifier.
    :param sentence: the sentence identifier within it.
    :return: the key both tables address a span by.
    """
    return f"{article}:{sentence}"


def article_of(document: str) -> str:
    """`PMC1233920:M01009` -> `PMC1233920`, the article the sentence is from.

    :param document: a sentence key.
    :return: the PMC identifier it belongs to.
    """
    return document.split(":", 1)[0]


def parse_sentences(lines: Iterable[str]) -> dict[str, str]:
    """The sentence table's rows, keyed by `sentence_id`.

    :param lines: the sentence table's rows.
    :return: each sentence's text, by document key.
    :raises ValueError: on a row that is not three tab-separated fields.
    """
    texts: dict[str, str] = {}
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != _SENTENCE_COLUMNS:
            raise ValueError(
                f"{SENTENCES}:{number} has {len(fields)} fields, expected "
                f"{_SENTENCE_COLUMNS}: {line!r}"
            )
        article, sentence, text = fields
        texts[sentence_id(article, sentence)] = text
    return texts


def parse_annotations(lines: Iterable[str]) -> list[ExternalMention]:
    """The annotation table's rows as mentions, offsets read as written.

    :param lines: the annotation table's rows.
    :return: one identifier-less mention per row, half-open.
    :raises ValueError: on a row that is not five tab-separated fields or
        whose offsets are not integers.
    """
    mentions: list[ExternalMention] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != _ANNOTATION_COLUMNS:
            raise ValueError(
                f"{ANNOTATIONS}:{number} has {len(fields)} fields, expected "
                f"{_ANNOTATION_COLUMNS}: {line!r}"
            )
        article, sentence, start, end, surface = fields
        try:
            first, last = int(start), int(end)
        except ValueError as error:
            raise ValueError(
                f"{ANNOTATIONS}:{number} has non-integer offsets "
                f"({start!r}, {end!r})"
            ) from error
        mentions.append(
            ExternalMention(
                document=sentence_id(article, sentence),
                start=first,
                end=last,
                surface=surface,
                external_id=None,
            )
        )
    return mentions


def split_misplaced(
    texts: Mapping[str, str],
    mentions: Sequence[ExternalMention],
    limit: float = MISPLACED_LIMIT,
) -> tuple[list[ExternalMention], list[ExternalMention]]:
    """Partition the mentions by whether they address their own surface form.

    :param texts: each sentence's text, by document key.
    :param mentions: the mentions to check.
    :param limit: share of misplaced rows the corpus is still read at.
    :return: those that address their surface form, and those that do not.
    :raises ValueError: if more than `limit` of them are misplaced, which is
        what a wholesale offset-convention error looks like and what a handful
        of bad rows does not.
    """
    placed: list[ExternalMention] = []
    misplaced: list[ExternalMention] = []
    for mention in mentions:
        text = texts.get(mention.document, "")
        if text[mention.start : mention.end] == mention.surface:
            placed.append(mention)
        else:
            misplaced.append(mention)

    if mentions and len(misplaced) > limit * len(mentions):
        shown = "; ".join(
            f"{mention.document}[{mention.start}:{mention.end}] is "
            f"{texts.get(mention.document, '')[mention.start : mention.end]!r}"
            f", annotated as {mention.surface!r}"
            for mention in misplaced[:_REPORTED]
        )
        raise ValueError(
            f"{len(misplaced)} of {len(mentions)} enzymeNER offsets do not "
            f"address the spans they annotate, past the "
            f"{limit:.0%} a few bad rows explains: the corpus on "
            f"disk is not the one this loader reads half-open. {shown}"
        )
    return placed, misplaced


def load_enzymener(
    root: str | os.PathLike[str], misplaced_limit: float = MISPLACED_LIMIT
) -> EnzymeNER:
    """Read the corpus at `root`.

    :param root: the corpus directory, holding both tables.
    :param misplaced_limit: share of rows allowed to miss their own surface
        form before the corpus is refused rather than read without them.
    :return: the sentences, their articles, and the annotated spans.
    :raises FileNotFoundError: if either table is missing.
    :raises ValueError: if a row is malformed, an annotation names a sentence
        the corpus does not carry, or too many offsets miss their surface.
    """
    directory = pathlib.Path(root)
    texts = parse_sentences(
        (directory / SENTENCES).read_text(encoding=ENCODING).splitlines()
    )
    mentions = parse_annotations(
        (directory / ANNOTATIONS).read_text(encoding=ENCODING).splitlines()
    )

    unknown = sorted(
        {
            mention.document
            for mention in mentions
            if mention.document not in texts
        }
    )
    if unknown:
        raise ValueError(
            f"{len(unknown)} annotated sentences are not in {SENTENCES}, so "
            f"their spans address nothing: {unknown[:_REPORTED]}"
        )

    placed, misplaced = split_misplaced(texts, mentions, misplaced_limit)
    return EnzymeNER(
        texts=texts,
        articles={document: article_of(document) for document in sorted(texts)},
        mentions=tuple(placed),
        misplaced=tuple(misplaced),
    )


__all__ = [
    "ANNOTATIONS",
    "ENCODING",
    "MISPLACED_LIMIT",
    "SENTENCES",
    "EnzymeNER",
    "article_of",
    "load_enzymener",
    "parse_annotations",
    "parse_sentences",
    "sentence_id",
    "split_misplaced",
]
