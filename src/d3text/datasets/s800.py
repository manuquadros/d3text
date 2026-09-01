"""S800: 800 abstracts with hand-assigned NCBI taxids, one per species span.

The corpus `end` offset is inclusive and this package's is half-open, so
`load_s800` converts and then checks every mention against the text it
addresses: a one-character-short span tokenizes and matches nothing, lowering
a score with nothing anywhere disagreeing. See the evaluation page of the
documentation.
"""

import os
import pathlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from d3text.identifier_bridge import ExternalMention

ANNOTATIONS = "S800.tsv"
"""The annotation table, relative to the corpus root."""

ABSTRACTS = "abstracts"
"""Directory of document texts, relative to the corpus root."""

_COLUMNS = 5


@dataclass(frozen=True, slots=True)
class S800:
    """The corpus: its texts, its PubMed IDs, and its annotated spans.

    `mentions` carries half-open offsets into the matching `texts` entry, and
    every one of them has been checked to address its own `surface` — see the
    module docstring for why that check is not optional.
    """

    texts: Mapping[str, str]
    pubmed_ids: Mapping[str, str]
    mentions: tuple[ExternalMention, ...]


def parse_annotations(lines: Iterable[str]) -> list[ExternalMention]:
    """The annotation table's rows as mentions with half-open offsets.

    The `+ 1` on `end` is the whole conversion, and the one thing here a
    reader has to believe; `load_s800` proves it against the text.

    :param lines: the annotation table's rows.
    :return: one mention per row, with half-open offsets.
    :raises ValueError: on a row that is not five tab-separated fields or
        whose offsets are not integers.
    """
    mentions: list[ExternalMention] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        fields = line.rstrip("\n").split("\t")
        if len(fields) != _COLUMNS:
            raise ValueError(
                f"{ANNOTATIONS}:{number} has {len(fields)} fields, expected "
                f"{_COLUMNS}: {line!r}"
            )
        taxid, doc_id, start, end, surface = fields
        try:
            first, last = int(start), int(end)
        except ValueError as error:
            raise ValueError(
                f"{ANNOTATIONS}:{number} has non-integer offsets "
                f"({start!r}, {end!r})"
            ) from error
        mentions.append(
            ExternalMention(
                document=document_of(doc_id),
                start=first,
                end=last + 1,
                surface=surface,
                external_id=taxid,
            )
        )
    return mentions


def document_of(doc_id: str) -> str:
    """`species001:21183147` -> `species001`, the text file's stem.

    :param doc_id: the annotation table's document identifier.
    :return: the stem of the file holding that document's text.
    """
    return doc_id.split(":", 1)[0]


def pubmed_id_of(doc_id: str) -> str:
    """`species001:21183147` -> `21183147`, or the whole id if it has none.

    :param doc_id: the annotation table's document identifier.
    :return: the PubMed ID it carries, or the identifier itself.
    """
    _, _, pubmed_id = doc_id.partition(":")
    return pubmed_id or doc_id


def check_offsets(
    texts: Mapping[str, str], mentions: Sequence[ExternalMention]
) -> None:
    """Assert every mention addresses its own surface form.

    Self-checking by construction: the table carries the surface text beside
    the offsets, so the corpus validates its own coordinate convention.

    :param texts: each document's text, by document stem.
    :param mentions: the mentions to check.
    :raises ValueError: naming the first few disagreements, or a mention whose
        document has no text.
    """
    wrong: list[str] = []
    for mention in mentions:
        text = texts.get(mention.document)
        if text is None:
            wrong.append(f"{mention.document} has no text")
        elif text[mention.start : mention.end] != mention.surface:
            wrong.append(
                f"{mention.document}[{mention.start}:{mention.end}] is "
                f"{text[mention.start : mention.end]!r}, annotated as "
                f"{mention.surface!r}"
            )
        if len(wrong) >= 5:
            break

    if wrong:
        raise ValueError(
            "S800's offsets do not address the spans it annotates, so the "
            "corpus on disk is not the one this loader converts (its `end` "
            "is inclusive): " + "; ".join(wrong)
        )


def load_s800(root: str | os.PathLike[str]) -> S800:
    """Read the corpus at `root`.

    :param root: the corpus directory, holding the table and the texts.
    :return: the texts, the PubMed IDs, and the annotated spans.
    :raises FileNotFoundError: if the annotation table or a document named by
        it is missing.
    :raises ValueError: if a row is malformed or an offset does not address
        its annotated surface form.
    """
    directory = pathlib.Path(root)
    table = directory / ANNOTATIONS
    lines = table.read_text(encoding="utf8").splitlines()
    mentions = parse_annotations(lines)

    pubmed_ids: dict[str, str] = {}
    for line in lines:
        if not line.strip():
            continue
        doc_id = line.rstrip("\n").split("\t")[1]
        pubmed_ids[document_of(doc_id)] = pubmed_id_of(doc_id)

    texts = {
        document: (directory / ABSTRACTS / f"{document}.txt").read_text(
            encoding="utf8"
        )
        for document in pubmed_ids
    }

    check_offsets(texts, mentions)
    return S800(texts=texts, pubmed_ids=pubmed_ids, mentions=tuple(mentions))


__all__ = [
    "ABSTRACTS",
    "ANNOTATIONS",
    "S800",
    "check_offsets",
    "document_of",
    "load_s800",
    "parse_annotations",
    "pubmed_id_of",
]
