"""Three-way distant-supervision targets, one per encoded token.

The document-level objective localizes badly — the best operating point
measured anywhere over six arms is 29.5% precision at 29.5% recall, from a head
firing on 42% of all tokens — so a token-level tagger needs token-level targets,
and BRENDA has no span annotation to give it. What it has is per-document
entity links and a table of surface forms per entity, which is enough to place
labels by matching, but only if the labels admit what matching cannot know.

Hence three values, not two:

===============  =======================================================
`POSITIVE`       matches a surface form of an entity in *this* document's
                 gold set
`IGNORE_INDEX`   matches a surface form of some *other* entity
`NEGATIVE`       matches nothing
===============  =======================================================

**The third value is a target, not a class.** The tagger's output space is
unchanged; `IGNORE_INDEX` marks tokens the loss does not read, which is why it
is spelled as torch's own `ignore_index` default rather than as a third label.

Two-way labelling is the trap. Over 300 validation fulltexts a document matches
a median of 87 distinct entities against a median of 3 gold ones, so 97% of what
matches is not gold-linked; calling all of it negative teaches BRENDA's notion
of *salience* rather than entity-hood, and suppresses hardest exactly where a
novel entity resembles an uncurated one. Abstaining costs ~2.8% of tokens and
keeps ~96% of the negative signal.

**Matching runs once per document, not once per window.** The 512-token windows
overlap by a 20-token stride, so a mention near a boundary lives in two of them
and one split across a boundary lives whole in neither. Labels are therefore
placed on the document's *characters* and projected onto every window's offset
map, which makes the two windows agree by construction and costs one pass over
the text rather than one per window.

**The text has to be the text the encodings were built from**, which is
`d3text.corpus.document_text(abstract, fulltext)` — abstract and body joined
with a newline and *then* stripped of JATS tags. It is not `encode_split`'s
`fulltext` column, which strips tags from the body alone and never sees the
abstract; offsets taken against that string do not address the stored
`input_ids`.
"""

import collections.abc
import re
from dataclasses import dataclass
from typing import Any

import h5py
import hdf5plugin
import numpy
from numpy.typing import NDArray

from d3text.surface_forms import SurfaceFormIndex

IGNORE_INDEX = -100
"""Target for a token the loss must skip.

`torch.nn.functional.cross_entropy`'s own default `ignore_index`, so an array
of these is usable as a target tensor with no translation step. `models.
masked_token_cross_entropy` carries the same default for the same reason.
"""

NEGATIVE = 0
POSITIVE = 1

MAX_MENTION_GAP = 3
"""Characters allowed between two words of one multi-word mention.

The words of a form are matched against the words of the text, so whatever
punctuation separates them is not compared — which is the point, since
`3beta-hydroxysteroid: oxygen oxidoreductase` is one BRENDA synonym written
with three different separators. Bounding the gap is what stops that
indifference from joining words across a paragraph.
"""

_LABEL_DTYPE = numpy.int8


@dataclass(frozen=True, slots=True)
class Mention:
    """A character span of the document, and what it could be naming.

    `entity_ids` is a set because a surface form is not owned by one entity:
    `AS-A` names four separate enzymes, and a species nested inside a strain
    designation yields both. Whether the span is `POSITIVE` or ignored is not
    decided here — it depends on the document's gold set, which a mention knows
    nothing about.
    """

    start: int
    end: int
    entity_ids: frozenset[str]


def find_mentions(
    text: str,
    index: SurfaceFormIndex,
    max_gap: int = MAX_MENTION_GAP,
) -> list[Mention]:
    """Every surface form of any entity, located in `text`.

    Longest match first, and matches do not overlap: `Streptomyces
    griseocarneus` is one mention of one bacterium rather than that plus a
    mention of the genus `Streptomyces`. The consequence worth knowing is that
    a long non-gold form covering a short gold one yields `IGNORE_INDEX` where
    `POSITIVE` was available — abstention, which is the direction this whole
    scheme errs in.
    """
    words = [
        (match.group(), match.start(), match.end())
        for match in re.finditer(r"[^\W_]+", text)
    ]

    mentions: list[Mention] = []
    position = 0
    while position < len(words):
        if not index.may_start(words[position][0]):
            position += 1
            continue

        reach = _contiguous_run(words, position, max_gap, index.max_words)
        matched = 0
        for length in range(reach, 0, -1):
            window = words[position : position + length]
            entity_ids = index.lookup([word for word, _, _ in window])
            if entity_ids:
                mentions.append(
                    Mention(
                        start=window[0][1],
                        end=window[-1][2],
                        entity_ids=entity_ids,
                    )
                )
                matched = length
                break

        position += matched or 1

    return mentions


def _contiguous_run(
    words: list[tuple[str, int, int]],
    position: int,
    max_gap: int,
    limit: int,
) -> int:
    """How many words from `position` are close enough to be one mention."""
    span = 1
    while position + span < len(words) and span < limit:
        if words[position + span][1] - words[position + span - 1][2] > max_gap:
            break
        span += 1
    return span


def character_labels(
    length: int,
    mentions: collections.abc.Iterable[Mention],
    gold_entity_ids: collections.abc.Set[str],
) -> NDArray[numpy.int8]:
    """One target per character of the document.

    A mention is `POSITIVE` when any entity it could name is in the document's
    gold set — any rather than all, because an ambiguous form that includes the
    curated entity is evidence for it, and demanding the form be unambiguous
    would throw away every acronym BRENDA shares between enzymes.
    """
    labels = numpy.full(length, NEGATIVE, dtype=_LABEL_DTYPE)
    for mention in mentions:
        labels[mention.start : mention.end] = (
            POSITIVE
            if mention.entity_ids & frozenset(gold_entity_ids)
            else IGNORE_INDEX
        )
    return labels


def project_onto_tokens(
    labels: NDArray[numpy.int8], offset_mapping: Any
) -> NDArray[numpy.int8]:
    """Read `labels` off for each token of an `offset_mapping`.

    `offset_mapping` is what a fast tokenizer returns beside the `input_ids`
    the encodings store — `[window, token, 2]` character bounds into the same
    string `labels` was built over. The result has the window geometry, so it
    lines up element-wise with the stored `input_ids`.

    A token spanning both a positive and an ignored character is positive: the
    order is `POSITIVE` over `IGNORE_INDEX` over `NEGATIVE`, so a subword that
    straddles a mention boundary is never asserted negative on the strength of
    the half of it that fell outside.

    Special and padding tokens carry an empty `(0, 0)` span and are ignored
    outright — a `[PAD]` contributing to the loss would be a divisor bug of
    exactly the kind this module exists to avoid.
    """
    offsets = numpy.asarray(offset_mapping)
    if offsets.ndim < 2 or offsets.shape[-1] != 2:
        msg = (
            "offset_mapping must end in a size-2 axis of character bounds; "
            f"got shape {offsets.shape}"
        )
        raise ValueError(msg)

    starts = offsets[..., 0].astype(numpy.int64)
    ends = offsets[..., 1].astype(numpy.int64)

    positive = numpy.concatenate(
        ([0], numpy.cumsum(labels == POSITIVE, dtype=numpy.int64))
    )
    ignored = numpy.concatenate(
        ([0], numpy.cumsum(labels == IGNORE_INDEX, dtype=numpy.int64))
    )

    low = numpy.clip(starts, 0, labels.shape[0])
    high = numpy.maximum(numpy.clip(ends, 0, labels.shape[0]), low)

    projected = numpy.where(
        positive[high] - positive[low] > 0,
        POSITIVE,
        numpy.where(ignored[high] - ignored[low] > 0, IGNORE_INDEX, NEGATIVE),
    ).astype(_LABEL_DTYPE)
    projected[ends <= starts] = IGNORE_INDEX

    return projected


def document_token_labels(
    text: str,
    index: SurfaceFormIndex,
    gold_entity_ids: collections.abc.Set[str],
    offset_mapping: Any,
) -> NDArray[numpy.int8]:
    """The three-way targets for one document, in its encodings' geometry."""
    mentions = find_mentions(text, index)
    return project_onto_tokens(
        character_labels(len(text), mentions, gold_entity_ids), offset_mapping
    )


def store_token_labels(
    store: h5py.File, pubmed_id: str, labels: NDArray[numpy.int8]
) -> None:
    """Write one document's targets into an open label store.

    The store is a **parallel HDF5 artifact keyed by pubmed id**, mirroring the
    encodings file rather than riding on the split frames, for three reasons.
    The targets are produced offline against a tokenizer and the BRENDA entity
    tables, and a frame column would recompute both on every run. The frames
    carry no token geometry, so a column could only hold character spans and
    would have to be projected at load time anyway. And `BrendaDataset` narrows
    its frame to four columns and emits six keys, so a new column dies at that
    narrowing unless both are widened — a reader keyed on pubmed id needs
    neither change, since that is already how the encodings are addressed.
    """
    key = str(pubmed_id)
    if key in store:
        del store[key]
    store.create_dataset(
        name=key,
        data=labels,
        dtype="int8",
        compression=hdf5plugin.Zstd(clevel=22),
    )


def load_token_labels(store: h5py.File, pubmed_id: str) -> NDArray[numpy.int8]:
    """One document's targets, as written by `store_token_labels`.

    :raises KeyError: if the store holds no targets for `pubmed_id`.
    """
    key = str(pubmed_id)
    if key not in store:
        msg = f"{key} has no token labels in {store.filename}"
        raise KeyError(msg)
    return numpy.asarray(store[key][:], dtype=_LABEL_DTYPE)


__all__ = [
    "IGNORE_INDEX",
    "MAX_MENTION_GAP",
    "NEGATIVE",
    "POSITIVE",
    "Mention",
    "character_labels",
    "document_token_labels",
    "find_mentions",
    "load_token_labels",
    "project_onto_tokens",
    "store_token_labels",
]
