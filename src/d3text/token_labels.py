"""Three-way distant-supervision targets, one per encoded token.

The document-level objective localizes badly — the best operating point
measured anywhere over six arms is 29.5% precision at 29.5% recall, from a head
firing on 42% of all tokens — so a token-level tagger needs token-level targets,
and BRENDA has no span annotation to give it. What it has is per-document
entity links and a table of surface forms per entity, which is enough to place
labels by matching, but only if the labels admit what matching cannot know.

Hence three outcomes, not two:

=================  =====================================================
an **entity type** matches a surface form of an entity in *this*
                   document's gold set, and that entity's type is the
                   target
`IGNORE_INDEX`     matches a surface form of some *other* entity
`OUTSIDE`          matches nothing
=================  =====================================================

**The middle value is a target, not a class.** The tagger's output space is one
column per entity type plus `OUTSIDE` — the `O` of an ordinary tagger — and
`IGNORE_INDEX` marks tokens the loss does not read, which is why it is spelled
as torch's own `ignore_index` default rather than as an extra label.

**The codes are recorded inside the artifact.** `LabelSpace` reads the type set
and its order off `d3text.schema.BRENDA_SCHEMA`, and `write_label_space` stamps
that order onto the store's root attributes. Nothing in an array of small
integers says which column is which type; a store written under one order and
read under another does not fail, it scores every type against another type's
target — the same trap `d3text.checkpoint` records a vocabulary against, for
the same reason.

**The labels are flat, not BIO.** The specification is "per token, an entity
type or `O`", so two mentions of the *same* type with no token between them
read as one span. The separator normally supplies that token — mentions are
word-aligned and BRENDA's forms are separated by punctuation or whitespace, and
punctuation is its own token — but two same-type mentions divided only by a
space are genuinely indistinguishable from one. A span tagger that needs the
boundary wants a `B-`/`I-` scheme over the same codes, which is a change to
this module and a regeneration of the artifact.

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
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import h5py
import hdf5plugin
import numpy
from numpy.typing import NDArray

from d3text.schema import BRENDA_SCHEMA, Schema
from d3text.surface_forms import SurfaceFormIndex

IGNORE_INDEX = -100
"""Target for a token the loss must skip.

`torch.nn.functional.cross_entropy`'s own default `ignore_index`, so an array
of these is usable as a target tensor with no translation step. `models.
masked_token_cross_entropy` carries the same default for the same reason.
"""

OUTSIDE = 0
"""The tagger's `O`: a token no surface form of any entity covers."""

NEGATIVE = OUTSIDE
"""`OUTSIDE` under the three-way vocabulary the targets are described in.

Kept as a name because "negative" is what this target *is* — an assertion that
the token names no entity — while `OUTSIDE` is what the tagger's column is
called. They have to be the same integer, and saying so once here is cheaper
than a reader working out that they are.
"""

MAX_MENTION_GAP = 3
"""Characters allowed between two words of one multi-word mention.

The words of a form are matched against the words of the text, so whatever
punctuation separates them is not compared — which is the point, since
`3beta-hydroxysteroid: oxygen oxidoreductase` is one BRENDA synonym written
with three different separators. Bounding the gap is what stops that
indifference from joining words across a paragraph.
"""

_LABEL_DTYPE = numpy.int8


@dataclass(frozen=True)
class LabelSpace:
    """What each integer target means: the entity types, in code order.

    `OUTSIDE` is always 0 and the types take 1, 2, 3, ... in the order they are
    declared. That order is the whole content of this object, and it is the
    reason the object exists: an array of small integers says nothing about
    which code is an enzyme and which a strain, so a store written under one
    order and read under another produces no error at all — it scores every
    type against another type's target. A width change would at least fail
    loudly; a re-permutation does not. `d3text.checkpoint` records a vocabulary
    beside a state dict against exactly that, and `write_label_space` records
    this beside the targets for the same reason.

    Built from a `Schema` rather than declared, so the type set has one
    definition: `d3text.datasets.brenda` derives the class head's columns from
    the same object.
    """

    types: tuple[str, ...]
    prefixes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.types:
            raise ValueError("a label space must declare at least one type")
        if len(self.types) != len(self.prefixes):
            raise ValueError(
                f"{len(self.types)} type names against "
                f"{len(self.prefixes)} ID prefixes"
            )
        for names, what in (
            (self.types, "type names"),
            (self.prefixes, "ID prefixes"),
        ):
            if len(set(names)) != len(names):
                raise ValueError(f"duplicate {what}: {list(names)}")

        # int8 holds -128..127, so the codes fit until a schema declares 127
        # entity types; `IGNORE_INDEX` is -100 and so cannot collide with a
        # code, which is what keeps "the loss skips this token" orthogonal to
        # "this token is of type t".
        highest = len(self.types)
        if highest > numpy.iinfo(_LABEL_DTYPE).max:
            raise ValueError(
                f"{highest} entity types do not fit "
                f"{numpy.dtype(_LABEL_DTYPE).name} targets"
            )

    @classmethod
    def from_schema(cls, schema: Schema) -> "LabelSpace":
        """The label space of `schema`, in its declaration order."""
        return cls(
            types=tuple(
                entity_type.name for entity_type in schema.entity_types
            ),
            prefixes=tuple(
                entity_type.prefix for entity_type in schema.entity_types
            ),
        )

    @property
    def codes(self) -> tuple[int, ...]:
        """The entity-type codes, in declaration order.

        `OUTSIDE` is deliberately not among them: it is the absence of a type,
        and every caller that iterates the types wants to skip it.
        """
        return tuple(range(1, len(self.types) + 1))

    @property
    def by_prefix(self) -> dict[str, int]:
        """Entity-ID prefix -> its code (`"enz"` -> the enzyme code)."""
        return dict(zip(self.prefixes, self.codes))

    def code_of(self, entity_id: str) -> int:
        """The code of a prefixed entity ID, e.g. `enz3494`.

        :raises KeyError: if no declared prefix starts `entity_id`, which means
            the gold set and this space were built from different schemas.
        """
        return _code_of(entity_id, self.by_prefix)

    def type_of(self, code: int) -> str:
        """The entity type a code names.

        :raises KeyError: if `code` is not one of `codes`.
        """
        if code not in self.codes:
            raise KeyError(f"{code} is not an entity-type code of {self}")
        return self.types[code - 1]


def _code_of(entity_id: str, by_prefix: Mapping[str, int]) -> int:
    for prefix, code in by_prefix.items():
        if entity_id.startswith(prefix):
            return code
    msg = (
        f"{entity_id!r} wears none of the declared ID prefixes "
        f"{sorted(by_prefix)}"
    )
    raise KeyError(msg)


BRENDA_LABELS = LabelSpace.from_schema(BRENDA_SCHEMA)
"""The label space of the BRENDA corpus, which is the only one there is yet."""


@dataclass(frozen=True, slots=True)
class Mention:
    """A character span of the document, and what it could be naming.

    `entity_ids` is a set because a surface form is not owned by one entity:
    `AS-A` names four separate enzymes, and a species nested inside a strain
    designation yields both. Which type the span carries, or whether it is
    ignored, is not decided here — it depends on the document's gold set, which
    a mention knows nothing about.
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
    a type was available — abstention, which is the direction this whole scheme
    errs in.
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
    space: LabelSpace = BRENDA_LABELS,
) -> NDArray[numpy.int8]:
    """One target per character of the document.

    A mention carries the *type* of a gold entity it could be naming — any such
    entity rather than all, because an ambiguous form that includes the curated
    entity is evidence for it, and demanding the form be unambiguous would
    throw away every acronym BRENDA shares between enzymes.

    Three resolutions, and two of them abstain rather than guess:

    - Several candidates of the **same** type is that type. `AS-A` names four
      separate enzymes and every one of them makes the token an enzyme, so
      ambiguity about *which* entity is not ambiguity about the target.
    - Gold candidates of **different** types — a species nested in a strain
      designation names both — is `IGNORE_INDEX`. A flat scheme has one code
      per token, so choosing either type would assert that the other is wrong
      here, and the token is genuinely evidence for both.
    - A **gold** candidate of one type beside a **non-gold** candidate of
      another resolves to the gold one's type. The non-gold match is exactly
      what `IGNORE_INDEX` exists not to assert, and the gold link is curated
      fact; this is the typed reading of the rule that already had a positive
      beat an ignore.
    """
    by_prefix = space.by_prefix
    gold = frozenset(gold_entity_ids)
    labels = numpy.full(length, OUTSIDE, dtype=_LABEL_DTYPE)
    for mention in mentions:
        labels[mention.start : mention.end] = _mention_label(
            mention, gold, by_prefix
        )
    return labels


def _mention_label(
    mention: Mention,
    gold_entity_ids: frozenset[str],
    by_prefix: Mapping[str, int],
) -> int:
    """`mention`'s target: one type's code, or `IGNORE_INDEX`."""
    matched = mention.entity_ids & gold_entity_ids
    if not matched:
        return IGNORE_INDEX
    codes = {_code_of(entity_id, by_prefix) for entity_id in matched}
    return codes.pop() if len(codes) == 1 else IGNORE_INDEX


def project_onto_tokens(
    labels: NDArray[numpy.int8],
    offset_mapping: Any,
    space: LabelSpace = BRENDA_LABELS,
) -> NDArray[numpy.int8]:
    """Read `labels` off for each token of an `offset_mapping`.

    `offset_mapping` is what a fast tokenizer returns beside the `input_ids`
    the encodings store — `[window, token, 2]` character bounds into the same
    string `labels` was built over. The result has the window geometry, so it
    lines up element-wise with the stored `input_ids`.

    A token covering characters of **one** type and any number of ignored or
    outside characters takes that type: a subword straddling a mention boundary
    is never asserted `OUTSIDE` on the strength of the half of it that fell
    outside. A token covering **two** types — two adjacent mentions, one
    subword spanning both — is `IGNORE_INDEX`, for the reason a mention naming
    two types is: a flat scheme has one code per token, and either choice would
    assert the other type is wrong there.

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

    def running(value: int) -> NDArray[numpy.int64]:
        return numpy.concatenate(
            ([0], numpy.cumsum(labels == value, dtype=numpy.int64))
        )

    typed = numpy.stack([running(code) for code in space.codes])
    ignored = running(IGNORE_INDEX)

    low = numpy.clip(starts, 0, labels.shape[0])
    high = numpy.maximum(numpy.clip(ends, 0, labels.shape[0]), low)

    covered = typed[:, high] - typed[:, low] > 0
    how_many = covered.sum(axis=0)
    codes = numpy.asarray(space.codes, dtype=_LABEL_DTYPE)

    projected = numpy.where(
        how_many == 1,
        codes[covered.argmax(axis=0)],
        numpy.where(
            (how_many > 1) | (ignored[high] - ignored[low] > 0),
            IGNORE_INDEX,
            OUTSIDE,
        ),
    ).astype(_LABEL_DTYPE)
    projected[ends <= starts] = IGNORE_INDEX

    return projected


def document_token_labels(
    text: str,
    index: SurfaceFormIndex,
    gold_entity_ids: collections.abc.Set[str],
    offset_mapping: Any,
    space: LabelSpace = BRENDA_LABELS,
) -> NDArray[numpy.int8]:
    """The typed targets for one document, in its encodings' geometry."""
    mentions = find_mentions(text, index)
    return project_onto_tokens(
        character_labels(len(text), mentions, gold_entity_ids, space),
        offset_mapping,
        space,
    )


TOKEN_LABELS_FORMAT = 1
"""Version of the store's own layout, stamped on its root attributes."""

_FORMAT_ATTRIBUTE = "d3text_token_labels_format"
_TYPES_ATTRIBUTE = "label_types"
_PREFIXES_ATTRIBUTE = "label_prefixes"
_CODES_ATTRIBUTE = "label_codes"
_IGNORE_ATTRIBUTE = "ignore_index"
_OUTSIDE_ATTRIBUTE = "outside_index"


def write_label_space(
    store: h5py.File, space: LabelSpace = BRENDA_LABELS
) -> None:
    """Record what the store's integer targets mean, on its root attributes.

    Written once, when the store is created. `store_token_labels` refuses to
    write into a store that has not got this, because targets whose meaning
    lives only in the code that produced them are the failure `LabelSpace`
    exists to prevent — and a store already full of them cannot be repaired,
    only regenerated.
    """
    store.attrs[_FORMAT_ATTRIBUTE] = TOKEN_LABELS_FORMAT
    store.attrs[_TYPES_ATTRIBUTE] = list(space.types)
    store.attrs[_PREFIXES_ATTRIBUTE] = list(space.prefixes)
    store.attrs[_CODES_ATTRIBUTE] = list(space.codes)
    store.attrs[_IGNORE_ATTRIBUTE] = IGNORE_INDEX
    store.attrs[_OUTSIDE_ATTRIBUTE] = OUTSIDE


def read_label_space(store: h5py.File) -> LabelSpace:
    """The label space a store's targets were written under.

    :raises KeyError: if the store records none, which is either a store from
        before they were recorded or a file that is not one of these at all.
        Neither can be labelled against safely, and the distinction does not
        help: a store of unattributed codes has to be regenerated either way.
    :raises ValueError: if it records a different `IGNORE_INDEX` or `OUTSIDE`
        than this module uses, or codes that are not 1..n in order.
    """
    if _FORMAT_ATTRIBUTE not in store.attrs:
        msg = (
            f"{store.filename} records no label space, so what its integer "
            "targets mean is unknown; regenerate it"
        )
        raise KeyError(msg)

    space = LabelSpace(
        types=tuple(_strings(store.attrs[_TYPES_ATTRIBUTE])),
        prefixes=tuple(_strings(store.attrs[_PREFIXES_ATTRIBUTE])),
    )

    recorded = {
        _CODES_ATTRIBUTE: [int(code) for code in store.attrs[_CODES_ATTRIBUTE]],
        _IGNORE_ATTRIBUTE: int(store.attrs[_IGNORE_ATTRIBUTE]),
        _OUTSIDE_ATTRIBUTE: int(store.attrs[_OUTSIDE_ATTRIBUTE]),
    }
    expected = {
        _CODES_ATTRIBUTE: list(space.codes),
        _IGNORE_ATTRIBUTE: IGNORE_INDEX,
        _OUTSIDE_ATTRIBUTE: OUTSIDE,
    }
    if recorded != expected:
        msg = (
            f"{store.filename} was written under {recorded}, which this "
            f"build does not use ({expected})"
        )
        raise ValueError(msg)

    return space


def _strings(attribute: Any) -> list[str]:
    """An HDF5 string attribute as `str`, whichever way h5py handed it back."""
    return [
        value.decode("utf8") if isinstance(value, bytes) else str(value)
        for value in attribute
    ]


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
    if _FORMAT_ATTRIBUTE not in store.attrs:
        msg = (
            f"{store.filename} records no label space; call "
            "`write_label_space` before writing targets into it"
        )
        raise KeyError(msg)

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
    "BRENDA_LABELS",
    "IGNORE_INDEX",
    "MAX_MENTION_GAP",
    "NEGATIVE",
    "OUTSIDE",
    "TOKEN_LABELS_FORMAT",
    "LabelSpace",
    "Mention",
    "character_labels",
    "document_token_labels",
    "find_mentions",
    "load_token_labels",
    "project_onto_tokens",
    "read_label_space",
    "store_token_labels",
    "write_label_space",
]
