"""Three-way distant-supervision targets, one per encoded token.

An entity type where a token matches a surface form of an entity in this
document's gold set, `IGNORE_INDEX` where it matches a form of some other
entity, `OUTSIDE` where it matches nothing. See the distant-supervision page of
the documentation for why the middle value is a target rather than a class, and
why the spans are stored beside the codes.
"""

import ast
import collections.abc
import functools
import hashlib
import inspect
import linecache
import os
import re
import sys
import textwrap
import types
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import h5py
import hdf5plugin
import numpy
from numpy.typing import ArrayLike, NDArray

from d3text import surface_forms
from d3text.constraints import EntityId, NonNegative
from d3text.schema import BRENDA_SCHEMA, Schema
from d3text.surface_forms import (
    SurfaceFormIndex,
    index_digest,
    word_spans,
)

IGNORE_INDEX = -100
"""Target for a token the loss must skip.

`torch.nn.functional.cross_entropy`'s own default `ignore_index`, so an array
of these is usable as a target tensor with no translation step.
"""

OUTSIDE = 0
"""The tagger's `O`: a token no surface form of any entity covers."""

NEGATIVE = OUTSIDE
"""`OUTSIDE` under the three-way vocabulary the targets are described in."""

MAX_MENTION_GAP = 3
"""Characters allowed between two words of one multi-word mention."""

_LABEL_DTYPE = numpy.int8


@dataclass(frozen=True)
class LabelSpace:
    """What each integer target means: the entity types, in code order.

    `OUTSIDE` is 0 and the types take 1, 2, 3, ... in the order the schema
    declares them. A store written under one order and read under another
    scores every type against another type's target with no shape ever
    disagreeing, which is why `write_label_space` records this beside the
    codes.
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
        """The label space of `schema`, in its declaration order.

        :param schema: declares the entity types, their order and their ID
            prefixes.
        :return: the space those types define.
        """
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

        `OUTSIDE` is deliberately not among them: it is the absence of a type.
        """
        return tuple(range(1, len(self.types) + 1))

    @property
    def by_prefix(self) -> dict[str, int]:
        """Entity-ID prefix -> its code (`"enz"` -> the enzyme code)."""
        return dict(zip(self.prefixes, self.codes))

    def code_of(self, entity_id: EntityId) -> int:
        """The code of a prefixed entity ID, e.g. `enz3494`.

        :param entity_id: a prefixed BRENDA entity ID.
        :return: the code of the type its prefix names.
        :raises KeyError: if no declared prefix starts `entity_id`, which means
            the gold set and this space were built from different schemas.
        """
        return _code_of(entity_id, self.by_prefix)

    def type_of(self, code: int) -> str:
        """The entity type a code names.

        :param code: one of `codes`.
        :return: the type's name.
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
"""The label space of the BRENDA corpus, the only one there is yet."""


@dataclass(frozen=True, slots=True)
class Mention:
    """A character span of the document, and what it could be naming.

    `entity_ids` is a set because a surface form is not owned by one entity.
    `fuzzy` marks a near-miss rather than a known form, and forces the mention
    to `IGNORE_INDEX` however its candidates fall: it may withhold a type,
    never assert one.
    """

    start: int
    end: int
    entity_ids: frozenset[str]
    fuzzy: bool = False


def find_mentions(
    text: str,
    index: SurfaceFormIndex,
    max_gap: NonNegative = MAX_MENTION_GAP,
) -> list[Mention]:
    """Every surface form of any entity, located in `text`.

    Longest match first, and matches do not overlap. A word the exact index
    finds nothing for is tried once against `index.fuzzy_ids` and recorded as a
    `fuzzy` mention if that hits.

    :param text: the document text to search.
    :param index: the surface forms to search for.
    :param max_gap: characters allowed between two words of one mention.
    :return: the mentions found, in text order.
    """
    words = word_spans(text)

    mentions: list[Mention] = []
    position = 0
    while position < len(words):
        word, start, end = words[position]
        matched = 0

        if index.may_start(word):
            reach = _contiguous_run(words, position, max_gap, index.max_words)
            for length in range(reach, 0, -1):
                window = words[position : position + length]
                entity_ids = index.lookup([w for w, _, _ in window])
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

        if not matched:
            fuzzy_ids = index.fuzzy_ids(word)
            if fuzzy_ids:
                mentions.append(
                    Mention(
                        start=start, end=end, entity_ids=fuzzy_ids, fuzzy=True
                    )
                )

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


def gold_entity_mention_spans(
    mentions: collections.abc.Iterable[Mention],
    gold_entity_ids: collections.abc.Set[str],
) -> dict[str, tuple[tuple[int, int], ...]]:
    """Every gold entity's own mention spans, by entity ID.

    Fuzzy mentions are excluded: `find_mentions` already read them as
    near-misses rather than known forms, so a lucky overlap with the gold set
    must not anchor an entity's representation, the same exclusion
    `_mention_type` makes for the same reason.

    :param mentions: the mentions to read, as `find_mentions` returns them.
    :param gold_entity_ids: the entities this document is linked to.
    :return: entity ID -> its own `(start, end)` spans, in text order. An
        entity absent here has no textual anchor in this document.
    """
    by_entity: dict[str, list[tuple[int, int]]] = {}
    for mention in mentions:
        if mention.fuzzy:
            continue
        for entity_id in mention.entity_ids & gold_entity_ids:
            by_entity.setdefault(entity_id, []).append(
                (mention.start, mention.end)
            )
    return {entity_id: tuple(spans) for entity_id, spans in by_entity.items()}


def _entity_token_presence(
    text_length: int,
    spans: collections.abc.Iterable[tuple[int, int]],
    offset_mapping: ArrayLike,
) -> NDArray[numpy.int8]:
    """Which tokens of `offset_mapping` fall inside any of `spans`.

    The running-cumsum projection `project_onto_tokens` uses, simplified to
    presence: one entity's own spans need no disambiguation against
    another's, unlike a type code.

    :param text_length: the document text's length in characters.
    :param spans: the entity's own `(start, end)` character spans.
    :param offset_mapping: the encodings' character bounds, as
        `project_onto_tokens` reads them.
    :return: one flag per token, shaped like `offset_mapping`'s leading axes.
    """
    mask = numpy.zeros(text_length, dtype=bool)
    for start, end in spans:
        mask[start:end] = True

    offsets = numpy.asarray(offset_mapping)
    starts = offsets[..., 0].astype(numpy.int64)
    ends = offsets[..., 1].astype(numpy.int64)
    running = numpy.concatenate(([0], numpy.cumsum(mask, dtype=numpy.int64)))
    low = numpy.clip(starts, 0, text_length)
    high = numpy.maximum(numpy.clip(ends, 0, text_length), low)
    covered = (running[high] - running[low]) > 0
    covered[ends <= starts] = False
    return covered.astype(_LABEL_DTYPE)


def character_labels(
    length: int,
    mentions: collections.abc.Iterable[Mention],
    gold_entity_ids: collections.abc.Set[str],
    space: LabelSpace = BRENDA_LABELS,
) -> NDArray[numpy.int8]:
    """One target per character of the document.

    :param length: the document text's length in characters.
    :param mentions: the mentions to paint, as `find_mentions` returns them.
    :param gold_entity_ids: the entities this document is linked to.
    :param space: the label space the codes are written in.
    :return: one code per character.
    """
    return character_labels_from_spans(
        length, mention_spans(mentions, gold_entity_ids, space)
    )


SPAN_COLUMNS = 4
"""Width of a `mention_spans` row: start, end, type code, gold flag."""

SPAN_START, SPAN_END, SPAN_TYPE, SPAN_GOLD = range(SPAN_COLUMNS)
"""Column offsets into a `mention_spans` row."""

_SPAN_DTYPE = numpy.int32
"""Character offsets into a fulltext, which runs to hundreds of thousands."""


def mention_spans(
    mentions: collections.abc.Iterable[Mention],
    gold_entity_ids: collections.abc.Set[str],
    space: LabelSpace = BRENDA_LABELS,
) -> NDArray[numpy.int32]:
    """Every mention as a row `(start, end, type_code, gold)`.

    Character coordinates of the document text, half-open. The last two columns
    are not a restatement of each other: `gold` is whether the loss may read
    the type at all, while `type_code` is what the candidates point at even
    when it may not.

    :param mentions: the mentions to record, as `find_mentions` returns them.
    :param gold_entity_ids: the entities this document is linked to.
    :param space: the label space the type codes are written in.
    :return: an `[n_mentions, SPAN_COLUMNS]` array.
    """
    by_prefix = space.by_prefix
    gold = frozenset(gold_entity_ids)
    rows = [
        (mention.start, mention.end, *_mention_type(mention, gold, by_prefix))
        for mention in mentions
    ]
    return numpy.array(rows, dtype=_SPAN_DTYPE).reshape(len(rows), SPAN_COLUMNS)


def character_labels_from_spans(
    length: int, spans: NDArray[numpy.int32]
) -> NDArray[numpy.int8]:
    """Paint `spans` back onto `length` characters.

    The inverse of the projection, and the reason the two stored
    representations cannot drift: `character_labels` *is* this call.

    :param length: the document text's length in characters.
    :param spans: rows as `mention_spans` writes them.
    :return: one code per character.
    :raises ValueError: if `spans` is not `[n_mentions, SPAN_COLUMNS]`.
    """
    rows = numpy.asarray(spans)
    if rows.ndim != 2 or rows.shape[1] != SPAN_COLUMNS:
        msg = (
            f"mention spans must be [n_mentions, {SPAN_COLUMNS}]; "
            f"got shape {rows.shape}"
        )
        raise ValueError(msg)

    labels = numpy.full(length, OUTSIDE, dtype=_LABEL_DTYPE)
    for start, end, code, is_gold in rows.tolist():
        labels[start:end] = code if is_gold else IGNORE_INDEX
    return labels


def mentioned_types(
    spans: NDArray[numpy.int32], min_chars: NonNegative | Mapping[int, int] = 0
) -> frozenset[int]:
    """Every entity-type code appearing anywhere in `spans`, gold or not.

    :param spans: rows as `mention_spans` writes them.
    :param min_chars: shortest mention whose type is counted, uniformly or as a
        `code -> cutoff` mapping; a code the mapping does not name is not
        gated.
    :return: the codes present, `OUTSIDE` rows excluded.
    """
    if spans.size == 0:
        return frozenset()
    lengths = spans[:, SPAN_END] - spans[:, SPAN_START]
    codes = spans[:, SPAN_TYPE]
    cutoffs: int | NDArray[numpy.int32] = (
        numpy.fromiter(
            (min_chars.get(int(code), 0) for code in codes), dtype=lengths.dtype
        )
        if isinstance(min_chars, Mapping)
        else min_chars
    )
    long_enough = lengths >= cutoffs
    return frozenset(
        int(code)
        for code, ok in zip(codes.tolist(), long_enough.tolist())
        if code != OUTSIDE and ok
    )


def _mention_type(
    mention: Mention,
    gold_entity_ids: frozenset[str],
    by_prefix: Mapping[str, int],
) -> tuple[int, int]:
    """`mention`'s type code and whether that code may be asserted.

    A fuzzy mention never counts as matching the gold set here, even when one
    of its candidate entities is gold: `find_mentions` already read `word` as a
    near-miss rather than a known form, and a near-miss of the *right* entity
    is exactly as unverified as one of the wrong one. Forcing `matched` empty
    is what keeps every fuzzy mention `IGNORE_INDEX` rather than letting a
    lucky overlap with the gold set turn an abstention into an assertion.
    """
    matched = (
        frozenset() if mention.fuzzy else mention.entity_ids & gold_entity_ids
    )
    candidates = matched or mention.entity_ids
    codes = {_code_of(entity_id, by_prefix) for entity_id in candidates}
    code = codes.pop() if len(codes) == 1 else OUTSIDE
    return code, int(bool(matched) and code != OUTSIDE)


def project_onto_tokens(
    labels: NDArray[numpy.int8],
    offset_mapping: ArrayLike,
    space: LabelSpace = BRENDA_LABELS,
) -> NDArray[numpy.int8]:
    """Read `labels` off for each token of an `offset_mapping`.

    :param labels: one code per character, as `character_labels` paints them.
    :param offset_mapping: `[window, token, 2]` character bounds into the same
        string, as a fast tokenizer returns beside the `input_ids`.
    :param space: the label space `labels` is written in.
    :return: one code per token, in the offset mapping's window geometry.
    :raises ValueError: if `offset_mapping` does not end in a size-2 axis, or
        reaches past the end of `labels`.
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

    reach = int(max(ends.max(), starts.max())) if ends.size else 0
    if reach > labels.shape[0]:
        msg = (
            f"offset_mapping reaches character {reach}, past the "
            f"{labels.shape[0]} characters labels covers"
        )
        raise ValueError(msg)

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


@dataclass(frozen=True, slots=True)
class DocumentLabels:
    """One document's targets: the per-token codes and the spans behind them.

    `text_length` is the thing neither of them holds: a consumer painting the
    spans back onto a character array would otherwise have to guess it as the
    last mention's `end`, silently shortening every document whose text outruns
    its last match.
    """

    codes: NDArray[numpy.int8]
    spans: NDArray[numpy.int32]
    text_length: int
    entity_token_masks: Mapping[str, NDArray[numpy.int8]] = field(
        default_factory=dict
    )
    """One gold entity's own mention presence, by entity ID.

    Shaped like `codes`; a token is 1 where some non-fuzzy mention matched to
    that entity ID covers it. An entity absent here has no textual anchor in
    this document -- the same "cannot build a representation" case as a
    dictionary coverage miss, and a caller reads both back as no entry.
    """

    def __post_init__(self) -> None:
        if self.spans.ndim != 2 or self.spans.shape[1] != SPAN_COLUMNS:
            msg = (
                f"mention spans must be [n_mentions, {SPAN_COLUMNS}]; "
                f"got shape {self.spans.shape}"
            )
            raise ValueError(msg)
        if self.text_length < 0:
            raise ValueError(f"negative text length {self.text_length}")
        for entity_id, mask in self.entity_token_masks.items():
            if mask.shape != self.codes.shape:
                msg = (
                    f"entity {entity_id!r} carries a mask of shape "
                    f"{mask.shape}, which does not match codes' shape "
                    f"{self.codes.shape}"
                )
                raise ValueError(msg)


def document_token_labels(
    text: str,
    index: SurfaceFormIndex,
    gold_entity_ids: collections.abc.Set[str],
    offset_mapping: ArrayLike,
    space: LabelSpace = BRENDA_LABELS,
) -> DocumentLabels:
    """The typed targets for one document, in its encodings' geometry.

    :param text: the text the encodings were built from, which is
        `d3text.corpus.document_text`'s output and not the `fulltext` column.
    :param index: the surface forms to match.
    :param gold_entity_ids: the entities this document is linked to.
    :param offset_mapping: the encodings' character bounds.
    :param space: the label space to write the codes in.
    :return: the codes and the spans they were projected from.
    :raises ValueError: if `offset_mapping` does not address `text`, either
        malformed or reaching past its length.
    """
    mentions = find_mentions(text, index)
    spans = mention_spans(mentions, gold_entity_ids, space)
    codes = project_onto_tokens(
        character_labels_from_spans(len(text), spans),
        offset_mapping,
        space,
    )
    entity_token_masks = {
        entity_id: _entity_token_presence(
            len(text), entity_spans, offset_mapping
        )
        for entity_id, entity_spans in gold_entity_mention_spans(
            mentions, gold_entity_ids
        ).items()
    }
    return DocumentLabels(
        codes=codes,
        spans=spans,
        text_length=len(text),
        entity_token_masks=entity_token_masks,
    )


_LABELLING_CONSTANT_TYPES = (
    bool,
    bytes,
    float,
    frozenset,
    int,
    re.Pattern,
    str,
)
"""Value types a module-level name must have to count as a labelling constant.

A module or an instance such as `BRENDA_LABELS` is excluded: what those decide
is either recorded on the store beside the rules — the label space, the layout
version — or is unreachable without editing a rule the walk does fingerprint.
A class is not a constant: one the sweep constructs is hashed as a rule, and
one it is only handed, `LabelSpace` or `SurfaceFormIndex`, is covered by the
label space, the index digest and its methods' own fingerprints.
"""


def labelling_rules() -> dict[str, str]:
    """A fingerprint per rule that decides which spans a document gets.

    Covers every function `document_token_labels` reaches inside this module
    and `d3text.surface_forms`, every class of theirs those functions
    construct, and every plain constant they read. It says nothing about the
    behaviour of `rapidfuzz` or `wordfreq`, which the lockfiles pin.

    :return: each rule's qualified name -> its fingerprint.
    :raises OSError: if this package's source is unreachable, which leaves the
        labelling unfingerprintable rather than unchanged.
    """
    sources, constants = _labelling_path()
    rules = dict(sources)
    for name, module, attribute in constants:
        rules[name] = _fingerprint(_constant_repr(getattr(module, attribute)))
    return rules


def _constant_repr(value: object) -> str:
    """A constant's repr in the form every interpreter writes it in.

    A `frozenset` iterates in the order its elements hash, and `PYTHONHASHSEED`
    randomises string hashing per process, so its bare repr differs between two
    runs of the same build. Hashing that would refuse every store at random and
    blame a constant nobody touched.
    """
    if isinstance(value, frozenset):
        elements = sorted(_constant_repr(element) for element in value)
        return "frozenset({" + ", ".join(elements) + "})"
    return repr(value)


@functools.cache
def _labelling_path() -> (
    tuple[dict[str, str], tuple[tuple[str, types.ModuleType, str], ...]]
):
    """The rules and constants `document_token_labels` reaches.

    Only the rule fingerprints are frozen here: source cannot change within a
    process, while a constant is read afresh on every call so that a rebound
    value is seen.
    """
    modules = (sys.modules[__name__], surface_forms)
    methods = {
        name: value
        for name, value in vars(surface_forms.SurfaceFormIndex).items()
        if isinstance(value, types.FunctionType)
    }

    sources: dict[str, str] = {}
    constants: dict[str, tuple[types.ModuleType, str]] = {}
    pending: list[Callable[..., Any]] = [document_token_labels]
    while pending:
        rule = pending.pop()
        name = _rule_name(rule)
        if name in sources:
            continue
        sources[name] = _source_fingerprint(rule)

        tree = _rule_tree(rule)
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                for module in modules:
                    value = vars(module).get(node.id)
                    declared = _declared_rule(
                        value, module, constructed=node.id in called
                    )
                    if declared is not None:
                        pending.append(declared)
                    elif isinstance(value, _LABELLING_CONSTANT_TYPES):
                        key = f"{_short_name(module.__name__)}.{node.id}"
                        constants[key] = (module, node.id)
            elif isinstance(node, ast.Attribute) and node.attr in methods:
                pending.append(methods[node.attr])

    return sources, tuple(
        (name, module, attribute)
        for name, (module, attribute) in sorted(constants.items())
    )


def _declared_rule(
    value: object, module: types.ModuleType, *, constructed: bool
) -> Callable[..., Any] | None:
    """`value` as a rule `module` declares, or None if it is not one.

    A function is unwrapped first, so a memoized rule is fingerprinted by the
    code it memoizes rather than by the wrapper `functools` put around it. A
    class counts only where the sweep constructs it, since only then do its
    field defaults — `Mention.fuzzy`, which the exact branch never passes —
    and its `__post_init__` run on the sweep's output.
    """
    if isinstance(value, type):
        if constructed and value.__module__ == module.__name__:
            return value
        return None
    if not callable(value):
        return None
    unwrapped = inspect.unwrap(value)
    if (
        isinstance(unwrapped, types.FunctionType)
        and unwrapped.__module__ == module.__name__
    ):
        return unwrapped
    return None


def _rule_tree(rule: Callable[..., Any]) -> ast.Module:
    """`rule`'s source, decorators included, dedented out of any class body.

    beartype's import hook recompiles this package so that a decorated
    function's code starts at its `def`, and `inspect.getsource` then drops
    the decorators; they are restored from the file, which is why the lookup
    needs the unwrapped function's own filename and line.
    """
    declared = inspect.unwrap(rule)
    tree = ast.parse(textwrap.dedent(inspect.getsource(declared)))
    head = tree.body[0]
    code = getattr(declared, "__code__", None)
    if code is not None and isinstance(
        head, (ast.AsyncFunctionDef, ast.FunctionDef)
    ):
        line = code.co_firstlineno + head.lineno - 1
        head.decorator_list = _decorators(code.co_filename, line)
    return tree


def _decorators(path: str, line: int) -> list[ast.expr]:
    """The decorators `path` writes on the function defined at `line`."""
    source = "".join(linecache.getlines(path))
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
            and node.lineno == line
        ):
            return node.decorator_list
    return []


def _source_fingerprint(rule: Callable[..., Any]) -> str:
    """A fingerprint of `rule`'s code, blind to its prose and its layout.

    Bare strings are dropped — a docstring, and the note a dataclass field
    carries under it — and the tree is unparsed rather than hashed as written,
    so reformatting or rewriting the explanation of a rule does not invalidate
    every store that rule labelled.
    """
    tree = _rule_tree(rule)
    for node in ast.walk(tree):
        if isinstance(
            node,
            (
                ast.AsyncFunctionDef,
                ast.ClassDef,
                ast.FunctionDef,
                ast.Module,
            ),
        ):
            kept = [
                statement
                for statement in node.body
                if not _is_bare_string(statement)
            ]
            node.body = kept or node.body
    return _fingerprint(ast.unparse(tree))


def _is_bare_string(statement: ast.stmt) -> bool:
    """Whether `statement` is prose: a docstring, or a field's note."""
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


def _rule_name(rule: Callable[..., Any]) -> str:
    """A rule's name as the store records it, e.g. `token_labels.lookup`."""
    return f"{_short_name(rule.__module__)}.{rule.__qualname__}"


def _short_name(dotted: str) -> str:
    """The last segment of a dotted name."""
    return dotted.rsplit(".", 1)[-1]


def _fingerprint(text: str) -> str:
    """A short, stable hash of one rule."""
    return hashlib.sha256(text.encode("utf8")).hexdigest()[:16]


def _rule_lines(rules: Mapping[str, str]) -> list[str]:
    """The rules as `name=fingerprint`, sorted, as the store holds them."""
    return [f"{name}={rules[name]}" for name in sorted(rules)]


def _rules_digest(rules: Mapping[str, str]) -> str:
    """One hash over a whole set of rules, for naming it in a refusal."""
    lines = "\n".join(_rule_lines(rules))
    return hashlib.sha256(lines.encode("utf8")).hexdigest()


TOKEN_LABELS_FORMAT = 5
"""Version of the store's own layout, stamped on its root attributes."""

_FORMAT_ATTRIBUTE = "d3text_token_labels_format"
_TYPES_ATTRIBUTE = "label_types"
_PREFIXES_ATTRIBUTE = "label_prefixes"
_CODES_ATTRIBUTE = "label_codes"
_IGNORE_ATTRIBUTE = "ignore_index"
_OUTSIDE_ATTRIBUTE = "outside_index"
_DIGEST_ATTRIBUTE = "surface_form_index_digest"
_SOURCES_ATTRIBUTE = "surface_form_index_sources"
_RULES_ATTRIBUTE = "labelling_rules"
_TEXT_LENGTH_ATTRIBUTE = "text_length"
_CODES_DATASET = "codes"
_SPANS_DATASET = "spans"
_ENTITY_IDS_DATASET = "entity_ids"
_ENTITY_MASKS_DATASET = "entity_masks"


@dataclass(frozen=True)
class IndexStamp:
    """What determined the surface-form index a store's targets came from.

    `digest` is the whole comparison, since it moves with the datasets pooled,
    with the extractors that pooled them and with the filters applied to the
    result. `sources` is judged on nothing and exists only so a refusal can
    name the inputs the store was built from rather than two hashes.
    """

    digest: str
    sources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.digest:
            raise ValueError("an index stamp must carry a digest")

    @classmethod
    def from_index(
        cls,
        index: SurfaceFormIndex,
        sources: collections.abc.Iterable[str] = (),
    ) -> "IndexStamp":
        """The stamp of `index`, naming the inputs it was pooled from.

        :param index: the surface forms the targets are matched against.
        :param sources: the inputs this invocation pooled that index from, as
            it named them.
        :return: the stamp to record on the store.
        """
        return cls(digest=index_digest(index), sources=tuple(sources))


def write_label_space(
    store: h5py.File,
    space: LabelSpace = BRENDA_LABELS,
    *,
    stamp: IndexStamp,
) -> None:
    """Record what the store's targets mean and what produced them.

    Written once, when the store is created; `store_token_labels` refuses a
    store that has not got it.

    The index is a caller's choice and so arrives as `stamp`; the rules that
    read it are a property of this build, so they are read off the code.

    :param store: an open, writable label store.
    :param space: the space its codes will be written in.
    :param stamp: the surface-form index its targets will be matched against.
    :raises OSError: if this package's source is unreachable, which leaves the
        labelling unfingerprintable.
    """
    store.attrs[_FORMAT_ATTRIBUTE] = TOKEN_LABELS_FORMAT
    store.attrs[_TYPES_ATTRIBUTE] = list(space.types)
    store.attrs[_PREFIXES_ATTRIBUTE] = list(space.prefixes)
    store.attrs[_CODES_ATTRIBUTE] = list(space.codes)
    store.attrs[_IGNORE_ATTRIBUTE] = IGNORE_INDEX
    store.attrs[_OUTSIDE_ATTRIBUTE] = OUTSIDE
    store.attrs[_DIGEST_ATTRIBUTE] = stamp.digest
    store.attrs.create(
        _SOURCES_ATTRIBUTE,
        numpy.array(stamp.sources, dtype=h5py.string_dtype("utf-8")),
    )
    store.attrs.create(
        _RULES_ATTRIBUTE,
        numpy.array(
            _rule_lines(labelling_rules()),
            dtype=h5py.string_dtype("utf-8"),
        ),
    )


def read_label_space(store: h5py.File) -> LabelSpace:
    """The label space a store's targets were written under.

    :param store: an open label store.
    :return: the recorded space.
    :raises KeyError: if the store records none, which means it has to be
        regenerated.
    :raises ValueError: if it was written under another layout version, or
        records a different `IGNORE_INDEX` or `OUTSIDE` than this module uses,
        or codes that are not 1..n in order.
    """
    check_format(store)

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


def check_format(store: h5py.File) -> int:
    """The layout version a store was written under, if this build reads it.

    :param store: an open label store.
    :return: the recorded version.
    :raises KeyError: if the store is stamped with no version at all.
    :raises ValueError: if it is stamped with another version, which calls for
        a regeneration rather than a migration.
    """
    if _FORMAT_ATTRIBUTE not in store.attrs:
        msg = (
            f"{store.filename} records no label space, so what its integer "
            f"targets mean is unknown; {_regenerate(store)}"
        )
        raise KeyError(msg)

    recorded = int(store.attrs[_FORMAT_ATTRIBUTE])
    if recorded != TOKEN_LABELS_FORMAT:
        msg = (
            f"{store.filename} is a format-{recorded} label store and this "
            f"build writes and reads format {TOKEN_LABELS_FORMAT}; "
            f"{_regenerate(store)}"
        )
        raise ValueError(msg)
    return recorded


def read_index_stamp(store: h5py.File) -> IndexStamp:
    """What the store records its targets were matched against.

    :param store: an open label store.
    :return: the recorded stamp.
    :raises KeyError: if the store records no label space, or no surface-form
        index.
    :raises ValueError: if it was written under another layout version.
    """
    check_format(store)

    if _DIGEST_ATTRIBUTE not in store.attrs:
        msg = (
            f"{store.filename} records no surface-form index, so which "
            "strings its targets were matched against is unknown; "
            f"{_regenerate(store)}"
        )
        raise KeyError(msg)

    return IndexStamp(
        digest=_string(store.attrs[_DIGEST_ATTRIBUTE]),
        sources=tuple(_strings(store.attrs[_SOURCES_ATTRIBUTE])),
    )


def check_labelling_rules(store: h5py.File) -> dict[str, str]:
    """The rules a store's targets were placed by, if this build shares them.

    The index digest answers which strings name entities; this answers what
    was done with that answer. A widened `MAX_MENTION_GAP`, a reordered window
    search or a guard added inside `fuzzy_ids` relabels a corpus against a
    byte-identical index, so the two questions need separate stamps.

    :param store: an open label store.
    :return: the recorded fingerprints.
    :raises KeyError: if the store records no label space, or no labelling
        rules.
    :raises ValueError: if it was written under another layout version, or by
        rules this build no longer labels by.
    :raises OSError: if this package's source is unreachable.
    """
    recorded = read_labelling_rules(store)
    current = labelling_rules()
    if recorded == current:
        return recorded

    moved = sorted(
        name
        for name in set(recorded) | set(current)
        if recorded.get(name) != current.get(name)
    )
    msg = (
        f"{store.filename} holds targets placed by labelling rules "
        f"{_rules_digest(recorded)[:12]}, but this build labels by "
        f"{_rules_digest(current)[:12]}; {', '.join(moved)} changed since, "
        "so the same string would be labelled differently — "
        f"{_regenerate(store)}"
    )
    raise ValueError(msg)


def read_labelling_rules(store: h5py.File) -> dict[str, str]:
    """What the store records its targets were placed by.

    :param store: an open label store.
    :return: the recorded fingerprints, by rule name.
    :raises KeyError: if the store records no label space, or no labelling
        rules.
    :raises ValueError: if it was written under another layout version.
    """
    check_format(store)

    if _RULES_ATTRIBUTE not in store.attrs:
        msg = (
            f"{store.filename} records no labelling rules, so which code "
            "placed its targets is unknown; "
            f"{_regenerate(store)}"
        )
        raise KeyError(msg)

    recorded: dict[str, str] = {}
    for line in _strings(store.attrs[_RULES_ATTRIBUTE]):
        name, _, fingerprint = line.partition("=")
        recorded[name] = fingerprint
    return recorded


def check_index(store: h5py.File, stamp: IndexStamp) -> IndexStamp:
    """The store's index stamp, if `stamp` and this build produced it.

    :param store: an open label store.
    :param stamp: the index the caller is about to label against.
    :return: the recorded stamp.
    :raises KeyError: if the store records no label space, no surface-form
        index, or no labelling rules.
    :raises ValueError: if it was written under another layout version,
        against another surface-form index, or by other labelling rules.
    :raises OSError: if this package's source is unreachable.
    """
    recorded = read_index_stamp(store)
    if recorded.digest != stamp.digest:
        msg = (
            f"{store.filename} holds targets matched against surface-form "
            f"index {recorded.digest[:12]}, pooled from "
            f"{_pooled_from(recorded)}, but this run matches against index "
            f"{stamp.digest[:12]}, pooled from {_pooled_from(stamp)}; the two "
            "disagree about which strings name entities, so the file's halves "
            f"would label the same string differently — {_regenerate(store)}"
        )
        raise ValueError(msg)

    check_labelling_rules(store)
    return recorded


def store_index_digest(path: str | os.PathLike[str] | None) -> str | None:
    """The surface-form index digest recorded by the store at `path`.

    Opens the store for this one attribute, so a caller that wants to record
    or compare a run's label provenance need not hold the file open.

    :param path: a label store, or an empty path for a run that reads none.
    :return: the recorded digest, or None where there is no store to read.
    :raises KeyError: if the store records no label space, or no surface-form
        index.
    :raises ValueError: if it was written under another layout version.
    """
    if not path:
        return None

    with h5py.File(path, "r") as store:
        return read_index_stamp(store).digest


def store_labelling_rules_digest(
    path: str | os.PathLike[str] | None,
) -> str | None:
    """The labelling-rules digest recorded by the store at `path`.

    Opens the store for this one attribute, so a caller that wants to record
    or compare a run's label provenance need not hold the file open. Answers
    a different question than `store_index_digest`: that one names which
    strings the targets were matched against, this one names what the sweep
    did with that answer, and either can move while the other does not.

    :param path: a label store, or an empty path for a run that reads none.
    :return: the recorded digest, or None where there is no store to read.
    :raises KeyError: if the store records no label space, or no labelling
        rules.
    :raises ValueError: if it was written under another layout version.
    """
    if not path:
        return None

    with h5py.File(path, "r") as store:
        return _rules_digest(read_labelling_rules(store))


def _regenerate(store: h5py.File) -> str:
    """How to rebuild a refused store, spelled as the command that does it."""
    return (
        "regenerate it with `precompute-token-labels <base_model> "
        f"<entity_tables> {store.filename} <dataset> [dataset ...]`"
    )


def _pooled_from(stamp: IndexStamp) -> str:
    """The inputs a stamp names, for a refusal message."""
    return ", ".join(stamp.sources) if stamp.sources else "unrecorded inputs"


def _string(value: Any) -> str:
    """One HDF5 string, whichever way h5py handed it back."""
    return value.decode("utf8") if isinstance(value, bytes) else str(value)


def _strings(attribute: Any) -> list[str]:
    """An HDF5 string attribute as `str`, whichever way h5py handed it back."""
    return [_string(value) for value in attribute]


def store_token_labels(
    store: h5py.File, pubmed_id: str, labels: DocumentLabels
) -> None:
    """Write one document's targets into an open label store.

    One group per pubmed id, holding the per-token `codes` and the character
    `spans` they were projected from. It takes a `DocumentLabels` rather than
    the two arrays so a store of codes with no spans cannot be written at all.

    :param store: an open, writable label store carrying a label space.
    :param pubmed_id: the document's key; an existing group is replaced.
    :param labels: the codes and spans to write.
    :raises KeyError: if the store records no label space, no surface-form
        index, or no labelling rules.
    :raises ValueError: if it was written under another layout version, or by
        other labelling rules.
    :raises OSError: if this package's source is unreachable.
    """
    if _FORMAT_ATTRIBUTE not in store.attrs:
        msg = (
            f"{store.filename} records no label space; call "
            "`write_label_space` before writing targets into it"
        )
        raise KeyError(msg)
    read_index_stamp(store)
    check_labelling_rules(store)

    key = str(pubmed_id)
    if key in store:
        del store[key]
    group = store.create_group(key)
    group.attrs[_TEXT_LENGTH_ATTRIBUTE] = labels.text_length
    _write_array(group, _CODES_DATASET, labels.codes, "int8")
    _write_array(group, _SPANS_DATASET, labels.spans, "int32")

    entity_ids = sorted(labels.entity_token_masks)
    masks = (
        numpy.stack([labels.entity_token_masks[eid] for eid in entity_ids])
        if entity_ids
        else numpy.zeros((0, *labels.codes.shape), dtype=_LABEL_DTYPE)
    )
    group.create_dataset(
        _ENTITY_IDS_DATASET,
        data=numpy.array(entity_ids, dtype=h5py.string_dtype("utf-8")),
    )
    _write_array(group, _ENTITY_MASKS_DATASET, masks, "int8")


def _write_array(
    group: h5py.Group, name: str, data: NDArray[Any], dtype: str
) -> None:
    """One compressed dataset, or an uncompressed one if it is empty.

    A filter needs chunks and a chunk cannot be zero-sized, so a document that
    matched nothing would fail to store under Zstd.
    """
    group.create_dataset(
        name=name,
        data=data,
        dtype=dtype,
        **({"compression": hdf5plugin.Zstd(clevel=22)} if data.size else {}),
    )


def load_token_labels(
    store: h5py.File,
    pubmed_id: str,
    space: LabelSpace = BRENDA_LABELS,
) -> DocumentLabels:
    """One document's targets, as written by `store_token_labels`.

    :param store: an open label store.
    :param pubmed_id: the document to read.
    :param space: the label space the caller will read the codes under, checked
        against the one the store records rather than assumed.
    :return: the stored codes and spans.
    :raises KeyError: if the store holds no targets for `pubmed_id`, or records
        no label space.
    :raises ValueError: if it was written under another layout version, or
        under a label space other than `space`.
    """
    recorded = read_label_space(store)
    if recorded != space:
        msg = (
            f"{store.filename} records the label space {recorded}, but its "
            f"codes are being read as {space}; every type would be read as "
            "another type — regenerate the store, or read it under the space "
            "it records"
        )
        raise ValueError(msg)

    key = str(pubmed_id)
    if key not in store:
        msg = f"{key} has no token labels in {store.filename}"
        raise KeyError(msg)

    group = store[key]
    entity_ids = _strings(group[_ENTITY_IDS_DATASET][:])
    masks = numpy.asarray(group[_ENTITY_MASKS_DATASET][:], dtype=_LABEL_DTYPE)
    entity_token_masks = {
        entity_id: masks[position]
        for position, entity_id in enumerate(entity_ids)
    }

    return DocumentLabels(
        codes=numpy.asarray(group[_CODES_DATASET][:], dtype=_LABEL_DTYPE),
        spans=numpy.asarray(group[_SPANS_DATASET][:], dtype=_SPAN_DTYPE),
        text_length=int(group.attrs[_TEXT_LENGTH_ATTRIBUTE]),
        entity_token_masks=entity_token_masks,
    )


__all__ = [
    "BRENDA_LABELS",
    "IGNORE_INDEX",
    "MAX_MENTION_GAP",
    "NEGATIVE",
    "OUTSIDE",
    "SPAN_COLUMNS",
    "SPAN_END",
    "SPAN_GOLD",
    "SPAN_START",
    "SPAN_TYPE",
    "TOKEN_LABELS_FORMAT",
    "DocumentLabels",
    "IndexStamp",
    "LabelSpace",
    "Mention",
    "character_labels",
    "character_labels_from_spans",
    "check_format",
    "check_index",
    "check_labelling_rules",
    "document_token_labels",
    "find_mentions",
    "gold_entity_mention_spans",
    "labelling_rules",
    "load_token_labels",
    "mention_spans",
    "mentioned_types",
    "project_onto_tokens",
    "read_index_stamp",
    "read_label_space",
    "read_labelling_rules",
    "store_index_digest",
    "store_labelling_rules_digest",
    "store_token_labels",
    "write_label_space",
]
