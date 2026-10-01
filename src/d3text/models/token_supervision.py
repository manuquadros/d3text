"""Reading the precomputed token targets in the geometry the model scores.

The store holds per-window codes; the model scores the *aggregated* document,
so the codes cross that merge through `aggregate_embeddings` itself. Sharing
that one axis lets `resolve_mentions` ground a tagged span in the stored
mentions it overlaps, with neither the document text nor a tokenizer.
"""

import contextlib
import functools
import logging
import os
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import cast

import numpy
import torch
from jaxtyping import Bool, Int64
from numpy.typing import NDArray
from torch import Tensor

from d3text import encodings_store, token_labels
from d3text.constraints import NonNegative
from d3text.linking_eval import TaggedSpan
from d3text.mention_metrics import PredictedMention, token_predicted_mentions
from d3text.surface_forms import SurfaceFormIndex
from d3text.utils import WINDOW_LENGTH, WINDOW_STRIDE, aggregate_embeddings

from .model_types import BatchItem

logger = logging.getLogger(__name__)

_BYTES_PER_MB = 10**6


@dataclass(frozen=True)
class StoredMention:
    """One exact mention the store holds, placed on the aggregated token axis.

    :param entity_ids: every entity the mention's surface form could name, of
        any type and gold or not.
    :param positions: the aggregated-axis tokens covering it, sorted.
    """

    entity_ids: frozenset[str]
    positions: Int64[Tensor, " positions"]


def _document_labels_bytes(labels: token_labels.DocumentLabels | None) -> int:
    """Real memory one cached label group holds.

    :param labels: the group to size, or None for a cached miss.
    :return: the bytes it holds, as `log_cache_stats` reports them.

    `codes`, `ambiguous`, `spans`, `anchors` and `entity_token_masks` are
    arrays, sized by `nbytes`. `candidate_ids` is what `load_token_labels`
    returns, a `CandidatePack`, sized by its own `nbytes` -- the flat ID
    strings, not one `frozenset` object per mention row. A `DocumentLabels`
    built directly with a plain tuple of frozensets (as tests and
    `document_token_labels` do) falls back to summing `sys.getsizeof` over
    each set and its members.
    """
    if labels is None:
        return 0
    total = (
        labels.codes.nbytes
        + labels.ambiguous.nbytes
        + labels.spans.nbytes
        + labels.anchors.nbytes
        + sum(mask.nbytes for mask in labels.entity_token_masks.values())
    )
    candidates = labels.candidate_ids
    if isinstance(candidates, token_labels.CandidatePack):
        total += candidates.nbytes
    else:
        for candidate_set in candidates:
            total += sys.getsizeof(candidate_set)
            total += sum(sys.getsizeof(eid) for eid in candidate_set)
    return total


@dataclass(frozen=True)
class _Derived:
    """Per-document products the raw group's own window geometry determines.

    The window merge selects by the mask alone, so `source`, built once from
    an index tensor, tells every field sharing this geometry which flat
    `window * tokens + column` cell landed at each aggregated-axis position.

    :param source: the flat index selected for each aggregated-axis token.
    :param mentions: `exact_mentions`'s own return value, computed once; None
        until the first call computes it.
    """

    source: Int64[Tensor, " token"]
    mentions: tuple[StoredMention, ...] | None = None


def _derived_bytes(derived: _Derived) -> int:
    """Real memory one document's derived products hold.

    :param derived: the products to size.
    :return: the bytes they hold, as `log_cache_stats` reports them.
    """
    total = derived.source.numel() * derived.source.element_size()
    if derived.mentions is not None:
        for mention in derived.mentions:
            total += (
                mention.positions.numel() * mention.positions.element_size()
            )
            total += sys.getsizeof(mention.entity_ids)
            total += sum(sys.getsizeof(eid) for eid in mention.entity_ids)
    return total


def _source_index(mask: NDArray[numpy.int64]) -> Int64[Tensor, " token"]:
    """The flat `window * tokens + column` cell kept per aggregated token.

    Module-level and bound with `functools.partial`, never a closure: the
    beartype import hook memoises a nested function per `def` execution, so
    each closure, with its copy of `mask`, would live as long as the process.
    """
    windows, tokens = mask.shape
    return (
        aggregate_embeddings(
            torch.arange(windows * tokens).reshape(windows, tokens, 1),
            torch.as_tensor(mask),
        )
        .squeeze(-1)
        .to(torch.int64)
    )


def _stored_mentions(
    labels: token_labels.DocumentLabels,
    source: Int64[Tensor, " token"],
    mask: NDArray[numpy.int64],
) -> tuple[StoredMention, ...]:
    """`exact_mentions`'s value: each anchor placed where the merge put its
    window's token, one `StoredMention` per candidate-bearing row. Module
    level for the reason `_source_index` gives."""
    tokens = mask.shape[1]
    rows, window, start, end = torch.as_tensor(
        labels.anchors, dtype=torch.int64
    ).T
    low = torch.searchsorted(source, window * tokens + start).tolist()
    high = torch.searchsorted(source, window * tokens + end).tolist()

    found: dict[int, list[int]] = {}
    for row, first, last in zip(rows.tolist(), low, high):
        found.setdefault(row, []).extend(range(first, last))
    return tuple(
        StoredMention(
            entity_ids=entity_ids,
            positions=torch.tensor(
                sorted(set(found.get(row, ()))), dtype=torch.int64
            ),
        )
        for row, entity_ids in enumerate(labels.candidate_ids)
        if entity_ids
    )


def live_mentions(
    text: str,
    encoding: encodings_store.Encoding,
    index: SurfaceFormIndex,
) -> tuple[StoredMention, ...]:
    """Every exact mention of `text`, matched now, not read from a store.

    Built the way `TokenLabelReader.exact_mentions` builds the stored ones,
    over `encoding`'s own geometry, so for the text and index a label store
    entry was built from the two agree. Gold plays no part in either.

    :param text: the document text `encoding`'s offsets index.
    :param encoding: the document's windows, as the model reads them.
    :param index: the surface forms to match.
    :return: one entry per mention naming a candidate, fuzzy and ambiguous
        ones excluded, as `exact_mentions` returns them.
    """
    labels = token_labels.document_token_labels(
        text, index, frozenset(), encoding["offset_mapping"]
    )
    mask = numpy.asarray(encoding["attention_mask"], dtype=numpy.int64)
    return _stored_mentions(labels, _source_index(mask), mask)


@dataclass
class _Entry:
    """One document's cached raw group plus whatever of its derived products
    have been computed so far, their bytes charged as a single cost."""

    labels: token_labels.DocumentLabels | None
    labels_cost: int
    derived: _Derived | None = None
    derived_cost: int = 0

    @property
    def total_cost(self) -> int:
        return self.labels_cost + self.derived_cost


class _LabelCache:
    """`TokenLabelReader`'s per-document cache, unbounded by design.

    The dataset bounds it, flat after one pass; a byte budget the store
    outgrew would make every later document a permanent miss, re-decoded from
    the store on each lookup. Bytes, raw group and `_Derived` products as one
    cost per document, are still counted for `log_cache_stats`.
    """

    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}
        self._used = 0
        # Counted at `TokenLabelReader._load`'s cache check, reset per pass
        # by `log_cache_stats`: more than one miss per document per run says
        # something is reading around the cache.
        self.hits = 0
        self.misses = 0

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def get(self, key: str) -> token_labels.DocumentLabels | None:
        """Look up a cached label group; caller checks `in` first for a miss."""
        return self._entries[key].labels

    def set(self, key: str, value: token_labels.DocumentLabels | None) -> None:
        """Cache `value` under `key`, replacing any earlier entry.

        :param key: the document ID to store it under.
        :param value: the label group, charged its real
            `_document_labels_bytes` cost for reporting.
        """
        existing = self._entries.get(key)
        if existing is not None:
            self._used -= existing.total_cost
        cost = _document_labels_bytes(value)
        self._entries[key] = _Entry(labels=value, labels_cost=cost)
        self._used += cost

    def get_or_compute_source(
        self, key: str, compute: Callable[[], Int64[Tensor, " token"]]
    ) -> Int64[Tensor, " token"]:
        """This document's aggregated-axis source index, computed once.

        :param key: the document ID `set` cached the raw group under.
        :param compute: builds the source index on a cache miss.
        :return: the cached (or freshly built) source index.
        """
        entry = self._entries.get(key)
        if entry is not None and entry.derived is not None:
            return entry.derived.source
        source = compute()
        self._store_derived(key, _Derived(source=source))
        return source

    def get_or_compute_mentions(
        self, key: str, compute: Callable[[], tuple[StoredMention, ...]]
    ) -> tuple[StoredMention, ...]:
        """This document's exact mentions, computed once.

        :param key: the document ID `set` cached the raw group under.
        :param compute: builds the mention tuple on a cache miss; called
            only once the shared source index is already cached (by an
            earlier `get_or_compute_source`), so this costs no extra
            `aggregate_embeddings` call of its own.
        :return: the cached (or freshly built) mentions.
        """
        entry = self._entries.get(key)
        if (
            entry is not None
            and entry.derived is not None
            and entry.derived.mentions is not None
        ):
            return entry.derived.mentions
        mentions = compute()
        entry = self._entries.get(key)
        if entry is not None and entry.derived is not None:
            self._store_derived(key, replace(entry.derived, mentions=mentions))
        return mentions

    def _store_derived(self, key: str, derived: _Derived) -> None:
        """Attach `derived` to `key`'s entry; a document `set` never cached
        (the store lacks it, or nothing loaded it) has nothing to attach
        them to."""
        entry = self._entries.get(key)
        if entry is None:
            return
        cost = _derived_bytes(derived)
        self._used += cost - entry.derived_cost
        entry.derived = derived
        entry.derived_cost = cost


class TokenLabelReader:
    """One run's handle on a token-label store, space- and tokenizer-checked
    once at open.

    :param path: the store to open for reading.
    :param space: the label space this run's tagger head is sized to.
    :param base_model: the checkpoint this run tokenizes with. Required, not
        defaulted, so a caller cannot silently build a reader the tokenizer
        check never runs against.
    :raises KeyError: if the store records no label space or no tokenizer.
    :raises ValueError: if it was written under another layout version,
        records another label space, or was tokenized by another base model
        or at another window geometry than this run merges codes under; or if
        it is a store of the older HDF5 layout.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        space: token_labels.LabelSpace = token_labels.BRENDA_LABELS,
        *,
        base_model: str,
    ) -> None:
        self._store = token_labels.TokenLabelStore(path)
        try:
            recorded = token_labels.read_label_space(self._store)
            if recorded != space:
                msg = (
                    f"{os.fspath(path)} records the label space {recorded}, "
                    f"but this model's tagger head is sized to {space}; its "
                    "codes would be scored against the wrong columns — "
                    "regenerate the store, or build the model over the "
                    "space it records"
                )
                raise ValueError(msg)
            # Window length and stride are this process's own constants,
            # the ones `_source_index` merges at, so they are not the
            # caller's to pass.
            token_labels.check_reader_tokenizer(
                self._store, base_model, WINDOW_LENGTH, WINDOW_STRIDE
            )
        except (KeyError, ValueError):
            self._store.close()
            raise
        self.space = space
        self._label_cache = _LabelCache()

    def close(self) -> None:
        self._store.close()

    def refuse_wholly_missing_sources(
        self,
        splits: Mapping[str, Iterable[tuple[str, int]]],
    ) -> None:
        """Refuse splits with a corpus source absent from the label store.

        :param splits: split name to its `(source, pubmed_id)` rows.
        :raises ValueError: if every row of a source is absent from the store.
        """
        stored = frozenset(self._store.keys())
        missing: dict[str, list[str]] = {}
        for split, documents in splits.items():
            source_present: dict[str, bool] = {}
            for source, pubmed_id in documents:
                source_present[source] = (
                    source_present.get(source, False)
                    or str(pubmed_id) in stored
                )
            wholly_missing = sorted(
                source
                for source, present in source_present.items()
                if not present
            )
            if wholly_missing:
                missing[split] = wholly_missing

        if missing:
            detail = ", ".join(
                f"{split}: {sources}" for split, sources in missing.items()
            )
            msg = (
                f"{self._store.path} holds no token labels for any row of "
                f"source(s) in {detail}. "
                f"{token_labels.regeneration_hint(self._store.path)}."
            )
            raise ValueError(msg)

    def _load(self, pubmed_id: int | str) -> token_labels.DocumentLabels | None:
        """One document's raw label group, or None if the store lacks it.

        Shared by `document_codes`, `mentioned_types`, `entity_positions` and
        `exact_mentions` through a per-instance cache, so a document already
        read this pass — including every gold entity `entity_positions` reads
        off the same document — costs one store read rather than one per
        call.
        """
        key = str(pubmed_id)
        if key in self._label_cache:
            self._label_cache.hits += 1
            return self._label_cache.get(key)

        self._label_cache.misses += 1
        try:
            labels = token_labels.load_token_labels(
                self._store, key, self.space
            )
        except KeyError:
            labels = None
        self._label_cache.set(key, labels)
        return labels

    def log_cache_stats(self, step: str) -> None:
        """Report and reset this pass's label-cache hit rate.

        Mirrors the CPU embeddings cache's reporting in
        `Model.log_pass_stats`, which calls this. The cache never evicts, so
        past the first pass every lookup should hit; a miss on a later pass,
        or a held size still growing, is a document being read around the
        cache.

        :param step: which pass this covers, for the log line.
        """
        cache = self._label_cache
        total = cache.hits + cache.misses
        hit_rate = 100 * cache.hits / total if total else 0.0
        logger.info(
            "Token-label cache (%s pass): %d/%d hits (%.1f%%), %d documents "
            "cached, %d MB held",
            step,
            cache.hits,
            total,
            hit_rate,
            len(cache._entries),
            cache._used // _BYTES_PER_MB,
        )
        cache.hits = 0
        cache.misses = 0

    def _aggregated_source(
        self, key: str, mask: NDArray[numpy.int64]
    ) -> Int64[Tensor, " token"]:
        """This document's aggregated-axis source index, cached per document.

        The merge picks positions from the mask alone, so the index is the
        same for every per-token field aggregated over this mask.

        :param key: the document's cache key, as `_load` uses it.
        :param mask: the `[windows, tokens]` attention mask, already
            reshaped and validated against the field about to be gathered.
        :return: the flat index selected for each aggregated-axis token.
        """
        return self._label_cache.get_or_compute_source(
            key, functools.partial(_source_index, mask)
        )

    def mentioned_types(
        self,
        pubmed_id: int | str,
        min_chars: NonNegative | Mapping[int, int] = 0,
    ) -> frozenset[int] | None:
        """Every entity-type code matched anywhere in the document, or None.

        :param pubmed_id: the document to read.
        :param min_chars: shortest mention counted, uniformly or per type code.
        :return: the codes present, or None when the store holds nothing for
            this document — outside what the store covers, not a document that
            mentions nothing.
        """
        labels = self._load(pubmed_id)
        if labels is None:
            return None
        return token_labels.mentioned_types(labels.spans, min_chars=min_chars)

    def document_codes(
        self,
        pubmed_id: int | str,
        window_attention_mask: object,
    ) -> Int64[Tensor, " token"] | None:
        """One document's targets on the aggregated token axis, or None.

        :param pubmed_id: the document to read.
        :param window_attention_mask: the document's own mask as the batch item
            carries it; leading collation axes are flattened away.
        :return: one target per aggregated token, or None when the store holds
            no targets — the caller's to skip or to mask, since only it knows
            whether that is a truncated split or a stale store.
        :raises ValueError: if the stored codes and the mask disagree in window
            geometry, which means the store was built against different
            encodings.
        """
        key = str(pubmed_id)
        labels = self._load(key)
        if labels is None:
            return None

        mask = numpy.asarray(window_attention_mask)
        mask = mask.reshape(-1, mask.shape[-1]).astype(numpy.int64)
        if labels.codes.shape != mask.shape:
            msg = (
                f"document {key} stores codes of shape {labels.codes.shape} "
                f"against encodings of shape {mask.shape}; the label store "
                "was built from different encodings — regenerate it"
            )
            raise ValueError(msg)

        source = self._aggregated_source(key, mask)
        flat = torch.as_tensor(labels.codes, dtype=torch.int64).reshape(-1)
        return flat[source]

    def document_ambiguous(
        self,
        pubmed_id: int | str,
        window_attention_mask: object,
    ) -> Bool[Tensor, " token"] | None:
        """One document's ambiguous-mention flags, aggregated axis, or None.

        Mirrors `document_codes` exactly, over the `ambiguous` channel
        instead of `codes`.

        :param pubmed_id: the document to read.
        :param window_attention_mask: the document's own mask as the batch item
            carries it; leading collation axes are flattened away.
        :return: one flag per aggregated token, or None when the store holds
            no targets — the caller's to skip or to mask, since only it knows
            whether that is a truncated split or a stale store.
        :raises ValueError: if the stored ambiguous mask and the mask disagree
            in window geometry, which means the store was built against
            different encodings.
        """
        key = str(pubmed_id)
        labels = self._load(key)
        if labels is None:
            return None

        mask = numpy.asarray(window_attention_mask)
        mask = mask.reshape(-1, mask.shape[-1]).astype(numpy.int64)
        if labels.ambiguous.shape != mask.shape:
            msg = (
                f"document {key} stores an ambiguous mask of shape "
                f"{labels.ambiguous.shape} against encodings of shape "
                f"{mask.shape}; the label store was built from different "
                "encodings — regenerate it"
            )
            raise ValueError(msg)

        source = self._aggregated_source(key, mask)
        flat = torch.as_tensor(labels.ambiguous).reshape(-1)
        return flat[source] > 0

    def entity_positions(
        self,
        pubmed_id: int | str,
        entity_id: str,
        window_attention_mask: object,
    ) -> Int64[Tensor, " positions"] | None:
        """One gold entity's own mention token positions, aggregated axis.

        Mirrors `document_codes`'s aggregation, keyed by entity ID rather than
        read off the type-code channel, so a relation argument's
        representation can be pooled from its own mention(s) rather than from
        a learned column.

        :param pubmed_id: the document to read.
        :param entity_id: the entity whose mention span(s) to look up.
        :param window_attention_mask: the document's own mask as the batch
            item carries it; leading collation axes are flattened away.
        :return: the aggregated-axis token indices, sorted; None when the
            store holds nothing for this document, or nothing for this
            entity -- no textual anchor and a dictionary coverage miss read
            back alike.
        :raises ValueError: if the stored mask and the batch mask disagree in
            window geometry, which means the store was built against
            different encodings.
        """
        key = str(pubmed_id)
        labels = self._load(key)
        if labels is None:
            return None

        entity_mask = labels.entity_token_masks.get(str(entity_id))
        if entity_mask is None:
            return None

        mask = numpy.asarray(window_attention_mask)
        mask = mask.reshape(-1, mask.shape[-1]).astype(numpy.int64)
        if entity_mask.shape != mask.shape:
            msg = (
                f"document {key} stores a mask of shape {entity_mask.shape} "
                f"for entity {entity_id!r} against encodings of shape "
                f"{mask.shape}; the label store was built from different "
                "encodings -- regenerate it"
            )
            raise ValueError(msg)

        source = self._aggregated_source(key, mask)
        flat = torch.as_tensor(entity_mask).reshape(-1)
        positions = torch.nonzero(flat[source] > 0, as_tuple=True)[0]
        return positions.to(torch.int64) if positions.numel() else None

    def exact_mentions(
        self,
        pubmed_id: int | str,
        window_attention_mask: object,
    ) -> tuple[StoredMention, ...] | None:
        """Every exact mention of one document, in text order, or None.

        Reads the anchors of every dictionary match, not only gold ones, and
        so never `entity_positions`' gold-only masks. Each window's token index
        is itself run through `aggregate_embeddings`, so an anchor lands where
        the merge put that token and a mention in an overlap is counted once.

        :param pubmed_id: the document to read.
        :param window_attention_mask: the document's own mask as the batch
            item carries it; leading collation axes are flattened away.
        :return: one entry per exact mention, fuzzy ones excluded; None when
            the store holds nothing for this document.
        :raises ValueError: if the stored codes and the mask disagree in window
            geometry, which means the store was built against different
            encodings.
        """
        key = str(pubmed_id)
        labels = self._load(key)
        if labels is None:
            return None

        mask = numpy.asarray(window_attention_mask)
        mask = mask.reshape(-1, mask.shape[-1]).astype(numpy.int64)
        if labels.codes.shape != mask.shape:
            msg = (
                f"document {key} stores codes of shape {labels.codes.shape} "
                f"against encodings of shape {mask.shape}; the label store "
                "was built from different encodings — regenerate it"
            )
            raise ValueError(msg)

        source = self._aggregated_source(key, mask)
        return self._label_cache.get_or_compute_mentions(
            key, functools.partial(_stored_mentions, labels, source, mask)
        )

    def _gold_entity_positions(
        self,
        pubmed_id: int | str,
        window_attention_mask: object,
    ) -> dict[str, Int64[Tensor, " positions"]]:
        """Every gold entity's own aggregated-axis positions, one document.

        Calls `entity_positions` once per gold entity the document's label
        group names, relying on `_load`'s cache so the group itself is read
        once regardless of how many entities it carries.

        :param pubmed_id: the document to read.
        :param window_attention_mask: the document's own mask as the batch
            item carries it; leading collation axes are flattened away.
        :return: entity ID -> its positions; a document the store holds
            nothing for, and an entity with no textual anchor, both
            contribute nothing rather than raising.
        """
        labels = self._load(pubmed_id)
        if labels is None:
            return {}

        found = {
            entity_id: self.entity_positions(
                pubmed_id, entity_id, window_attention_mask
            )
            for entity_id in labels.entity_token_masks
        }
        return {
            entity_id: positions
            for entity_id, positions in found.items()
            if positions is not None
        }


def resolve_mentions(
    predicted: Sequence[PredictedMention],
    stored: Sequence[StoredMention],
    space: token_labels.LabelSpace = token_labels.BRENDA_LABELS,
) -> list[PredictedMention]:
    """Give each predicted span the candidates of the mentions it overlaps.

    Narrowing never picks: a candidate set the document names no
    single-candidate form of stays whole. Its input is every dictionary match,
    gold or not, so nothing here reads a gold-only mask — see the models page
    of the documentation.

    :param predicted: the tagger's spans, on the aggregated token axis.
    :param stored: the same document's exact mentions, as
        `TokenLabelReader.exact_mentions` returns them.
    :param space: the label space the spans' type codes are written in.
    :return: the same spans in the same order, each carrying the candidate IDs
        of its own tagged type from every mention it overlaps, narrowed to the
        IDs the document also names through a single-candidate mention wherever
        that intersection is non-empty. An empty set is NIL — a typed span the
        store grounds in nothing — rather than a failure.
    :raises KeyError: if a span wears a type code `space` does not declare,
        which means the tagger head and this label space were built over
        different schemas.
    """
    covering: dict[int, set[str]] = {}
    for mention in stored:
        for position in mention.positions.tolist():
            covering.setdefault(position, set()).update(mention.entity_ids)
    unambiguous = frozenset(
        entity_id
        for mention in stored
        if len(mention.entity_ids) == 1
        for entity_id in mention.entity_ids
    )
    resolved: list[PredictedMention] = []
    for span in predicted:
        prefix = space.prefix_of(span.type_code)

        covered: set[str] = set()
        for position in range(span.start, span.end):
            covered |= covering.get(position, set())
        candidates = frozenset(
            entity_id for entity_id in covered if entity_id.startswith(prefix)
        )
        resolved.append(
            replace(span, entity_ids=(candidates & unambiguous) or candidates)
        )
    return resolved


def char_spans_from_predictions(
    predicted: Sequence[PredictedMention],
    offset_mapping: NDArray[numpy.integer] | Tensor,
    attention_mask: NDArray[numpy.integer] | Tensor,
    text: str,
    document: str,
    space: token_labels.LabelSpace = token_labels.BRENDA_LABELS,
) -> list[TaggedSpan]:
    """Ground a tagger's aggregated-axis spans in one document's own text.

    `offset_mapping` crosses the same window merge as the embeddings. No
    BRENDA mention store is consulted, which is what makes this usable
    against an external corpus's own annotation offsets.

    :param predicted: the tagger's spans, on the aggregated token axis.
    :param offset_mapping: the document's windowed char-offset mapping, as
        `precompute-encodings` stores it.
    :param attention_mask: the document's windowed attention mask, same
        `[windows, tokens]` shape `offset_mapping` aggregates against.
    :param text: the document's own text, sliced at each span's char bounds.
    :param document: the external corpus's document key, carried onto every
        `TaggedSpan` unchanged.
    :param space: the label space `predicted`'s type codes are written in.
    :return: one `TaggedSpan` per predicted mention, in the same order.
    """
    offsets = aggregate_embeddings(
        torch.as_tensor(numpy.asarray(offset_mapping), dtype=torch.int64),
        torch.as_tensor(numpy.asarray(attention_mask)),
    )
    spans = []
    for mention in predicted:
        char_start = int(offsets[mention.start, 0])
        char_end = int(offsets[mention.end - 1, 1])
        spans.append(
            TaggedSpan(
                document=document,
                start=char_start,
                end=char_end,
                surface=text[char_start:char_end],
                entity_type=space.type_of(mention.type_code),
            )
        )
    return spans


def store_batch_item(
    encoding: encodings_store.Encoding, document_id: int
) -> BatchItem:
    """One stored document's windows, as the model methods take a batch item.

    :param encoding: the document, as the encodings store returns it.
    :param document_id: what `get_token_embeddings` keys its caches on — a
        BRENDA document's pubmed id, or the id
        `encodings_store.external_document_id` mints for a document of an
        external corpus, which has no pubmed id.
    :return: the item, carrying the document's token ids and mask alone:
        nothing built from a stored document reads gold, so it has neither
        a class nor a relation field.
    """
    return {
        "id": torch.tensor(document_id),
        "doc_id": torch.zeros(encoding["input_ids"].shape[0]),
        "sequence": {
            "input_ids": torch.as_tensor(encoding["input_ids"]),
            "attention_mask": torch.as_tensor(encoding["attention_mask"]),
        },
    }


def resolve_token_tagger(model: object) -> Callable[[Tensor], Tensor] | None:
    """The model's span tagger, as `predicted_spans_from_store` takes it.

    No concrete model types `token_tagger` as a `Callable`, yet a non-None
    one always is, so the cast restates that. What to do about a checkpoint
    carrying none is the caller's.

    :param model: a loaded checkpoint. Typed loosely (`object`, not
        `factory.ConfigurableModel`) because the command tests drive their
        callers through stub models beartype would otherwise refuse.
    :return: the tagger, or None where the checkpoint detects no span.
    """
    return cast(
        Callable[[Tensor], Tensor] | None,
        getattr(model, "token_tagger", None),
    )


def _pubmed_document_id(key: str) -> int:
    """`key` read as the pubmed id a BRENDA document is stored under.

    Validated rather than converted: `get_token_embeddings` keys its caches on
    this number, and everything at or below zero is reserved for the documents
    of corpora that issue no pubmed id, so a key reading as one of those would
    be served an external document's activations.
    """
    document_id = int(key)
    if document_id <= 0:
        msg = (
            f"{key!r} is not a pubmed id, so it names no BRENDA document: "
            f"an id at or below zero belongs to the external corpora."
        )
        raise ValueError(msg)
    return document_id


def _group_key(corpus: str | None, document: str) -> str:
    """The store key `document` is written under for `corpus`.

    :param corpus: which corpus's groups to read, or None for a BRENDA
        document keyed by its bare pubmed id.
    :param document: the document id, as `texts` keys it.
    :return: the store key.
    """
    return (
        document
        if corpus is None
        else encodings_store.external_key(corpus, document)
    )


def readable_documents(
    store: encodings_store.EncodingsStore,
    corpus: str | None,
    documents: Iterable[str],
) -> frozenset[str]:
    """Which of `documents` `predicted_spans_from_store` would actually read.

    The same key `predicted_spans_from_store` reads under, so the two never
    disagree.

    :param store: an open encodings store.
    :param corpus: which corpus's documents to read (`"s800"` or
        `"enzymener"`), or None for BRENDA documents, keyed by the bare
        pubmed id.
    :param documents: the document ids to check.
    :return: the subset of `documents` the store holds.
    """
    return frozenset(
        document
        for document in documents
        if _group_key(corpus, document) in store
    )


def predicted_spans_from_encoding(
    encoding: encodings_store.Encoding,
    document_id: int,
    text: str,
    document: str,
    get_token_embeddings: Callable[
        [Sequence[BatchItem]], tuple[Tensor, Tensor]
    ],
    hidden: Callable[[Tensor, Tensor], Tensor],
    token_tagger: Callable[[Tensor], Tensor],
    autocast: Callable[[], contextlib.AbstractContextManager[object]],
    space: token_labels.LabelSpace = token_labels.BRENDA_LABELS,
) -> list[TaggedSpan]:
    """Run a tagger over one document's encoding.

    Takes the calls a forward needs, not a model, for the reason
    `predicted_spans_from_store` gives.

    :param encoding: the document's windows, as the encodings store returns
        them or `encodings_store.document_encodings` builds them.
    :param document_id: what `get_token_embeddings` keys its caches on, as
        `store_batch_item` takes it.
    :param text: the document's full text, which `encoding`'s offsets index.
    :param document: the document's id, as each span records it.
    :param get_token_embeddings: the trained model's own, e.g.
        `model.get_token_embeddings`.
    :param hidden: the trained model's own, e.g. `model.hidden`.
    :param token_tagger: the trained model's own, e.g. `model.token_tagger`
        — the caller's to confirm is not `None` before passing it.
    :param autocast: the trained model's own, e.g. `model.autocast_context`,
        for the reason `predicted_spans_from_store` gives.
    :param space: the label space `token_tagger`'s codes are written in.
    :return: one `TaggedSpan` per predicted mention.
    """
    item = store_batch_item(encoding, document_id)
    with torch.no_grad():
        embeddings, mask = get_token_embeddings([item])
        with autocast():
            token_logits = token_tagger(hidden(embeddings, mask))
    length = int(mask[0].sum())
    codes = token_logits[0, :length].argmax(dim=-1).cpu().numpy()
    return char_spans_from_predictions(
        token_predicted_mentions(codes),
        encoding["offset_mapping"],
        encoding["attention_mask"],
        text=text,
        document=document,
        space=space,
    )


def predicted_spans_from_store(
    store: encodings_store.EncodingsStore,
    corpus: str | None,
    texts: Mapping[str, str],
    get_token_embeddings: Callable[
        [Sequence[BatchItem]], tuple[Tensor, Tensor]
    ],
    hidden: Callable[[Tensor, Tensor], Tensor],
    token_tagger: Callable[[Tensor], Tensor],
    autocast: Callable[[], contextlib.AbstractContextManager[object]],
    space: token_labels.LabelSpace = token_labels.BRENDA_LABELS,
) -> list[TaggedSpan]:
    """Run a tagger over the `texts` documents `store` holds.

    Takes the calls a forward needs, not a model, whose attributes type as
    `Tensor | Module`. A document the store lacks is skipped. The
    embeddings cache is keyed by the store key's own id, never one counted
    off `texts`, which would name another document in the next call.

    :param store: an open encodings store.
    :param corpus: which corpus's documents to read (`"s800"` or
        `"enzymener"`), or None for BRENDA documents, keyed by the bare
        pubmed id.
    :param texts: document id to full text, as `load_s800`/`load_enzymener`
        return, or pubmed id to `corpus.document_text` output.
    :param get_token_embeddings: the trained model's own, e.g.
        `model.get_token_embeddings`.
    :param hidden: the trained model's own, e.g. `model.hidden`.
    :param token_tagger: the trained model's own, e.g. `model.token_tagger`
        — the caller's to confirm is not `None` before passing it.
    :param autocast: the trained model's own, e.g. `model.autocast_context`.
        `get_token_embeddings` returns its embeddings in the model's AMP
        dtype while `hidden` and `token_tagger` keep fp32 weights, so the
        two only meet under the autocast training ran them in.
    :param space: the label space `token_tagger`'s codes are written in.
    :return: one `TaggedSpan` per predicted mention, across every readable
        document.
    :raises ValueError: with `corpus` None, if a key of `texts` is not a
        positive pubmed id — which means these documents were not written by
        the BRENDA path of `precompute-encodings`.
    """
    spans: list[TaggedSpan] = []
    for document, text in texts.items():
        key = _group_key(corpus, document)
        encoding = store.get(key)
        if encoding is None:
            continue

        document_id = (
            _pubmed_document_id(document)
            if corpus is None
            else encodings_store.external_document_id(key)
        )
        spans.extend(
            predicted_spans_from_encoding(
                encoding,
                document_id,
                text,
                document,
                get_token_embeddings,
                hidden,
                token_tagger,
                autocast,
                space,
            )
        )
    return spans


def padded_targets(
    rows: list[Int64[Tensor, " token"]],
    length: int,
    ignore_index: int = token_labels.IGNORE_INDEX,
) -> Int64[Tensor, "document token"]:
    """Stack per-document target rows to `length`, padding with the mask.

    Padding is `ignore_index` rather than a class: a pad contributing to the
    loss would be the divisor bug `masked_token_cross_entropy` exists to avoid.

    :param rows: one target row per document.
    :param length: the padded token axis to stack to.
    :param ignore_index: the value marking a token the loss must skip.
    :return: the padded targets.
    """
    padded = torch.full((len(rows), length), ignore_index, dtype=torch.int64)
    for row_index, row in enumerate(rows):
        padded[row_index, : row.shape[0]] = row
    return padded


def document_lengths(attention_mask: Tensor) -> list[int]:
    """Unpadded token count per document of a batch-level attention mask.

    :param attention_mask: the batch's mask.
    :return: one count per document.
    """
    return [int(count) for count in attention_mask.sum(dim=1).tolist()]


__all__ = [
    "StoredMention",
    "TokenLabelReader",
    "char_spans_from_predictions",
    "document_lengths",
    "live_mentions",
    "padded_targets",
    "predicted_spans_from_store",
    "readable_documents",
    "resolve_mentions",
    "store_batch_item",
]
