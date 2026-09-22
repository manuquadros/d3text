"""Reading the precomputed token targets in the geometry the model scores.

The store holds per-window codes; the model scores the *aggregated* document.
The codes are carried across that merge by running them through
`aggregate_embeddings` itself rather than restating its overlap arithmetic. The
label space is verified at open, not assumed, since a store written under a
permuted schema holds codes whose integers mean different types. Sharing that
one axis is what lets `resolve_mentions` ground a tagged span in the stored
mentions it overlaps, with neither the document text nor a tokenizer.
"""

import logging
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import cast

import h5py
import numpy
import torch
from jaxtyping import Bool, Int64
from numpy.typing import NDArray
from torch import Tensor

from d3text import encodings_store, token_labels
from d3text.constraints import NonNegative
from d3text.linking_eval import TaggedSpan
from d3text.mention_metrics import PredictedMention, token_predicted_mentions
from d3text.utils import aggregate_embeddings

from .model_types import BatchItem

logger = logging.getLogger(__name__)

# Bounds `TokenLabelReader`'s per-document cache by the bytes its entries
# hold, not their count: a `DocumentLabels` carries one int8 array per gold
# entity plus one candidate-ID set per mention, so its size scales with a
# document's entity density the same way the embeddings cache's did before
# `models.base.ByteBudgetCache` -- a count that reads as modest is the one
# that gets the run killed. Not built by reusing that class: its `set`
# hardcodes a tensor cost function and a `(str, int)` key, both pinned by its
# own tests.
_LABEL_CACHE_MAX_BYTES = 64_000_000


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
    :return: the byte cost to charge against the cache's budget.

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

    `aggregate_embeddings`'s window-by-window selection depends only on the
    mask, never on the values being merged, so `source` -- built once from an
    index tensor -- tells every field sharing this document's geometry which
    flat `window * tokens + column` cell landed at each aggregated-axis
    position; gathering through it costs one indexing op instead of a second
    walk over the windows.

    :param source: the flat index selected for each aggregated-axis token.
    :param mentions: `exact_mentions`'s own return value, computed once; None
        until the first call computes it.
    """

    source: Int64[Tensor, " token"]
    mentions: tuple[StoredMention, ...] | None = None


def _derived_bytes(derived: _Derived) -> int:
    """Real memory one document's derived products hold.

    :param derived: the products to size.
    :return: the byte cost to charge against the shared cache budget.
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
    """Bounds `TokenLabelReader`'s cache by real bytes, never evicting.

    Mirrors `models.base.ByteBudgetCache`'s accounting -- charge each entry
    its real cost via `_document_labels_bytes`, decline an entry that would
    cross the budget on the way in -- as a small parallel class rather than a
    shared one; see `_LABEL_CACHE_MAX_BYTES` for why. Not its eviction: a
    label group has one source, so there is no dearer one to keep room for.

    Also holds each document's `_Derived` products, in the same entry as its
    raw group so the two share one cost and one budget: a document whose raw
    group was declined has nothing to attach derived products to either, and
    a derived product that would push a cached document over the budget is
    itself declined -- the caller still gets the value it computed, just
    uncached, same as any other budget decline here.
    """

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self._entries: dict[str, _Entry] = {}
        self._used = 0
        # Counted at `TokenLabelReader._load`'s cache check, reported and
        # reset once per `run_epoch` pass by `log_cache_stats` -- otherwise a
        # decline that forces a repeat HDF5 read every pass is invisible
        # short of timing whole epochs and reasoning backwards.
        self.hits = 0
        self.misses = 0
        self.declines = 0

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def get(self, key: str) -> token_labels.DocumentLabels | None:
        """Look up a cached label group; caller checks `in` first for a miss."""
        return self._entries[key].labels

    def set(self, key: str, value: token_labels.DocumentLabels | None) -> bool:
        """Cache `value` under `key` unless doing so would cross the budget.

        :param key: the document ID to store it under.
        :param value: the label group, charged its real
            `_document_labels_bytes` cost.
        :return: whether `value` was actually cached.
        """
        cost = _document_labels_bytes(value)
        existing = self._entries.get(key)
        used = self._used - (0 if existing is None else existing.total_cost)
        if used + cost > self.max_bytes:
            self.declines += 1
            return False
        self._entries[key] = _Entry(labels=value, labels_cost=cost)
        self._used = used + cost
        return True

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
        if entry is not None:
            self._store_derived(entry, _Derived(source=source))
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
        if entry is not None and entry.derived is not None:
            self._store_derived(
                entry, replace(entry.derived, mentions=mentions)
            )
        return mentions

    def _store_derived(self, entry: _Entry, derived: _Derived) -> None:
        """Cache `derived` for `key` unless doing so would cross the budget."""
        cost = _derived_bytes(derived)
        used = self._used - entry.derived_cost
        if used + cost > self.max_bytes:
            return
        entry.derived = derived
        entry.derived_cost = cost
        self._used = used + cost


class TokenLabelReader:
    """One run's handle on a token-label store, space-checked once at open."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        space: token_labels.LabelSpace = token_labels.BRENDA_LABELS,
    ) -> None:
        self._store = h5py.File(path, "r")
        recorded = token_labels.read_label_space(self._store)
        if recorded != space:
            self._store.close()
            msg = (
                f"{os.fspath(path)} records the label space {recorded}, but "
                f"this model's tagger head is sized to {space}; its codes "
                "would be scored against the wrong columns — regenerate the "
                "store, or build the model over the space it records"
            )
            raise ValueError(msg)
        self.space = space
        self._label_cache = _LabelCache(_LABEL_CACHE_MAX_BYTES)

    def close(self) -> None:
        self._store.close()

    def _load(self, pubmed_id: int | str) -> token_labels.DocumentLabels | None:
        """One document's raw label group, or None if the store lacks it.

        Shared by `document_codes`, `mentioned_types`, `entity_positions` and
        `exact_mentions` through a per-instance cache, so a document already
        read this pass — including every gold entity `entity_positions` reads
        off the same document — costs one HDF5 group read rather than one per
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

        Mirrors the CPU embeddings cache's reporting in `run_epoch` -- a
        decline that forces a repeat HDF5 read every pass would otherwise
        only show up as a slower wall clock.

        :param step: which pass this covers, for the log line.
        """
        cache = self._label_cache
        total = cache.hits + cache.misses
        hit_rate = 100 * cache.hits / total if total else 0.0
        logger.info(
            "Token-label cache (%s pass): %d/%d hits (%.1f%%), %d "
            "declines, %d documents cached, %d/%d MB used",
            step,
            cache.hits,
            total,
            hit_rate,
            cache.declines,
            len(cache._entries),
            cache._used // 10**6,
            cache.max_bytes // 10**6,
        )
        cache.hits = 0
        cache.misses = 0
        cache.declines = 0

    def _aggregated_source(
        self, key: str, mask: NDArray[numpy.int64]
    ) -> Int64[Tensor, " token"]:
        """This document's aggregated-axis source index, cached per document.

        `document_codes`, `document_ambiguous`, `entity_positions` and
        `exact_mentions` each aggregate a different per-token field over this
        same `[windows, tokens]` mask; since the merge picks positions from
        the mask alone, the flat `window * tokens + column` index it selects
        for every aggregated-axis token is identical across all four, and
        needs computing only once per document.

        :param key: the document's cache key, as `_load` uses it.
        :param mask: the `[windows, tokens]` attention mask, already
            reshaped and validated against the field about to be gathered.
        :return: the flat index selected for each aggregated-axis token.
        """
        windows, tokens = mask.shape

        def compute() -> Int64[Tensor, " token"]:
            return (
                aggregate_embeddings(
                    torch.arange(windows * tokens).reshape(windows, tokens, 1),
                    torch.as_tensor(mask),
                )
                .squeeze(-1)
                .to(torch.int64)
            )

        return self._label_cache.get_or_compute_source(key, compute)

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

        windows, tokens = mask.shape
        source = self._aggregated_source(key, mask)

        def compute_mentions() -> tuple[StoredMention, ...]:
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

        return self._label_cache.get_or_compute_mentions(key, compute_mentions)

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

    `offset_mapping` is windowed exactly like the embeddings a tagger scores —
    `[windows, tokens, 2]` char bounds, one row of windows per document — so it
    is carried across the same window merge via `aggregate_embeddings` rather
    than a second aggregation arithmetic. Once on the aggregated axis, a
    span's char bounds are just its first token's start and its last token's
    end; no BRENDA mention store is consulted, which is what makes this usable
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


def store_batch_item(group: h5py.Group, document_id: int) -> BatchItem:
    """One stored document's windows, as the model methods take a batch item.

    :param group: the document's finished encodings-store group.
    :param document_id: what `get_token_embeddings` keys its caches on — a
        BRENDA document's pubmed id, or the id
        `encodings_store.external_document_id` mints for a document of an
        external corpus, which has no pubmed id.
    :return: the item, carrying the document's token ids and mask alone:
        nothing built from a store group reads gold, so it has neither a
        class nor a relation field.
    """
    return {
        "id": torch.tensor(document_id),
        "doc_id": torch.zeros(group["input_ids"].shape[0]),
        "sequence": {
            "input_ids": torch.as_tensor(group["input_ids"][:]),
            "attention_mask": torch.as_tensor(group["attention_mask"][:]),
        },
    }


def resolve_token_tagger(model: object) -> Callable[[Tensor], Tensor] | None:
    """The model's span tagger, as `predicted_spans_from_store` takes it.

    `nn.Module.__getattr__`'s fallback types `token_tagger` differently on
    each concrete model (`nn.Linear | None`, or `Tensor | Module` for one
    that never declares it at class level), and none of those is the
    `Callable[[Tensor], Tensor]` the spans are read through, so the cast
    restates what the absence of the attribute already decides: a non-None
    `token_tagger` is always callable. What a caller does about a checkpoint
    that carries none differs -- refusing to run at all, or skipping one
    block of a report -- so that stays the caller's.

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
    """`key` read as the pubmed id a BRENDA group is stored under.

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


def predicted_spans_from_store(
    store: h5py.File,
    corpus: str | None,
    texts: Mapping[str, str],
    get_token_embeddings: Callable[
        [Sequence[BatchItem]], tuple[Tensor, Tensor]
    ],
    hidden: Callable[[Tensor], Tensor],
    token_tagger: Callable[[Tensor], Tensor],
    space: token_labels.LabelSpace = token_labels.BRENDA_LABELS,
) -> list[TaggedSpan]:
    """Run a tagger over the `texts` documents `store` holds a group for.

    Takes the three calls a forward needs rather than a model object: every
    concrete model types `token_tagger` and `hidden` through
    `nn.Module.__getattr__`'s fallback (`Tensor | Module`, see this
    project's docstring conventions), which no `Callable` Protocol matches
    structurally, so the caller resolves and narrows them once instead.

    Looks each document of `texts` up under the key `corpus` says it was
    written as, forwards its windowed `input_ids`/`attention_mask` through
    `get_token_embeddings`/`hidden`/`token_tagger` the way
    `score_token_detection` does for a BRENDA document, and grounds the
    aggregated-axis argmax in `text` via `char_spans_from_predictions`. A
    document the store holds no group for, or one a precompute pass never
    finished, is skipped — the same silence
    `linking_corpora._organism_gold`/`_enzyme_gold` already use for an
    absent corpus.

    An external corpus's document is given the id
    `encodings_store.external_document_id` mints for its store key, which is
    a property of that key alone. An id counted off this call's `texts`, or
    off one store's key order, names a different document in the next call or
    the next store while the cache it keys outlives both -- and two documents
    of one token count under one id are traded for each other with nothing
    raising. A BRENDA document is keyed by its own pubmed id, so it reads the
    embedding the store already holds for it, and a key that is not a pubmed
    id is refused rather than converted.

    :param store: an open encodings store.
    :param corpus: which corpus's groups to read (`"s800"` or
        `"enzymener"`), or None for BRENDA documents, whose group key is
        the bare pubmed id.
    :param texts: document id to full text, as `load_s800`/`load_enzymener`
        return, or pubmed id to `corpus.document_text` output.
    :param get_token_embeddings: the trained model's own, e.g.
        `model.get_token_embeddings`.
    :param hidden: the trained model's own, e.g. `model.hidden`.
    :param token_tagger: the trained model's own, e.g. `model.token_tagger`
        — the caller's to confirm is not `None` before passing it.
    :param space: the label space `token_tagger`'s codes are written in.
    :return: one `TaggedSpan` per predicted mention, across every readable
        document.
    :raises ValueError: with `corpus` None, if a key of `texts` is not a
        positive pubmed id — which means these documents were not written by
        the BRENDA path of `precompute-encodings`.
    """
    spans: list[TaggedSpan] = []
    for document, text in texts.items():
        key = (
            document
            if corpus is None
            else encodings_store.external_key(corpus, document)
        )
        group = store.get(key)
        if not encodings_store.is_finished_group(group):
            continue

        document_id = (
            _pubmed_document_id(document)
            if corpus is None
            else encodings_store.external_document_id(key)
        )
        item = store_batch_item(group, document_id)
        with torch.no_grad():
            embeddings, mask = get_token_embeddings([item])
            token_logits = token_tagger(hidden(embeddings))
        length = int(mask[0].sum())
        codes = token_logits[0, :length].argmax(dim=-1).cpu().numpy()
        spans.extend(
            char_spans_from_predictions(
                token_predicted_mentions(codes),
                group["offset_mapping"][:],
                group["attention_mask"][:],
                text=text,
                document=document,
                space=space,
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
    "padded_targets",
    "predicted_spans_from_store",
    "resolve_mentions",
    "store_batch_item",
]
