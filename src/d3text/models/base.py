"""Base model class and the helpers shared by every model in this package.

`Model` provides base transformer loading, AMP and gradient checkpointing,
token embedding lookup and logit pooling; the concrete subclasses live in their
own modules and import from here. See the models page of the documentation for
the pooling modes, the loss divisors and the AMP rules.
"""

import atexit
import contextlib
import functools
import itertools
import logging
import math
import os
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from enum import StrEnum
from typing import ClassVar, NamedTuple, Self, TypeVar, assert_never, cast

import lmdb
import numpy as np
import torch
import torch.nn as nn
import transformers
from d3text import runtime, tracking
from d3text.constraints import NonNegativeReal, Positive, UnitInterval
from d3text.embeddings_store import (
    AGGREGATED,
    EmbeddingsStore,
    LayerBoundaryStore,
    ProvenanceError,
    StoreProvenance,
    boundary_name,
    sub_databases,
    unfrozen_counts,
)
from d3text.progress import batch_progress, split_documents
from d3text.runtime import select_amp_dtype
from d3text.training.update import BatchUpdate
from d3text.utils import WINDOW_LENGTH, WINDOW_STRIDE, aggregate_embeddings
from jaxtyping import Bool, Float, Int64, Integer
from sklearn.metrics import average_precision_score, f1_score
from sklearn.metrics import classification_report
from torch import Tensor
from torch.autograd.profiler import record_function
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers.masking_utils import create_bidirectional_mask

from .config import ModelConfig, TokenLossWeighting, machine_config
from .heads import PermutationBatchNorm1d
from .model_types import BatchItem

logger = logging.getLogger(__name__)

BYTES_PER_MB = 10**6
CpuCacheKey = tuple[str, int, int]


class _CacheEntry(NamedTuple):
    """A cached tensor, the bytes it is charged, and where it came from."""

    value: Tensor
    cost: int
    from_store: bool


class ByteBudgetCache:
    """A document cache bounded by the bytes its entries hold.

    :param max_bytes: ceiling on the total size of the cached tensors.

    An aggregated entry has one row per token of a whole paper. A layer-boundary
    entry has one row per token-window position, including overlap and padding.
    Either varies too much for a budget counted in entries to name a quantity
    that can be compared against free memory. The ceiling is enforced on the
    way in rather than at the call site, so a document too large for what is
    left is declined while the cache stays open for the next, smaller one.

    What an entry saves depends on where it came from, so the budget is spent
    on the dearer source first. A document the embeddings store served can be
    read again from disk; one whose only other source is a base-model forward
    cannot be had for less. Only a forward-only admission evicts, and only
    store-sourced entries: a forward-only entry is evicted for nothing, and a
    store hit that does not fit beside what is already here is declined
    rather than displacing a peer that cost exactly as much. Without the
    first rule, whoever filled the budget first held it for the life of the
    process and a promoted store hit denied the space to a forward-only
    document not yet seen; without the second, a working set larger than the
    budget would have each store hit evict the entry the next pass wants,
    since a pass reads its split once and in order.
    """

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self._entries: dict[CpuCacheKey, _CacheEntry] = {}
        self._used = 0

    @property
    def used_bytes(self) -> int:
        """Total size of the cached tensors."""
        return self._used

    def size(self) -> int:
        """Number of documents currently cached."""
        return len(self._entries)

    def get(self, key: CpuCacheKey) -> Tensor | None:
        """Look up a cached activation.

        :param key: the `cpu_cache_key` it was stored under.
        :return: the tensor, or None if it is not cached.
        """
        entry = self._entries.get(key)
        return None if entry is None else entry.value

    def _used_without(self, key: CpuCacheKey) -> int:
        """Bytes currently held, excluding `key`'s own entry if cached."""
        cached = self._entries.get(key)
        return self._used - (0 if cached is None else cached.cost)

    def _evictable_for(self, key: CpuCacheKey) -> list[CpuCacheKey]:
        """What a forward-only `key` may drop, oldest admitted first.

        Insertion order rather than recency: a pass reads every document of
        its split once, so no entry is more recently used than another by the
        time the next admission asks, and a recency order would be sampler
        noise. `key`'s own entry is never a victim — an overwrite already
        frees it, and dropping it here would refund its bytes twice.
        """
        return [
            cached
            for cached, entry in self._entries.items()
            if cached != key and entry.from_store
        ]

    def _pinned_bytes(self, key: CpuCacheKey) -> int:
        """Bytes no admission of `key` may free: the forward-only entries."""
        return sum(
            entry.cost
            for cached, entry in self._entries.items()
            if cached != key and not entry.from_store
        )

    def would_admit(
        self, key: CpuCacheKey, cost: int, from_store: bool = False
    ) -> bool:
        """Whether `set` would admit a `cost`-byte value under `key`.

        :param key: the `cpu_cache_key` the entry would be stored under.
        :param cost: the candidate value's `numel * element_size`, computed
            without materializing it (e.g. still on-device) so a caller can
            skip a host copy it already knows will be declined.
        :param from_store: whether the embeddings store would be the
            candidate's source, which is a different question: it must fit
            in what is free, while a forward-only candidate may also have
            the room store-sourced entries are holding.
        :return: True if it fits in what `set` can offer it.
        """
        floor = (
            self._used_without(key) if from_store else self._pinned_bytes(key)
        )
        return floor + cost <= self.max_bytes

    def set(
        self, key: CpuCacheKey, value: Tensor, from_store: bool = False
    ) -> None:
        """Cache `value`, evicting store-sourced entries for the room.

        Only for a forward-only `value`, and only as much as it needs.
        Nothing is evicted for one that does not fit even then: a document
        larger than the budget would otherwise empty the cache of everything
        the store served and still be declined.

        :param key: the `cpu_cache_key` to store it under.
        :param value: the activation, charged its real `numel * element_size`.
        :param from_store: whether the embeddings store served `value`, which
            both makes the entry evictable later and bars it from evicting
            now.
        """
        cost = value.numel() * value.element_size()
        if not self.would_admit(key, cost, from_store):
            return

        used = self._used_without(key)
        victims = [] if from_store else self._evictable_for(key)
        for victim in victims:
            if used + cost <= self.max_bytes:
                break
            used -= self._entries.pop(victim).cost

        self._entries[key] = _CacheEntry(value, cost, from_store)
        self._used = used + cost

    def full(self) -> bool:
        """Whether the budget is spent.

        :return: True once the cached bytes reach the ceiling. It is the call
            site's short-circuit only; `set` enforces the ceiling regardless.
        """
        return self._used >= self.max_bytes

    def clear(self) -> None:
        """Drop every entry and release the budget they held."""
        self._entries.clear()
        self._used = 0


def build_cpu_embeddings_cache(megabytes: int) -> ByteBudgetCache | None:
    """Build the process-wide embeddings cache from its configured budget.

    :param megabytes: `MachineConfig.cpu_embeddings_cache_mb`, where 1 MB is
        10**6 bytes so that the number compares against `free --si`.
    :return: the cache, or None when the budget is 0 and there is none.
    """
    return ByteBudgetCache(megabytes * BYTES_PER_MB) if megabytes else None


mconfig = machine_config()
cpu_embeddings_cache = build_cpu_embeddings_cache(
    mconfig.cpu_embeddings_cache_mb
)

# Counted at the cache lookup in `get_token_embeddings`, reported and reset
# once per `run_epoch` pass — otherwise a run's hit rate is only inferable by
# timing whole epochs and reasoning backwards from the wall clock.
cpu_cache_hits = 0
cpu_cache_misses = 0


def cpu_cache_key(
    base_model: str,
    document_id: int,
    *,
    unfrozen_top_layers: int = 0,
) -> CpuCacheKey:
    """Identify a cached activation by its model and layer boundary.

    The cache is process-wide and one process can hold more than one base model
    or frozen boundary, so a document id alone cannot identify an activation.

    :param base_model: the base model whose forward produced the activation.
    :param document_id: the document the activation belongs to — a pubmed id,
        or the id `encodings_store.external_document_id` mints for a document
        of a corpus that issues none. Not `BatchItem`'s `doc_id`, which is the
        document's position in its batch and identifies nothing outside it.
    :param unfrozen_top_layers: the number of trainable top layers after the
        cached boundary, or 0 for an aggregated frozen-trunk output.
    :return: the cache key for that model, boundary, and document.
    """
    return base_model, unfrozen_top_layers, document_id


def _store_provenance(base_model: str) -> StoreProvenance:
    """The stamp a store this run creates records: the encodings' geometry."""
    return StoreProvenance(
        base_model=base_model,
        max_length=WINDOW_LENGTH,
        stride=WINDOW_STRIDE,
        forward_dtype=str(
            select_amp_dtype("cuda" if torch.cuda.is_available() else "cpu")
        ),
    )


_Store = TypeVar("_Store", EmbeddingsStore, LayerBoundaryStore)


def _frozen(base_model: str, path: str | os.PathLike[str]) -> bool:
    """Whether `config.toml` bars the run from writing `base_model`'s store."""
    if base_model not in mconfig.frozen_embeddings_stores:
        return False
    logger.info(
        "frozen_embeddings_stores lists the base model of the store at %s, "
        "so this run does not write or create it.",
        path,
    )
    return True


def _open_existing(
    open_store: Callable[[bool], _Store],
    base_model: str,
    path: str | os.PathLike[str],
) -> _Store:
    """Open a store that exists, writable unless the machine forbids it.

    The writable open is attempted and, if the path refuses it, the store is
    read as it stands.
    """
    if _frozen(base_model, path):
        return open_store(False)
    try:
        store = open_store(True)
    except (lmdb.Error, OSError) as error:
        logger.warning(
            "Cannot open the store at %s for writing (%s); it is read as it "
            "stands and will not grow.",
            path,
            error,
        )
        return open_store(False)
    return store


@functools.cache
def embeddings_store(
    base_model: str,
) -> EmbeddingsStore | LayerBoundaryStore | None:
    """Where a frozen trunk's rows come from, opened once, or `None`.

    Lazy, because importing `d3text.models` must not touch the filesystem.
    The base model's env answers with its aggregated sub-database if it has
    one; failing that, with its boundary of fewest unfrozen layers, from
    which the model derives the rows; failing both, with a new aggregated
    sub-database. An existing store is opened writable unless
    `frozen_embeddings_stores` lists the base model or the path refuses a
    writable open, and a listed base model gets no new store. The run tops up
    a writable aggregated sub-database with the documents it embeds; a
    boundary serving it is read, and nothing is put into it. A store that
    cannot be opened, or that a different base model wrote, disables itself
    and the run recomputes the embeddings.

    :param base_model: the base model the store has to have been written by.
    :return: the open store, or None if there is none or it is unusable.
    """
    path = mconfig.embeddings_store.get(base_model)
    if not path:
        return None
    try:
        names = sub_databases(path) if os.path.exists(path) else frozenset()
        boundaries = unfrozen_counts(names)
        if AGGREGATED in names:
            store = _open_existing(
                lambda writable: EmbeddingsStore(
                    path, base_model, WINDOW_LENGTH, writable=writable
                ),
                base_model,
                path,
            )
            if store.writable:
                logger.info(
                    "Topping up the aggregated embeddings at %s: each "
                    "document it lacks goes in when the base model embeds "
                    "it, until a write is refused.",
                    path,
                )
        elif boundaries:
            # Cached there, so a run at this boundary shares the reader.
            return layer_boundary_store(base_model, boundaries[0])
        elif _frozen(base_model, path):
            return None
        else:
            store = EmbeddingsStore.create(path, _store_provenance(base_model))
            logger.info(
                "No aggregated embeddings at %s, so this run stores them: "
                "each document goes in the first time the base model embeds "
                "it, until a write is refused, and later passes read it from "
                "there.",
                path,
            )
    except (lmdb.Error, OSError) as error:
        logger.warning(
            "Cannot open the embeddings store at %s (%s); embeddings will be "
            "computed by the base model as though none were configured.",
            path,
            error,
        )
        return None
    except ProvenanceError as error:
        logger.warning(
            "%s Embeddings will be computed by the base model as though none "
            "were configured.",
            error,
        )
        return None

    store.min_free_gib = mconfig.embeddings_store_min_free_gib
    # Nothing else owns the store, so exit is the only place to report its hit
    # rate; atexit runs last-registered-first, so logging still works then.
    atexit.register(store.close)
    return store


@functools.cache
def layer_boundary_store(
    base_model: str, unfrozen_top_layers: int
) -> LayerBoundaryStore | None:
    """One boundary of the base model's env, opened once, or `None`.

    A boundary the env lacks, or an env not there yet, is created, and a run
    at that boundary puts into it the prefixes it computes, until a write is
    refused. An existing boundary is opened writable unless
    `frozen_embeddings_stores` lists the base model or the path refuses a
    writable open, and a listed base model gets no new boundary. A run at
    the boundary tops up a writable one the same way; a frozen-trunk run
    that `embeddings_store` serves from it puts nothing into it.

    :param base_model: the base model the store has to have been written by.
    :param unfrozen_top_layers: the boundary, as the number of top encoder
        layers run over the stored rows.
    :return: the open store, or None if there is none or it is unusable.
    """
    path = mconfig.embeddings_store.get(base_model)
    if not path:
        return None
    try:
        names = sub_databases(path) if os.path.exists(path) else frozenset()
        if boundary_name(unfrozen_top_layers) in names:
            store = _open_existing(
                lambda writable: LayerBoundaryStore(
                    path,
                    base_model,
                    unfrozen_top_layers,
                    WINDOW_LENGTH,
                    writable=writable,
                ),
                base_model,
                path,
            )
        elif _frozen(base_model, path):
            return None
        else:
            store = LayerBoundaryStore.create(
                path, _store_provenance(base_model), unfrozen_top_layers
            )
            logger.info(
                "No layer-boundary prefixes at %s for %d unfrozen layer(s), "
                "so this run stores them: each document's frozen prefix goes "
                "in the first time the trunk computes it, until a write is "
                "refused, and later passes replay only the top layers from "
                "there.",
                path,
                unfrozen_top_layers,
            )
    except (lmdb.Error, OSError) as error:
        logger.warning(
            "Cannot open the layer-boundary store at %s (%s); the trunk's "
            "frozen prefix will be recomputed as though none were "
            "configured.",
            path,
            error,
        )
        return None
    except ProvenanceError as error:
        logger.warning(
            "%s The trunk's frozen prefix will be recomputed as though "
            "none were configured.",
            error,
        )
        return None

    store.min_free_gib = mconfig.embeddings_store_min_free_gib
    atexit.register(store.close)
    return store


def document_token_count(item: BatchItem) -> int:
    """How many rows `aggregate_embeddings` produces for `item`.

    Measured by running the aggregation over a zero-width tensor rather than by
    reimplementing its overlap arithmetic, since a second copy of that
    arithmetic would be a hole in the check this number serves.

    :param item: one batch item, carrying its chunk geometry.
    :return: the number of aggregated rows it yields.
    """
    masks = item["sequence"]["attention_mask"]
    masks = masks.reshape(-1, masks.shape[-1])
    empty = torch.empty((masks.shape[0], masks.shape[1], 0))

    return aggregate_embeddings(empty, masks).shape[0]


def _eval_frozen_submodules(module: nn.Module) -> None:
    """Pin every subtree of `module` with no trainable parameter to eval.

    Stops at the first wholly frozen subtree, since `eval` already recurses. A
    parameter-less leaf keeps whatever mode the caller's `train(mode)` set.

    :param module: the module to walk, already in the caller's chosen mode.
    :return: None; `module` is edited in place.
    """
    params = list(module.parameters())
    if not params:
        return
    if all(not p.requires_grad for p in params):
        module.eval()
    else:
        for child in module.children():
            _eval_frozen_submodules(child)


class Step(StrEnum):
    TRAINING = "training"
    VALIDATION = "validation"


def label_columns(
    labels: Sequence[str], sentinel: str
) -> tuple[int, Int64[Tensor, " kept"]]:
    """Locate `sentinel` among `labels` and list every other column.

    The head scores one extra column the targets do not carry, so loss and
    evaluation run on the others. Locating it by name keeps those columns
    correct if it ever stops being the last one.

    :param labels: the head's labels, in column order.
    :param sentinel: the extra label, `OOS` on the class head.
    :return: the sentinel's column and the indices of every other column.
    :raises ValueError: if `sentinel` is not among `labels`.
    """
    index = labels.index(sentinel)
    return index, torch.tensor(
        [column for column in range(len(labels)) if column != index],
        dtype=torch.int64,
    )


def balanced_class_weights(
    targets: Int64[Tensor, " relation"],
    num_classes: Positive,
    weights: Float[Tensor, " relation"] | None = None,
) -> Float[Tensor, " classes"]:
    """Inverse-frequency class weights for one batch of relation targets.

    Per batch rather than precomputed because the candidate pairs are proposed
    by the current span tagger's groundings, so there is no dataset frequency
    to derive them from. A zero weighted count is replaced by 1 to stay
    finite. Such a class is absent or carries only zero-weight elements, so
    its weight is gathered at most to be multiplied by 0, which is why it
    must be finite; a label-smoothed loss is the exception, reading every
    class's weight.

    :param targets: the batch's relation targets.
    :param num_classes: width of the relation head.
    :param weights: per-element weight each target counts with; `None` counts
        every target once. A down-weighted element counts toward its class's
        frequency by the same fraction it contributes to the loss.
    :return: one weight per class.
    """
    counts = torch.bincount(targets, weights=weights, minlength=num_classes)
    total = targets.numel() if weights is None else weights.sum()
    return total / (num_classes * torch.where(counts > 0, counts, 1.0))


def _log_one_minus_p_t(
    preds: Float[Tensor, "row logits"], targets: Int64[Tensor, " row"]
) -> Float[Tensor, " row"]:
    """`log(1 - p_t)` per row, summed over the non-target classes in log space.

    `1 - softmax(...)` rounds to exactly 0 once `p_t` rounds to 1, taking the
    log with it; the logsumexp over the other classes stays finite there.
    """
    off_target = nn.functional.one_hot(targets, preds.shape[-1]).bool()
    return (
        preds.log_softmax(dim=-1)
        .masked_fill(off_target, float("-inf"))
        .logsumexp(dim=-1)
    )


def _weighted_mean(values: Tensor, log_weight: Tensor) -> Tensor:
    """`sum(w * values) / sum(w)` for `w = exp(log_weight)`, 0 if `w` is 0.

    Shifts `log_weight` by its detached max before exponentiating. A weighted
    mean is unchanged by scaling every weight, and the largest scaled weight
    is exactly 1, so the divisor is never subnormal and the backward pass
    (`exp` passes `upstream * w`) never multiplies a zero by an infinity,
    however large the focal exponent. A mass of exactly 0 (every weight 0)
    returns 0 rather than 0 / 0.

    :param values: per-element values.
    :param log_weight: per-element log-weight, `-inf` for weight 0.
    :return: the scalar weighted mean.
    """
    top = log_weight.detach().max()
    top = torch.where(torch.isfinite(top), top, torch.zeros_like(top))
    weight = (log_weight - top).exp()
    mass = weight.sum()
    return (weight * values).sum() / torch.where(
        mass > 0, mass, torch.ones_like(mass)
    )


def focal_cross_entropy(
    preds: Float[Tensor, "relation logits"],
    targets: Int64[Tensor, " relation"],
    gamma: NonNegativeReal,
    label_smoothing: UnitInterval = 0.0,
) -> Float[Tensor, ""]:
    """Cross-entropy with each element scaled by `(1 - p_t) ** gamma`.

    Normalised by the modulation mass rather than the row count: under a plain
    mean an easy pair still divides the denominator, so proposing more of them
    would shrink the loss on the rare positives. The mass is not floored, so
    the result is a true weighted mean whatever the batch size.

    :param preds: per-pair logits.
    :param targets: per-pair class targets.
    :param gamma: the focusing exponent; 0 is plain cross-entropy.
    :param label_smoothing: passed through to the per-element cross-entropy.
    :return: the scalar loss.
    """
    elementwise = nn.functional.cross_entropy(
        preds, targets, reduction="none", label_smoothing=label_smoothing
    )
    log_modulation = (
        gamma * _log_one_minus_p_t(preds, targets)
        if gamma > 0
        else torch.zeros_like(elementwise)
    )
    return _weighted_mean(elementwise, log_modulation)


def masked_token_cross_entropy(
    preds: Float[Tensor, "token logits"],
    targets: Int64[Tensor, " token"],
    ignore_index: int = -100,
    weighting: TokenLossWeighting = "unweighted",
    focal_gamma: NonNegativeReal = 2.0,
    ambiguous: Bool[Tensor, " token"] | None = None,
    downweight: UnitInterval = 0.0,
) -> Float[Tensor, ""]:
    """Cross-entropy over the tokens `targets` does not mask out.

    Every scheme is a weighted mean divided by the weight mass, not the token
    count; an all-masked batch returns a differentiable zero, not a NaN.

    :param preds: per-token logits.
    :param targets: per-token targets, masked with `ignore_index`.
    :param ignore_index: the target value marking a token the loss must skip.
    :param weighting: `unweighted`, `balanced` (per-batch inverse frequency
        over the kept tokens) or `focal`.
    :param focal_gamma: the focusing exponent, read only under `focal`.
    :param ambiguous: kept tokens whose target is real but unverified (a
        comma-joined surface-form collision); `None` gives every kept token
        full weight.
    :param downweight: the weight an ambiguous token keeps; `0.0` excludes it
        from both the numerator and the divisor, `1.0` cancels the
        down-weight entirely.
    :return: the scalar loss.
    """
    kept = targets != ignore_index
    if not bool(kept.any()):
        return preds.sum() * 0.0

    kept_preds = preds[kept]
    kept_targets = targets[kept]

    elementwise = nn.functional.cross_entropy(
        kept_preds, kept_targets, reduction="none"
    )
    weight = torch.ones_like(elementwise)
    if ambiguous is not None:
        weight[ambiguous[kept]] = downweight

    if weighting == "balanced":
        class_weight = balanced_class_weights(
            kept_targets, preds.shape[-1], weights=weight
        )
        weight = weight * class_weight[kept_targets]

    log_weight = weight.log()
    if weighting == "focal" and focal_gamma > 0:
        log_weight = log_weight + focal_gamma * _log_one_minus_p_t(
            kept_preds, kept_targets
        )
    return _weighted_mean(elementwise, log_weight)


def masked_bce_with_logits(
    logits: Float[Tensor, "document class"],
    targets: Float[Tensor, "document class"],
    abstain: Bool[Tensor, "document class"] | None = None,
    pos_weight: Tensor | None = None,
    downweight: UnitInterval = 0.0,
) -> Float[Tensor, ""]:
    """BCE-with-logits, weighted-mean over the `(document, class)` pairs.

    The divisor is the weight sum, not the pair count, for the reason
    `masked_token_cross_entropy` divides by the kept count.

    :param logits: per-document class logits.
    :param targets: per-document class targets.
    :param abstain: negative targets this run has decided not to fully enforce;
        None reduces to a plain `BCEWithLogitsLoss(reduction="mean")`.
    :param pos_weight: passed through to the per-element loss.
    :param downweight: the weight an abstained pair keeps; 0.0 excludes it from
        both the numerator and the divisor.
    :return: the scalar loss.
    """
    elementwise = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none"
    )
    if abstain is None:
        return elementwise.mean()

    weight = torch.ones_like(elementwise)
    weight[abstain] = downweight
    kept = weight > 0
    if not bool(kept.any()):
        return elementwise.sum() * 0.0
    return (elementwise * weight).sum() / weight.sum()


def load_base_model(base_model: str) -> transformers.PreTrainedModel:
    """Load a frozen transformer base.

    Tolerates legacy configs that lack a `model_type` key by falling back to an
    explicit BERT config, since `AutoConfig` reads that key to choose the
    architecture and old-format repos omit it.

    :param base_model: the checkpoint name to load.
    :return: the loaded transformer.
    """
    try:
        cfg = transformers.AutoConfig.from_pretrained(base_model)
    except ValueError:
        cfg = transformers.BertConfig.from_pretrained(base_model)
    return transformers.AutoModel.from_pretrained(base_model, config=cfg)


# Elements per slice when reducing a [document, token, logits] tensor; a
# budget in elements rather than tokens keeps the slice's size independent of
# the batch's shape.
_POOL_CHUNK_ELEMENTS = 14_000_000


def pool_chunk_tokens(documents: int, width: int) -> int:
    """Tokens per slice for a `[documents, token, width]` reduction.

    Narrower slices for a wider batch, floored at one token so a batch wide
    enough to exceed the budget on a single token still advances.

    :param documents: rows in the batch.
    :param width: the reduction's feature width.
    :return: how many tokens one slice may cover.
    """
    return max(1, _POOL_CHUNK_ELEMENTS // max(1, documents * width))


class _ChunkedLogSumExp(torch.autograd.Function):
    """`logsumexp` over the token dimension, in float32, one slice at a time.

    Materialising the float32 copy in one piece costs about half the peak of a
    training step. Only the summation order differs, and that difference does
    not survive the cast back to bfloat16.
    """

    @staticmethod
    def forward(
        ctx, logits: Float[Tensor, "document token logits"], chunk: int
    ) -> Float[Tensor, "document logits"]:
        documents, tokens, width = logits.shape
        peak = logits.new_full(
            (documents, width), -float("inf"), dtype=torch.float32
        )
        for start in range(0, tokens, chunk):
            peak = torch.maximum(
                peak, logits[:, start : start + chunk].float().amax(dim=1)
            )

        # A column that is entirely -inf would make `x - peak` a NaN; shifting
        # it by zero instead lets it underflow to the -inf torch returns.
        shift = peak.masked_fill(~peak.isfinite(), 0.0)

        total = torch.zeros_like(peak)
        for start in range(0, tokens, chunk):
            total += (
                (logits[:, start : start + chunk].float() - shift.unsqueeze(1))
                .exp()
                .sum(dim=1)
            )

        pooled = shift + total.log()
        ctx.save_for_backward(logits, pooled)
        ctx.chunk = chunk
        return pooled

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(  # type: ignore[override]
        ctx, grad_pooled: Float[Tensor, "document logits"]
    ) -> tuple[Float[Tensor, "document token logits"], None]:
        logits, pooled = ctx.saved_tensors
        grad = torch.empty_like(logits)
        upstream = grad_pooled.unsqueeze(1)
        out = pooled.unsqueeze(1)
        for start in range(0, logits.shape[1], ctx.chunk):
            stop = start + ctx.chunk
            grad[:, start:stop] = (
                upstream * (logits[:, start:stop].float() - out).exp()
            ).to(logits.dtype)
        return grad, None


def token_counts(
    mask: Bool[Tensor, "document token"],
) -> Float[Tensor, " document"]:
    """Real tokens per document, floored at one.

    The floor keeps an all-padding document finite without touching any real
    document's count.

    :param mask: the batch's attention mask.
    :return: each document's token count.
    """
    return mask.sum(dim=1).clamp(min=1).to(torch.float32)


class _ChunkedMean(torch.autograd.Function):
    """`mean` over the token dimension, in float32, one slice at a time.

    Same bargain as `_ChunkedLogSumExp`, and simpler: a mean spreads its
    gradient evenly, so backward reads none of the input. With a mask, padded
    positions are kept out of both the sum and the divisor.
    """

    @staticmethod
    def forward(
        ctx,
        logits: Float[Tensor, "document token logits"],
        chunk: int,
        mask: Bool[Tensor, "document token"] | None,
    ) -> Float[Tensor, "document logits"]:
        documents, tokens, width = logits.shape
        total = logits.new_zeros((documents, width), dtype=torch.float32)
        for start in range(0, tokens, chunk):
            piece = logits[:, start : start + chunk].float()
            if mask is not None:
                piece = piece * mask[:, start : start + chunk].unsqueeze(-1)
            total += piece.sum(dim=1)
        ctx.shape = logits.shape
        ctx.dtype = logits.dtype
        # `save_for_backward`, not `ctx`: it registers the mask's version
        # counter, so an in-place edit before backward raises rather than
        # scattering the gradient over the wrong tokens.
        ctx.save_for_backward(mask)
        if mask is None:
            ctx.counts = tokens
            return total / tokens
        counts = token_counts(mask)
        ctx.counts = counts
        return total / counts.unsqueeze(1)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(  # type: ignore[override]
        ctx, grad_pooled: Float[Tensor, "document logits"]
    ) -> tuple[Float[Tensor, "document token logits"], None, None]:
        (mask,) = ctx.saved_tensors
        if mask is None:
            grad = (grad_pooled / ctx.counts).unsqueeze(1).expand(ctx.shape)
        else:
            grad = (grad_pooled / ctx.counts.unsqueeze(1)).unsqueeze(
                1
            ) * mask.unsqueeze(-1)
        return grad.to(ctx.dtype), None, None


def reject_empty_token_dim(logits: Float[Tensor, "..."], dim: int = 1) -> None:
    """Refuse to pool a document that has no tokens.

    The four poolings disagree completely on an empty reduction, and none of
    them can answer what a document with no text predicts.

    :param logits: the tensor about to be pooled.
    :param dim: the token dimension.
    :raises ValueError: if that dimension is empty.
    """
    if logits.shape[dim] == 0:
        msg = (
            f"cannot pool logits of shape {tuple(logits.shape)}: dimension "
            f"{dim} holds no tokens, so the document has no text to score"
        )
        raise ValueError(msg)


def pool_token_dim(
    logits: Float[Tensor, "document token logits"],
    pooling: str,
    mask: Bool[Tensor, "document token"] | None = None,
) -> Float[Tensor, "document logits"]:
    """Pool the token dimension without a float32 copy of the whole tensor.

    Every mode routes through here so the pooled values cannot depend on which
    path ran.

    :param logits: per-token logits, padding already filled with a large
        negative value.
    :param pooling: one of `logmeanexp`, `logsumexp`, `max`, `mean`.
    :param mask: the batch's attention mask; without it `logmeanexp` and `mean`
        normalise by the padded length, so a document's pooled logits depend on
        how long its batch companions were.
    :return: one logit vector per document.
    """
    reject_empty_token_dim(logits)
    documents, tokens, width = logits.shape
    if pooling == "max":
        # Exact and free: widening to float32 is injective, so the maximum of
        # the widened values is the widening of the maximum. No copy needed.
        return torch.amax(logits, dim=1)

    chunk = pool_chunk_tokens(documents, width)
    if pooling == "mean":
        return _ChunkedMean.apply(logits, chunk, mask).to(logits.dtype)

    pooled = _ChunkedLogSumExp.apply(logits, chunk)
    if pooling == "logmeanexp":
        if mask is None:
            pooled = pooled - math.log(tokens)
        else:
            pooled = pooled - token_counts(mask).log().unsqueeze(1)
    return pooled.to(logits.dtype)


class _TrunkTop(nn.Module):
    """The one call site both trunk paths compile through.

    Registers nothing, so `state_dict()` gains no second key for the layers
    `owner.base_model` already owns; `owner` sits in a list so
    `nn.Module.__setattr__` never registers it as a submodule.
    """

    def __init__(self, owner: "Model") -> None:
        super().__init__()
        self._owner_ref = [owner]

    def forward(
        self,
        hidden_states: Float[Tensor, "window token embedding"],
        attention_mask: Integer[Tensor, "window token"],
    ) -> Float[Tensor, "window token embedding"]:
        """Run the trainable top layers eagerly; `.compile()` traces this.

        :param hidden_states: one row of hidden states per window, at the
            frozen/trainable boundary, always at `owner.amp_dtype` — the one
            dtype both trunk paths cast to before calling here.
        :param attention_mask: the matching per-window padding mask, always
            a real tensor — never `None` — so dynamo never guards on that.
        :return: the top layers' output, the same shape as `hidden_states`.
        """
        return self._owner_ref[0]._replay_top_layers_eager(
            hidden_states, attention_mask
        )


def _run_hidden_layer(
    layer: nn.Sequential,
    x: Float[Tensor, "document token features"],
    mask: Bool[Tensor, "document token"],
) -> Float[Tensor, "document token features"]:
    """Run one `build_layers` block, passing `mask` only to its norm.

    `layer` is `Linear, GELU, dropout`, optionally followed by a norm
    submodule; every submodule but `PermutationBatchNorm1d` takes the
    single-tensor `nn.Module` call, so the mask is threaded through only
    where the batch-norm route needs it to exclude padding from its
    statistics.

    :param layer: one block from `Model.hidden_layers`.
    :param x: that block's input.
    :param mask: which positions of `x` carry a real token.
    :return: the block's output, the same shape as `x`.
    """
    for module in layer:
        x = (
            module(x, mask)
            if isinstance(module, PermutationBatchNorm1d)
            else module(x)
        )
    return x


@contextlib.contextmanager
def _throwaway_buffers(layer: nn.Module) -> Iterator[None]:
    """Point `layer`'s buffers at copies, discarded on exit.

    Swapped rather than frozen: toggling `track_running_stats` instead
    changes which tensors batch norm saves for backward, and the checkpoint
    rejects a recompute that saves a different set.
    """
    originals = [
        (module, name, buffer)
        for module in layer.modules()
        for name, buffer in module.named_buffers(recurse=False)
    ]
    for module, name, buffer in originals:
        setattr(module, name, buffer.clone())
    try:
        yield
    finally:
        for module, name, buffer in originals:
            setattr(module, name, buffer)


def _checkpoint_contexts(
    layer: nn.Module,
) -> tuple[
    contextlib.AbstractContextManager[None],
    contextlib.AbstractContextManager[None],
]:
    """`torch.utils.checkpoint`'s `context_fn` for one hidden block.

    The recompute re-runs a forward whose running-stat update the step
    already made; repeating it would leave eval-time statistics depending on
    whether gradient checkpointing was on.
    """
    return contextlib.nullcontext(), _throwaway_buffers(layer)


class Model(torch.nn.Module):
    """Base class implementing the machinery every model shares.

    Base transformer loading, AMP and gradient checkpointing, token embedding
    lookup, logit pooling, and one epoch's forward and loss accumulation
    (`run_epoch`). The epoch *schedule* around it — optimizer, LR scheduler,
    early stopping, the best-epoch snapshot — belongs to
    `d3text.training.trainer.Trainer`.
    """

    # Assigned in subclass __init__ / registered as buffers; annotated here so
    # nn.Module.__getattr__ doesn't collapse them to `Tensor | Module`.
    hidden_layers: nn.ModuleList
    _checkpoint_hidden: bool
    base_model: transformers.PreTrainedModel
    classes: list[str]
    class_columns: Tensor
    # Set only by a model with a span tagger; declared here so a caller holding
    # a `ConfigurableModel` union can assign it from a checkpoint's vocabulary.
    training_entity_ids: frozenset[str] | None
    # Built by `freeze_base_model` when `config.unfrozen_top_layers` is set
    # and `base_model` has an `encoder.layer` stack; stays None for a frozen
    # trunk or a base model `compile_trunk` has nothing to compile for.
    _trunk_top: _TrunkTop | None

    # Parked by `prefetch_layer_boundary_reads` before each yield and cleared
    # by its `finally` (or consumed first), so an unwrapped call sees `None`.
    _parked_layer_boundary_reads: (
        tuple[Sequence[BatchItem], list[Future[Tensor | None] | None]] | None
    ) = None

    # The one thread every layer-boundary `store.get` runs on, never replaced:
    # a second pool would race `LayerBoundaryStore`'s unlocked counters.
    _layer_boundary_pool: ThreadPoolExecutor | None = None

    # Time blocked on layer-boundary reads this pass, and over how many
    # batches; per pass because a running total of waiting means nothing.
    _layer_boundary_wait_seconds: float = 0.0
    _layer_boundary_wait_batches: int = 0

    # Read when `config.selection_metrics` is empty. Empty here, so a class
    # that sets neither fails at the first validation, not falls back to loss.
    default_selection_metrics: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        config: ModelConfig | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__()

        # A bare `ModelConfig()` says `ETEBrendaModel` whatever the subclass,
        # and would trip that class's label-store requirement.
        self.config = (
            config
            if config is not None
            else ModelConfig(model_class=type(self).__name__)
        )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.amp_dtype = select_amp_dtype(self.device)

        self.ramp_epochs: int = self.config.ramp_epochs
        self.entity_logits_pooling = self.config.entity_logits_pooling
        self._trunk_top = None

    @staticmethod
    def _mask_padding(
        class_logits: Float[Tensor, "document token classes"],
        attention_mask: Bool[Tensor, "document token"],
    ) -> None:
        """Fill padded positions in place with the dtype's lowest finite value.

        Taken from the logits' own dtype because no fixed sentinel fits every
        autocast dtype: fp16 tops out at 65504, so `masked_fill_` refuses
        -1e9 outright there. Finite rather than -inf because the masked mean
        multiplies the fill by zero, and `-inf * 0` is NaN.

        :param class_logits: per-token logits, edited in place.
        :param attention_mask: which positions carry a real token.
        """
        class_logits.masked_fill_(
            ~attention_mask.unsqueeze(-1), torch.finfo(class_logits.dtype).min
        )

    def freeze_base_model(self) -> None:
        """Freeze `base_model`'s parameters and put it in eval mode.

        Call once the subclass has built it. `config.unfrozen_top_layers`
        leaves that many top encoder layers trainable; frozen `nn.Linear`
        weights are then stored in `amp_dtype`. Warns once when a wholly
        frozen trunk has no usable store to read from.

        :raises NotImplementedError: `unfrozen_top_layers` is set and this
            base model exposes no `encoder.layer` stack to unfreeze from.
        :raises ValueError: `unfrozen_top_layers` exceeds the number of
            encoder layers the base model has.
        """
        for param in self.base_model.parameters():
            param.requires_grad = False

        unfrozen = self.config.unfrozen_top_layers
        if unfrozen:
            try:
                encoder_module = self.base_model.get_submodule("encoder.layer")
            except AttributeError as exc:
                raise NotImplementedError(
                    f"{type(self.base_model).__name__} exposes no "
                    "`encoder.layer` stack, so `unfrozen_top_layers` does "
                    "not know which modules to unfreeze for it"
                ) from exc
            if not isinstance(encoder_module, nn.ModuleList):
                raise NotImplementedError(
                    f"{type(self.base_model).__name__}'s encoder.layer is "
                    f"a {type(encoder_module).__name__}, not the "
                    "nn.ModuleList `unfrozen_top_layers` expects"
                )
            encoder_layers = encoder_module
            if unfrozen > len(encoder_layers):
                raise ValueError(
                    f"unfrozen_top_layers={unfrozen} exceeds "
                    f"{type(self.base_model).__name__}'s "
                    f"{len(encoder_layers)} encoder layers"
                )
            for layer in encoder_layers[len(encoder_layers) - unfrozen :]:
                for param in layer.parameters():
                    param.requires_grad = True
            # Built even uncompiled: `_replay_top_layers` asserts it exists.
            self._trunk_top = _TrunkTop(self)
        elif embeddings_store(self.config.base_model) is None:
            # Here, once per model, not per batch at the lookup; nothing else
            # says what going without a store costs.
            logger.warning(
                "The trunk is frozen (unfrozen_top_layers=0) and no usable "
                "embeddings store is configured, so %s is re-run over every "
                "document the CPU embeddings cache (%d MB) cannot hold, on "
                "every pass over the data, for output that cannot change. "
                "`precompute-embeddings` writes a store; "
                "`embeddings_store` in config.toml points a run at one.",
                self.config.base_model,
                mconfig.cpu_embeddings_cache_mb,
            )

        for module in self.base_model.modules():
            if isinstance(module, nn.Linear) and not any(
                param.requires_grad
                for param in module.parameters(recurse=False)
            ):
                module.to(self.amp_dtype)

        self.base_model.eval()

    def compile_trunk(self) -> bool:
        """Compile the trainable top encoder layers, not the whole model.

        Compiles `_trunk_top` in place; a no-op for a wholly frozen trunk,
        which builds no wrapper.

        :return: whether a graph is installed for the trunk wrapper.
        """
        if self._trunk_top is None:
            return False
        return runtime.compile_model(self._trunk_top)

    def trunk_is_compiled(self) -> bool:
        """Whether the trunk wrapper's own `__call__` dispatches to a graph.

        Reads live state the way `runtime.is_compiled` does, and for the
        same reason: a backend failure at any later forward clears the graph,
        so this is what a finished run's `compiled` tag should be read from,
        not `compile_trunk`'s return value.

        :return: whether a graph is installed for the trunk wrapper.
        """
        if self._trunk_top is None:
            return False
        return runtime.is_compiled(self._trunk_top)

    def train(self, mode: bool = True) -> Self:
        """Set training mode on every submodule but `base_model`'s frozen part.

        `nn.Module.train` recurses and would undo `freeze_base_model`. Read
        off `_modules`, since a composing model pins a composed model's base
        through that model's own `train`.

        :param mode: whether the trainable parts are in training mode.
        :return: this model.
        """
        super().train(mode)
        base_model = self._modules.get("base_model")
        if base_model is not None:
            if self.config.unfrozen_top_layers:
                base_model.train(mode)
                if mode:
                    _eval_frozen_submodules(base_model)
            else:
                base_model.eval()
        return self

    def _pool_logits(
        self,
        logits: Float[Tensor, "..."],
        dim: int = 1,
        mask: Bool[Tensor, "document token"] | None = None,
    ) -> Float[Tensor, "..."]:
        """Pool per-token logits to a document vector along `dim`.

        Selected by `ModelConfig.entity_logits_pooling` and computed in
        float32, then cast back. The `[document, token, logits]` case — every
        call site in the models — goes through `pool_token_dim` a slice at a
        time; the general path serves any other shape or `dim`, whose rows
        carry no padding.

        :param logits: the logits to pool.
        :param dim: the token dimension.
        :param mask: the batch's attention mask, so the normalisers stay
            per-document.
        :return: the pooled logits.
        """
        pooling = self.entity_logits_pooling
        reject_empty_token_dim(logits, dim)
        if logits.ndim == 3 and dim == 1:
            if pooling not in ("logsumexp", "logmeanexp", "max", "mean"):
                raise ValueError(f"Unknown pooling: {pooling}")
            return pool_token_dim(logits, pooling, mask)
        if mask is not None:
            raise ValueError(
                "mask is only supported for [document, token, logits] "
                "pooling along dim=1"
            )

        x = logits.float()
        if pooling == "logsumexp":
            pooled = torch.logsumexp(x, dim=dim)
        elif pooling == "logmeanexp":
            pooled = torch.logsumexp(x, dim=dim) - math.log(x.shape[dim])
        elif pooling == "max":
            pooled = torch.amax(x, dim=dim)
        elif pooling == "mean":
            pooled = torch.mean(x, dim=dim)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
        return pooled.to(logits.dtype)

    def _pool_logits_segments(
        self,
        logits: Float[Tensor, "row logits"],
        segment: Int64[Tensor, " row"],
        num_segments: int,
        counts: Int64[Tensor, " segment"],
    ) -> Float[Tensor, "segment logits"]:
        """Pool rows into segments: one output vector per segment id.

        The segmented counterpart of `_pool_logits(rows, dim=0)` in a fixed
        number of kernels instead of one launch per segment. Every segment must
        own at least one row.

        :param logits: the rows to pool.
        :param segment: each row's segment id.
        :param num_segments: how many segments there are.
        :param counts: rows per segment, which is `logmeanexp`'s divisor.
        :return: one pooled vector per segment.
        """
        x = logits.float()
        index = segment.unsqueeze(-1).expand_as(x)
        zeros = x.new_zeros((num_segments, x.shape[-1]))
        pooling = self.entity_logits_pooling
        if pooling in ("max", "mean"):
            pooled = zeros.scatter_reduce(
                0,
                index,
                x,
                reduce="amax" if pooling == "max" else "mean",
                include_self=False,
            )
        elif pooling in ("logsumexp", "logmeanexp"):
            # Per-segment max shift, detached since it cancels analytically
            # (keeps backward off the amax). An all -inf segment shifts by zero,
            # else `x - peak` is NaN; it underflows to -inf like the plain op.
            peak = zeros.scatter_reduce(
                0, index, x, reduce="amax", include_self=False
            ).detach()
            shift = peak.masked_fill(~peak.isfinite(), 0.0)
            summed = zeros.scatter_add(0, index, (x - shift[segment]).exp())
            pooled = shift + summed.log()
            if pooling == "logmeanexp":
                pooled = pooled - counts.float().log().unsqueeze(-1)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
        return pooled.to(logits.dtype)

    def register_class_columns(self) -> None:
        """Find the OOS column and remember the others.

        Call once `self.classes` is set. Non-persistent, since it is derived
        from them and an older checkpoint would otherwise be missing the key.
        """
        self.oos_index, class_columns = label_columns(self.classes, "OOS")
        self.register_buffer("class_columns", class_columns, persistent=False)

    def drop_oos(
        self, class_logits: Float[Tensor, "... class"]
    ) -> Float[Tensor, "... class"]:
        """Class logits without the OOS column, to the width of the targets.

        :param class_logits: the head's full-width logits.
        :return: the columns the targets carry.
        """
        return class_logits.index_select(-1, self.class_columns)

    @property
    def known_classes(self) -> list[str]:
        """Class names in column order, minus OOS.

        :return: the columns `drop_oos` keeps, and so the labels the losses and
            the reports are computed over.
        """
        return [self.classes[column] for column in self.class_columns.tolist()]

    def epoch_loss_weights(self, epoch: int) -> dict[str, float]:
        """The multiplier applied to each named loss this epoch, if any.

        Keys match `run_epoch`'s losses, so a logged weight sits beside the
        loss it scaled. Only the model that ramps an objective overrides this.

        :param epoch: the epoch about to run.
        :return: objective name -> its multiplier, omitting the unscaled ones.
        """
        return {}

    def autocast_context(self, enabled=True):
        """An autocast context in this model's AMP dtype.

        :param enabled: whether autocasting is on.
        :return: the context manager to run the forward under.
        """
        return torch.autocast(
            device_type=self.device,
            dtype=self.amp_dtype,
            enabled=enabled,
        )

    def build_layers(self, embedding_size: int) -> None:
        in_features = embedding_size
        self._checkpoint_hidden = False

        if self.config.common_hidden_block:
            # Common layers setup
            self.hidden_layers = nn.ModuleList()
            self.dropout = (
                nn.Dropout(self.config.dropout)
                if self.config.dropout
                else nn.Identity()
            )

            for layer_size in self.config.hidden_layers:
                layer = nn.Sequential(
                    nn.Linear(in_features, layer_size), nn.GELU(), self.dropout
                )

                match self.config.normalization:
                    case "layer":
                        layer.append(nn.LayerNorm(layer_size))
                    case "batch":
                        layer.append(PermutationBatchNorm1d(layer_size))
                    case "none":
                        pass
                    case unreachable:
                        assert_never(unreachable)

                self.hidden_layers.append(layer)
                in_features = layer_size
        else:
            self.hidden_layers = nn.ModuleList()

        self.hidden_block_output_size = in_features

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for all compatible modules.

        Not the base model: checkpointing every layer (frozen ones included)
        forced an ~11x slower epoch for no gradient benefit on the frozen
        ones. Removed.
        """
        self._checkpoint_hidden = any(
            param.requires_grad for param in self.hidden_layers.parameters()
        )

    def hidden(
        self,
        x: Float[Tensor, "document token features"],
        mask: Bool[Tensor, "document token"],
    ) -> Float[Tensor, "document token features"]:
        """Run the common hidden block; identity when it has no layers.

        A method, not a closure over `self` assigned per instance: beartype
        memoises every function it decorates, which would keep the model
        alive for the life of the process.

        :param x: token embeddings.
        :param mask: which tokens are real.
        :return: the transformed token features.
        """
        for layer in self.hidden_layers:
            if self._checkpoint_hidden:
                x = torch.utils.checkpoint.checkpoint(
                    _run_hidden_layer,
                    layer,
                    x,
                    mask,
                    use_reentrant=False,
                    context_fn=functools.partial(_checkpoint_contexts, layer),
                )
            else:
                x = _run_hidden_layer(cast(nn.Sequential, layer), x, mask)
        return x

    def compute_losses(
        self,
        batch: Sequence[BatchItem],
        epoch: int,
    ) -> dict[str, Tensor]:
        """One batch's losses, keyed by objective name; per subclass.

        A key present in one batch of an epoch must be present in every batch
        of it, since `run_epoch` accumulates under these names.

        :param batch: the batch to run.
        :param epoch: the epoch number, read only by a model that ramps.
        :return: one loss per objective.
        """
        raise NotImplementedError

    def evaluate_model(
        self,
        data: DataLoader,
        tau_cls: UnitInterval = 0.5,
        prefix: str = "test",
        log_reports: bool = True,
        step: int | None = None,
    ) -> dict[str, float]:
        """Score `data`; see the concrete override for the keys it reports.

        Validation and the final test split both score through here, so the
        two can never disagree about what a key means.

        :param data: the split to score.
        :param tau_cls: threshold binarizing the class logits.
        :param prefix: the tracking-key prefix the scores are reported
            under, so a validation pass never overwrites `test/*`.
        :param log_reports: whether to log the per-class text reports to the
            console and as run artifacts, in addition to the metrics. Off for
            a per-epoch validation pass, which would otherwise print and
            write one such report every epoch.
        :param step: the tracking step the metrics are logged under; `None`
            for a one-off evaluation, the epoch number for a validation pass.
        :return: the scores, keyed under `prefix`.
        :raises NotImplementedError: no override exists for this model class.
        """
        raise NotImplementedError

    def run_epoch(
        self,
        data: DataLoader,
        epoch: int,
        update: BatchUpdate,
    ) -> tuple[dict[str, float], int]:
        """Train one epoch: every batch through `compute_losses` and `update`.

        Shared by every subclass — only `compute_losses` differs between them.
        Training only: validation scores through `evaluate_model`.
        `prefetch_layer_boundary_reads` wraps the loop so a configured
        layer-boundary store's reads for one batch overlap the previous
        batch's replay; see its docstring.

        :param data: the split to train on.
        :param epoch: the epoch number.
        :param update: `Trainer`'s batch update.
        :return: the summed losses by objective, and how many batches ran.
        """
        epoch_loss_sums: dict[str, Tensor] = {}
        n_batches = 0

        for batch in self.prefetch_layer_boundary_reads(batch_progress(data)):
            update.zero_grad()

            losses = self.compute_losses(batch, epoch)
            n_batches += 1

            update(*losses.values())

            for key, value in losses.items():
                detached = value.detach()
                if key in epoch_loss_sums:
                    epoch_loss_sums[key] = epoch_loss_sums[key] + detached
                else:
                    epoch_loss_sums[key] = detached.clone()

            del losses

        self.log_pass_stats(Step.TRAINING)
        epoch_losses = {
            key: value.item() for key, value in epoch_loss_sums.items()
        }
        return epoch_losses, n_batches

    def log_pass_stats(self, step: Step) -> None:
        """Log the embedding caches' and layer-boundary store's counters.

        The CPU cache's counters and the layer-boundary wait are reset, so
        the next pass reports only its own.

        :param step: which pass it was, for the log line.
        """
        if cpu_embeddings_cache is not None:
            global cpu_cache_hits, cpu_cache_misses
            total = cpu_cache_hits + cpu_cache_misses
            hit_rate = 100 * cpu_cache_hits / total if total else 0.0
            logger.info(
                "CPU embeddings cache (%s pass): %d/%d hits (%.1f%%), "
                "%d documents cached, %d/%d MB used",
                step,
                cpu_cache_hits,
                total,
                hit_rate,
                cpu_embeddings_cache.size(),
                cpu_embeddings_cache.used_bytes // BYTES_PER_MB,
                cpu_embeddings_cache.max_bytes // BYTES_PER_MB,
            )
            cpu_cache_hits = 0
            cpu_cache_misses = 0

        token_labels_reader = getattr(self, "_token_labels", None)
        if token_labels_reader is not None:
            token_labels_reader.log_cache_stats(step)
            getattr(self, "log_missing_token_labels")(step)

        if self.config.unfrozen_top_layers:
            layer_store = self._layer_boundary_store()
            if layer_store is not None:
                # Coverage is cumulative like the embeddings store's below;
                # the wait is this pass's own, so it is reset once logged.
                logger.info(
                    "Layer-boundary store (cumulative, through the %s "
                    "pass): %s",
                    step,
                    layer_store.summary(),
                )
                logger.info(
                    "Waited %.3f s for prefetched layer-boundary reads "
                    "over %d batch(es) this %s pass",
                    self._layer_boundary_wait_seconds,
                    self._layer_boundary_wait_batches,
                    step,
                )
                self._layer_boundary_wait_seconds = 0.0
                self._layer_boundary_wait_batches = 0
        else:
            store = embeddings_store(self.config.base_model)
            if store is not None:
                # Cumulative, not per-pass: the counters are what `close`
                # reports at process exit, and resetting them here would
                # leave that total covering only the last pass.
                logger.info(
                    "Embeddings store (cumulative, through the %s pass): %s",
                    step,
                    store.summary(),
                )

    def batch_input_tensors(
        self,
        batch: Sequence[BatchItem],
    ) -> dict[str, Integer[Tensor, "sequence token"]]:
        """Concatenate each document's chunk sequences into one tensor per key.

        Every dimension but the last is flattened away, because the same item
        arrives 2-D from `BrendaDataset` and 3-D from `default_collate`, where
        the leading 1 is an artefact of batching a one-element list rather than
        a document axis.

        :param batch: the batch's items.
        :return: one `[sum(n_chunks), token]` tensor per encoding key.
        """
        return {
            key: torch.concat(
                tuple(
                    doc["sequence"][key].reshape(
                        -1, doc["sequence"][key].shape[-1]
                    )
                    for doc in batch
                ),
                dim=0,
            )
            for key in ("input_ids", "attention_mask")
        }

    @record_function("get_token_embeddings")
    def get_token_embeddings(
        self, batch: Sequence[BatchItem]
    ) -> tuple[
        Float[Tensor, "batch max_doc_len embedding"],
        Bool[Tensor, "batch max_doc_len"],
    ]:
        """Token embeddings for a batch, from the cheapest available source.

        The in-process cache, then the precomputed store, then the frozen base
        model. With `config.unfrozen_top_layers` set, the in-process cache and
        layer-boundary store hold the frozen prefix; hits replay only the
        trainable top layers.

        :param batch: the batch's items.
        :return: the padded embeddings and their mask.
        """
        trunk_trainable = bool(self.config.unfrozen_top_layers)

        if trunk_trainable:
            inputs, missing = self._resolve_layer_boundary_cached(batch)
        else:
            inputs, missing = self._resolve_cached(batch, trunk_trainable)
        if missing:
            self._embed_missing(missing, inputs, trunk_trainable)

        return self._pad_and_mask(inputs)

    def _replay_top_layers(
        self,
        prefix: Float[Tensor, "window token embedding"],
        attention_mask: Integer[Tensor, "window token"],
        attention_mask_cpu: Integer[Tensor, "window token"],
    ) -> Float[Tensor, "window token embedding"]:
        """Run hidden states at the frozen/trainable boundary through the top.

        The one call both trunk paths share. The compiled branch is not given
        `attention_mask_cpu`: a guarded input it does not need would compile
        a second graph the first time a batch's padding differs.

        :param prefix: one row of hidden states per window, at
            `self.amp_dtype`, whether a store's or a fresh forward's.
        :param attention_mask: the matching per-window padding mask, never
            `None`.
        :param attention_mask_cpu: the same mask, still on the host.
        :return: the top layers' output, the same shape as `prefix`.
        """
        assert self._trunk_top is not None, (
            "called with unfrozen_top_layers unset or no encoder.layer "
            "stack; freeze_base_model builds no wrapper for either"
        )
        if self.trunk_is_compiled():
            return self._trunk_top(prefix, attention_mask)
        return self._replay_top_layers_eager(
            prefix, attention_mask, attention_mask_cpu
        )

    def _replay_top_layers_eager(
        self,
        prefix: Float[Tensor, "window token embedding"],
        attention_mask: Integer[Tensor, "window token"],
        attention_mask_cpu: Integer[Tensor, "window token"] | None = None,
        layers: int | None = None,
    ) -> Float[Tensor, "window token embedding"]:
        """Run a layer-boundary prefix through the trainable top layers.

        Builds the extended mask `BertModel.forward` would. Given
        `attention_mask_cpu` (eager path only), decides padding from the host
        copy rather than having `create_bidirectional_mask` read the device.

        :param prefix: one row of hidden states per window.
        :param attention_mask: the matching per-window padding mask.
        :param attention_mask_cpu: the same mask, still on the host, or
            `None` from the traced call, which keeps the old device check.
        :param layers: how many top layers to run, or `None` for
            `config.unfrozen_top_layers`.
        :return: the top layers' output, the same shape as `prefix`.
        """
        encoder_layers = cast(
            nn.ModuleList, self.base_model.get_submodule("encoder.layer")
        )
        if layers is None:
            layers = self.config.unfrozen_top_layers
        top_layers = encoder_layers[len(encoder_layers) - layers :]

        if attention_mask_cpu is None:
            extended_mask = create_bidirectional_mask(
                config=self.base_model.config,
                inputs_embeds=prefix,
                attention_mask=attention_mask,
            )
        else:
            no_padding = bool(attention_mask_cpu.all())
            extended_mask = create_bidirectional_mask(
                config=self.base_model.config,
                inputs_embeds=prefix,
                attention_mask=None if no_padding else attention_mask,
                allow_is_bidirectional_skip=no_padding,
            )
        hidden_states = prefix
        for layer in top_layers:
            hidden_states = layer(hidden_states, extended_mask)
        return hidden_states

    def _layer_boundary_store(self) -> LayerBoundaryStore | None:
        """The store for this run's frozen/trainable boundary, or `None`.

        Shared by `prefetch_layer_boundary_reads` and
        `_resolve_layer_boundary_cached`, which must agree on exactly which
        store a batch's items read from.

        :return: the run's layer-boundary store, or `None` if the trunk
            has no partially-frozen boundary or no store is configured
            for it.
        """
        if not self.config.unfrozen_top_layers:
            return None
        return layer_boundary_store(
            self.config.base_model, self.config.unfrozen_top_layers
        )

    def _submit_layer_boundary_reads(
        self,
        store: LayerBoundaryStore,
        pool: ThreadPoolExecutor,
        batch: Sequence[BatchItem],
    ) -> list[Future[Tensor | None] | None]:
        """One `store.get` future per batch item, submitted to `pool`.

        Shared by `prefetch_layer_boundary_reads` and
        `_resolve_layer_boundary_cached` so the two submit identically.

        :param store: the store to read each item's prefix from.
        :param pool: the single-worker pool each read is submitted to.
        :param batch: the batch's items, in the order to return futures for.
        :return: one future per cache miss, and `None` for each CPU-cache hit,
            in `batch` order.
        """
        futures: list[Future[Tensor | None] | None] = []
        for item in batch:
            document_id = int(item["id"].item())
            expected_windows = int(item["doc_id"].shape[-1])
            cache_key = cpu_cache_key(
                self.config.base_model,
                document_id,
                unfrozen_top_layers=self.config.unfrozen_top_layers,
            )
            cached = (
                None
                if cpu_embeddings_cache is None
                else cpu_embeddings_cache.get(cache_key)
            )
            if cached is not None and cached.shape[0] == expected_windows:
                futures.append(None)
            else:
                futures.append(
                    pool.submit(
                        store.get,
                        document_id,
                        expected_windows=expected_windows,
                    )
                )
        return futures

    def _layer_boundary_worker(self) -> ThreadPoolExecutor:
        """The one background thread every layer-boundary read runs on.

        :return: the model's single-worker pool for layer-boundary reads.
        """
        if self._layer_boundary_pool is None:
            self._layer_boundary_pool = ThreadPoolExecutor(max_workers=1)
        return self._layer_boundary_pool

    def prefetch_layer_boundary_reads(
        self, batches: Iterable[Sequence[BatchItem]]
    ) -> Iterator[Sequence[BatchItem]]:
        """Issue batch `k + 1`'s store reads while batch `k` is processed.

        A pass-through when there is no layer-boundary store. The park is
        cleared in `finally`, so an early exit never leaves one behind.

        :param batches: the batch loop to wrap, e.g. `batch_progress(data)`.
        :return: the same batches, unchanged, one lookahead deep.
        """
        store = self._layer_boundary_store()
        if store is None:
            yield from batches
            return

        pool = self._layer_boundary_worker()
        try:
            iterator = iter(batches)
            try:
                batch = next(iterator)
            except StopIteration:
                return
            futures = self._submit_layer_boundary_reads(store, pool, batch)

            for next_batch in iterator:
                self._parked_layer_boundary_reads = (batch, futures)
                futures = self._submit_layer_boundary_reads(
                    store, pool, next_batch
                )
                yield batch
                batch = next_batch

            self._parked_layer_boundary_reads = (batch, futures)
            yield batch
        finally:
            self._parked_layer_boundary_reads = None
            self._layer_boundary_pool = None
            pool.shutdown(cancel_futures=True)

    @torch.compiler.disable
    def _resolve_layer_boundary_cached(
        self, batch: Sequence[BatchItem]
    ) -> tuple[list[Tensor | None], list[tuple[int, BatchItem]]]:
        """Resolve each item against the configured layer-boundary store.

        Consumes a park only if its batch is this one by identity; otherwise
        reads are submitted to the background pool here. Time blocked on
        them is summed into `_layer_boundary_wait_seconds`.

        :param batch: the batch's items.
        :return: one slot per batch item, `None` where still unresolved
            (a full forward is needed), and the `(index, item)` pairs left
            unresolved, in batch order.
        """
        inputs: list[Tensor | None] = [None] * len(batch)
        missing: list[tuple[int, BatchItem]] = []

        store = self._layer_boundary_store()
        futures: list[Future[Tensor | None] | None] | None = None
        if store is not None:
            self._layer_boundary_wait_batches += 1
            parked = self._parked_layer_boundary_reads
            if parked is not None and parked[0] is batch:
                self._parked_layer_boundary_reads = None
                futures = parked[1]
            else:
                if parked is not None:
                    # Another batch's park: discarded, not risked against this
                    # one's window counts. A read already running keeps running.
                    self._parked_layer_boundary_reads = None
                    for stale in parked[1]:
                        if stale is not None:
                            stale.cancel()
                futures = self._submit_layer_boundary_reads(
                    store, self._layer_boundary_worker(), batch
                )

        global cpu_cache_hits, cpu_cache_misses
        for ix, item in enumerate(batch):
            document_id = int(item["id"].item())
            expected_windows = int(item["doc_id"].shape[-1])
            cache_key = cpu_cache_key(
                self.config.base_model,
                document_id,
                unfrozen_top_layers=self.config.unfrozen_top_layers,
            )
            cached = (
                None
                if cpu_embeddings_cache is None
                else cpu_embeddings_cache.get(cache_key)
            )
            if cached is not None and cached.shape[0] == expected_windows:
                cpu_cache_hits += 1
                future = None if futures is None else futures[ix]
                if future is not None:
                    future.cancel()
            else:
                if cpu_embeddings_cache is not None:
                    cpu_cache_misses += 1
                if futures is not None:
                    future = futures[ix]
                    if future is None:
                        assert store is not None
                        future = self._layer_boundary_worker().submit(
                            store.get,
                            document_id,
                            expected_windows=expected_windows,
                        )
                    wait_start = time.monotonic()
                    cached = future.result()
                    self._layer_boundary_wait_seconds += (
                        time.monotonic() - wait_start
                    )
                    if cached is not None and cpu_embeddings_cache is not None:
                        cpu_embeddings_cache.set(
                            cache_key, cached, from_store=True
                        )
                else:
                    cached = None
            if cached is None:
                missing.append((ix, item))
                continue

            attention_mask = item["sequence"]["attention_mask"].reshape(
                -1, item["sequence"]["attention_mask"].shape[-1]
            )
            device_mask = attention_mask.to(self.device, non_blocking=True)
            with self.autocast_context():
                replayed = self._replay_top_layers(
                    cached.to(self.device, dtype=self.amp_dtype),
                    device_mask,
                    attention_mask,
                )
            inputs[ix] = aggregate_embeddings(replayed, attention_mask).to(
                dtype=self.amp_dtype
            )

        return inputs, missing

    def _derived_embedding(
        self, store: LayerBoundaryStore, item: BatchItem
    ) -> Float[Tensor, "token embedding"] | None:
        """A frozen trunk's aggregated rows, derived from a stored boundary.

        Replays the boundary's top layers over the stored prefix and
        aggregates the windows as `utils.embed_document` does, landing on
        the host as an aggregated store's hit does.

        :param store: the boundary to derive from.
        :param item: the document.
        :return: one row per token, or None if the boundary cannot answer.
        """
        prefix = store.get(
            int(item["id"].item()),
            expected_windows=int(item["doc_id"].shape[-1]),
        )
        if prefix is None:
            return None
        attention_mask = item["sequence"]["attention_mask"].reshape(
            -1, item["sequence"]["attention_mask"].shape[-1]
        )
        with torch.no_grad(), self.autocast_context():
            replayed = self._replay_top_layers_eager(
                prefix.to(self.device, dtype=self.amp_dtype),
                attention_mask.to(self.device),
                attention_mask,
                layers=store.unfrozen_top_layers,
            )
        return aggregate_embeddings(replayed, attention_mask).cpu()

    @torch.compiler.disable
    def _resolve_cached(
        self,
        batch: Sequence[BatchItem],
        trunk_trainable: bool,
    ) -> tuple[list[Tensor | None], list[tuple[int, BatchItem]]]:
        """Resolve each item against the CPU cache, then the precomputed store.

        Neither is consulted when the trunk trains. A store hit is promoted
        into the cache outside inference mode, for the reason
        `_write_resolved_embeddings` gives.

        :param batch: the batch's items.
        :param trunk_trainable: whether `config.unfrozen_top_layers` is set.
        :return: one slot per batch item, `None` where still unresolved, and
            the `(index, item)` pairs left unresolved, in batch order.
        :raises RuntimeError: if a store hit's bf16 magnitude exceeds
            fp16's finite range and `self.amp_dtype` is fp16, so the cast
            would otherwise turn it into `inf` silently.
        """
        global cpu_cache_hits, cpu_cache_misses

        inputs: list[Tensor | None] = [None] * len(batch)
        missing: list[tuple[int, BatchItem]] = []
        store = (
            None
            if trunk_trainable
            else embeddings_store(self.config.base_model)
        )
        promotion_context = (
            contextlib.nullcontext()
            if store is None or cpu_embeddings_cache is None
            else torch.inference_mode(False)
        )

        with promotion_context:
            for ix, item in enumerate(batch):
                document_id: int = int(item["id"].item())
                if not trunk_trainable and (
                    cpu_embeddings_cache is not None or store is not None
                ):
                    # A row-count disagreement is a miss and needs no
                    # eviction: whichever source answers writes the same key.
                    expected_tokens = document_token_count(item)
                    if cpu_embeddings_cache is not None:
                        cpu_cached = cpu_embeddings_cache.get(
                            cpu_cache_key(self.config.base_model, document_id)
                        )
                        if (
                            cpu_cached is not None
                            and cpu_cached.shape[0] == expected_tokens
                        ):
                            cpu_cache_hits += 1
                            inputs[ix] = cpu_cached
                            continue
                        cpu_cache_misses += 1
                    if store is not None:
                        stored = (
                            self._derived_embedding(store, item)
                            if isinstance(store, LayerBoundaryStore)
                            else store.get(
                                document_id, expected_tokens=expected_tokens
                            )
                        )
                        if stored is not None:
                            embedding = stored.to(dtype=self.amp_dtype)
                            if self.amp_dtype is torch.float16 and bool(
                                (
                                    torch.isfinite(stored)
                                    & ~torch.isfinite(embedding)
                                ).any()
                            ):
                                msg = (
                                    f"document {document_id}: the "
                                    "embeddings store holds a bf16 value "
                                    "whose magnitude exceeds fp16's finite "
                                    "range (65504); the cast to fp16 "
                                    "would silently turn it into inf"
                                )
                                raise RuntimeError(msg)
                            if cpu_embeddings_cache is not None:
                                # Marked as the store's, so a forward-only
                                # document may evict it later.
                                cpu_embeddings_cache.set(
                                    cpu_cache_key(
                                        self.config.base_model, document_id
                                    ),
                                    embedding,
                                    from_store=True,
                                )
                            inputs[ix] = embedding
                            continue
                missing.append((ix, item))

        return inputs, missing

    def _embed_missing(
        self,
        missing: list[tuple[int, BatchItem]],
        inputs: list[Tensor | None],
        trunk_trainable: bool,
    ) -> None:
        """Run one batched forward for `missing` and fill their slots.

        A trainable trunk runs or reads its frozen prefix and hands the output
        to `_replay_top_layers`, the same route a stored prefix takes; a
        frozen trunk keeps the one `base_model` call under `no_grad`.

        :param missing: `(index, item)` pairs `_resolve_cached` left
            unresolved, in batch order.
        :param inputs: the batch's slot list; filled in place at each
            `missing` index.
        :param trunk_trainable: whether `config.unfrozen_top_layers` is set;
            True keeps the top layers' gradients and caches the frozen prefix.
        :return: None; `inputs` is mutated in place.
        """
        grad_context = (
            contextlib.nullcontext() if trunk_trainable else torch.no_grad()
        )
        with grad_context:
            batched_inputs = self.batch_input_tensors(
                [item for _, item in missing]
            )
            attention_mask_cpu = batched_inputs["attention_mask"]
            attention_mask = attention_mask_cpu.to(
                self.device, non_blocking=True
            )
            input_ids = batched_inputs["input_ids"].to(
                self.device, dtype=torch.int, non_blocking=True
            )
            with self.autocast_context():
                if trunk_trainable:
                    output = self._embed_missing_trainable_trunk(
                        input_ids,
                        attention_mask,
                        attention_mask_cpu,
                        [item for _, item in missing],
                    )
                else:
                    output = self.base_model(
                        input_ids=input_ids, attention_mask=attention_mask
                    ).last_hidden_state.detach()

        out_iter = iter(output)
        # `aggregate_embeddings` reads lengths off this mask on the host;
        # handing it the CPU copy already at hand, rather than the device
        # one the forward above needed, keeps that read sync-free.
        masks_iter = iter(attention_mask_cpu)
        self._write_resolved_embeddings(
            missing, inputs, out_iter, masks_iter, trunk_trainable
        )

        # Both names hold the hidden states (the iterator keeps views), so
        # both go to end their residency before the padding.
        del output, out_iter

    def _embed_missing_trainable_trunk(
        self,
        input_ids: Integer[Tensor, "window token"],
        attention_mask: Integer[Tensor, "window token"],
        attention_mask_cpu: Integer[Tensor, "window token"],
        items: Sequence[BatchItem],
    ) -> Float[Tensor, "window token embedding"]:
        """Run the frozen prefix eagerly, then the top layers via the wrapper.

        Mirrors `precompute_embeddings.embed_document_and_prefix`; call it
        under `self.autocast_context()`. Padding is decided from the host
        mask, safe here because this never runs inside the compiled graph.
        A layer-boundary store this run is building gets each item's prefix.

        :param input_ids: the batch's token ids, on `self.device`.
        :param attention_mask: the matching per-window padding mask.
        :param attention_mask_cpu: the same mask, still on the host.
        :param items: the batch items the windows belong to, in order.
        :return: the trunk's output, the same shape a whole-model forward's
            `last_hidden_state` would be.
        """
        encoder_layers = cast(
            nn.ModuleList, self.base_model.get_submodule("encoder.layer")
        )
        frozen_layers = len(encoder_layers) - self.config.unfrozen_top_layers

        hidden_states = self.base_model.get_submodule("embeddings")(
            input_ids=input_ids
        )
        no_padding = bool(attention_mask_cpu.all())
        extended_mask = create_bidirectional_mask(
            config=self.base_model.config,
            inputs_embeds=hidden_states,
            attention_mask=None if no_padding else attention_mask,
            allow_is_bidirectional_skip=no_padding,
        )
        for layer in encoder_layers[:frozen_layers]:
            hidden_states = layer(hidden_states, extended_mask)

        store = self._layer_boundary_store()
        if store is not None and store.writable:
            # Rounded through the store's bf16 before the top layers see it,
            # so the pass that builds the store trains on the values every
            # later pass reads back from it.
            hidden_states = hidden_states.to(torch.bfloat16).to(self.amp_dtype)
        if (
            store is not None and store.writable
        ) or cpu_embeddings_cache is not None:
            for item, prefix in zip(
                items,
                hidden_states.split(
                    [int(item["doc_id"].shape[-1]) for item in items]
                ),
                strict=True,
            ):
                document_id = int(item["id"].item())
                cache_key = cpu_cache_key(
                    self.config.base_model,
                    document_id,
                    unfrozen_top_layers=self.config.unfrozen_top_layers,
                )
                if store is not None and store.writable:
                    store.put(document_id, prefix)
                if cpu_embeddings_cache is not None:
                    cost = prefix.numel() * prefix.element_size()
                    if cpu_embeddings_cache.would_admit(cache_key, cost):
                        with torch.inference_mode(False):
                            cached_prefix = prefix.detach().to("cpu", copy=True)
                        cpu_embeddings_cache.set(
                            cache_key,
                            cached_prefix,
                            from_store=store is not None and store.writable,
                        )

        # One dtype from either path, so the compiled wrapper guards on one
        # dtype instead of recompiling between this output and a store hit.
        return self._replay_top_layers(
            hidden_states.to(dtype=self.amp_dtype),
            attention_mask,
            attention_mask_cpu,
        )

    @torch.compiler.disable
    def _write_resolved_embeddings(
        self,
        missing: list[tuple[int, BatchItem]],
        inputs: list[Tensor | None],
        out_iter: Iterator[Tensor],
        masks_iter: Iterator[Tensor],
        trunk_trainable: bool,
    ) -> None:
        """Aggregate each missing document's windows and fill its slot.

        Leaves inference mode so a cache entry written during validation can
        be trained through later; compiler-disabled because dynamo drops a
        captured `inference_mode(False)` rather than honouring it.

        :param missing: `(index, item)` pairs `_resolve_cached` left
            unresolved, in batch order.
        :param inputs: the batch's slot list; filled in place at each
            `missing` index.
        :param out_iter: iterator over the forward's flat hidden states,
            one row per window across every missing document.
        :param masks_iter: iterator over the matching flat attention mask,
            on the host -- `aggregate_embeddings` reads it there.
        :param trunk_trainable: whether `config.unfrozen_top_layers` is
            set; True skips the cache entirely.
        :return: None; `inputs` is mutated in place.
        """
        # Not needed on the trunk-trainable path, which writes to neither
        # cache.
        cache_context = (
            contextlib.nullcontext()
            if trunk_trainable
            else torch.inference_mode(False)
        )
        store = (
            None
            if trunk_trainable
            else embeddings_store(self.config.base_model)
        )
        with cache_context:
            for ix, item in missing:
                number_of_sequences_for_item = item["doc_id"].shape[-1]
                outs = torch.stack(
                    tuple(
                        itertools.islice(out_iter, number_of_sequences_for_item)
                    )
                ).to(dtype=self.amp_dtype)
                masks = torch.stack(
                    tuple(
                        itertools.islice(
                            masks_iter, number_of_sequences_for_item
                        )
                    )
                )
                doc_embedding = aggregate_embeddings(outs, masks)
                if (
                    store is not None
                    and not isinstance(store, LayerBoundaryStore)
                    and store.writable
                ):
                    # Rounded through the store's bf16 before the heads see
                    # it, so the pass that builds the store trains on the
                    # values every later pass reads back from it.
                    doc_embedding = doc_embedding.to(torch.bfloat16).to(
                        self.amp_dtype
                    )
                    store.put(int(item["id"].item()), doc_embedding)
                inputs[ix] = doc_embedding

                # No split gate: a validation document saves a forward per
                # epoch just as a training one does.
                if not trunk_trainable and cpu_embeddings_cache is not None:
                    cache_key = cpu_cache_key(
                        self.config.base_model, int(item["id"].item())
                    )
                    cost = doc_embedding.numel() * doc_embedding.element_size()
                    # Checked on the still-on-device tensor: a document
                    # the cache will decline must not pay for the copy
                    # to host RAM first.
                    if cpu_embeddings_cache.would_admit(cache_key, cost):
                        cpu_embeddings_cache.set(
                            cache_key,
                            # Budgeted in host RAM; a device tensor
                            # would pin VRAM.
                            doc_embedding.cpu(),
                        )

    def _pad_and_mask(
        self, inputs: list[Tensor | None]
    ) -> tuple[
        Float[Tensor, "batch max_doc_len embedding"],
        Bool[Tensor, "batch max_doc_len"],
    ]:
        """Move every resolved slot onto the model device, pad, and mask.

        :param inputs: one resolved embedding per batch item, in batch
            order; every slot must already be filled.
        :return: the padded embeddings and their mask.
        """
        # Hits reach the card only now, after the hidden states are released,
        # so they never share it with the forward's peak.
        embeddings = [
            emb.to(self.device, non_blocking=True)
            for emb in cast(list[Tensor], inputs)
        ]
        max_doc_len = max(emb.shape[0] for emb in embeddings)
        padded_embeddings = pad_sequence(
            embeddings, batch_first=True, padding_value=0.0
        )
        attention_masks = torch.zeros(
            (len(embeddings), max_doc_len),
            dtype=torch.bool,
            device=self.device,
        )
        for i, emb in enumerate(embeddings):
            attention_masks[i, : emb.shape[0]] = True

        return padded_embeddings, attention_masks


def print_epoch_stats(
    losses: dict[str, float], denominator: int, step: Step
) -> dict[str, float]:
    """Print the epoch's average losses, and return them keyed for tracking.

    Returning what it prints is the point: the console and the tracking server
    cannot disagree about an epoch's numbers.

    :param losses: the epoch's summed losses by objective.
    :param denominator: how many batches they were summed over.
    :param step: whether this was a training or a validation pass.
    :return: the averages, under their tracking keys.
    """
    for obj, loss in losses.items():
        logger.info("Average (%s) %s loss: %.4f", obj, step, loss / denominator)

    total_loss = sum(losses.values())
    logger.info("Average %s loss: %.4f", step, total_loss / denominator)

    # `loss_` rather than the bare objective name: MLflow charts a key with
    # no unit and no legend, so `training/class` left the reader to guess
    # whether the axis was a loss, a score, or a count.
    return {
        f"{step}/loss_{obj}": value / denominator
        for obj, value in {**losses, "total": total_loss}.items()
    }


def epoch_rate_metrics(
    batches: int, seconds: float, step: Step
) -> dict[str, float]:
    """How long the epoch took, and how fast it went, keyed for tracking.

    Rate is in batches rather than documents because `TokenBudgetBatchSampler`
    makes the document count per batch a function of document length.

    :param batches: how many batches ran.
    :param seconds: the epoch's wall-clock time.
    :param step: whether this was a training or a validation pass.
    :return: the timing metrics, under their tracking keys.
    """
    metrics = {f"{step}/epoch_seconds": seconds}
    if seconds > 0:
        metrics[f"{step}/batches_per_second"] = batches / seconds

    return metrics


def typed_relation_f1(
    true: np.ndarray,
    pred: np.ndarray,
    labels: np.ndarray,
    none_index: int,
    suffix: str = "",
    prefix: str = "test",
) -> dict[str, float]:
    """Macro- and micro-F1 over the typed relation labels, `none` excluded.

    :param true: gold labels for the scored rows.
    :param pred: predicted labels for the same rows.
    :param labels: the label values scored over.
    :param none_index: the label to exclude.
    :param suffix: appended to both keys, so a second scoring rule's F1s chart
        beside the default rule's instead of overwriting them.
    :param prefix: the tracking-key prefix, `test` for a one-off evaluation
        or `validation` for `Trainer`'s per-epoch selection score.
    :return: the two scores, or nothing when there is no row to score or the
        schema declares no typed label.
    """
    typed = np.array([label for label in labels if label != none_index])
    if not true.size or not typed.size:
        return {}

    return {
        f"{prefix}/relation_macro_f1_typed{suffix}": f1_score(
            true, pred, labels=typed, average="macro", zero_division=0
        ),
        f"{prefix}/relation_micro_f1_typed{suffix}": f1_score(
            true, pred, labels=typed, average="micro", zero_division=0
        ),
    }


def relation_metrics(
    true: np.ndarray,
    pred: np.ndarray,
    labels: np.ndarray,
    none_index: int,
    prefix: str = "test",
) -> dict[str, float]:
    """Relation scores over the candidate pairs, with `none` held separate.

    A macro-F1 across all three labels is dominated by `none`, which is both
    the majority class and the one nobody asked about. `none_share` records
    which pair distribution this pass actually met, since the candidates come
    from the current span tagger's groundings rather than from the corpus.

    :param true: gold labels for the candidate pairs.
    :param pred: predicted labels for the same pairs.
    :param labels: the label values scored over.
    :param none_index: the label to hold separate.
    :param prefix: the tracking-key prefix, `test` for a one-off evaluation
        or `validation` for `Trainer`'s per-epoch selection score.
    :return: the scores, under their tracking keys.
    """
    metrics = {f"{prefix}/relation_candidate_pairs": float(true.size)}
    if not true.size:
        # A split can yield no candidate at all; the count is then the
        # finding, and `f1_score` refuses an empty array anyway.
        return metrics

    metrics[f"{prefix}/relation_accuracy"] = float((true == pred).mean())
    metrics[f"{prefix}/relation_none_share"] = float(
        (true == none_index).mean()
    )
    metrics.update(
        typed_relation_f1(true, pred, labels, none_index, prefix=prefix)
    )

    return metrics


def support_metrics(
    tasks: Mapping[str, tuple[np.ndarray, np.ndarray]],
    prefix: str = "test",
) -> dict[str, float]:
    """Gold and predicted positive counts per task, keyed for tracking.

    These are what tell one micro-F1 of zero from another: a head predicting
    nothing and a head predicting the wrong labels score identically.
    `labels_predicted` counts the columns ever used, which is how a head
    collapsed onto one frequent label shows up.

    :param tasks: task name -> its `(gold, predicted)` indicator matrices.
    :param prefix: the tracking-key prefix, `test` for a one-off evaluation
        or `validation` for `Trainer`'s per-epoch selection score.
    :return: the counts, under their tracking keys.
    """
    metrics: dict[str, float] = {}
    for task, (true, pred) in tasks.items():
        metrics[f"{prefix}/{task}_gold_positives"] = float(true.sum())
        metrics[f"{prefix}/{task}_predicted_positives"] = float(pred.sum())
        metrics[f"{prefix}/{task}_labels_predicted"] = float(
            (pred.sum(axis=0) > 0).sum()
        )

    return metrics


def micro_ap_metrics(
    task: str,
    true: np.ndarray,
    probs: np.ndarray,
    prefix: str = "test",
) -> dict[str, float]:
    """Micro-averaged average precision for one head, keyed for tracking.

    A diverged head scores every column NaN, which `average_precision_score`
    refuses outright; raising here would end the pass with everything already
    measured unlogged, since the dict goes out in a single call at the end.

    :param task: the head scored, naming the key it is logged under.
    :param true: gold indicators, one row per document.
    :param probs: the head's scores for the same rows and columns.
    :param prefix: the tracking-key prefix, `test` for a one-off evaluation
        or `validation` for `Trainer`'s per-epoch selection score.
    :return: `{prefix}/{task}_micro_ap`, NaN when the scores cannot be ranked.
    """
    key = f"{prefix}/{task}_micro_ap"
    try:
        metrics = {
            key: float(average_precision_score(true, probs, average="micro"))
        }
        logger.info("micro-AP: %s", metrics[key])
    except ValueError as exc:
        # Nothing was ranked, so either end of the scale would be a claim.
        metrics = {key: float("nan")}
        logger.warning("micro-AP: undefined (%s); logged as nan", exc)

    return metrics


def coverage_metrics(
    data: DataLoader, scored: int, prefix: str = "test"
) -> dict[str, float]:
    """How many of the split's documents the pass actually scored.

    The planned count is logged at run setup, before anything has been read,
    and the two come apart whenever the frame and the encodings file disagree —
    which shrinks every metric's denominator without shrinking the number a run
    list shows beside them.

    :param data: the loader the pass ran over.
    :param scored: how many documents reached the model.
    :param prefix: which split this coverage describes, `test` for a one-off
        evaluation or `validation` for `Trainer`'s per-epoch pass.
    :return: the counts, keyed under `dataset/` so the three sit together in a
        run table.
    """
    metrics = {f"dataset/{prefix}_documents_scored": float(scored)}

    planned = split_documents(data)
    if planned is not None:
        metrics[f"dataset/{prefix}_documents_missing"] = float(planned - scored)

    return metrics


def class_predictions(
    data: DataLoader,
    logits: list[Tensor],
    true: list[Tensor],
    tau: UnitInterval,
    prefix: str,
    metrics: dict[str, float],
    step: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Concatenate one pass's batches of class logits, or bail out empty.

    :param data: the loader the pass ran over, for `coverage_metrics`.
    :param logits: each batch's OOS-dropped class logits, CPU tensors.
    :param true: each batch's gold class indicators, CPU tensors.
    :param tau: threshold binarizing the class probabilities.
    :param prefix: the tracking-key prefix the scores are reported under.
    :param metrics: the caller's metrics dict, updated in place.
    :param step: the tracking step logged when the pass scored nothing.
    :return: `(cls_true, cls_pred, cls_probs)`, or None when `logits` is
        empty; the caller must then return `metrics` as-is.
    """
    if not logits:
        logger.warning("No samples found.")
        metrics.update(coverage_metrics(data, 0, prefix=prefix))
        tracking.log_metrics(metrics, step=step)
        return None
    cls_logits = torch.cat(logits, dim=0).numpy()
    cls_true = torch.cat(true, dim=0).numpy().astype(int)
    cls_probs = 1.0 / (1.0 + np.exp(-cls_logits))
    cls_pred = (cls_probs >= tau).astype(int)
    metrics.update(coverage_metrics(data, cls_true.shape[0], prefix=prefix))
    metrics.update(
        support_metrics({"class": (cls_true, cls_pred)}, prefix=prefix)
    )
    return cls_true, cls_pred, cls_probs


def class_report_metrics(
    cls_true: np.ndarray,
    cls_pred: np.ndarray,
    cls_probs: np.ndarray,
    prefix: str,
    known_classes: Sequence[str],
    log_reports: bool,
    include_ap: bool = True,
) -> dict[str, float]:
    """Micro-F1, optionally micro-AP, and the per-class text report.

    :param cls_true: gold class indicators, one row per document.
    :param cls_pred: binarized predictions for the same rows.
    :param cls_probs: the probabilities `cls_pred` was thresholded from.
    :param prefix: the tracking-key prefix the scores are reported under.
    :param known_classes: the class names in column order.
    :param log_reports: whether to log the per-class report to the console
        and as a run artifact.
    :param include_ap: whether to add a micro-AP metric; `ete.py` never has.
    :return: the micro-F1 metric, and micro-AP's when `include_ap` is set.
    """
    metrics: dict[str, float] = {}
    logger.info("\n=== Entity CLASS metrics (multilabel, document-level) ===")
    metrics[f"{prefix}/class_micro_f1"] = f1_score(
        cls_true, cls_pred, average="micro", zero_division=0
    )
    logger.info("micro-F1: %s", metrics[f"{prefix}/class_micro_f1"])
    if include_ap:
        metrics.update(
            micro_ap_metrics("class", cls_true, cls_probs, prefix=prefix)
        )
    report = classification_report(
        y_true=cls_true,
        y_pred=cls_pred,
        target_names=known_classes,
        zero_division=0,
    )
    if log_reports:
        logger.info(report)
        tracking.log_text(str(report), f"{prefix}/class_report.txt")
    return metrics
