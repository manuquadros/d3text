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
import operator
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any, assert_never, cast

import lmdb
import numpy as np
import torch
import torch.nn as nn
import transformers
from cacheout import Cache
from d3text.embeddings_store import EmbeddingsStore, ProvenanceError
from d3text.progress import batch_progress, split_documents
from d3text.training.update import BatchUpdate
from d3text.utils import aggregate_embeddings
from jaxtyping import Bool, Float, Int64, Integer
from sklearn.metrics import f1_score
from torch import Tensor
from torch.autograd.profiler import record_function
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from .config import (
    ModelConfig,
    TokenLossWeighting,
    machine_config,
    save_model_config,
)
from .heads import PermutationBatchNorm1d
from .model_types import BatchItem

logger = logging.getLogger(__name__)

mconfig = machine_config()
if mconfig.cpu_embeddings_cache_size:
    cpu_embeddings_cache = Cache(maxsize=mconfig.cpu_embeddings_cache_size)
else:
    cpu_embeddings_cache = None


def cpu_cache_key(base_model: str, doc_id: int) -> tuple[str, int]:
    """Identify a cached activation by the base model that produced it.

    The cache is process-wide and one process holds more than one base model,
    so a document id alone would let two base models of equal hidden width
    serve one trial's activations to the next.

    :param base_model: the base model whose forward produced the activation.
    :param doc_id: the document the activation belongs to.
    :return: the cache key for that pair.
    """
    return base_model, doc_id


@functools.cache
def embeddings_store(base_model: str) -> EmbeddingsStore | None:
    """The configured embeddings store, opened once, or `None` without one.

    Lazy, because importing `d3text.models` must not touch the filesystem. A
    store that cannot be opened, or that a different base model wrote, disables
    itself and the run recomputes the embeddings.

    :param base_model: the base model the store has to have been written by.
    :return: the open store, or None if there is none or it is unusable.
    """
    if not mconfig.embeddings_store:
        return None
    try:
        store = EmbeddingsStore(mconfig.embeddings_store, base_model)
    except lmdb.Error as error:
        logger.warning(
            "Cannot open the embeddings store at %s (%s); embeddings will be "
            "computed by the base model as though none were configured.",
            mconfig.embeddings_store,
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

    # Nothing owns the store — this function is cached and the reader lives as
    # long as the process — so process exit is the only place its hit rate can
    # be reported. `logging.shutdown` registers itself when logging is first
    # imported, and atexit runs last-registered-first, so this still has a
    # working handler when it fires.
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


class Step(StrEnum):
    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "testing"


def get_pool_fn(pooling: str):
    if pooling == "max":
        return lambda x: torch.amax(x, dim=0)
    elif pooling == "mean":
        return lambda x: torch.mean(x, dim=0)
    elif pooling == "logsumexp":
        return lambda x: torch.logsumexp(x, dim=0)
    else:
        raise ValueError(f"Unknown pooling: {pooling}")


def ordered_entities(entity_index: Mapping[str, int]) -> list[str]:
    """Entity names ordered by the logit column they are scored in.

    :param entity_index: entity name -> its column, which must be exactly
        `0..N-1` since the model treats an index as a position.
    :return: the names in column order.
    """
    ordered = sorted(entity_index.items(), key=operator.itemgetter(1))
    if [index for _, index in ordered] != list(range(len(ordered))):
        raise ValueError(
            "entity_index must map its "
            f"{len(entity_index)} names onto contiguous indices 0..N-1, "
            f"got {sorted(entity_index.values())}"
        )
    return [name for name, _ in ordered]


def label_columns(
    labels: Sequence[str], sentinel: str
) -> tuple[int, Int64[Tensor, " kept"]]:
    """Locate `sentinel` among `labels` and list every other column.

    The heads score one extra column the targets do not carry, so loss and
    evaluation run on the others. Locating it by name keeps those columns
    correct if it ever stops being the last one.

    :param labels: the head's labels, in column order.
    :param sentinel: the extra label, `UNK` for entities or `OOS` for classes.
    :return: the sentinel's column and the indices of every other column.
    :raises ValueError: if `sentinel` is not among `labels`.
    """
    index = labels.index(sentinel)
    return index, torch.tensor(
        [column for column in range(len(labels)) if column != index],
        dtype=torch.int64,
    )


def balanced_class_weights(
    targets: Int64[Tensor, " relation"], num_classes: int
) -> Float[Tensor, " classes"]:
    """Inverse-frequency class weights for one batch of relation targets.

    Per batch rather than precomputed because the candidate pairs are proposed
    by the current entity head, so there is no dataset frequency to derive them
    from. An absent class's count is clamped, since its weight is never read.

    :param targets: the batch's relation targets.
    :param num_classes: width of the relation head.
    :return: one weight per class.
    """
    counts = torch.bincount(targets, minlength=num_classes)
    return targets.numel() / (num_classes * counts.clamp(min=1))


def focal_cross_entropy(
    preds: Float[Tensor, "relation logits"],
    targets: Int64[Tensor, " relation"],
    gamma: float,
    label_smoothing: float = 0.0,
) -> Float[Tensor, ""]:
    """Cross-entropy with each element scaled by `(1 - p_t) ** gamma`.

    Normalised by the modulation mass rather than the row count: under a plain
    mean an easy pair still divides the denominator, so proposing more of them
    would shrink the loss on the rare positives.

    :param preds: per-pair logits.
    :param targets: per-pair class targets.
    :param gamma: the focusing exponent; 0 is plain cross-entropy.
    :param label_smoothing: passed through to the per-element cross-entropy.
    :return: the scalar loss.
    """
    elementwise = nn.functional.cross_entropy(
        preds, targets, reduction="none", label_smoothing=label_smoothing
    )
    p_t = preds.softmax(dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1)
    modulation = (1 - p_t) ** gamma
    return (modulation * elementwise).sum() / modulation.sum().clamp(min=1.0)


def masked_token_cross_entropy(
    preds: Float[Tensor, "token logits"],
    targets: Int64[Tensor, " token"],
    ignore_index: int = -100,
    weighting: TokenLossWeighting = "unweighted",
    focal_gamma: float = 2.0,
) -> Float[Tensor, ""]:
    """Cross-entropy over the tokens `targets` does not mask out.

    The divisor is the unmasked count, not the token count: dividing by the
    whole sequence would scale every real token's loss by the share of the
    document that happened to be masked. An all-masked batch returns a
    differentiable zero rather than a NaN.

    :param preds: per-token logits.
    :param targets: per-token targets, masked with `ignore_index`.
    :param ignore_index: the target value marking a token the loss must skip.
    :param weighting: `unweighted`, `balanced` (per-batch inverse frequency
        over the kept tokens) or `focal`.
    :param focal_gamma: the focusing exponent, read only under `focal`.
    :return: the scalar loss.
    """
    kept = targets != ignore_index
    if not bool(kept.any()):
        return preds.sum() * 0.0

    kept_preds = preds[kept]
    kept_targets = targets[kept]

    if weighting == "focal":
        return focal_cross_entropy(kept_preds, kept_targets, gamma=focal_gamma)

    if weighting == "balanced":
        weight = balanced_class_weights(kept_targets, preds.shape[-1])
        return nn.functional.cross_entropy(
            kept_preds, kept_targets, weight=weight
        )

    elementwise = nn.functional.cross_entropy(
        kept_preds, kept_targets, reduction="none"
    )
    return elementwise.sum() / kept.sum()


def masked_bce_with_logits(
    logits: Float[Tensor, "document class"],
    targets: Float[Tensor, "document class"],
    abstain: Bool[Tensor, "document class"] | None = None,
    pos_weight: Tensor | None = None,
    downweight: float = 0.0,
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


# Elements per slice when reducing a [document, token, logits] tensor. A fixed
# *token* width made the slice `[documents, 2048, entities]`, which grows with
# the batch: measured, every document beyond the first added ~0.157 GiB to peak
# memory whatever its length, so the batch size a card could hold stopped
# tracking the batch's token budget. Budgeting elements keeps the slice the
# same size whatever shape the batch has.
#
# 14M elements is 28 MB in bfloat16 — what one slice cost at the old width for
# a single document over the 6862-entity head.
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
        # Through `save_for_backward` rather than onto `ctx`: the mask is an
        # input, so this registers its version counter and an in-place edit
        # between forward and backward raises instead of silently scattering
        # the gradient over the wrong tokens.
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


def has_bf16_hardware() -> bool:
    """Whether this GPU runs bfloat16 in silicon rather than by emulation.

    `torch.cuda.is_bf16_supported()` answers a different question and returns
    True on cards with no bf16 units at all. Asked by compute capability, which
    is readable on every torch version.

    :return: whether bf16 arithmetic is native here.
    """
    if not torch.cuda.is_available():
        return False

    return torch.cuda.get_device_capability() >= (8, 0)


def select_amp_dtype() -> torch.dtype:
    """Pick bf16 only where the active backend can run it in silicon.

    Each backend is asked independently: compute capability is meaningless
    under HIP, so the device-name allowlist is the sole authority for ROCm.

    :return: the autocast dtype to use.
    """
    is_rocm = getattr(torch.version, "hip", None) is not None

    if is_rocm:
        device_name = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
        )
        bf16_ok = any(k in device_name for k in ("MI200", "MI250", "MI3"))
    else:
        bf16_ok = has_bf16_hardware()

    return torch.bfloat16 if bf16_ok else torch.float16


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
    base_model: transformers.PreTrainedModel
    _neg_inf: Tensor
    classes: list[str]
    class_columns: Tensor
    entities: list[str]
    entity_columns: Tensor

    def __init__(
        self,
        config: ModelConfig | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__()

        self.config = config if config is not None else ModelConfig()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.amp_dtype = select_amp_dtype()

        self.ramp_epochs: int = self.config.ramp_epochs
        self.entity_logits_pooling = self.config.entity_logits_pooling

        self.register_buffer("_neg_inf", torch.tensor(-1e9))

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
            # Shift by the per-segment max before exponentiating, as
            # torch.logsumexp does. The shift is detached because it cancels
            # analytically, which both keeps the gradient the plain softmax and
            # keeps the backward off `scatter_reduce`'s amax. A segment that is
            # entirely -inf would make `x - peak` a NaN, so such a segment is
            # shifted by zero instead and left to underflow to the -inf the
            # unsegmented op returns.
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

    def register_entity_columns(self) -> None:
        """Find the UNK column and remember the others.

        Call once `self.entities` is set. Non-persistent, since it is derived
        from them and an older checkpoint would otherwise be missing the key.
        """
        self.unk_index, entity_columns = label_columns(self.entities, "UNK")
        self.register_buffer("entity_columns", entity_columns, persistent=False)

    def drop_unk(
        self, entity_logits: Float[Tensor, "... entity"]
    ) -> Float[Tensor, "... entity"]:
        """Entity logits without the UNK column, to the width of the targets.

        :param entity_logits: the head's full-width logits.
        :return: the columns the targets carry.
        """
        return entity_logits.index_select(-1, self.entity_columns)

    @property
    def known_entities(self) -> list[str]:
        """Entity names in column order, minus UNK.

        :return: the columns `drop_unk` keeps, aligned with `entity_index` and
            with `class_matrix`'s rows.
        """
        return [
            self.entities[column] for column in self.entity_columns.tolist()
        ]

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

            def hidden_forward(x):
                for layer in self.hidden_layers:
                    x = layer(x)
                return x

            self.hidden = hidden_forward
        else:
            self.hidden = nn.Identity()

        self.hidden_block_output_size = in_features

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for all compatible modules.

        The base model is not among them: it is frozen and only ever runs under
        `no_grad`, so there is no activation graph to trade against
        recomputation.
        """
        if hasattr(self, "hidden_layers"):

            def hidden_with_checkpoint(x):
                for layer in self.hidden_layers:
                    x = torch.utils.checkpoint.checkpoint(
                        layer, x, use_reentrant=False
                    )
                return x

            if any(
                param.requires_grad for param in self.hidden_layers.parameters()
            ):
                self.hidden = hidden_with_checkpoint
        else:
            self.hidden = nn.Identity()

    @property
    def loss_fn(self) -> nn.Module:
        """The loss function for this model type.

        :return: the loss module.
        """
        raise NotImplementedError

    def compute_batch(
        self,
        batch: Any,
    ) -> float:
        """Compute loss for a batch and perform the optimization step.

        :param batch: the batch to run.
        :return: the batch's loss value.
        """
        raise NotImplementedError

    def compute_losses(
        self,
        batch: Sequence[BatchItem],
        step: Step,
        epoch: int,
    ) -> dict[str, Tensor]:
        """One batch's losses, keyed by objective name; per subclass.

        A key present in one batch of an epoch must be present in every batch
        of it, since `run_epoch` accumulates under these names. `step` is what
        lets a ramped model score validation under its final weight while
        training still follows the schedule.

        :param batch: the batch to run.
        :param step: whether this is a training or a validation pass.
        :param epoch: the epoch number, read only by a model that ramps.
        :return: one loss per objective.
        """
        raise NotImplementedError

    def run_epoch(
        self,
        data: DataLoader,
        step: Step,
        epoch: int,
        update: BatchUpdate,
    ) -> tuple[dict[str, float], int]:
        """Run every batch through `compute_losses` and the optimizer step.

        Shared by every subclass — only `compute_losses` differs between them.

        :param data: the split to run over.
        :param step: whether this is a training or a validation pass.
        :param epoch: the epoch number.
        :param update: `Trainer`'s batch update, ignored on a validation pass.
        :return: the summed losses by objective, and how many batches ran.
        """
        epoch_losses: dict[str, float] = {}
        n_batches = 0
        grad_context = (
            contextlib.nullcontext()
            if step == Step.TRAINING
            else torch.inference_mode()
        )

        with grad_context:
            for batch in batch_progress(data):
                if step == Step.TRAINING:
                    update.zero_grad()

                losses = self.compute_losses(batch, step, epoch)
                n_batches += 1

                if step == Step.TRAINING:
                    update(*losses.values())

                for key, value in losses.items():
                    epoch_losses[key] = (
                        epoch_losses.get(key, 0.0) + value.detach().cpu().item()
                    )

                del losses

        return epoch_losses, n_batches

    def save_config(self, path: str) -> None:
        save_model_config(self.config.model_dump(), path)

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
        model. The store's matrices were computed under different autocast
        settings, so a run that reads it gets slightly different activations —
        but the same ones every epoch, which the live path cannot promise.

        :param batch: the batch's items.
        :return: the padded embeddings and their mask.
        """
        inputs: list[None | Tensor] = [None] * len(batch)
        missing: list[tuple[int, BatchItem]] = []
        store = embeddings_store(self.config.base_model)

        for ix, item in enumerate(batch):
            doc_id: int = int(item["id"].item())
            if cpu_embeddings_cache is not None:
                cpu_cached = cpu_embeddings_cache.get(
                    cpu_cache_key(self.config.base_model, doc_id)
                )
                if cpu_cached is not None:
                    inputs[ix] = cpu_cached
                    continue
            if store is not None:
                stored = store.get(
                    doc_id, expected_tokens=document_token_count(item)
                )
                if stored is not None:
                    # Not written to the CPU cache: that cache exists to spare
                    # a base-model forward, and this document has already been
                    # spared one. Filling it here would evict documents whose
                    # only other source *is* the forward.
                    inputs[ix] = stored.to(dtype=self.amp_dtype)
                    continue
            missing.append((ix, item))

        if missing:
            with torch.no_grad():
                batched_inputs = self.batch_input_tensors(
                    [item for _, item in missing]
                )
                with self.autocast_context():
                    output = (
                        self.base_model(
                            input_ids=batched_inputs["input_ids"].to(
                                self.device, dtype=torch.int, non_blocking=True
                            ),
                            attention_mask=batched_inputs["attention_mask"].to(
                                self.device, non_blocking=True
                            ),
                        )
                        .last_hidden_state.detach()
                        .cpu()
                    )

            out_iter = iter(output)
            masks_iter = iter(batched_inputs["attention_mask"])
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
                inputs[ix] = doc_embedding

                # No split gate: a cached document skips one frozen
                # base-model forward per epoch whichever split it came from,
                # so reserving the one shared budget for training documents
                # buys nothing and leaves validation permanently cold.
                if (
                    cpu_embeddings_cache is not None
                    and not cpu_embeddings_cache.full()
                ):
                    cpu_embeddings_cache.set(
                        cpu_cache_key(
                            self.config.base_model, int(item["id"].item())
                        ),
                        doc_embedding,
                    )

        # Every slot is filled above (cache hit or freshly computed).
        embeddings = cast(list[Tensor], inputs)
        max_doc_len = max(emb.shape[0] for emb in embeddings)
        padded_embeddings = pad_sequence(
            embeddings, batch_first=True, padding_value=0.0
        )
        attention_masks = torch.zeros(
            (len(embeddings), max_doc_len), dtype=torch.bool
        )
        for i, emb in enumerate(embeddings):
            attention_masks[i, : emb.shape[0]] = True

        return padded_embeddings.to(
            self.device, non_blocking=True
        ), attention_masks.to(self.device, non_blocking=True)


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


def relation_metrics(
    true: np.ndarray,
    pred: np.ndarray,
    labels: np.ndarray,
    none_index: int,
) -> dict[str, float]:
    """Relation scores over the candidate pairs, with `none` held separate.

    A macro-F1 across all three labels is dominated by `none`, which is both
    the majority class and the one nobody asked about. `none_share` records
    which pair distribution this pass actually met, since the candidates come
    from the current entity head rather than from the corpus.

    :param true: gold labels for the candidate pairs.
    :param pred: predicted labels for the same pairs.
    :param labels: the label values scored over.
    :param none_index: the label to hold separate.
    :return: the scores, under their tracking keys.
    """
    metrics = {"test/relation_candidate_pairs": float(true.size)}
    if not true.size:
        # The hard mask proposes the candidates, so a split can yield none at
        # all. The count is the finding; a score over zero pairs is not one,
        # and `f1_score` refuses an empty array outright.
        return metrics

    metrics["test/relation_accuracy"] = float((true == pred).mean())
    metrics["test/relation_none_share"] = float((true == none_index).mean())

    typed = np.array([label for label in labels if label != none_index])
    if typed.size:
        metrics["test/relation_macro_f1_typed"] = f1_score(
            true, pred, labels=typed, average="macro", zero_division=0
        )
        metrics["test/relation_micro_f1_typed"] = f1_score(
            true, pred, labels=typed, average="micro", zero_division=0
        )

    return metrics


def support_metrics(
    tasks: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, float]:
    """Gold and predicted positive counts per task, keyed for tracking.

    These are what tell one micro-F1 of zero from another: a head predicting
    nothing and a head predicting the wrong labels score identically.
    `labels_predicted` counts the columns ever used, which is how a head
    collapsed onto one frequent label shows up.

    :param tasks: task name -> its `(gold, predicted)` indicator matrices.
    :return: the counts, under their tracking keys.
    """
    metrics: dict[str, float] = {}
    for task, (true, pred) in tasks.items():
        metrics[f"test/{task}_gold_positives"] = float(true.sum())
        metrics[f"test/{task}_predicted_positives"] = float(pred.sum())
        metrics[f"test/{task}_labels_predicted"] = float(
            (pred.sum(axis=0) > 0).sum()
        )

    return metrics


def coverage_metrics(data: DataLoader, scored: int) -> dict[str, float]:
    """How many of the split's documents the pass actually scored.

    The planned count is logged at run setup, before anything has been read,
    and the two come apart whenever the frame and the encodings file disagree —
    which shrinks every metric's denominator without shrinking the number a run
    list shows beside them.

    :param data: the loader the pass ran over.
    :param scored: how many documents reached the model.
    :return: the counts, keyed under `dataset/` so the three sit together in a
        run table.
    """
    metrics = {"dataset/test_documents_scored": float(scored)}

    planned = split_documents(data)
    if planned is not None:
        metrics["dataset/test_documents_missing"] = float(planned - scored)

    return metrics
