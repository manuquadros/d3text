import itertools
import logging
import math
import operator
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from enum import StrEnum
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import transformers
from cacheout import Cache
from d3text import tracking
from d3text.progress import batch_progress
from d3text.utils import aggregate_embeddings
from jaxtyping import Bool, Float, Int16, Int64, Integer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    label_ranking_average_precision_score,
)
from torch import Tensor
from torch.autograd.profiler import record_function
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import trange

from .config import (
    ModelConfig,
    embedding_dims,
    machine_config,
    optimizers,
    save_model_config,
    schedulers,
)
from .model_types import BatchedLogits, BatchItem, IndexedRelation

logger = logging.getLogger(__name__)

mconfig = machine_config()
if mconfig.cpu_embeddings_cache_size:
    cpu_embeddings_cache = Cache(maxsize=mconfig.cpu_embeddings_cache_size)
else:
    cpu_embeddings_cache = None


class Step(StrEnum):
    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "testing"


GRAD_CLIP_NORM = 1.0


def get_pool_fn(pooling: str):
    if pooling == "max":
        return lambda x: torch.amax(x, dim=0)
    elif pooling == "mean":
        return lambda x: torch.mean(x, dim=0)
    elif pooling == "logsumexp":
        return lambda x: torch.logsumexp(x, dim=0)
    else:
        raise ValueError(f"Unknown pooling: {pooling}")


def get_batch_entities(
    batch: Sequence[BatchItem], device: str = "cuda"
) -> tuple[Int16[Tensor, " entities"], ...]:
    """Get tuple indicating the entities tagged for each document.

    :return: Tuple whose positions correspond to sequences found in
        the batch. Each sequence mapped to the entities found in its
        respective document.
    """
    seqs = []
    for doc in batch:
        entities = (
            doc["entities"].nonzero()[:, 1].to(device=device, dtype=torch.int16)
        )
        seqs.append(entities)

    return tuple(seqs)


def ordered_entities(entity_index: Mapping[str, int]) -> list[str]:
    """Entity names ordered by the logit column they are scored in.

    The model treats an entity's index as a *position* in the entity logit
    vector, so the indices must be exactly ``0..N-1``; anything else would make
    ``entities[i]`` name a different entity than column ``i`` scores.
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

    The heads score one extra column that the targets do not carry — UNK for
    entities, OOS for classes — so loss and evaluation run on the other columns.
    Locating the sentinel by name keeps those columns correct if it ever stops
    being the last one.

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

    Candidate pairs are proposed per batch by the entity hard mask, so the
    `none` share is a property of the current entity head rather than of the
    corpus: there is no dataset frequency to precompute, and the weights have to
    be re-derived every batch.

    A class absent from `targets` would divide by zero. Its weight is never read
    — `cross_entropy` gathers weights by target value — so clamping the count is
    enough to keep the tensor finite.
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

    Suppresses the loss from pairs the model already scores confidently, which
    is most of what the hard mask proposes. Unlike a fixed class weight this
    tracks the entity head: as the mask sharpens and stops emitting junk pairs,
    the down-weighting relaxes on its own. `gamma == 0` is plain cross-entropy.

    Normalising by the modulation mass rather than by the row count is what
    makes that work. Under a plain `.mean()` an easy pair still divides the
    denominator, so proposing more of them shrinks the loss on the rare
    positives — the dilution this weighting exists to remove. Dividing by the
    mass instead keeps an easy pair out of *both* sides. The clamp guards the
    degenerate batch in which every pair is already scored confidently: the
    numerator vanishes with the mass, so the loss decays to zero instead of
    exploding.
    """
    elementwise = nn.functional.cross_entropy(
        preds, targets, reduction="none", label_smoothing=label_smoothing
    )
    p_t = preds.softmax(dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1)
    modulation = (1 - p_t) ** gamma
    return (modulation * elementwise).sum() / modulation.sum().clamp(min=1.0)


def load_base_model(base_model: str) -> transformers.PreTrainedModel:
    """Load a frozen transformer base, tolerating legacy configs that lack a
    ``model_type`` key (e.g. ``prajjwal1/bert-mini``).

    ``AutoModel.from_pretrained`` delegates to ``AutoConfig.from_pretrained``,
    which reads ``model_type`` from ``config.json`` to choose the architecture.
    Old-format repos omit it and raise ``ValueError``; fall back to an explicit
    BERT config in that case (every model in ``embedding_dims`` is BERT-based).
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

    Narrower slices for a wider batch, so `documents * tokens * width` stays
    put. At least one token, or a batch wide enough to exceed the budget on a
    single token would not advance.
    """
    return max(1, _POOL_CHUNK_ELEMENTS // max(1, documents * width))


class _ChunkedLogSumExp(torch.autograd.Function):
    """`logsumexp` over the token dimension, in float32, one slice at a time.

    `torch.logsumexp(logits.float(), dim=1)` first materialises a float32 copy
    of the entity logits — the largest tensor in the step, and twice the size
    of the bfloat16 original — and autograd holds it, plus a gradient of the
    same shape, until backward has run. Together those two are about half the
    peak of a training step.

    This walks the token dimension in slices, performing the same two-pass
    shift-and-sum `torch.logsumexp` performs. Only the order in which the
    exponentials are summed differs, and that difference does not survive the
    cast back to bfloat16: the pooled logits are bitwise unchanged. Backward
    needs no float32 copy either — the gradient of a logsumexp is
    ``grad * exp(x - out)``, which this recomputes slice by slice from the
    input and the (tiny) output it saved, rather than reading a stored one.
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


class _ChunkedMean(torch.autograd.Function):
    """`mean` over the token dimension, in float32, one slice at a time.

    Same bargain as `_ChunkedLogSumExp`, and simpler: a mean spreads its
    gradient evenly, so backward reads none of the input at all.
    """

    @staticmethod
    def forward(
        ctx, logits: Float[Tensor, "document token logits"], chunk: int
    ) -> Float[Tensor, "document logits"]:
        documents, tokens, width = logits.shape
        total = logits.new_zeros((documents, width), dtype=torch.float32)
        for start in range(0, tokens, chunk):
            total += logits[:, start : start + chunk].float().sum(dim=1)
        ctx.shape = logits.shape
        ctx.dtype = logits.dtype
        ctx.tokens = tokens
        return total / tokens

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(  # type: ignore[override]
        ctx, grad_pooled: Float[Tensor, "document logits"]
    ) -> tuple[Float[Tensor, "document token logits"], None]:
        grad = (grad_pooled / ctx.tokens).unsqueeze(1).expand(ctx.shape)
        return grad.to(ctx.dtype), None


def reject_empty_token_dim(logits: Float[Tensor, "..."], dim: int = 1) -> None:
    """Refuse to pool a document that has no tokens.

    The four supported poolings disagree completely on an empty reduction:
    `logsumexp` returns `-inf` (a confidently correct negative), `mean` returns
    `NaN` that propagates into the epoch's loss with nothing in the log to
    attribute it, `logmeanexp` dies inside `math.log`, and only `max` names the
    dimension. None of them can answer what a document with no text predicts,
    so the answer is given once, here.
    """
    if logits.shape[dim] == 0:
        msg = (
            f"cannot pool logits of shape {tuple(logits.shape)}: dimension "
            f"{dim} holds no tokens, so the document has no text to score"
        )
        raise ValueError(msg)


def pool_token_dim(
    logits: Float[Tensor, "document token logits"], pooling: str
) -> Float[Tensor, "document logits"]:
    """Pool the token dimension without a float32 copy of the whole tensor.

    The float32 is right — pooling thousands of tokens in bfloat16 is where the
    precision actually matters — but it does not have to exist all at once.
    Every mode routes through here so the pooled values cannot depend on which
    path ran.
    """
    reject_empty_token_dim(logits)
    documents, tokens, width = logits.shape
    if pooling == "max":
        # Exact and free: widening to float32 is injective, so the maximum of
        # the widened values is the widening of the maximum. No copy needed.
        return torch.amax(logits, dim=1)

    chunk = pool_chunk_tokens(documents, width)
    if pooling == "mean":
        return _ChunkedMean.apply(logits, chunk).to(logits.dtype)

    pooled = _ChunkedLogSumExp.apply(logits, chunk)
    if pooling == "logmeanexp":
        pooled = pooled - math.log(tokens)
    return pooled.to(logits.dtype)


class Model(torch.nn.Module):
    """Base model class implementing common functionality.

    This class provides the basic structure and utilities for all models:
    - Base transformer model initialization
    - Training loop with early stopping
    - Validation
    - Model saving/loading
    - Common layer setup (dropout, hidden layers)

    Attributes:
        config: Model configuration parameters
        base_model: Pre-trained transformer model
        tokenizer: Associated tokenizer
        device: Training device (CPU/GPU)
        best_score: Best validation score achieved
        best_model_state: State dict of best model
    """

    # Assigned in subclass __init__ / registered as buffers; annotated here so
    # nn.Module.__getattr__ doesn't collapse them to `Tensor | Module`.
    base_model: transformers.PreTrainedModel
    _neg_inf: Tensor
    classes: list[str]
    class_columns: Tensor
    # Plain attributes rather than buffers: they are per-epoch telemetry, and a
    # buffer would follow the parameters into every checkpoint.
    _grad_norm_sum: Tensor | None
    _grad_norm_clipped: Tensor | None

    def __init__(
        self,
        config: ModelConfig | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__()

        self.config = config if config is not None else ModelConfig()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = torch.amp.GradScaler(self.device)

        is_rocm = getattr(torch.version, "hip", None) is not None
        device_name = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
        )
        bf16_ok = (not is_rocm) and getattr(
            torch.cuda, "is_bf16_supported", lambda: False
        )()

        if is_rocm and not any(
            k in device_name for k in ("MI200", "MI250", "MI300", "MI3")
        ):
            bf16_ok = False
        self.amp_dtype = torch.bfloat16 if bf16_ok else torch.float16

        self.ramp_epochs: int = self.config.ramp_epochs
        self.entity_logits_pooling = self.config.entity_logits_pooling

        self.checkpoint = "checkpoint.pt"
        self.best_model_state: dict[str, Any] | None
        self.register_buffer("_neg_inf", torch.tensor(-1e9))
        self._reset_grad_norms()

    def _pool_logits(
        self,
        logits: Float[Tensor, "..."],
        dim: int = 1,
    ) -> Float[Tensor, "..."]:
        """Pool per-token logits to a document vector along `dim`.

        Selected by `entity_logits_pooling` (from `ModelConfig`):

        - ``logsumexp``: smooth-max — one strong token can carry the document;
          adds up to ``+log(T)`` for diffuse classes, so it is length-biased.
          The default: a single mention should suffice for detection.
        - ``logmeanexp``: ``logsumexp - log(T)``; length-invariant smooth-mean,
          but dilutes a lone mention in a long document.
        - ``max``: hard max; length-invariant.
        - ``mean``: arithmetic mean.

        Computed in float32, then cast back to the input dtype.

        The `[document, token, logits]` case — every call site in the models —
        is pooled a slice at a time by `pool_token_dim`, which never holds a
        float32 copy of the whole tensor. The general path below still serves
        any other shape or `dim`.
        """
        pooling = self.entity_logits_pooling
        reject_empty_token_dim(logits, dim)
        if logits.ndim == 3 and dim == 1:
            if pooling not in ("logsumexp", "logmeanexp", "max", "mean"):
                raise ValueError(f"Unknown pooling: {pooling}")
            return pool_token_dim(logits, pooling)

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

        The segmented counterpart of ``_pool_logits(rows, dim=0)`` — segment
        ``g`` gets exactly what pooling its own rows in isolation would give —
        in a fixed number of kernels instead of one launch per segment. Every
        segment must own at least one row.

        `counts` is the number of rows per segment; `logmeanexp` needs it
        because its divisor is the segment's own size, not the row count of
        `logits`.
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

    def register_class_columns(self) -> None:
        """Find the OOS column and remember the others. Call once `self.classes`
        is set.

        Non-persistent: derived from `self.classes`, so it must not enter the
        checkpoint (an older checkpoint would then be missing the key).
        """
        self.oos_index, class_columns = label_columns(self.classes, "OOS")
        self.register_buffer("class_columns", class_columns, persistent=False)

    def drop_oos(
        self, class_logits: Float[Tensor, "... class"]
    ) -> Float[Tensor, "... class"]:
        """Class logits without the OOS column, to the width of the targets."""
        return class_logits.index_select(-1, self.class_columns)

    @property
    def known_classes(self) -> list[str]:
        """Class names in column order, minus OOS: the columns `drop_oos` keeps,
        and so the labels the losses and the reports are computed over."""
        return [self.classes[column] for column in self.class_columns.tolist()]

    def _update(self, *losses: Float[Tensor, ""]) -> None:
        loss: Float[Tensor, ""] = torch.stack(losses).sum()

        if hasattr(self, "scaler"):
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            self._record_grad_norm(
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(), GRAD_CLIP_NORM
                )
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self._record_grad_norm(
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(), GRAD_CLIP_NORM
                )
            )
            self.optimizer.step()

    def _reset_grad_norms(self) -> None:
        """Drop the previous epoch's gradient-norm accumulators."""
        self._grad_norm_sum = None
        self._grad_norm_clipped = None
        self._grad_norm_steps = 0

    def _record_grad_norm(self, grad_norm: Tensor) -> None:
        """Accumulate one step's pre-clip gradient norm, without a sync.

        `clip_grad_norm_` returns the norm it measured *before* clipping, which
        is the only informative one — after clipping it is `GRAD_CLIP_NORM` by
        construction on every step that clipped at all. The sum is kept on the
        accelerator and read once per epoch: an `.item()` per optimizer step
        would serialise the training loop against the device.
        """
        norm = grad_norm.detach()
        clipped = (norm > GRAD_CLIP_NORM).to(norm.dtype)
        if self._grad_norm_sum is None or self._grad_norm_clipped is None:
            self._grad_norm_sum = norm.clone()
            self._grad_norm_clipped = clipped
        else:
            self._grad_norm_sum = self._grad_norm_sum + norm
            self._grad_norm_clipped = self._grad_norm_clipped + clipped
        self._grad_norm_steps += 1

    def _grad_norm_metrics(self) -> dict[str, float]:
        """The epoch's mean pre-clip gradient norm and its clipping rate.

        Empty when no optimizer step ran — a validation-only pass, or a model
        whose `run_epoch` never calls `_update` — so that nothing logs a
        gradient statistic for an epoch that computed no gradients. A clipping
        rate pinned at 1.0 is the signal that `GRAD_CLIP_NORM` is doing the
        optimising rather than the learning rate.
        """
        if not self._grad_norm_steps or self._grad_norm_sum is None:
            return {}

        steps = float(self._grad_norm_steps)
        metrics = {"training/grad_norm": self._grad_norm_sum.item() / steps}
        if self._grad_norm_clipped is not None:
            metrics["training/grad_clip_rate"] = (
                self._grad_norm_clipped.item() / steps
            )

        return metrics

    def epoch_loss_weights(self, epoch: int) -> dict[str, float]:
        """The multiplier applied to each named loss this epoch, if any.

        Keys match `run_epoch`'s `losses` dict, so a logged
        `loss_weight/relation` sits beside the `training/relation` it scaled —
        without which a loss curve that bends because the ramp moved is
        indistinguishable from one that bends because the model changed.
        Overridden by the subclasses that ramp: the schedule itself lives in
        `get_loss_weights`, whose second return value names a different
        objective in each of them.
        """
        return {}

    def get_loss_weights(
        self, epoch: int, w0: float = 0.1
    ) -> tuple[float, float]:
        """Compute weights for entity and relation given the epoch.

        :param epoch: current epoch index (0-based)
        :param w0: initial relation weight
        - epoch: current epoch index (0-based)
        - ramp_epochs: how many epochs to linearly ramp relation loss
        - w0: initial relation weight
        """
        if not self.ramp_epochs:
            return 1.0, 1.0
        t = min(1.0, epoch / float(self.ramp_epochs))
        w_rel = w0 + (1.0 - w0) * t  # ramps from w0 -> 1.0
        # w_ent = 1.0 - 0.3 * w_rel  # decays from 1.0 -> 0.7
        w_ent = 1.0
        return w_ent, w_rel

    def autocast_context(self, enabled=True):
        """Select the dtype for autocasting dynamically.

        The value of self.amp_dtype is a function of the support of the GPU
        for Bfloat16.
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
                    case _:
                        pass

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
        """Enable gradient checkpointing for all compatible modules."""
        if hasattr(self.base_model, "gradient_checkpointing_enable") and any(
            param.requires_grad for param in self.base_model.parameters()
        ):
            self.base_model.gradient_checkpointing_enable()

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

    def unfreeze_encoder_layers(self, n: int = 2):
        layers = sorted(
            {
                int(name.split("encoder.layer.")[1].split(".")[0])
                for name in self.base_model.state_dict()
                if "encoder.layer." in name
            }
        )
        start = max(0, len(layers) - n)
        target_layers = layers[start:]

        for name, param in self.base_model.named_parameters():
            if any(f"encoder.layer.{i}." in name for i in target_layers):
                param.requires_grad = True
                logger.info("Trainable: %s", name)

    @property
    def loss_fn(self) -> nn.Module:
        """Return the appropriate loss function for this model type"""
        raise NotImplementedError

    def compute_batch(
        self,
        batch: Any,
    ) -> float:
        """Compute loss for a batch and perform optimization step.
        Returns the loss value for this batch."""
        raise NotImplementedError

    def run_epoch(
        self, data: DataLoader, step: Step, epoch: int
    ) -> tuple[dict[str, float], int]:
        """Process all batches; implemented per model subclass."""
        raise NotImplementedError

    def _setup_training(
        self,
    ) -> tuple[
        torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler | None
    ]:
        """Setup optimizer and learning rate scheduler.

        Returns:
            Tuple of (optimizer, scheduler)
        """
        optimizer = optimizers[self.config.optimizer](
            self.parameters(), lr=self.config.lr
        )

        scheduler = None
        match self.config.lr_scheduler:
            case "exponential":
                scheduler = schedulers["exponential"](optimizer, gamma=0.95)
            case "reduce_on_plateau":
                scheduler = schedulers["reduce_on_plateau"](
                    optimizer, min_lr=0.0001, patience=2, factor=0.5
                )

        return optimizer, scheduler

    def train_model(
        self,
        train_data: DataLoader,
        val_data: DataLoader | None = None,
        save_checkpoint: bool = True,
        output_loss: bool = True,
    ) -> float | None:
        """Generic training loop for all models"""
        self.optimizer, self.scheduler = self._setup_training()

        self.stop_counter = 0
        self.best_model_state = None
        self.best_val_loss = float("inf")
        self.best_epoch = -1
        epochs_run = 0
        stopped_early = False

        for epoch in trange(
            self.config.num_epochs,
            dynamic_ncols=True,
            position=0,
            desc="Epochs",
            leave=True,
        ):
            self.train()
            self._reset_grad_norms()
            tracking.log_metrics(
                {
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    **{
                        f"loss_weight/{objective}": weight
                        for objective, weight in self.epoch_loss_weights(
                            epoch
                        ).items()
                    },
                },
                step=epoch,
            )
            started = time.perf_counter()
            losses, denominator = self.run_epoch(
                data=train_data, step=Step.TRAINING, epoch=epoch
            )
            train_seconds = time.perf_counter() - started
            epochs_run = epoch + 1

            tracking.log_metrics(
                {
                    **print_epoch_stats(
                        losses=losses,
                        denominator=denominator,
                        step=Step.TRAINING,
                    ),
                    **self._grad_norm_metrics(),
                    **epoch_rate_metrics(
                        batches=denominator,
                        seconds=train_seconds,
                        step=Step.TRAINING,
                    ),
                },
                step=epoch,
            )

            if val_data is not None:
                val_loss = self.validate_model(val_data=val_data, epoch=epoch)

                if self.scheduler is not None:
                    if self.config.lr_scheduler == "reduce_on_plateau":
                        # ReduceLROnPlateau.step takes the monitored metric, not
                        # an epoch; it is not an LRScheduler subclass.
                        cast(
                            torch.optim.lr_scheduler.ReduceLROnPlateau,
                            self.scheduler,
                        ).step(val_loss)
                    else:
                        self.scheduler.step()

                logger.info("Average validation loss: %.5f", val_loss)

                if epoch <= self.ramp_epochs:
                    self.stop_counter = 0
                early_stop = self.early_stop(
                    val_loss, epoch=epoch, save_checkpoint=save_checkpoint
                )
                tracking.log_metrics(
                    {
                        "early_stopping/epochs_without_improvement": float(
                            self.stop_counter
                        )
                    },
                    step=epoch,
                )
                if early_stop:
                    stopped_early = True
                    if save_checkpoint and self.best_model_state is not None:
                        logger.info(
                            "Model converged. Loading the best epoch's "
                            "parameters."
                        )
                        self.load_state_dict(self.best_model_state, strict=True)
                    break

            logger.info("-" * 50)

        if val_data is not None:
            # The summary the run list is scanned by. `epochs_after_best`
            # answers what `best_val_loss` alone cannot: a run that stopped
            # with several epochs since its best had converged, while one that
            # ended at its best was still improving when `num_epochs` ran out.
            tracking.log_metrics(
                {
                    "best_val_loss": self.best_val_loss,
                    "best_epoch": float(self.best_epoch),
                    "epochs_run": float(epochs_run),
                    "epochs_after_best": float(
                        epochs_run - 1 - self.best_epoch
                    ),
                    "stopped_early": float(stopped_early),
                }
            )
            if output_loss:
                return self.best_val_loss
        return None

    def early_stop(
        self, val_loss: float, epoch: int, save_checkpoint: bool
    ) -> bool:
        """Stop training after `self.config.patience` epochs have passed
        without improvement to `metric` according to the `goal`. Most likely
        we will want to minimize validation loss.

        If `save_checkpoint` is True, store the best model state in
        `self.best_model_state`.

        `epoch` is carried here rather than tracked in `train_model` so that
        the epoch and the loss it belongs to are written by the same
        comparison; two comparisons in two places is how `best_epoch` came to
        disagree with `best_val_loss` in the first place.
        """
        if val_loss <= self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self.stop_counter = 0
            if save_checkpoint:
                self.best_model_state = self._cpu_state_dict()
        else:
            self.stop_counter += 1

        if self.stop_counter > self.config.patience:
            return True
        else:
            return False

    def _cpu_state_dict(self) -> dict[str, Any]:
        """A detached CPU copy of the current parameters.

        `deepcopy(self.state_dict())` preserved each tensor's device, so on
        CUDA the best-epoch snapshot was a second resident copy of the whole
        model — the frozen base model included, 0.4 GiB of it — pinned for the
        rest of the run and briefly doubled at every improving epoch, since the
        new copy is built before the old one is dropped. Nothing ever reads it
        on-device: it is `torch.save`d, or loaded back once at convergence, and
        `load_state_dict` copies each tensor to its parameter's own device
        either way.

        `copy=True` is load-bearing, and only on CPU runs: `.to("cpu")` on a
        tensor already there returns *self*, which would leave the snapshot
        aliasing the live parameters and tracking them as training continued.
        """
        return {
            key: (
                value.detach().to("cpu", copy=True)
                if isinstance(value, Tensor)
                else deepcopy(value)
            )
            for key, value in self.state_dict().items()
        }

    def save_model(self, path: str) -> None:
        try:
            torch.save(self.best_model_state, path)
        except NameError:
            logger.warning("The model has not been trained yet...")

    def validate_model(
        self,
        val_data: DataLoader,  # , w_ent: float = 1.0, w_rel: float = 1.0
        epoch: int,
    ) -> float:
        self.eval()
        started = time.perf_counter()
        losses, denominator = self.run_epoch(
            data=val_data, step=Step.VALIDATION, epoch=epoch
        )
        seconds = time.perf_counter() - started

        tracking.log_metrics(
            {
                **print_epoch_stats(
                    losses=losses,
                    denominator=denominator,
                    step=Step.VALIDATION,
                ),
                **epoch_rate_metrics(
                    batches=denominator,
                    seconds=seconds,
                    step=Step.VALIDATION,
                ),
            },
            step=epoch,
        )

        return sum(losses.values()) / denominator

    def save_config(self, path: str) -> None:
        save_model_config(self.config.model_dump(), path)

    def batch_input_tensors(
        self,
        batch: Sequence[BatchItem],
    ) -> dict[str, Integer[Tensor, "sequence token"]]:
        """Concatenate each document's chunk sequences into a single
        ``[sum(n_chunks), token]`` tensor per key.

        Every dimension but the last is flattened away, because the same item
        reaches here under two shapes: ``BrendaDataset[[...]]`` yields a 2-D
        ``[n_chunks, token]``, while the `DataLoader` collates that through
        `default_collate` and hands over a 3-D ``[1, n_chunks, token]`` — the
        leading 1 is an artefact of batching a one-element list, not a
        document axis. Concatenating the 3-D form on dim 0 stacks documents
        along the *chunk* axis instead of extending it, and raises as soon as
        two documents differ in chunk count, which is every real batch.

        Flattening to a single row per chunk (rather than one row per token)
        is what ``get_token_embeddings`` unpacks back into documents, via
        ``doc_id.shape[-1]``.
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
        """Get token embeddings for a batch with caching support."""
        inputs: list[None | Tensor] = [None] * len(batch)
        missing: list[tuple[int, BatchItem]] = []

        for ix, item in enumerate(batch):
            doc_id: int = int(item["id"].item())
            if cpu_embeddings_cache is not None:
                cpu_cached = cpu_embeddings_cache.get(doc_id)
                if cpu_cached is not None:
                    inputs[ix] = cpu_cached
                else:
                    missing.append((ix, item))
            else:
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
                    cpu_embeddings_cache.set(item["id"].item(), doc_embedding)

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

    Returning what it prints is the point: `train_model` logs this dict to
    MLflow rather than re-deriving the averages, so the console and the
    tracking server cannot disagree about an epoch's numbers.
    """
    for obj, loss in losses.items():
        logger.info("Average (%s) %s loss: %.4f", obj, step, loss / denominator)

    total_loss = sum(losses.values())
    logger.info("Average %s loss: %.4f", step, total_loss / denominator)

    return {
        f"{step}/{obj}": value / denominator
        for obj, value in {**losses, "total": total_loss}.items()
    }


def epoch_rate_metrics(
    batches: int, seconds: float, step: Step
) -> dict[str, float]:
    """How long the epoch took, and how fast it went, keyed for tracking.

    Wall-clock is what makes two runs' loss curves comparable as *choices*:
    a configuration that reaches the same validation loss in half the epochs
    has not won anything if each of its epochs costs twice as much. Rate is in
    batches rather than documents because `TokenBudgetBatchSampler` makes the
    document count per batch a function of document length, so `run_epoch`
    counts batches and nothing downstream knows better.
    """
    metrics = {f"{step}/seconds": seconds}
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
    the majority class and the one nobody asked about; what ranks runs is the
    score over the typed labels alone. `none_share` is logged beside it because
    the candidate set is proposed by the *current entity head* rather than by
    the corpus — the same checkpoint can face a different pair distribution
    from one run to the next, and this is the only record of which one it met.
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
    nothing at all and a head predicting the wrong labels score identically,
    and only the predicted-positive count separates them. `labels_predicted`
    counts the *columns* ever used rather than the positives, which is how a
    head collapsed onto one frequent label shows up.
    """
    metrics: dict[str, float] = {}
    for task, (true, pred) in tasks.items():
        metrics[f"test/{task}_gold_positives"] = float(true.sum())
        metrics[f"test/{task}_predicted_positives"] = float(pred.sum())
        metrics[f"test/{task}_labels_predicted"] = float(
            (pred.sum(axis=0) > 0).sum()
        )

    return metrics


class PermutationBatchNorm1d(nn.BatchNorm1d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = torch.permute(input, (0, 2, 1))
        out = torch.permute(super().forward(input), (0, 2, 1))
        return out


class BrendaClassificationModel(Model):
    # Registered buffers; annotated so access resolves to Tensor, not Module.
    class_matrix: Tensor
    entity_pos_weight: Tensor
    class_pos_weight: Tensor
    entity_columns: Tensor

    def __init__(
        self,
        classes: Mapping[str, set[str]],
        class_matrix: Float[Tensor, "entity class"],
        entity_index: dict[str, int],
        config: None | ModelConfig = None,
        entity_freqs: Float[Tensor, " entities"] | None = None,
        class_freqs: Float[Tensor, " classes"] | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__(config, device=device)
        self.classes = list(classes.keys()) + ["OOS"]

        # Derived from `entity_index`, not from `classes`, so that
        # `entities[i]` is always the entity scored by entity logit column `i`.
        # Flattening `classes.values()` only yields that order while the
        # per-class entity sets stay disjoint.
        self.entities = ordered_entities(entity_index) + ["UNK"]

        # The dataset does not include a `none` class, so we add one.
        self.num_of_entities = len(self.entities)
        self.num_of_classes = len(self.classes)

        self.register_entity_columns()
        self.register_class_columns()

        self.build_layers(embedding_size=embedding_dims[self.config.base_model])

        self.base_model = load_base_model(self.config.base_model)

        for param in self.base_model.parameters():
            param.requires_grad = False

        self.enable_gradient_checkpointing()

        # Initialize class matrix mapping each entity index to its entity
        # class index.
        self.entity_to_index = entity_index
        self.register_buffer("class_matrix", class_matrix)

        if entity_freqs is not None:
            entity_pos_w = (
                (1 - entity_freqs).clamp(1e-5, 1 - 1e-5)
                / entity_freqs.clamp(1e-5, 1 - 1e-5)
            ).clamp(max=50.0)
        else:
            entity_pos_w = torch.ones(len(entity_index))
        if class_freqs is not None:
            class_pos_w = (
                (1 - class_freqs).clamp(1e-5, 1 - 1e-5)
                / class_freqs.clamp(1e-5, 1 - 1e-5)
            ).clamp(max=20.0)
        else:
            class_pos_w = torch.ones(len(classes))

        self.register_buffer("entity_pos_weight", entity_pos_w)
        self.register_buffer("class_pos_weight", class_pos_w)

        self.classifier = ClassificationHead(
            input_size=self.hidden_block_output_size,
            n_entities=self.num_of_entities,
            n_classes=self.num_of_classes,
            entity_freqs=entity_freqs,
            class_freqs=class_freqs,
            unk_index=self.unk_index,
            oos_index=self.oos_index,
        )

        self.entity_threshold = self.config.entity_entropy_threshold
        self.consistency_weight = getattr(
            self.config, "consistency_weight", 0.1
        )
        self.evaluation = False

    def register_entity_columns(self) -> None:
        """Find the UNK column and remember the others. Call once
        `self.entities` is set.

        Non-persistent: derived from `self.entities`, so it must not enter the
        checkpoint (an older checkpoint would then be missing the key).
        """
        self.unk_index, entity_columns = label_columns(self.entities, "UNK")
        self.register_buffer("entity_columns", entity_columns, persistent=False)

    def drop_unk(
        self, entity_logits: Float[Tensor, "... entity"]
    ) -> Float[Tensor, "... entity"]:
        """Entity logits without the UNK column, to the width of the targets."""
        return entity_logits.index_select(-1, self.entity_columns)

    @property
    def known_entities(self) -> list[str]:
        """Entity names in column order, minus UNK: the columns `drop_unk`
        keeps, aligned with `entity_index` and with `class_matrix`'s rows."""
        return [
            self.entities[column] for column in self.entity_columns.tolist()
        ]

    def _consistency_loss(
        self, entity_logits: torch.Tensor, class_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Penalize cases where an entity is predicted but the class head
        does not agree with that entity's class.

        Uses only the 'proper' columns: drops UNK (entity) and OOS (class),
        leveraging self.class_matrix [E-1, C-1].
        """
        if self.consistency_weight <= 0:
            return torch.tensor(
                0.0, device=entity_logits.device, dtype=entity_logits.dtype
            )

        with torch.autocast(device_type=self.device, enabled=False):
            # probabilities in fp32 for stable reductions
            pe = torch.sigmoid(self.drop_unk(entity_logits)).float()
            pc = torch.sigmoid(self.drop_oos(class_logits)).float()

            # pick, for each entity row, its class probability from class head:
            # pc_for_entity: [B, E-1] where each column i = pc[:, class_of_entity_i]
            # class_matrix: [E-1, C-1]; do a gather via matmul because rows are one-hot
            pc_for_entity = pc @ self.class_matrix.T  # [B, E-1]

            penalty = pe * (1.0 - pc_for_entity)  # [B, E-1]

            # average over batch and entities (avoid NaNs)
            cons = penalty.mean()

        return cons.to(entity_logits.dtype)

    def run_epoch(
        self, data: DataLoader, step: Step, epoch: int
    ) -> tuple[dict[str, float], int]:
        """Process all batches, computing loss and printing diagnostics.

        :param epoch: epoch number
        :param train_data: DataLoader for the training data
        :returns: combined losses for epoch and the denominator for loss
            averaging.
        """
        epoch_ent_loss = 0.0
        epoch_class_loss = 0.0
        n_batches = 0

        w_ent, w_class = self.get_loss_weights(epoch)

        for batch in batch_progress(data):
            if step == Step.TRAINING:
                self.optimizer.zero_grad(set_to_none=True)

            ent_loss, class_loss = self.compute_batch_losses(batch)
            n_batches += 1

            ent_loss_scaled = ent_loss * w_ent
            class_loss_scaled = class_loss * w_class

            if step == Step.TRAINING:
                self._update(ent_loss_scaled, class_loss_scaled)

            epoch_ent_loss += ent_loss_scaled.detach().cpu().item()
            epoch_class_loss += class_loss_scaled.detach().cpu().item()
            del ent_loss, class_loss, ent_loss_scaled, class_loss_scaled

        losses = {
            "entity": epoch_ent_loss,
            "class": epoch_class_loss,
        }

        return losses, n_batches

    def epoch_loss_weights(self, epoch: int) -> dict[str, float]:
        w_ent, w_class = self.get_loss_weights(epoch)
        return {"entity": w_ent, "class": w_class}

    @property
    def entity_loss_fn(self) -> nn.Module:
        # weights = torch.ones(self.num_of_entities - 1, device=self.device)
        return nn.BCEWithLogitsLoss(
            reduction="mean", pos_weight=self.entity_pos_weight
        )

    @property
    def class_loss_fn(self) -> nn.Module:
        # weights = torch.ones(self.num_of_classes - 1, device=self.device)
        # weights[-1] = 0
        return nn.BCEWithLogitsLoss(
            reduction="mean", pos_weight=self.class_pos_weight
        )

    def compute_entity_loss(
        self,
        predictions: tuple[Tensor, Tensor],
        targets: tuple[Tensor, Tensor],
        class_scale: float = 1,
    ) -> tuple[Float[Tensor, ""], Float[Tensor, ""]]:
        entity_loss = self.entity_loss_fn(
            self.drop_unk(predictions[0]).float(),
            targets[0].float(),
        )
        class_loss = self.class_loss_fn(
            self.drop_oos(predictions[1]).float(),
            targets[1].float(),
        )

        cons = self._consistency_loss(predictions[0], predictions[1])
        class_loss = class_loss + self.consistency_weight * cons

        return entity_loss, class_loss

    def compute_batch_losses(
        self, batch: Sequence[BatchItem]
    ) -> tuple[Float[Tensor, ""], Float[Tensor, ""]]:
        ent_true, class_true = self.ground_truth(batch)
        entity_logits, class_logits = self.get_batch_logits(batch)

        return self.compute_entity_loss(
            predictions=(entity_logits, class_logits),
            targets=(ent_true, class_true),
        )

    def get_batch_logits(
        self,
        batch: Sequence[BatchItem],
        gold_relations: list[IndexedRelation] | None = None,
    ) -> tuple[
        Float[Tensor, "sequence entities"],
        Float[Tensor, "sequence classes"],
    ]:
        token_embeddings, token_att_mask = self.get_token_embeddings(batch)
        token_embeddings = token_embeddings.to(self.device, non_blocking=True)
        token_att_mask = token_att_mask.to(self.device, non_blocking=True)

        entity_logits, class_logits = self(
            token_embeddings,
            token_att_mask,
        )

        return (
            entity_logits,
            class_logits,
        )

    def ground_truth(
        self,
        batch: Sequence[BatchItem],
    ) -> tuple[
        Float[Tensor, "batch entities"],
        Float[Tensor, "batch classes"],
    ]:
        """Get ground truth for each document in the batch

        :param: Batch of documents.
        :return: Tuple containing:
            - Multi-hot encoded tensor, where each position of dim 2
              specifies whether the entity corresponding to that index occurs in
              the particular document along dim 1.
            - Idem for class labels
        """
        entity_targets = torch.concat(
            tuple(doc["entities"] for doc in batch)
        ).to(self.device)

        class_targets = torch.concat(tuple(doc["classes"] for doc in batch)).to(
            self.device
        )

        return entity_targets.float(), class_targets.float()

    def evaluate_model(
        self, test_data: DataLoader, tau_ids: float = 0.5, tau_cls: float = 0.5
    ) -> dict[str, float]:
        """Document-level multilabel evaluation for entity IDs and classes.

        Returns what it prints, and logs the same dict to the active tracking
        run — the `print_epoch_stats` contract, for the same reason: a number
        computed twice is a number that can disagree with itself. An empty dict
        means the split produced no samples.
        """
        self.eval()
        metrics: dict[str, float] = {}
        all_id_logits, all_id_true = [], []
        all_cls_logits, all_cls_true = [], []

        with torch.no_grad():
            for batch in batch_progress(
                test_data, desc="Evaluating", position=0, leave=True
            ):
                id_logits_doc, cls_logits_doc = self.get_batch_logits(batch)
                id_true_doc, cls_true_doc = self.ground_truth(batch)

                # logits, narrowed to the columns the targets carry
                all_id_logits.append(
                    self.drop_unk(id_logits_doc).detach().float().cpu()
                )
                all_cls_logits.append(
                    self.drop_oos(cls_logits_doc).detach().float().cpu()
                )

                # TRUE LABELS (fix the bug: append *_true, not logits)
                all_id_true.append(id_true_doc.detach().to(torch.int64).cpu())
                all_cls_true.append(cls_true_doc.detach().to(torch.int64).cpu())

        if not all_id_logits:
            logger.warning("No samples found.")
            return metrics

        # concat
        id_logits = torch.cat(all_id_logits, dim=0).numpy()
        id_true = torch.cat(all_id_true, dim=0).numpy().astype(int)

        cls_logits = torch.cat(all_cls_logits, dim=0).numpy()
        cls_true = torch.cat(all_cls_true, dim=0).numpy().astype(int)

        # probabilities
        id_probs = 1.0 / (1.0 + np.exp(-id_logits))
        cls_probs = 1.0 / (1.0 + np.exp(-cls_logits))

        # binarize for F1 / report
        id_pred = (id_probs >= tau_ids).astype(int)
        cls_pred = (cls_probs >= tau_cls).astype(int)

        # ======= METRICS =======

        metrics.update(
            support_metrics(
                {"entity": (id_true, id_pred), "class": (cls_true, cls_pred)}
            )
        )

        logger.info("\n=== Entity ID metrics (multilabel, document-level) ===")
        try:
            metrics["test/entity_micro_f1"] = f1_score(
                id_true, id_pred, average="micro", zero_division=0
            )
            logger.info("micro-F1: %s", metrics["test/entity_micro_f1"])
        except ValueError:
            logger.info("micro-F1: (no positives or predictions) 0.0")

        # Probability-aware multilabel metrics (no threshold)
        try:
            metrics["test/entity_lrap"] = label_ranking_average_precision_score(
                id_true, id_probs
            )
            logger.info("LRAP: %s", metrics["test/entity_lrap"])
            metrics["test/entity_micro_ap"] = average_precision_score(
                id_true, id_probs, average="micro"
            )
            logger.info("micro-AP: %s", metrics["test/entity_micro_ap"])
        except ValueError:
            logger.info("LRAP / micro-AP: undefined (no positives)")

        # macro-F1 over frequent IDs only
        support = id_true.sum(axis=0)
        keep = np.where(support >= 10)[0]
        if keep.size > 0:
            metrics["test/entity_macro_f1_support10"] = f1_score(
                id_true[:, keep],
                id_pred[:, keep],
                average="macro",
                zero_division=0,
            )
            logger.info(
                "macro-F1 (support>=10): %s",
                metrics["test/entity_macro_f1_support10"],
            )
        else:
            logger.info(
                "macro-F1 (support>=10): n/a (no labels meet support threshold)"
            )

        logger.info(
            "\n=== Entity CLASS metrics (multilabel, document-level) ==="
        )
        metrics["test/class_micro_f1"] = f1_score(
            cls_true, cls_pred, average="micro", zero_division=0
        )
        logger.info("micro-F1: %s", metrics["test/class_micro_f1"])
        metrics["test/class_micro_ap"] = average_precision_score(
            cls_true, cls_probs, average="micro"
        )
        logger.info("micro-AP: %s", metrics["test/class_micro_ap"])
        report = classification_report(
            y_true=cls_true,
            y_pred=cls_pred,  # <- must be binary indicators
            target_names=self.known_classes,
            zero_division=0,
        )
        logger.info(report)
        tracking.log_text(str(report), "test/class_report.txt")

        tracking.log_metrics(metrics)

        return metrics

    @record_function("forward")
    def forward(
        self,
        embeddings: Float[Tensor, "document token embedding"],
        attention_mask: Bool[Tensor, "document token"],
    ) -> tuple[
        BatchedLogits,
        BatchedLogits,
    ]:
        """Forward pass

        :return: tuple containing:
            - Entity logits pooled by document.
            - Class logits pooled by document.
        """
        with self.autocast_context():
            hidden_output: Float[Tensor, "document token features"] = (
                self.hidden(embeddings)
            )
            unmasked_entity_logits, unmasked_class_logits = self.classifier(
                hidden_output
            )
            token_mask = attention_mask.unsqueeze(-1)
            entity_logits = torch.where(
                token_mask, unmasked_entity_logits, self._neg_inf
            )
            class_logits = torch.where(
                token_mask, unmasked_class_logits, self._neg_inf
            )

            return (
                self._pool_logits(entity_logits),
                self._pool_logits(class_logits),
            )


class NERClassificationModel(Model):
    """Simplified model for Named Entity Recognition (NER) without entity linking.

    This model predicts entity classes/types for each token in a document,
    aggregating predictions at the document level. Unlike BrendaClassificationModel,
    it does not perform entity linking (mapping to specific entity IDs).
    """

    # Registered buffer; annotated so access resolves to Tensor, not Module.
    class_pos_weight: Tensor

    def __init__(
        self,
        classes: Mapping[str, set[str]],
        config: None | ModelConfig = None,
        class_freqs: Float[Tensor, " classes"] | None = None,
        # Accept but ignore entity-linking arguments for compatibility
        class_matrix: Float[Tensor, "entity class"] | None = None,
        entity_index: dict[str, int] | None = None,
        entity_freqs: Float[Tensor, " entities"] | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__(config, device=device)

        # Add "OOS" (out-of-scope) class for tokens that don't belong to any entity class
        self.classes = list(classes.keys()) + ["OOS"]
        self.num_of_classes = len(self.classes)

        self.register_class_columns()

        # Build hidden layers
        self.build_layers(embedding_size=embedding_dims[self.config.base_model])

        # Initialize transformer base model
        self.base_model = load_base_model(self.config.base_model)

        # Freeze base model parameters initially
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.enable_gradient_checkpointing()

        # Setup class weights for handling imbalanced data
        if class_freqs is not None:
            class_pos_w = (
                (1 - class_freqs).clamp(1e-5, 1 - 1e-5)
                / class_freqs.clamp(1e-5, 1 - 1e-5)
            ).clamp(max=20.0)
        else:
            class_pos_w = torch.ones(len(classes))

        self.register_buffer("class_pos_weight", class_pos_w)

        # Simple classification head
        self.classifier = nn.Sequential(
            nn.Linear(
                in_features=self.hidden_block_output_size,
                out_features=self.hidden_block_output_size,
                bias=True,
            ),
            nn.GELU(),
            nn.Dropout(self.config.dropout)
            if self.config.dropout
            else nn.Identity(),
            nn.Linear(self.hidden_block_output_size, self.num_of_classes),
        )

        # Initialize classifier bias if frequencies provided
        if class_freqs is not None:
            initialize_classifier_bias(
                linear=cast(nn.Linear, self.classifier[-1]),
                freqs=class_freqs,
                sentinel_index=self.oos_index,
                sentinel_prior=0.9,
            )

    @property
    def class_loss_fn(self) -> nn.Module:
        """Binary cross-entropy loss for multilabel classification."""
        return nn.BCEWithLogitsLoss(
            reduction="mean", pos_weight=self.class_pos_weight
        )

    def run_epoch(
        self, data: DataLoader, step: Step, epoch: int
    ) -> tuple[dict[str, float], int]:
        """Process all batches, computing loss and printing diagnostics.

        :param epoch: epoch number
        :param data: DataLoader for the data
        :param step: training, validation, or testing step
        :returns: losses for epoch and the denominator for loss averaging
        """
        epoch_class_loss = 0.0
        n_batches = 0

        for batch in batch_progress(data):
            if step == Step.TRAINING:
                self.optimizer.zero_grad(set_to_none=True)

            class_loss = self.compute_batch_losses(batch)
            n_batches += 1

            if step == Step.TRAINING:
                self._update(class_loss)

            epoch_class_loss += class_loss.detach().cpu().item()
            del class_loss

        losses = {"class": epoch_class_loss}

        return losses, n_batches

    def compute_batch_losses(
        self, batch: Sequence[BatchItem]
    ) -> Float[Tensor, ""]:
        """Compute loss for a batch."""
        class_true = self.ground_truth(batch)
        class_logits = self.get_batch_logits(batch)

        class_loss = self.class_loss_fn(
            self.drop_oos(class_logits).float(),
            class_true.float(),
        )

        return class_loss

    def get_batch_logits(
        self,
        batch: Sequence[BatchItem],
    ) -> Float[Tensor, "sequence classes"]:
        """Get class logits for a batch."""
        token_embeddings, token_att_mask = self.get_token_embeddings(batch)
        token_embeddings = token_embeddings.to(self.device, non_blocking=True)
        token_att_mask = token_att_mask.to(self.device, non_blocking=True)

        class_logits = self(token_embeddings, token_att_mask)

        return class_logits

    def ground_truth(
        self,
        batch: Sequence[BatchItem],
    ) -> Float[Tensor, "batch classes"]:
        """Get ground truth class labels for each document in the batch.

        :param batch: Batch of documents.
        :return: Multi-hot encoded tensor, where each position specifies
                 whether the class corresponding to that index occurs in
                 the particular document.
        """
        class_targets = torch.concat(tuple(doc["classes"] for doc in batch)).to(
            self.device
        )

        return class_targets.float()

    @record_function("forward")
    def forward(
        self,
        embeddings: Float[Tensor, "document token embedding"],
        attention_mask: Bool[Tensor, "document token"],
    ) -> BatchedLogits:
        """Forward pass for NER classification.

        :param embeddings: Token embeddings from base model
        :param attention_mask: Attention mask for valid tokens
        :return: Class logits pooled by document
        """
        with self.autocast_context():
            # Pass through hidden layers
            hidden_output: Float[Tensor, "document token features"] = (
                self.hidden(embeddings)
            )

            # Get class logits
            unmasked_class_logits = self.classifier(hidden_output)

            # Mask invalid positions
            token_mask = attention_mask.unsqueeze(-1)
            class_logits = torch.where(
                token_mask, unmasked_class_logits, self._neg_inf
            )

            return self._pool_logits(class_logits)

    def evaluate_model(
        self, test_data: DataLoader, tau_cls: float = 0.5
    ) -> dict[str, float]:
        """Document-level multilabel evaluation for entity classes.

        Returns what it prints and logs the same dict to the active tracking
        run; an empty dict means the split produced no samples.
        """
        self.eval()
        metrics: dict[str, float] = {}
        all_cls_logits, all_cls_true = [], []

        with torch.no_grad():
            for batch in batch_progress(
                test_data, desc="Evaluating", position=0, leave=True
            ):
                cls_logits_doc = self.get_batch_logits(batch)
                cls_true_doc = self.ground_truth(batch)

                # logits
                all_cls_logits.append(cls_logits_doc.detach().float().cpu())

                # TRUE LABELS
                all_cls_true.append(cls_true_doc.detach().to(torch.int64).cpu())

        if not all_cls_logits:
            logger.warning("No samples found.")
            return metrics

        # concat
        cls_logits = torch.cat(all_cls_logits, dim=0).numpy()
        cls_true = torch.cat(all_cls_true, dim=0).numpy().astype(int)

        if cls_logits.shape[1] != cls_true.shape[1]:
            cls_logits = cls_logits[:, : cls_true.shape[1]]

        # probabilities
        cls_probs = 1.0 / (1.0 + np.exp(-cls_logits))

        # binarize for F1 / report
        cls_pred = (cls_probs >= tau_cls).astype(int)

        # ======= METRICS =======

        metrics.update(support_metrics({"class": (cls_true, cls_pred)}))

        logger.info(
            "\n=== Entity CLASS metrics (multilabel, document-level) ==="
        )
        metrics["test/class_micro_f1"] = f1_score(
            cls_true, cls_pred, average="micro", zero_division=0
        )
        logger.info("micro-F1: %s", metrics["test/class_micro_f1"])
        metrics["test/class_micro_ap"] = average_precision_score(
            cls_true, cls_probs, average="micro"
        )
        logger.info("micro-AP: %s", metrics["test/class_micro_ap"])
        report = classification_report(
            y_true=cls_true,
            y_pred=cls_pred,
            target_names=self.known_classes,
            zero_division=0,
        )
        logger.info(report)
        tracking.log_text(str(report), "test/class_report.txt")

        tracking.log_metrics(metrics)

        return metrics


class BiaffineRelationClassifier(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_relations: int,
        separate_predicate_layer: bool = False,
        biaff_hidden_size: int = 32,
    ):
        super().__init__()
        self.separate_predicate_layer = separate_predicate_layer
        self.hidden_linear = nn.Sequential(
            nn.Linear(
                in_features=hidden_size,
                out_features=biaff_hidden_size,
                bias=True,
            ),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        if separate_predicate_layer:
            self.hidden_linear_y = nn.Sequential(
                nn.Linear(
                    in_features=hidden_size,
                    out_features=biaff_hidden_size,
                    bias=True,
                ),
                nn.GELU(),
                nn.Dropout(0.1),
            )
        else:
            self.hidden_linear_y = self.hidden_linear

        self.bilinear = nn.Parameter(
            torch.randn(num_relations, biaff_hidden_size, biaff_hidden_size)
        )
        nn.init.xavier_uniform_(self.bilinear)
        self.linear = nn.Linear(biaff_hidden_size * 2, num_relations)
        self.bias = nn.Parameter(torch.zeros(num_relations))

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        # x, y: [B, D]
        x = self.hidden_linear(x)
        y = self.hidden_linear_y(y)
        bilinear_term = torch.einsum(
            "bi,rid,bj->br", x, self.bilinear, y
        )  # [B, R]
        linear_term = self.linear(torch.cat([x, y], dim=-1))  # [B, R]
        return bilinear_term + linear_term + self.bias


class ETEBrendaModel(
    BrendaClassificationModel,
):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.relations = ("HasEnzyme", "HasSpecies", "none")
        self.relations_none_index = self.relations.index("none")
        self.num_relations = len(self.relations)
        self.relation_classifier = BiaffineRelationClassifier(
            hidden_size=self.hidden_block_output_size,
            num_relations=len(self.relations),
            biaff_hidden_size=self.config.biaffine_hidden_size,
        )

        self.relation_label_smoothing = self.config.relation_label_smoothing
        self.relation_loss_weighting = self.config.relation_loss_weighting
        self.relation_focal_gamma = self.config.relation_focal_gamma

    def run_epoch(
        self,
        data: DataLoader,
        step: Step,
        epoch: int,
    ) -> tuple[dict[str, float], int]:
        """Process all batches, computing loss and printing diagnostics.

        :param epoch: epoch number
        :param train_data: DataLoader for the training data
        :returns: combined loss for epoch
        """
        epoch_ent_loss = 0.0
        epoch_class_loss = 0.0
        epoch_rel_loss = 0.0
        n_batches = 0
        w_ent, w_rel = self.get_loss_weights(epoch)

        for batch in batch_progress(data):
            if step == Step.TRAINING:
                self.optimizer.zero_grad(set_to_none=True)

            if n_batches == 0:
                logger.info(
                    "Epoch %d: w_ent=%.3f, w_rel=%.3f", epoch, w_ent, w_rel
                )

            ent_loss, class_loss, rel_loss = self.compute_batch_losses(batch)

            ent_loss_scaled = ent_loss * w_ent
            class_loss_scaled = class_loss * w_ent
            rel_loss_scaled = rel_loss * w_rel

            if step == Step.TRAINING:
                self._update(
                    ent_loss_scaled, class_loss_scaled, rel_loss_scaled
                )

            epoch_ent_loss += ent_loss_scaled.detach().cpu().item()
            epoch_class_loss += class_loss_scaled.detach().cpu().item()
            epoch_rel_loss += rel_loss_scaled.detach().cpu().item()
            n_batches += 1

            del (
                rel_loss_scaled,
                ent_loss_scaled,
                class_loss_scaled,
                rel_loss,
                ent_loss,
                class_loss,
            )

        losses = {
            "entity": epoch_ent_loss,
            "class": epoch_class_loss,
            "relation": epoch_rel_loss,
        }

        return losses, n_batches

    def epoch_loss_weights(self, epoch: int) -> dict[str, float]:
        # `run_epoch` scales the class loss by the *entity* weight here; the
        # pair's second element is the relation ramp, not a class weight.
        w_ent, w_rel = self.get_loss_weights(epoch)
        return {"entity": w_ent, "class": w_ent, "relation": w_rel}

    def ground_truth(
        self,
        batch: Sequence[BatchItem],
    ) -> tuple[
        Float[Tensor, "batch entities"],
        Float[Tensor, "batch classes"],
        list[IndexedRelation],
    ]:
        """Get ground truth for each document in the batch

        :param: Batch of documents.
        :return: Tuple containing:
            - Multi-hot encoded tensor, where each position of dim 2
              specifies whether the entity corresponding to that index occurs in
              the particular document along dim 1.
            - Idem for class labels
            - List of relations indexed to document identifiers
        """
        entity_targets, class_targets = super().ground_truth(batch)

        relation_targets = []
        for docix, doc in enumerate(batch):
            try:
                doc_relations = doc.get("relations", [{}])[0]
            except IndexError:
                continue

            for args, label in doc_relations.items():
                relation_targets.append(
                    IndexedRelation(
                        docix=docix,
                        subject=args[0],
                        object=args[1],
                        label=label.argmax(),
                    )
                )

        return entity_targets, class_targets, relation_targets

    def align_relation_predictions(
        self,
        true_relations: Sequence[IndexedRelation],
        rel_meta: dict[str, Tensor],
        rel_logits: Float[Tensor, "relation logits"] | None,
    ) -> (
        tuple[
            dict[str, Tensor],
            Float[Tensor, "relation logits"],
            Int64[Tensor, " relation"],
        ]
        | None
    ):
        if rel_logits is None or rel_logits.numel() == 0:
            return None

        device = rel_logits.device
        seq, subj, obj = (
            rel_meta[key].detach().to(device=device, dtype=torch.long)
            for key in ("sequence", "arg_pred_i", "arg_pred_j")
        )

        n_rows = rel_logits.size(0)
        assert (
            seq.numel() == n_rows
            and subj.numel() == n_rows
            and obj.numel() == n_rows
        ), "rel_meta fields must align with rel_logits rows"

        none_idx = int(self.relations_none_index)

        # The gold side is Python data — a Sequence of NamedTuples keyed by
        # entity *strings* — so its lookup is built host-side, as before. Only
        # the join against the candidate triples runs on the device.
        gold_by_key: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for tr in true_relations:
            try:
                subj_ix = int(self.entity_to_index[tr.subject])
                obj_ix = int(self.entity_to_index[tr.object])
            except KeyError:
                continue  # gold refers to entity not mapped in this doc/batch
            gold_by_key[(int(tr.docix), subj_ix, obj_ix)].append(int(tr.label))

        gold_triples: list[tuple[int, int, int]] = []
        gold_labels: list[int] = []
        for key, labels in gold_by_key.items():
            # If multiple labels exist, prefer any non-none; else first.
            # (Adjust policy if your schema allows multi-label relations.)
            gold_triples.append(key)
            gold_labels.append(
                next((lbl for lbl in labels if lbl != none_idx), labels[0])
            )
        gold_index = torch.tensor(
            gold_triples, dtype=torch.long, device=device
        ).reshape(-1, 3)

        # Pack (sequence, subject, object) into one int64 so that grouping is a
        # single `torch.unique` and the gold join a single `searchsorted`.
        # The radices are read off the data instead of being fixed bit widths:
        # the argument indices are argmaxes over the whole entity vocabulary,
        # whose size is a property of the dataset, not of this function. Their
        # product is bounded by batch x |entities|^2 and stays far inside int64.
        def _radix(candidate: Tensor, gold: Tensor) -> Tensor:
            """One past the largest index either side of the join uses."""
            highest = candidate.max()
            if gold.numel():  # shape metadata, not a device read
                highest = torch.maximum(highest, gold.max())
            return highest + 1

        radix_i = _radix(subj, gold_index[:, 1])
        radix_j = _radix(obj, gold_index[:, 2])

        def _pack(s: Tensor, i: Tensor, j: Tensor) -> Tensor:
            return (s * radix_i + i) * radix_j + j

        keys = _pack(seq, subj, obj)
        unique_keys, inverse, counts = torch.unique(
            keys, return_inverse=True, return_counts=True
        )
        n_groups = int(unique_keys.numel())

        pooled_logits = self._pool_logits_segments(
            rel_logits, inverse, n_groups, counts
        )

        # One scratch slot past the groups absorbs gold triples that no
        # candidate pair proposed; masking them out instead would need a
        # boolean index, whose data-dependent shape is itself a device sync.
        targets = torch.full(
            (n_groups + 1,), none_idx, dtype=torch.long, device=device
        )
        if gold_labels:
            gold_keys = _pack(
                gold_index[:, 0], gold_index[:, 1], gold_index[:, 2]
            )
            slot = torch.searchsorted(unique_keys, gold_keys).clamp(
                max=n_groups - 1
            )
            slot = torch.where(unique_keys[slot] == gold_keys, slot, n_groups)
            targets = targets.scatter(
                0,
                slot,
                torch.tensor(gold_labels, dtype=torch.long, device=device),
            )
        targets = targets[:n_groups]

        # `torch.unique` returns its groups sorted; restore the first-appearance
        # order the row loop produced, so the returned rows keep the order every
        # caller has seen so far.
        first_row = torch.full(
            (n_groups,), n_rows, dtype=torch.long, device=device
        ).scatter_reduce(
            0, inverse, torch.arange(n_rows, device=device), reduce="amin"
        )
        order = first_row.argsort()
        ordered_keys = unique_keys[order]

        pooled_meta = {
            "sequence": ordered_keys // (radix_i * radix_j),
            "arg_pred_i": (ordered_keys // radix_j) % radix_i,
            "arg_pred_j": ordered_keys % radix_j,
        }

        return pooled_meta, pooled_logits[order], targets[order]

    @record_function("compute_relation_loss")
    def compute_relation_loss(
        self,
        true_relations: Sequence[IndexedRelation],
        rel_meta: dict[str, Tensor],
        rel_logits: Float[Tensor, "relation logits"] | None,
    ) -> Float[Tensor, ""]:
        aligned_rel_preds = self.align_relation_predictions(
            true_relations=true_relations,
            rel_meta=rel_meta,
            rel_logits=rel_logits,
        )
        if aligned_rel_preds is None:
            return torch.tensor(0.0, device=self.device)

        _, preds, targets = aligned_rel_preds

        if self.relation_loss_weighting == "focal":
            return focal_cross_entropy(
                preds,
                targets,
                gamma=self.relation_focal_gamma,
                label_smoothing=self.relation_label_smoothing,
            )

        weight = (
            balanced_class_weights(targets, self.num_relations)
            if self.relation_loss_weighting == "balanced"
            else None
        )
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=weight,
            reduction="mean",
            label_smoothing=self.relation_label_smoothing,
        )
        return loss_fn(preds, targets)

    def get_batch_logits(
        self,
        batch: Sequence[BatchItem],
        gold_relations: list[IndexedRelation] | None = None,
    ) -> tuple[
        Float[Tensor, "sequence entities"],
        Float[Tensor, "sequence classes"],
        tuple[dict[str, Tensor], Float[Tensor, "pairs relations"]] | None,
    ]:
        token_embeddings, token_att_mask = self.get_token_embeddings(batch)
        entities_in_batch = get_batch_entities(batch)

        entity_logits, class_logits, relation_index_logits = self(
            token_embeddings,
            token_att_mask,
            entities_in_batch,
            gold_relations=gold_relations,
        )

        return (
            entity_logits,
            class_logits,
            relation_index_logits,
        )

    def compute_batch_losses(
        self, batch: Sequence[BatchItem]
    ) -> tuple[Float[Tensor, ""], Float[Tensor, ""], Float[Tensor, ""]]:
        """Compute loss for a batch."""
        ent_true, class_true, rel_true = self.ground_truth(batch)
        entity_logits, class_logits, relation_index_logits = (
            self.get_batch_logits(batch, gold_relations=rel_true)
        )

        ent_loss, class_loss = self.compute_entity_loss(
            predictions=(entity_logits, class_logits),
            targets=(ent_true, class_true),
        )

        if relation_index_logits is not None:
            rel_index, rel_logits = relation_index_logits
        else:
            rel_index, rel_logits = ({}, None)

        relation_loss = self.compute_relation_loss(
            true_relations=rel_true,
            rel_meta=rel_index,
            rel_logits=rel_logits,
        )

        return ent_loss, class_loss, relation_loss

    def compute_batch_true_x_pred(
        self, batch: Sequence[BatchItem]
    ) -> dict[str, dict[str, np.ndarray]]:
        """Returns y_true, y_pred arrays for each task tackled by the model."""
        entity_logits: Float[Tensor, "sequence entities"]
        class_logits: Float[Tensor, "sequence classes"]
        relation_index_logits: (
            tuple[dict[str, Tensor], Float[Tensor, "pairs relations"]] | None
        )
        entity_logits, class_logits, relation_index_logits = (
            self.get_batch_logits(batch)
        )

        entity_truth: Float[Tensor, "batch entities"]
        class_truth: Float[Tensor, "batch classes"]
        rel_truth: list[IndexedRelation]
        entity_truth, class_truth, rel_truth = self.ground_truth(batch)
        relations_true = np.array([], dtype=int)
        relations_pred = np.array([], dtype=int)

        def _none_predictions():
            """Return none predictions for every gold label in this batch."""
            relations_true = np.array(
                [rel.label for rel in rel_truth], dtype=int
            )
            relations_pred = np.full(
                len(rel_truth), int(self.relations_none_index), dtype=int
            )
            return relations_true, relations_pred

        if rel_truth:
            if relation_index_logits:
                rel_meta: dict[str, Tensor]
                rel_logits: Float[Tensor, "pairs relations"]
                rel_meta, rel_logits = relation_index_logits
                aligned_rel_preds = self.align_relation_predictions(
                    true_relations=rel_truth,
                    rel_meta=rel_meta,
                    rel_logits=rel_logits,
                )
                if aligned_rel_preds is not None:
                    _, preds, targets = aligned_rel_preds
                    relations_true = (
                        targets.numpy(force=True).reshape(-1).astype(int)
                    )
                    relations_pred = preds.numpy(force=True)
                    relations_pred = (
                        relations_pred.argmax(axis=-1).reshape(-1).astype(int)
                    )
                else:
                    relations_true, relations_pred = _none_predictions()
            else:
                relations_true, relations_pred = _none_predictions()

        if relations_true.shape != relations_pred.shape:
            logger.warning(
                "relations_true %s != relations_pred %s",
                relations_true.shape,
                relations_pred.shape,
            )

        return {
            "entities": {
                "true": entity_truth.numpy(force=True),  # no squeeze
                "pred": torch.sigmoid(entity_logits.float())
                .round()
                .numpy(force=True),
            },
            "classes": {
                "true": class_truth.numpy(force=True),
                "pred": torch.sigmoid(class_logits.float())
                .round()
                .numpy(force=True),
            },
            "relations": {
                "true": np.asarray(relations_true).reshape(-1),
                "pred": np.asarray(relations_pred).reshape(-1),
            },
        }

    def _compute_relations_vectorized(
        self,
        entity_positions: Int64[Tensor, "n_entities 2"],
        entity_reprs: Float[Tensor, "n_entities features"],
        max_indices: Int64[Tensor, "document token"],
    ) -> tuple[dict[str, Tensor], Float[Tensor, "n_pairs relations"]] | None:
        """
        Compute relation logits for all valid entity pairs.
        Returns:
            - dict of raw tensors: {
                "doc": LongTensor[n_pairs],
                "arg_pred_i": LongTensor[n_pairs],
                "arg_pred_j": LongTensor[n_pairs],
            }
            - logits: FloatTensor[n_pairs, n_relations]
        """
        device = self.device
        doc_ids = entity_positions[:, 0]
        token_positions = entity_positions[:, 1]

        # `entity_preds` is a vector of integers indexing self.entities, hence
        # indicating to which entity the token was assigned by the entity
        # classifier.
        entity_preds: Int64[Tensor, " entities"] = max_indices[
            doc_ids, token_positions
        ]

        # Precompute indices and prepare output buffers
        unique_doc_ids = torch.unique(doc_ids)
        doc_batch = []
        arg_pred_i = []
        arg_pred_j = []
        reprs_i = []
        reprs_j = []

        for doc_id in unique_doc_ids:
            indices = torch.where(doc_ids == doc_id)[0]

            if len(indices) < 2:
                continue

            local_pos = token_positions[indices]
            local_preds = entity_preds[indices]
            unique_local_preds = torch.unique(local_preds)
            local_reprs = entity_reprs[indices]

            grouped_entity_positions = [
                local_pos[local_preds == pred] for pred in unique_local_preds
            ]
            pooled_reprs = torch.stack(
                [
                    local_reprs[local_preds == pred].mean(dim=0)
                    for pred in unique_local_preds
                ]
            )

            pairs = torch.combinations(
                torch.arange(len(grouped_entity_positions), device=device),
                r=2,
            )

            if len(pairs) == 0:
                continue

            i, j = pairs[:, 0], pairs[:, 1]
            pred_i = unique_local_preds[i]
            pred_j = unique_local_preds[j]

            n_pairs = len(i)
            doc_batch.append(
                torch.full((n_pairs,), doc_id, dtype=torch.long, device=device)
            )
            arg_pred_i.append(pred_i)
            arg_pred_j.append(pred_j)
            reprs_i.append(pooled_reprs[i])
            reprs_j.append(pooled_reprs[j])

        if reprs_i:
            all_repr_i = torch.cat(reprs_i, dim=0)
            all_repr_j = torch.cat(reprs_j, dim=0)
            logits = self.relation_classifier(all_repr_i, all_repr_j)

            meta = {
                "sequence": torch.cat(doc_batch),
                "arg_pred_i": torch.cat(arg_pred_i),
                "arg_pred_j": torch.cat(arg_pred_j),
            }
        else:
            return None

        return meta, logits

    @record_function("forward")
    def forward(
        self,
        embeddings: Float[Tensor, "document token embedding"],
        attention_mask: Bool[Tensor, "document token"],
        entities_in_batch: tuple[Int16[Tensor, " entities"], ...],
        gold_relations: list[IndexedRelation] | None = None,
    ) -> tuple[
        BatchedLogits,
        BatchedLogits,
        tuple[
            dict[str, Tensor],
            Float[Tensor, "pairs relations"],
        ]
        | None,
    ]:
        """Forward pass

        :return: tuple containing:
            - Entity logits pooled by document.
            - Class logits pooled by document.
            - Tuple containing:
                - Index of entity A, where dim=-1 corresponds to the entity
                  selected in entity_index
                - Index of entity B
                - Relation type logits
        """

        def _soft_entity_repr(
            doc_hidden: Float[Tensor, "tokens hidden_size"],
            doc_ent_logits: Float[Tensor, "tokens entities"],
            doc_mask: Bool[Tensor, " tokens"],
            ent_id: int,
        ) -> Float[Tensor, " hidden_size"]:
            with torch.autocast(device_type=self.device, enabled=False):
                scores = doc_ent_logits[:, ent_id].float()  # [T]
                scores = scores.masked_fill(~doc_mask, float("-inf"))
                w = torch.softmax(scores, dim=0)  # [T]
                rep = (w.unsqueeze(-1) * doc_hidden.float()).sum(dim=0)  # [H]
            return rep.to(doc_hidden.dtype)

        device = self.device
        with self.autocast_context():
            hidden_output: Float[Tensor, "document token features"] = (
                self.hidden(embeddings)
            )
            unmasked_entity_logits, unmasked_class_logits = self.classifier(
                hidden_output
            )
            token_mask = attention_mask.unsqueeze(-1)
            neg_inf = self._neg_inf
            entity_logits = torch.where(
                token_mask, unmasked_entity_logits, neg_inf
            )
            class_logits = torch.where(
                token_mask, unmasked_class_logits, neg_inf
            )

            # Nothing differentiable leaves this block: it yields a bool mask
            # and int64 indices, and the relation head's gradient reaches
            # `hidden_output` by *indexing* it with them, never through the
            # probabilities. Recorded by autograd, the four intermediates below
            # are each a full [document, token, entity] tensor — 864 MB apiece
            # at a p99-length batch — held for a backward that never reads
            # them. The arithmetic is unchanged, so the mask is bit-identical.
            # Sliced along the token dim for the same reason `pool_token_dim`
            # is: `torch.softmax` over the whole tensor, its clamp, its log and
            # the product are four more [document, token, entity] tensors, and
            # even freed immediately they set the peak of the whole step. Every
            # row of a softmax over the last dim is independent of every other,
            # so slicing changes no value — the mask is bitwise what the
            # unsliced expression gave.
            with torch.no_grad():
                entropies = []
                predictions = []
                chunk = pool_chunk_tokens(
                    entity_logits.shape[0], entity_logits.shape[2]
                )
                for start in range(0, entity_logits.shape[1], chunk):
                    entity_probs: Float[Tensor, "document token ent_probs"] = (
                        torch.softmax(
                            entity_logits[:, start : start + chunk],
                            dim=-1,
                        )
                    )
                    entropies.append(
                        -(
                            entity_probs * (entity_probs.clamp_min(1e-9)).log()
                        ).sum(-1)
                    )
                    predictions.append(entity_probs.argmax(dim=-1))
                    del entity_probs

                entropy = torch.cat(entropies, dim=1)
                max_indices = torch.cat(predictions, dim=1)
                del entropies, predictions

                hard_entity_mask: Bool[Tensor, "document token"]
                hard_entity_mask = (max_indices != self.unk_index) & (
                    entropy <= self.entity_threshold
                )
                del entropy

            rel_meta_logits = None
            if hard_entity_mask.any():
                # Select the predicted entity representations
                entity_positions: Int64[Tensor, "doc token"] = (
                    hard_entity_mask.nonzero(as_tuple=False)
                )
                if entity_positions.numel() >= 2:
                    entity_reprs = hidden_output[
                        entity_positions[:, 0],  # batch
                        entity_positions[:, 1],  # token
                    ]
                    rel_meta_logits = self._compute_relations_vectorized(
                        entity_positions, entity_reprs, max_indices
                    )

            gold_meta_logits = None
            if gold_relations is not None:
                batch, tokens, hidden_size = hidden_output.shape
                needed_by_doc: dict[int, set[int]] = {}
                for tr in gold_relations:
                    docix = int(tr.docix)
                    subj = int(self.entity_to_index.get(tr.subject, -1))
                    obj = int(self.entity_to_index.get(tr.object, -1))
                    if subj < 0 or obj < 0:
                        continue
                    needed_by_doc.setdefault(docix, set()).update((subj, obj))

                soft_repr_by_doc = {}
                for docix, ent_ids in needed_by_doc.items():
                    doc_hidden = hidden_output[docix]
                    doc_logits = unmasked_entity_logits[docix]
                    doc_mask = attention_mask[docix].to(torch.bool)
                    reps = {
                        eid: _soft_entity_repr(
                            doc_hidden=doc_hidden,
                            doc_ent_logits=doc_logits,
                            doc_mask=doc_mask,
                            ent_id=eid,
                        )
                        for eid in ent_ids
                    }
                    soft_repr_by_doc[docix] = reps

                rows_doc, rows_i, rows_j, rep_i, rep_j = [], [], [], [], []
                for tr in gold_relations:
                    doc_ix = int(tr.docix)
                    doc_reps = soft_repr_by_doc.get(doc_ix)
                    if not doc_reps:
                        continue
                    subj = int(self.entity_to_index.get(tr.subject, -1))
                    obj = int(self.entity_to_index.get(tr.object, -1))
                    if subj in doc_reps and obj in doc_reps:
                        rows_doc.append(doc_ix)
                        rows_i.append(subj)
                        rows_j.append(obj)
                        rep_i.append(doc_reps[subj])
                        rep_j.append(doc_reps[obj])

                if rep_i:
                    rep_i_t = torch.stack(rep_i, dim=0)
                    rep_j_t = torch.stack(rep_j, dim=0)
                    logits = self.relation_classifier(rep_i_t, rep_j_t)
                    gold_meta_logits = (
                        {
                            "sequence": torch.tensor(
                                rows_doc, device=device, dtype=torch.long
                            ),
                            "arg_pred_i": torch.tensor(
                                rows_i, device=device, dtype=torch.long
                            ),
                            "arg_pred_j": torch.tensor(
                                rows_j, device=device, dtype=torch.long
                            ),
                        },
                        logits,
                    )

            # ---- Merge hard-pair logits (if any) with gold-pair logits (if any)
            #
            # A (doc, subj, obj) triple can be produced by both the hard-entity
            # mask and the gold path. Keep at most one row per triple: prefer
            # the gold soft representation (richer signal) and drop the
            # overlapping hard-mask row. This stops the downstream aligner from
            # logsumexp-pooling two rows for the same triple, which would bias
            # its logits upward.
            merged = None
            if rel_meta_logits and gold_meta_logits:
                (m1, l1), (m2, l2) = rel_meta_logits, gold_meta_logits
                gold_keys = set(
                    zip(
                        m2["sequence"].tolist(),
                        m2["arg_pred_i"].tolist(),
                        m2["arg_pred_j"].tolist(),
                    )
                )
                hard_keep = [
                    r
                    for r, k in enumerate(
                        zip(
                            m1["sequence"].tolist(),
                            m1["arg_pred_i"].tolist(),
                            m1["arg_pred_j"].tolist(),
                        )
                    )
                    if k not in gold_keys
                ]
                keep_idx = torch.tensor(
                    hard_keep, device=device, dtype=torch.long
                )
                merged_meta = {
                    "sequence": torch.cat(
                        [m1["sequence"][keep_idx], m2["sequence"]]
                    ),
                    "arg_pred_i": torch.cat(
                        [m1["arg_pred_i"][keep_idx], m2["arg_pred_i"]]
                    ),
                    "arg_pred_j": torch.cat(
                        [m1["arg_pred_j"][keep_idx], m2["arg_pred_j"]]
                    ),
                }
                merged_logits = torch.cat([l1[keep_idx], l2], dim=0)
                merged = (merged_meta, merged_logits)
            else:
                merged = rel_meta_logits or gold_meta_logits

            return (
                self._pool_logits(entity_logits),
                self._pool_logits(class_logits),
                merged,
            )

    def evaluate_model(
        self,
        test_data: DataLoader,
        tau_ids: float = 0.5,
        tau_cls: float = 0.5,
        topk_ids: int | None = None,
    ) -> dict[str, float]:
        """
        Evaluate the end-to-end model from *document-level pooled logits*.
        - tau_ids / tau_cls: global thresholds for multilabel binarization
        - topk_ids: also keep top-K entity IDs per document

        Returns what it prints and logs the same dict to the active tracking
        run; an empty dict means the split produced no samples.
        """
        self.eval()
        metrics: dict[str, float] = {}
        all_id_logits, all_id_true = [], []
        all_cls_logits, all_cls_true = [], []
        all_rel_logits, all_rel_true = [], []  # we'll argmax rel later

        with torch.no_grad():
            # do NOT autocast around metric collection; keep numerics simple
            for batch in batch_progress(
                test_data, desc="Evaluating", position=0, leave=True
            ):
                # 1) pooled doc-level logits
                id_logits_doc, cls_logits_doc, rel_meta_logits = (
                    self.get_batch_logits(batch)
                )  # shapes: [B, num_ids], [B, num_classes], (meta, [N_pairs,R]) or None

                # 2) document-level multi-hot targets
                id_true_doc, cls_true_doc, rel_true_list = self.ground_truth(
                    batch
                )  # id_true_doc: [B,num_ids], cls_true_doc: [B,num_classes], rel_true_list: list[...]

                # logits narrowed to the columns the targets carry
                all_id_logits.append(
                    self.drop_unk(id_logits_doc).detach().float().cpu()
                )
                all_id_true.append(id_true_doc.detach().to(torch.int64).cpu())

                all_cls_logits.append(
                    self.drop_oos(cls_logits_doc).detach().float().cpu()
                )
                all_cls_true.append(cls_true_doc.detach().to(torch.int64).cpu())

                # 3) relations: reuse the training-time aligner so eval and
                #    training pool duplicates and assign targets identically
                #    (one row per (doc, subj, obj) triple).
                if rel_meta_logits is not None:
                    rel_meta, rel_logits = rel_meta_logits  # [N_pairs,R]
                    aligned = self.align_relation_predictions(
                        true_relations=rel_true_list,
                        rel_meta=rel_meta,
                        rel_logits=rel_logits,
                    )
                    if aligned is not None:
                        _, rel_logits_aligned, rel_targets = aligned
                        all_rel_logits.append(rel_logits_aligned.detach().cpu())
                        all_rel_true.append(rel_targets.detach().cpu())

        # ----- stack
        if not all_id_logits:
            logger.warning("No samples found.")
            return metrics

        id_logits = torch.cat(all_id_logits, dim=0).numpy()
        id_true = torch.cat(all_id_true, dim=0).numpy().astype(int)
        cls_logits = torch.cat(all_cls_logits, dim=0).numpy()
        cls_true = torch.cat(all_cls_true, dim=0).numpy().astype(int)

        # ---- IDs: probs -> binarize (threshold + optional top-K)
        id_probs = 1.0 / (1.0 + np.exp(-id_logits))
        id_pred = (id_probs >= tau_ids).astype(int)
        if topk_ids is not None and topk_ids > 0:
            # ensure at least top-K positives per doc (in addition to threshold)
            topk_idx = np.argpartition(
                -id_probs, kth=min(topk_ids, id_probs.shape[1] - 1), axis=1
            )[:, :topk_ids]
            rows = np.arange(id_probs.shape[0])[:, None]
            id_pred[rows, topk_idx] = 1

        # ---- CLASSES: probs -> binarize
        cls_probs = 1.0 / (1.0 + np.exp(-cls_logits))
        cls_pred = (cls_probs >= tau_cls).astype(int)

        # ---- sanity counts
        metrics.update(
            support_metrics(
                {"entity": (id_true, id_pred), "class": (cls_true, cls_pred)}
            )
        )
        logger.info(
            "\n[Entities] gold positives: %d | predicted positives: %d"
            " | classes with any preds: %d",
            int(id_true.sum()),
            int(id_pred.sum()),
            int((id_pred.sum(axis=0) > 0).sum()),
        )
        logger.info(
            "[Classes ] gold positives: %d | predicted positives: %d",
            int(cls_true.sum()),
            int(cls_pred.sum()),
        )

        # ======= METRICS =======

        # Entities (6k+ labels): prefer micro-F1 + LRAP; macro over frequent labels only
        logger.info("\n=== Entity ID metrics (multilabel, document-level) ===")
        try:
            metrics["test/entity_micro_f1"] = f1_score(
                id_true, id_pred, average="micro", zero_division=0
            )
            logger.info("micro-F1: %s", metrics["test/entity_micro_f1"])
        except ValueError:
            logger.info("micro-F1: (no positive labels or predictions) 0.0")

        try:
            metrics["test/entity_lrap"] = label_ranking_average_precision_score(
                id_true, id_probs
            )
            logger.info("LRAP: %s", metrics["test/entity_lrap"])
        except ValueError:
            logger.info("LRAP: undefined (no positives)")

        # macro-F1 over frequent labels
        support = id_true.sum(axis=0)
        keep = np.where(support >= 10)[0]  # tweak threshold as you like
        if keep.size > 0:
            metrics["test/entity_macro_f1_support10"] = f1_score(
                id_true[:, keep],
                id_pred[:, keep],
                average="macro",
                zero_division=0,
            )
            logger.info(
                "macro-F1 (support>=10): %s",
                metrics["test/entity_macro_f1_support10"],
            )
        else:
            logger.info(
                "macro-F1 (support>=10): n/a (no labels meet support threshold)"
            )

        logger.info(
            "\n=== Entity CLASS metrics (multilabel, document-level) ==="
        )
        metrics["test/class_micro_f1"] = f1_score(
            cls_true, cls_pred, average="micro", zero_division=0
        )
        logger.info("micro-F1: %s", metrics["test/class_micro_f1"])
        class_report = classification_report(
            y_true=cls_true,
            y_pred=cls_pred,
            target_names=self.known_classes,
            zero_division=0,
        )
        logger.info(class_report)
        tracking.log_text(str(class_report), "test/class_report.txt")

        # Relations (multiclass over candidate pairs)
        if all_rel_logits:
            rel_logits_np = torch.cat(all_rel_logits, dim=0).numpy()
            rel_true = torch.cat(all_rel_true, dim=0).numpy().astype(int)
            rel_pred = rel_logits_np.argmax(axis=1)

            logger.info(
                "\n=== Relation metrics (multiclass over candidate pairs) ==="
            )
            labels = np.arange(len(self.relations))
            metrics.update(
                relation_metrics(
                    true=rel_true,
                    pred=rel_pred,
                    labels=labels,
                    none_index=int(self.relations_none_index),
                )
            )
            relation_report = classification_report(
                y_true=rel_true,
                y_pred=rel_pred,
                labels=labels,
                target_names=list(self.relations),
                zero_division=0,
            )
            logger.info(relation_report)
            tracking.log_text(str(relation_report), "test/relation_report.txt")
        else:
            logger.info("\n(No relation pairs produced on this split.)")

        tracking.log_metrics(metrics)

        return metrics


class ClassificationHead(nn.Module):
    """Define a classification head for end-to-end models."""

    def __init__(
        self,
        input_size: int,
        n_entities: int,
        n_classes: int,
        entity_freqs: Float[Tensor, " entities"] | None = None,
        class_freqs: Float[Tensor, " classes"] | None = None,
        unk_index: int = -1,
        oos_index: int = -1,
    ) -> None:
        """Initialize the classification head.

        :param input_size: number of input features
        :param n_entities: number of output entities
        :param n_classes: number of output entity classes
        :param unk_index: column of the unsupervised UNK entity, which carries
            no frequency and so is seeded from a prior instead
        :param oos_index: idem for the OOS class
        """
        super().__init__()
        self.entity_classifier = nn.Sequential(
            nn.Linear(
                in_features=input_size,
                out_features=input_size,
                bias=True,
            ),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_size, n_entities),
        )
        # self.entity_classifier = nn.Linear(input_size, n_entities)
        self.class_classifier = nn.Linear(input_size, n_classes)
        if entity_freqs is not None:
            initialize_classifier_bias(
                linear=cast(nn.Linear, self.entity_classifier[-1]),
                freqs=entity_freqs,
                sentinel_index=unk_index,
            )
        if class_freqs is not None:
            initialize_classifier_bias(
                linear=cast(nn.Linear, self.class_classifier),
                freqs=class_freqs,
                sentinel_index=oos_index,
                sentinel_prior=0.9,
            )

    def forward(self, input: Tensor) -> tuple[Tensor, Tensor]:
        entity_logits = self.entity_classifier(input)
        class_logits = self.class_classifier(input)

        return entity_logits, class_logits


def initialize_classifier_bias(
    linear: torch.nn.Linear,
    freqs: torch.Tensor,
    eps: float = 1e-5,
    sentinel_index: int | None = -1,
    sentinel_prior: float = 0.1,
) -> None:
    """Initialize classifier bias using log odds from label frequencies.

    `freqs` covers the supervised labels only, in column order. `sentinel_index`
    names the head's one unsupervised column — UNK for an entity head, OOS for a
    class head — which has no frequency and is seeded from `sentinel_prior`
    instead. It defaults to the last column, where both models put it; pass
    `None` for a head with no sentinel column.
    """
    device = linear.weight.device
    dtype = linear.weight.dtype

    p = freqs.clamp(eps, 1 - eps).to(device=device, dtype=dtype)
    log_odds = torch.log(p) - torch.log1p(-p)  # logit(p)

    with torch.no_grad():
        if sentinel_index is None:
            if log_odds.numel() != linear.out_features:
                raise ValueError(
                    f"freqs len {log_odds.numel()} != out_features {linear.out_features}"
                )
            linear.bias.copy_(log_odds)
            return

        expected = linear.out_features - 1
        if log_odds.numel() != expected:
            raise ValueError(
                f"freqs len {log_odds.numel()} != expected {expected} "
                f"(out_features-1) for layer with a sentinel column"
            )

        sentinel = sentinel_index % linear.out_features
        kept = torch.tensor(
            [
                column
                for column in range(linear.out_features)
                if column != sentinel
            ],
            device=device,
        )
        bias = torch.empty(linear.out_features, device=device, dtype=dtype)
        bias[kept] = log_odds
        prior = max(min(sentinel_prior, 1 - eps), eps)
        bias[sentinel] = math.log(prior) - math.log1p(-prior)
        linear.bias.copy_(bias)
