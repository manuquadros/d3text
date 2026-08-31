"""Base model class and the helpers shared by every model in this package.

Split out of what used to be `models.py`. `Model` provides the machinery
common to every concrete model — base transformer loading, AMP/checkpointing,
token embedding lookup, logit pooling — while the concrete subclasses
(`NERClassificationModel`, `BrendaClassificationModel`, `ETEBrendaModel`) live
in their own modules and import from here.
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


@functools.cache
def embeddings_store(base_model: str) -> EmbeddingsStore | None:
    """The configured embeddings store, opened once, or `None` without one.

    Lazy rather than opened beside the cache above, for the reason the rest of
    the library defers its machine state: importing `d3text.models` must not
    touch the filesystem. A store that cannot be opened — a path that has moved,
    a half-written LMDB — disables itself and the run recomputes the embeddings,
    which is exactly what it would have done with no store configured. Losing
    the speed-up is not worth losing the run.

    `base_model` is what the store has to have been written by. A store that
    was not takes the same route as one that will not open: the run pays the
    base model's speed rather than training on somebody else's activations,
    and says so once. It is an argument rather than a field of the machine
    config because the store belongs to the machine and the model belongs to
    the run, and it is the pair that has to agree.
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

    Measured by running the aggregation over a zero-width tensor rather than
    by reimplementing its overlap arithmetic: the number exists to catch a
    store whose rows do not line up with the encodings, so a second, drifting
    copy of that arithmetic would be a hole in the very check it serves. The
    zero-width feature dimension is what makes it free — there are no values
    to slice, only rows to count.
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


def masked_token_cross_entropy(
    preds: Float[Tensor, "token logits"],
    targets: Int64[Tensor, " token"],
    ignore_index: int = -100,
    weighting: TokenLossWeighting = "unweighted",
    focal_gamma: float = 2.0,
) -> Float[Tensor, ""]:
    """Cross-entropy over the tokens `targets` does not mask out.

    The distant-supervision targets in `d3text.token_labels` carry a third
    value for the tokens that match a surface form of an entity this document
    was not annotated with. Those are the tokens nothing knows the answer for,
    and they are ~2.8% of the document.

    **The divisor is the unmasked count, not the token count** — the same trap
    `focal_cross_entropy` documents one level up. Summing the kept terms and
    dividing by the whole sequence length scales every real token's loss by the
    share of the document that happened to be masked, so a document with more
    uncurated entities in it teaches less about the ones it does have. That is
    the dilution the mask exists to remove, reintroduced by the reduction.

    `torch.nn.functional.cross_entropy(..., ignore_index=...)` is the other
    spelling of exactly this and divides the same way; this one exists so the
    divisor is visible at the call site rather than inherited from a default,
    and `tests/models/test_masked_loss.py` pins the two against each other.
    That equivalence holds only for the default `weighting="unweighted"` — the
    other two schemes have no single-call `nn.functional` spelling.

    **`weighting` mirrors `relation_loss_weighting`'s three-way choice on the
    relation head**, aimed at the same shape of imbalance: `OUTSIDE` is ~91% of
    kept tokens, so a plain average lets the majority class dominate the
    gradient. `balanced` reweights by per-batch inverse frequency over the kept
    tokens (`balanced_class_weights`, the same helper the relation head uses,
    and the same reason it is per-batch rather than precomputed: the kept set
    changes with the mask); the reduction is then `nn.CrossEntropyLoss`'s own
    weighted mean, dividing by the summed sample weights rather than the kept
    count. `focal` down-weights confidently-correct tokens instead
    (`focal_cross_entropy`, normalised by its own modulation mass). Passing
    `weighting="unweighted"` — the default — reproduces the previous behaviour
    exactly, weight tensor and all.

    An all-masked batch returns a differentiable zero rather than a NaN: it is
    reachable from a short document whose every match is uncurated, and losing
    a training run to it would be absurd.
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

    `abstain` marks a negative target this run has decided not to fully
    enforce: a document the class head is told carries none of a type, but
    whose text a dictionary match says otherwise. `None` is the ordinary
    case and reduces to a plain `BCEWithLogitsLoss(reduction="mean")`.

    `downweight` sets the weight an abstained pair keeps instead of being
    dropped outright. `0.0` — the default — is a hard abstain, excluded from
    both the numerator and the divisor: byte-identical to this function
    before the parameter existed. A value in `(0, 1]` keeps that fraction of
    the negative pressure rather than removing it (DEC-04's option 2, as
    opposed to option 1's hard mask). Unread when `abstain` is `None`.

    **The divisor is the weight sum, not the pair count** — the same
    reasoning `masked_token_cross_entropy` gives for tokens: dividing by the
    whole matrix would scale every asserted target's loss down by however
    many pairs this document happened to abstain or downweight, teaching
    less from the documents with more of them rather than the same amount
    from fewer terms.
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


def token_counts(
    mask: Bool[Tensor, "document token"],
) -> Float[Tensor, " document"]:
    """Real tokens per document, floored at one.

    The floor keeps a document that is all padding finite — dividing by its
    true count of zero would make the mean a NaN and `logmeanexp`'s
    ``log(counts)`` a ``-inf`` — while leaving every real document's count
    untouched.
    """
    return mask.sum(dim=1).clamp(min=1).to(torch.float32)


class _ChunkedMean(torch.autograd.Function):
    """`mean` over the token dimension, in float32, one slice at a time.

    Same bargain as `_ChunkedLogSumExp`, and simpler: a mean spreads its
    gradient evenly, so backward reads none of the input at all.

    With a mask, padded positions are zeroed out of the sum and the divisor
    is each document's real token count — the caller's `-1e9` fill would
    otherwise dominate the numerator, and the padded length the denominator.
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
        ctx.mask = mask
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
        if ctx.mask is None:
            grad = (grad_pooled / ctx.counts).unsqueeze(1).expand(ctx.shape)
        else:
            grad = (grad_pooled / ctx.counts.unsqueeze(1)).unsqueeze(
                1
            ) * ctx.mask.unsqueeze(-1)
        return grad.to(ctx.dtype), None, None


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
    logits: Float[Tensor, "document token logits"],
    pooling: str,
    mask: Bool[Tensor, "document token"] | None = None,
) -> Float[Tensor, "document logits"]:
    """Pool the token dimension without a float32 copy of the whole tensor.

    The float32 is right — pooling thousands of tokens in bfloat16 is where the
    precision actually matters — but it does not have to exist all at once.
    Every mode routes through here so the pooled values cannot depend on which
    path ran.

    `mask` is the attention mask of the batch whose padding the caller has
    already filled with a large negative value. Without it, `logmeanexp` and
    `mean` normalise by the *padded* length, so a document's pooled logits
    depend on how long its batch companions were — a short document batched
    with a long one is shifted by ``-log(T_pad / T_doc)`` on every column.
    With it, both modes normalise by each document's real token count, and
    `mean` also keeps the fill values out of its numerator. `logsumexp` and
    `max` never needed it: the fills vanish under both reductions.
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

    `torch.cuda.is_bf16_supported()` answers a different question: it defaults
    to `including_emulation=True` and so returns True on cards that have no
    bf16 units at all, which is how a Pascal card came to train under bf16
    autocast. Measured on a P100, that costs about 27% of the throughput of
    fp16 or fp32 and close to three times the peak memory — 10.4 GiB against
    3.5 GiB over 256 windows — on a card whose configured training run already
    peaked at 99.2% of its 16 GiB.

    Asked by compute capability, as `runtime.is_triton_compatible` asks its own
    question: bf16 units arrive with Ampere (8.0), and the capability is
    readable on every torch version, while the `including_emulation` keyword is
    not.
    """
    if not torch.cuda.is_available():
        return False

    return torch.cuda.get_device_capability() >= (8, 0)


def select_amp_dtype() -> torch.dtype:
    """Pick bf16 only where the active backend can run it in silicon.

    CUDA and ROCm answer this from different signals, so each backend must be
    asked independently rather than ANDing one backend's veto into the other's
    question. `has_bf16_hardware()` reads CUDA compute capability, which is
    meaningless under HIP — `get_device_capability` there returns gfx-derived
    numbers that would answer True even for a card with no bf16 units — so it
    is gated to CUDA, and the device-name allowlist is the sole authority for
    ROCm.

    "MI300" is dropped from the allowlist as redundant: it is a strict
    substring of "MI3", which is kept as a deliberate prefix match meant to
    also catch future MI3xx parts (MI325, MI350, ...) without naming each one.
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
    """Base model class implementing common functionality.

    This class provides the basic structure and utilities for all models:
    - Base transformer model initialization
    - One epoch's forward and loss computation (`run_epoch`)
    - Common layer setup (dropout, hidden layers)

    The epoch schedule around `run_epoch` — optimizer, LR scheduler, early
    stopping and the best-epoch snapshot — belongs to
    `d3text.training.trainer.Trainer`.

    Attributes:
        config: Model configuration parameters
        base_model: Pre-trained transformer model
        tokenizer: Associated tokenizer
        device: Training device (CPU/GPU)
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

        Selected by `entity_logits_pooling` (from `ModelConfig`):

        - ``logmeanexp``: ``logsumexp - log(T)``; length-invariant smooth-mean,
          but dilutes a lone mention in a long document. The default.
        - ``logsumexp``: smooth-max — one strong token can carry the document;
          adds up to ``+log(T)`` for diffuse classes, so it is length-biased.
        - ``max``: hard max; length-invariant.
        - ``mean``: arithmetic mean.

        Computed in float32, then cast back to the input dtype.

        The `[document, token, logits]` case — every call site in the models —
        is pooled a slice at a time by `pool_token_dim`, which never holds a
        float32 copy of the whole tensor; forwards pass their attention mask
        so the padded normalisers stay per-document (see `pool_token_dim`).
        The general path below still serves any other shape or `dim`, whose
        rows carry no padding.
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

    def epoch_loss_weights(self, epoch: int) -> dict[str, float]:
        """The multiplier applied to each named loss this epoch, if any.

        Keys match `run_epoch`'s `losses` dict, so a logged
        `loss_weight/relation` sits beside the `training/relation` it scaled —
        without which a loss curve that bends because the ramp moved is
        indistinguishable from one that bends because the model changed.
        Overridden by the one model that ramps an objective; every other loss
        trains at full weight from the first epoch and is absent from here.
        """
        return {}

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

        The base model is not among them: it is frozen, and only ever runs
        under `no_grad` in `get_token_embeddings`, so there is no activation
        graph to trade against recomputation.
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
        """Return the appropriate loss function for this model type"""
        raise NotImplementedError

    def compute_batch(
        self,
        batch: Any,
    ) -> float:
        """Compute loss for a batch and perform optimization step.
        Returns the loss value for this batch."""
        raise NotImplementedError

    def compute_losses(
        self,
        batch: Sequence[BatchItem],
        step: Step,
        epoch: int,
    ) -> dict[str, Tensor]:
        """One batch's losses, keyed by objective name; implemented per
        model subclass.

        Every key returned here is one `update` is asked to sum and
        optimize, and one `run_epoch` accumulates and reports under the same
        name — a key present in one batch of an epoch must be present in
        every batch of that epoch, since `NERClassificationModel` reports
        only ``class``, `BrendaClassificationModel` adds ``entity`` (and
        ``token`` when a token-label store is configured), and
        `ETEBrendaModel` adds ``relation`` already scaled by that epoch's
        ramp weight. `step` is what lets the ramped model score validation
        under its final weight while training still follows the schedule;
        a model with no ramp ignores both `step` and `epoch`.
        """
        raise NotImplementedError

    def run_epoch(
        self,
        data: DataLoader,
        step: Step,
        epoch: int,
        update: BatchUpdate,
    ) -> tuple[dict[str, float], int]:
        """Process all batches, applying `compute_losses` and the optimizer
        step.

        Shared by every model subclass — only `compute_losses` differs
        between them. `update` applies a training batch's losses to the
        weights; it is `Trainer`'s, not the model's, and is ignored on a
        validation pass.
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
        """Get token embeddings for a batch with caching support.

        Three sources, cheapest first: the in-process cache, the precomputed
        embeddings store, and the frozen base model. The base model is a pure
        function of the input ids and is never trained here, so the first two
        are not approximations of the third in kind — only in arithmetic. The
        store's matrices were computed under fp16 autocast and rounded to bf16,
        while the live forward runs under `amp_dtype`, so a run that reads the
        store gets slightly different activations from one that does not. It
        gets the *same* ones every epoch, which the live path cannot promise
        either.
        """
        inputs: list[None | Tensor] = [None] * len(batch)
        missing: list[tuple[int, BatchItem]] = []
        store = embeddings_store(self.config.base_model)

        for ix, item in enumerate(batch):
            doc_id: int = int(item["id"].item())
            if cpu_embeddings_cache is not None:
                cpu_cached = cpu_embeddings_cache.get(doc_id)
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

    Returning what it prints is the point: `Trainer.fit` logs this dict to
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


def coverage_metrics(data: DataLoader, scored: int) -> dict[str, float]:
    """How many of the split's documents the pass actually scored.

    `dataset/test_documents` is what the split frame *planned* to hold, and it
    is logged at run setup, before anything has been read. Every `test/*` score
    below is computed over the documents that reached the model instead, and
    the two come apart whenever the frame and the encodings file disagree:
    `BrendaDataset._getitems` drops a row whose pmid the HDF5 does not hold,
    and `batch_progress` drops a batch left empty by those drops. That shrinks
    the denominator of every metric here without shrinking the number a run
    list shows beside them.

    Keyed under `dataset/` rather than `test/` so the three sit together in a
    run table: `_scored` is only readable against the planned count, and
    `_missing` is the difference a run list can sort on, which no derived
    column could give it. `_missing` is 0 for a healthy split rather than
    absent, since an absent key cannot be told from a run of a version that
    did not log one; it is omitted altogether when the split size is unknown,
    which is the one case where the difference does not exist to be reported.
    """
    metrics = {"dataset/test_documents_scored": float(scored)}

    planned = split_documents(data)
    if planned is not None:
        metrics["dataset/test_documents_missing"] = float(planned - scored)

    return metrics
