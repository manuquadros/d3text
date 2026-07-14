"""The `Model` base class: everything a head-bearing model does that is not
about a particular set of heads.

Holds the frozen transformer base and its embedding cache, the optional common
hidden block, the document-level logit pooling, AMP and gradient checkpointing,
and one pass over a set of batches. `compute_losses` is the seam each subclass
fills in: the base class walks the batches, the subclass says only what one
batch costs and under what names.

A `Model` owns no optimizer, no scheduler and no early-stopping state: a run's
worth of those belongs to `d3text.training.Trainer`, which drives `run_epoch`
and hands it the `Optimization` to step.
"""

import itertools
import math
from collections.abc import Sequence
from enum import StrEnum
from typing import Protocol, cast, runtime_checkable

import torch
import torch.nn as nn
import transformers
from cacheout import Cache
from jaxtyping import Bool, Float, Int64, Integer
from torch import Tensor
from torch.autograd.profiler import record_function
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from d3text.utils import aggregate_embeddings

from .config import ModelConfig, machine_config, save_model_config
from .heads import PermutationBatchNorm1d
from .model_types import BatchItem

__all__ = [
    "Model",
    "Optimization",
    "Step",
    "cpu_embeddings_cache",
    "get_pool_fn",
    "label_columns",
    "load_base_model",
]

mconfig = machine_config()
if mconfig.cpu_embeddings_cache_size:
    cpu_embeddings_cache = Cache(maxsize=mconfig.cpu_embeddings_cache_size)
else:
    cpu_embeddings_cache = None


class Step(StrEnum):
    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "testing"


@runtime_checkable
class Optimization(Protocol):
    """What `run_epoch` needs of a training run in order to step it.

    The two calls the loop makes around one batch, and nothing else: how the
    gradients are scaled, clipped and applied is the `Trainer`'s business, and
    which optimizer and scheduler are behind them is its business too.

    Runtime-checkable because beartype checks this annotation with `isinstance`
    on every call, and a plain `Protocol` cannot be.
    """

    def zero_grad(self) -> None: ...

    def update(self, *losses: Float[Tensor, ""]) -> None: ...


def get_pool_fn(pooling: str):
    if pooling == "max":
        return lambda x: torch.amax(x, dim=0)
    elif pooling == "mean":
        return lambda x: torch.mean(x, dim=0)
    elif pooling == "logsumexp":
        return lambda x: torch.logsumexp(x, dim=0)
    else:
        raise ValueError(f"Unknown pooling: {pooling}")


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


class Model(torch.nn.Module):
    """Base model class implementing common functionality.

    This class provides the basic structure and utilities for all models:
    - Base transformer model initialization
    - One pass over a set of batches (`run_epoch`), stepping an `Optimization`
    - Common layer setup (dropout, hidden layers)
    - AMP autocasting and gradient checkpointing

    Attributes:
        config: Model configuration parameters
        base_model: Pre-trained transformer model
        device: Training device (CPU/GPU)
    """

    # Assigned in subclass __init__ / registered as buffers; annotated here so
    # nn.Module.__getattr__ doesn't collapse them to `Tensor | Module`.
    base_model: transformers.PreTrainedModel
    _neg_inf: Tensor
    classes: list[str]
    class_columns: Tensor

    def __init__(
        self,
        config: ModelConfig | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__()

        self.config = config if config is not None else ModelConfig()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

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

        self.register_buffer("_neg_inf", torch.tensor(-1e9))

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
        """
        x = logits.float()
        pooling = self.entity_logits_pooling
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
                print("Trainable:", name)

    @property
    def loss_fn(self) -> nn.Module:
        """Return the appropriate loss function for this model type"""
        raise NotImplementedError

    def compute_losses(
        self, batch: Sequence[BatchItem], epoch: int
    ) -> dict[str, Float[Tensor, ""]]:
        """One batch's losses, named and already weighted — the seam each model
        fills in.

        The values are what the optimizer steps on, so any epoch-dependent
        weighting a model applies to its objectives is applied here, not in the
        loop. The keys name the objectives this model trains: `run_epoch`
        accumulates under them and `print_epoch_stats` reports them.
        """
        raise NotImplementedError

    def on_epoch_start(self, step: Step, epoch: int) -> None:
        """Per-epoch diagnostics, before the first batch. A no-op by default."""

    def run_epoch(
        self,
        data: DataLoader,
        step: Step,
        epoch: int,
        optimization: Optimization | None = None,
    ) -> tuple[dict[str, float], int]:
        """Process every batch, stepping `optimization` once per batch.

        The loop steps if and only if it was given something to step, so a pass
        that is not meant to train cannot accidentally be handed an optimizer —
        and a training pass cannot silently fail to take one, which is what
        gating on `step` alone allowed.

        :raises ValueError: on `Step.TRAINING` with no `optimization`.
        :returns: the epoch's summed losses, by name, and the batch count they
            are to be averaged over.
        """
        if step is Step.TRAINING and optimization is None:
            raise ValueError("a training epoch needs an optimization to step")

        self.on_epoch_start(step, epoch)

        totals: dict[str, float] = {}
        n_batches = 0

        for batch in tqdm(
            data,
            dynamic_ncols=True,
            position=1,
            desc="Batches",
            leave=False,
        ):
            if optimization is not None:
                optimization.zero_grad()

            losses = self.compute_losses(batch, epoch)

            if optimization is not None:
                optimization.update(*losses.values())

            for name, loss in losses.items():
                totals[name] = (
                    totals.get(name, 0.0) + loss.detach().cpu().item()
                )
            n_batches += 1

            # The graph these losses hold is the epoch's peak memory; drop it
            # before the next forward, not after.
            del losses

        return totals, n_batches

    def save_config(self, path: str) -> None:
        save_model_config(self.config.model_dump(), path)

    def batch_input_tensors(
        self,
        batch: Sequence[BatchItem],
    ) -> dict[str, Integer[Tensor, "sequence token"]]:
        """Concatenate each document's ``[n_chunks, token]`` sequences along
        dim 0 into a single ``[sum(n_chunks), token]`` tensor per key.

        The per-document chunk tensors are concatenated as-is; flattening them
        into individual rows would collapse the result to 1-D and feed a
        mis-shaped ``input_ids`` to the base model (get_token_embeddings then
        slices this back into per-document chunks via ``doc_id``).
        """
        return {
            key: torch.concat(
                tuple(doc["sequence"][key] for doc in batch),
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

                if (
                    cpu_embeddings_cache is not None
                    and self.training
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
