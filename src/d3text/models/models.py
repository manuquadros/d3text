import itertools
import os
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from copy import deepcopy
from functools import partial
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import transformers
from cacheout import Cache
from d3text import data
from d3text.utils import (
    Token,
    aggregate_embeddings,
    merge_off_tokens,
    merge_predictions,
    split_and_tokenize,
    tokenize_cased,
)
from jaxtyping import Bool, Float, Int16, Int64, Integer, UInt8
from sklearn.metrics import classification_report
from torch import Tensor
from torch.autograd.profiler import record_function
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import BatchEncoding

from .config import (
    ModelConfig,
    embedding_dims,
    machine_config,
    optimizers,
    save_model_config,
    schedulers,
)
from .dict_tagger import DictTagger
from .model_types import BatchedLogits, IndexedRelation

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("medium")

mconfig = machine_config()
if mconfig.cuda_embeddings_cache_size:
    cuda_embeddings_cache = Cache(maxsize=mconfig.cuda_embeddings_cache_size)
else:
    cuda_embeddings_cache = None
if mconfig.cpu_embeddings_cache_size:
    cpu_embeddings_cache = Cache(maxsize=mconfig.cpu_embeddings_cache_size)
else:
    cpu_embeddings_cache = None


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
    batch: Sequence[dict[str, Tensor | BatchEncoding]], device: str = "cuda"
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

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()

        self.config = config if config is not None else ModelConfig()

        self.device = "cuda"
        self.scaler = torch.amp.GradScaler(self.device)

        if getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            print("bf16 supported")
            self.amp_dtype = torch.bfloat16
        else:
            print("bf16 not supported")
            self.amp_dtype = torch.float16

        is_rocm = getattr(torch.version, "hip", None) is not None
        device_name = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
        )
        print(device_name)
        bf16_ok = (not is_rocm) and getattr(
            torch.cuda, "is_bf16_supported", lambda: False
        )()

        if is_rocm and not any(
            k in device_name for k in ("MI200", "MI250", "MI300", "MI3")
        ):
            bf16_ok = False
        self.amp_dtype = torch.bfloat16 if bf16_ok else torch.float16
        print(self.amp_dtype)

        self.ramp_epochs: int = self.config.ramp_epochs

        self.checkpoint = "checkpoint.pt"
        self.best_model_state: dict[str, Any]

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
                print("Trainable:", name)

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

    def _setup_training(
        self,
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
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
        optimizer, scheduler = self._setup_training()

        self.stop_counter = 0
        self.best_model_state = None
        self.best_val_loss = float("inf")
        self.best_epoch = -1

        for epoch in trange(
            self.config.num_epochs,
            dynamic_ncols=True,
            position=0,
            desc="Epochs",
            leave=True,
        ):
            self.train()
            optimizer.zero_grad()
            loss = self.run_epoch(epoch=epoch, train_data=train_data)
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            if val_data is not None:
                val_loss = self.validate_model(
                    val_data=val_data, w_ent=w_ent, w_rel=w_rel
                )

                if scheduler is not None:
                    if self.config.lr_scheduler == "reduce_on_plateau":
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()

                tqdm.write(f"Average validation loss: {val_loss:.5f}")

                if epoch <= self.ramp_epochs:
                    self.stop_counter = 0
                early_stop = self.early_stop(
                    val_loss, save_checkpoint=save_checkpoint
                )
                if early_stop:
                    if save_checkpoint and self.best_model_state is not None:
                        print(
                            "Model converged. Loading the best epoch's parameters."
                        )
                        self.load_state_dict(self.best_model_state, strict=True)
                    break

            tqdm.write("-" * 50)

        if val_data is not None and output_loss:
            return self.best_val_loss
        return None

    def early_stop(self, val_loss: float, save_checkpoint: bool) -> bool:
        """Stop training after `self.config.patience` epochs have passed
        without improvement to `metric` according to the `goal`. Most likely
        we will want to minimize validation loss.

        If `save_checkpoint` is True, store the best model state in
        `self.best_model_state`.
        """
        if val_loss <= self.best_val_loss:
            self.best_val_loss = val_loss
            self.stop_counter = 0
            if save_checkpoint:
                self.best_model_state = deepcopy(self.state_dict())
        else:
            self.stop_counter += 1

        if self.stop_counter > self.config.patience:
            return True
        else:
            return False

    def save_model(self, path: str) -> None:
        try:
            torch.save(self.best_model_state, path)
        except NameError:
            print("The model has not been trained yet...")

    def validate_model(
        self, val_data: DataLoader, w_ent: float, w_rel: float
    ) -> float:
        self.eval()

        batch_ent_loss = 0.0
        batch_rel_loss = 0.0
        n_batches = 0

        with torch.inference_mode(), self.autocast_context():
            for batch in tqdm(
                val_data,
                dynamic_ncols=True,
                position=2,
                desc="Validation",
                leave=False,
            ):
                ent_loss, rel_loss = self.compute_batch_losses(batch)
                ent_loss = (ent_loss * w_ent).item()
                rel_loss = (rel_loss * w_rel).item()
                batch_ent_loss += ent_loss
                batch_rel_loss += rel_loss
                del rel_loss, ent_loss
                torch.cuda.empty_cache()
                n_batches += 1
            batch_loss = batch_ent_loss + batch_rel_loss
            loss = batch_loss / n_batches

        tqdm.write(
            "Average (entity) validation loss: "
            f"{batch_ent_loss / n_batches:.4f}"
        )
        tqdm.write(
            "Average (relation) validation loss: "
            f"{batch_rel_loss / n_batches:.4f}"
        )

        return loss

    def ids_to_tokens(
        self,
        ids: Iterable[int],
    ) -> list[str]:
        return self.tokenizer.convert_ids_to_tokens(ids)

    def save_config(self, path: str) -> None:
        save_model_config(self.config.model_dump(), path)


def print_epoch_stats(losses: dict[str, float], num_batches: int):
    for obj, loss in losses.items():
        tqdm.write(f"Average ({obj}) training loss: {loss / num_batches:.4f}")

    total_loss = sum(losses.values())
    tqdm.write(f"Average training loss: {total_loss / num_batches:.4f}")


class PermutationBatchNorm1d(nn.BatchNorm1d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = torch.permute(input, (0, 2, 1))
        out = torch.permute(super().forward(input), (0, 2, 1))
        return out


class BrendaClassificationModel(Model):
    def __init__(
        self, classes: Mapping[str, set[str]], config: None | ModelConfig = None
    ) -> None:
        super().__init__(config)
        self.classes = list(classes.keys()) + ["OOS"]

        self.entities = list(
            itertools.chain.from_iterable(classes.values())
        ) + ["UNK"]
        self.entity_to_class = {
            entity: cl for cl, ents in classes.items() for entity in ents
        }

        # The dataset does not include a `none` class, so we add one.
        self.num_of_entities = len(self.entities)
        self.num_of_classes = len(self.classes)

        self.build_layers(embedding_size=embedding_dims[self.config.base_model])

        self.base_model = transformers.AutoModel.from_pretrained(
            self.config.base_model
        )

        for param in self.base_model.parameters():
            param.requires_grad = False

        self.enable_gradient_checkpointing()

        # Initialize class matrix mapping each entity index to its entity
        # class index.
        class_matrix = torch.zeros(
            self.num_of_entities - 1,
            self.num_of_classes - 1,
            device=self.device,
        )
        self.entity_to_index = {
            eid: idx for idx, eid in enumerate(self.entities)
        }

        for entity_id, class_id in self.entity_to_class.items():
            ent_idx = self.entity_to_index[entity_id]
            class_idx = self.classes.index(class_id)
            class_matrix[ent_idx, class_idx] = 1
        self.register_buffer("class_matrix", class_matrix)

        self.classifier = ClassificationHead(
            input_size=self.hidden_block_output_size,
            n_entities=self.num_of_entities,
            n_classes=self.num_of_classes,
        )

        self.entity_logits_pooling = "logsumexp"
        self.entity_threshold = nn.Parameter(torch.tensor(0.7))
        self.evaluation = False

    def run_epoch(
        self, epoch: int, train_data: DataLoader
    ) -> Float[Tensor, ""]:
        """Process all batches, computing loss and printing diagnostics.

        :param epoch: epoch number
        :param train_data: DataLoader for the training data
        :returns: combined loss for epoch
        """
        epoch_ent_loss = torch.tensor(0.0)
        epoch_class_loss = torch.tensor(0.0)
        n_batches = 0

        for batch in tqdm(
            train_data,
            dynamic_ncols=True,
            position=1,
            desc="Batches",
            leave=False,
        ):
            ent_loss, class_loss = self.compute_batch_losses(batch)

            n_batches += 1
            epoch_ent_loss += ent_loss
            epoch_class_loss += class_loss
            del ent_loss, class_loss

        print_epoch_stats(
            losses={"entity": epoch_ent_loss, "class": epoch_class_loss},
            num_batches=n_batches,
        )

        return epoch_ent_loss + epoch_class_loss

    @property
    def entity_loss_fn(self) -> nn.Module:
        # weights = torch.ones(self.num_of_entities - 1, device=self.device)
        return nn.BCEWithLogitsLoss(reduction="mean")

    @property
    def class_loss_fn(self) -> nn.Module:
        # weights = torch.ones(self.num_of_classes - 1, device=self.device)
        # weights[-1] = 0
        return nn.BCEWithLogitsLoss(reduction="mean")

    def compute_entity_frequencies(
        self, dataloader: DataLoader, batch_accumulate: int = 512
    ) -> torch.Tensor:
        """Compute marginal frequency of each entity in the training dataset."""
        data = dataloader.dataset.data["entities"]

        all_entities = torch.stack(
            [
                torch.tensor(e, dtype=torch.float32)
                if not torch.is_tensor(e)
                else e.float()
                for e in data
            ]
        )

        freq = all_entities.mean(dim=0)  # shape: [num_entities]
        return freq.clamp(min=1e-5, max=1 - 1e-5)

    def batch_input_tensors(
        self,
        batch: Sequence[dict[str, Tensor | BatchEncoding]],
    ) -> dict[str, Integer[Tensor, "sequence token"]]:
        """Concatenate input tensors across the batch"""
        return {
            key: torch.concat(
                tuple(
                    itertools.chain.from_iterable(
                        map(lambda doc: doc["sequence"][key], batch)
                    )
                ),
                dim=0,
            )
            for key in ("input_ids", "attention_mask")
        }

    def get_token_embeddings(
        self, batch: Sequence[dict[str, Tensor | BatchEncoding]]
    ) -> tuple[
        Float[Tensor, "batch max_doc_len embedding"],
        UInt8[Tensor, "batch max_doc_len"],
    ]:
        device = self.device
        inputs: list[None | Tensor] = [None] * len(batch)
        missing: list[Tensor] = []

        for ix, item in enumerate(batch):
            doc_id: int = item["id"].item()
            if (
                cuda_embeddings_cache is not None
                and doc_id in cuda_embeddings_cache
            ):
                inputs[ix] = cuda_embeddings_cache.get(doc_id)
            else:
                cpu_cached = cpu_embeddings_cache.get(doc_id)
                if cpu_cached is not None:
                    inputs[ix] = cpu_cached.to(device, non_blocking=True)
                else:
                    missing.append((ix, item))

        if missing:
            with torch.no_grad():
                batched_inputs = self.batch_input_tensors(
                    [item for _, item in missing]
                )
                with self.autocast_context():
                    output = self.base_model(
                        input_ids=batched_inputs["input_ids"].to(
                            device, dtype=torch.int, non_blocking=True
                        ),
                        attention_mask=batched_inputs["attention_mask"].to(
                            device, non_blocking=True
                        ),
                    ).last_hidden_state

            out_iter = iter(output)
            masks_iter = iter(batched_inputs["attention_mask"])
            for ix, item in missing:
                number_of_sequences_for_item = item["doc_id"].shape[-1]
                outs = torch.stack(
                    tuple(
                        itertools.islice(out_iter, number_of_sequences_for_item)
                    )
                ).to(torch.float16)
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
                    not self.evaluation
                    and cuda_embeddings_cache is not None
                    and not cuda_embeddings_cache.full()
                ):
                    cuda_embeddings_cache.set(item["id"].item(), doc_embedding)
                elif not cpu_embeddings_cache.full():
                    cpu_embeddings_cache.set(
                        item["id"].item(), doc_embedding.cpu()
                    )

        max_doc_len = max(emb.shape[0] for emb in inputs)
        padded_embeddings = pad_sequence(
            inputs, batch_first=True, padding_value=0.0
        )
        attention_masks = torch.zeros(
            (len(inputs), max_doc_len), dtype=torch.uint8, device=device
        )
        for i, emb in enumerate(inputs):
            attention_masks[i, : emb.shape[0]] = 1

        return padded_embeddings, attention_masks

    def compute_entity_loss(
        self,
        predictions: tuple[Tensor, Tensor],
        targets: tuple[Tensor, Tensor],
        class_scale: float = 1,
    ) -> tuple[Float[Tensor, ""], Float[Tensor, ""]]:
        entity_loss = self.entity_loss_fn(
            predictions[0][..., :-1].float(),
            targets[0].float(),
        )
        class_loss = self.class_loss_fn(
            predictions[1][..., :-1].float(),
            targets[1].float(),
        )
        return entity_loss, class_loss

    def compute_batch_losses(
        self, batch: Sequence[Mapping[str, BatchEncoding | Tensor]]
    ) -> tuple[Float[Tensor, ""], Float[Tensor, ""]]:
        ent_true, class_true = self.ground_truth(batch)
        entity_logits, class_logits = self.get_batch_logits(batch)

        return self.compute_entity_loss(
            predictions=(entity_logits, class_logits),
            targets=(ent_true, class_true),
        )

    def get_batch_logits(
        self,
        batch: Sequence[dict[str, Tensor | BatchEncoding]],
        gold_relations: list[IndexedRelation] | None = None,
    ) -> tuple[
        Float[Tensor, "sequence entities"],
        Float[Tensor, "sequence classes"],
    ]:
        token_embeddings, token_att_mask = self.get_token_embeddings(batch)
        entities_in_batch = get_batch_entities(batch)

        entity_logits, class_logits = self(
            token_embeddings,
            token_att_mask,
            entities_in_batch,
        )

        return (
            entity_logits,
            class_logits,
        )

    def ground_truth(
        self,
        batch: Sequence[Mapping[str, BatchEncoding | Tensor]],
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

        class_targets = (
            entity_targets.to(dtype=self.class_matrix.dtype) @ self.class_matrix
        ).clamp(max=1)

        return entity_targets.float(), class_targets.float()

    @record_function("forward")
    def forward(
        self,
        embeddings: Float[Tensor, "document token embedding"],
        attention_mask: Integer[Tensor, "document token"],
        entities_in_batch: tuple[Int16[Tensor, " entities"], ...],
    ) -> tuple[
        BatchedLogits,
        BatchedLogits,
    ]:
        """Forward pass

        :return: tuple containing:
            - Entity logits pooled by document.
            - Class logits pooled by document.
        """
        device = self.device
        with self.autocast_context():
            hidden_output: Float[Tensor, "document token features"] = (
                self.hidden(embeddings)
            )
            unmasked_entity_logits, unmasked_class_logits = self.classifier(
                hidden_output
            )
            token_mask = attention_mask.to(
                dtype=torch.bool, device=device
            ).unsqueeze(-1)
            neg_inf = torch.tensor(-1e9, device=device)
            entity_logits = torch.where(
                token_mask, unmasked_entity_logits, neg_inf
            )
            class_logits = torch.where(
                token_mask, unmasked_class_logits, neg_inf
            )

            with torch.autocast(device_type=self.device, enabled=False):
                pooled_entities = torch.logsumexp(entity_logits, dim=1)
                pooled_classes = torch.logsumexp(class_logits, dim=1)
            return (
                pooled_entities.to(entity_logits.dtype),
                pooled_classes.to(class_logits.dtype),
            )


class BiaffineRelationClassifier(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_relations: int,
        separate_predicate_layer: bool = False,
    ):
        super().__init__()
        self.separate_predicate_layer = separate_predicate_layer
        biaff_hidden_size = 32
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
    def __init__(
        self, classes: Mapping[str, set[str]], config: None | ModelConfig = None
    ) -> None:
        super().__init__(
            classes,
            config,
        )

        self.relations = ("HasEnzyme", "HasSpecies", "none")
        self.relations_none_index = self.relations.index("none")
        self.num_relations = len(self.relations)
        self.relation_classifier = BiaffineRelationClassifier(
            hidden_size=self.hidden_block_output_size,
            num_relations=len(self.relations),
        )

        self.relation_label_smoothing = self.config.relation_label_smoothing

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
        w_ent = 1.0 - 0.3 * w_rel  # decays from 1.0 -> 0.7
        return w_ent, w_rel

    def run_epoch(self, epoch: int, train_data: DataLoader) -> float:
        """Process all batches, computing loss and printing diagnostics.

        :param epoch: epoch number
        :param train_data: DataLoader for the training data
        :returns: combined loss for epoch
        """
        epoch_ent_loss = 0.0
        epoch_rel_loss = 0.0
        n_batches = 0
        w_ent, w_rel = self.get_loss_weights(epoch)

        for batch in tqdm(
            train_data,
            dynamic_ncols=True,
            position=1,
            desc="Batches",
            leave=False,
        ):
            if n_batches == 0:
                tqdm.write(
                    f"Epoch {epoch}: w_ent={w_ent:.3f}, w_rel={w_rel:.3f}"
                )

            ent_loss, rel_loss = self.compute_batch_losses(batch)
            ent_loss_scaled = ent_loss * w_ent
            rel_loss_scaled = rel_loss * w_rel
            del ent_loss, rel_loss

            epoch_ent_loss += ent_loss_scaled.item()
            epoch_rel_loss += rel_loss_scaled.item()
            n_batches += 1

            # del loss
            del rel_loss_scaled, ent_loss_scaled
            torch.cuda.empty_cache()

        print_epoch_stats(
            {"entity": epoch_ent_loss, "relation": epoch_rel_loss}
        )

        return epoch_ent_loss + epoch_rel_loss

    def ground_truth(
        self,
        batch: Sequence[Mapping[str, BatchEncoding | Tensor]],
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

        pool_fn = get_pool_fn(self.entity_logits_pooling)

        def _as_list(x: Tensor):
            return x.detach().cpu().tolist()

        seq_list = _as_list(rel_meta.get("sequence"))
        subj_list = _as_list(rel_meta.get("arg_pred_i"))
        obj_list = _as_list(rel_meta.get("arg_pred_j"))

        n_rows = rel_logits.size(0)
        assert (
            len(seq_list) == n_rows
            and len(subj_list) == n_rows
            and len(obj_list) == n_rows
        ), "rel_meta fields must align with rel_logits rows"

        device = rel_logits.device
        num_rel = rel_logits.size(1)

        # Build grouping of row indices per (doc, subj_ix, obj_ix)
        groups = defaultdict(list)
        for row_idx, (d, i, j) in enumerate(zip(seq_list, subj_list, obj_list)):
            groups[(int(d), int(i), int(j))].append(row_idx)

        if not groups:
            return None

        # Build a quick lookup of gold labels per triple
        gold_by_key = defaultdict(list)
        for tr in true_relations:
            try:
                subj_ix = int(self.entity_to_index[tr.subject])
                obj_ix = int(self.entity_to_index[tr.object])
            except KeyError:
                continue  # gold refers to entity not mapped in this doc/batch
            gold_by_key[(int(tr.docix), subj_ix, obj_ix)].append(int(tr.label))

        # Prepare pooled outputs
        pooled_logits = []
        pooled_targets = []
        pooled_seq = []
        pooled_subj = []
        pooled_obj = []

        none_idx = self.relations_none_index

        # Pool each group's logits and assign target
        for (d, i, j), row_idxs in groups.items():
            # Stack rows -> [k, num_rel]
            group_logits = rel_logits[row_idxs]  # lives on device

            # Numerically stable pooling over duplicates
            with torch.autocast(device_type=self.device, enabled=False):
                pooled = torch.logsumexp(group_logits.float(), dim=0)
            pooled = pooled.to(rel_logits.dtype)

            # Target: default none, overwrite if gold(s) exist
            labels = gold_by_key.get((d, i, j))
            if labels:
                # If multiple labels exist, prefer any non-none; else first.
                # (Adjust policy if your schema allows multi-label relations.)
                if any(lbl != none_idx for lbl in labels):
                    target = next(lbl for lbl in labels if lbl != none_idx)
                else:
                    target = labels[0]
            else:
                target = int(none_idx)

            pooled_logits.append(pooled)
            pooled_targets.append(target)
            pooled_seq.append(d)
            pooled_subj.append(i)
            pooled_obj.append(j)

        pooled_logits = torch.stack(pooled_logits, dim=0).to(device)
        pooled_targets = torch.tensor(
            pooled_targets, dtype=torch.long, device=device
        )

        pooled_meta = {
            "sequence": torch.tensor(
                pooled_seq, dtype=torch.long, device=device
            ),
            "arg_pred_i": torch.tensor(
                pooled_subj, dtype=torch.long, device=device
            ),
            "arg_pred_j": torch.tensor(
                pooled_obj, dtype=torch.long, device=device
            ),
        }

        return pooled_meta, pooled_logits, pooled_targets

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
        else:
            _, preds, targets = aligned_rel_preds
            loss_fn = torch.nn.CrossEntropyLoss(
                reduction="mean", label_smoothing=self.relation_label_smoothing
            )
            loss = loss_fn(preds, targets)
            return loss

    def get_batch_logits(
        self,
        batch: Sequence[dict[str, Tensor | BatchEncoding]],
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
        self, batch: Sequence[dict[str, Tensor | BatchEncoding]]
    ) -> tuple[Float[Tensor, ""], Float[Tensor, ""]]:
        """Compute loss for a batch."""
        ent_true, class_true, rel_true = self.ground_truth(batch)
        entity_logits, class_logits, relation_index_logits = (
            self.get_batch_logits(batch, gold_relations=rel_true)
        )

        ent_loss = self.compute_entity_loss(
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

        return ent_loss, relation_loss

    def compute_batch_true_x_pred(
        self, batch: Sequence[dict[str, Tensor | BatchEncoding]]
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
            print(
                f"relations_true {relations_true.shape} "
                "!= relations_pred {relations_pred.shape}"
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
        entity_preds: Int64[Tensor, "entities"] = max_indices[
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
                    torch.logsumexp(local_reprs[local_preds == pred], dim=0)
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
        attention_mask: Integer[Tensor, "document token"],
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
            token_mask = attention_mask.to(
                dtype=torch.bool, device=device
            ).unsqueeze(-1)
            neg_inf = torch.tensor(-1e9, device=device)
            entity_logits = torch.where(
                token_mask, unmasked_entity_logits, neg_inf
            )
            class_logits = torch.where(
                token_mask, unmasked_class_logits, neg_inf
            )

            # Find entity positions
            entity_probs: Float[Tensor, "document token ent_probs"] = (
                torch.softmax(entity_logits, dim=-1)
            )
            entropy = -(
                entity_probs * (entity_probs.clamp_min(1e-9)).log()
            ).sum(-1)

            max_indices = entity_probs.argmax(dim=-1)
            hard_entity_mask: Bool[Tensor, "document token"]
            hard_entity_mask = (max_indices != self.num_of_entities - 1) & (
                entropy <= 0.8
            )

            def _pooled_output(relogits):
                with torch.autocast(device_type=self.device, enabled=False):
                    pooled_entities = torch.logsumexp(entity_logits, dim=1)
                    pooled_classes = torch.logsumexp(class_logits, dim=1)
                return (
                    pooled_entities.to(entity_logits.dtype),
                    pooled_classes.to(class_logits.dtype),
                    relogits,
                )

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
                needed_by_doc = {}
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
                    reps = soft_repr_by_doc.get(doc_ix)
                    if not reps:
                        continue
                    subj = int(self.entity_to_index.get(tr.subject, -1))
                    obj = int(self.entity_to_index.get(tr.object, -1))
                    if subj in reps and obj in reps:
                        rows_doc.append(doc_ix)
                        rows_i.append(subj)
                        rows_j.append(obj)
                        rep_i.append(reps[subj])
                        rep_j.append(reps[obj])

                if rep_i:
                    rep_i = torch.stack(rep_i, dim=0)
                    rep_j = torch.stack(rep_j, dim=0)
                    logits = self.relation_classifier(rep_i, rep_j)
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
            merged = None
            if rel_meta_logits and gold_meta_logits:
                (m1, l1), (m2, l2) = rel_meta_logits, gold_meta_logits
                merged_meta = {
                    "sequence": torch.cat([m1["sequence"], m2["sequence"]]),
                    "arg_pred_i": torch.cat(
                        [m1["arg_pred_i"], m2["arg_pred_i"]]
                    ),
                    "arg_pred_j": torch.cat(
                        [m1["arg_pred_j"], m2["arg_pred_j"]]
                    ),
                }
                merged_logits = torch.cat([l1, l2], dim=0)
                merged = (merged_meta, merged_logits)
            else:
                merged = rel_meta_logits or gold_meta_logits

            return _pooled_output(merged)

    def evaluate_model(
        self,
        test_data: DataLoader,
        tau_ids: float = 0.5,
        tau_cls: float = 0.5,
        topk_ids: int | None = None,
    ) -> None:
        """
        Evaluate the end-to-end model from *document-level pooled logits*.
        - tau_ids / tau_cls: global thresholds for multilabel binarization
        - topk_ids: also keep top-K entity IDs per document
        """
        import numpy as np
        from sklearn.metrics import (
            f1_score,
            label_ranking_average_precision_score,
        )

        self.eval()
        all_id_logits, all_id_true = [], []
        all_cls_logits, all_cls_true = [], []
        all_rel_logits, all_rel_true = [], []  # we'll argmax rel later

        with torch.no_grad():
            # do NOT autocast around metric collection; keep numerics simple
            for batch in tqdm(test_data, desc="Evaluating"):
                # 1) pooled doc-level logits
                id_logits_doc, cls_logits_doc, rel_meta_logits = (
                    self.get_batch_logits(batch)
                )  # shapes: [B, num_ids], [B, num_classes], (meta, [N_pairs,R]) or None

                # 2) document-level multi-hot targets
                id_true_doc, cls_true_doc, rel_true_list = self.ground_truth(
                    batch
                )  # id_true_doc: [B,num_ids], cls_true_doc: [B,num_classes], rel_true_list: list[...]

                all_id_logits.append(id_logits_doc.detach().float().cpu())
                all_id_true.append(id_true_doc.detach().to(torch.int64).cpu())

                all_cls_logits.append(cls_logits_doc.detach().float().cpu())
                all_cls_true.append(cls_true_doc.detach().to(torch.int64).cpu())

                # 3) relations: gather pooled pair logits + integer labels (if any)
                #    Use your existing aligner to get one row per (doc, s, o)
                if rel_meta_logits is not None:
                    rel_meta, rel_logits = rel_meta_logits  # [N_pairs,R]
                    # Build integer targets aligned to the pairs; default none
                    none_idx = getattr(
                        self, "relation_none_index", len(self.relations) - 1
                    )
                    targets = torch.full(
                        (rel_logits.size(0),),
                        none_idx,
                        dtype=torch.long,
                        device=rel_logits.device,
                    )

                    # Map (doc, i, j) -> row
                    key_to_row = {
                        (int(d), int(i), int(j)): r
                        for r, (d, i, j) in enumerate(
                            zip(
                                rel_meta["sequence"].tolist(),
                                rel_meta["arg_pred_i"].tolist(),
                                rel_meta["arg_pred_j"].tolist(),
                            )
                        )
                    }
                    for tr in rel_true_list:
                        try:
                            k = (
                                int(tr.docix),
                                int(self.entity_to_index[tr.subject]),
                                int(self.entity_to_index[tr.object]),
                            )
                            r = key_to_row.get(k)
                            if r is not None:
                                targets[r] = int(tr.label)
                        except KeyError:
                            pass

                    all_rel_logits.append(rel_logits.detach().cpu())
                    all_rel_true.append(targets.detach().cpu())

        # ----- stack
        if not all_id_logits:
            print("No samples found.")
            return

        id_logits = torch.cat(all_id_logits, dim=0).numpy()
        id_true = torch.cat(all_id_true, dim=0).numpy().astype(int)
        cls_logits = torch.cat(all_cls_logits.float(), dim=0).numpy()
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
        print(
            f"\n[Entities] gold positives: {int(id_true.sum())} | predicted positives: {int(id_pred.sum())} | classes with any preds: {int((id_pred.sum(axis=0) > 0).sum())}"
        )
        print(
            f"[Classes ] gold positives: {int(cls_true.sum())} | predicted positives: {int(cls_pred.sum())}"
        )

        # ======= METRICS =======

        # Entities (6k+ labels): prefer micro-F1 + LRAP; macro over frequent labels only
        print("\n=== Entity ID metrics (multilabel, document-level) ===")
        try:
            print(
                "micro-F1:",
                f1_score(id_true, id_pred, average="micro", zero_division=0),
            )
        except ValueError:
            print("micro-F1: (no positive labels or predictions) 0.0")

        try:
            print(
                "LRAP:",
                label_ranking_average_precision_score(id_true, id_probs),
            )
        except ValueError:
            print("LRAP: undefined (no positives)")

        # macro-F1 over frequent labels
        support = id_true.sum(axis=0)
        keep = np.where(support >= 10)[0]  # tweak threshold as you like
        if keep.size > 0:
            print(
                "macro-F1 (support>=10):",
                f1_score(
                    id_true[:, keep],
                    id_pred[:, keep],
                    average="macro",
                    zero_division=0,
                ),
            )
        else:
            print(
                "macro-F1 (support>=10): n/a (no labels meet support threshold)"
            )

        # Optionally, print per-label report for a *small* head of frequent IDs
        # idx_items = sorted(self.entity_to_index.items(), key=lambda kv: kv[1])
        # frequent_names = [idx_items[i][0] for i in keep[:50]]
        # print(classification_report(id_true[:, keep[:50]], id_pred[:, keep[:50]], target_names=frequent_names, zero_division=0))

        # Classes (small set): full report is fine
        print("\n=== Entity CLASS metrics (multilabel, document-level) ===")
        print(
            "micro-F1:",
            f1_score(cls_true, cls_pred, average="micro", zero_division=0),
        )
        print(
            classification_report(
                y_true=cls_true,
                y_pred=cls_pred,
                target_names=list(self.classes),
                zero_division=0,
            )
        )

        # Relations (multiclass over candidate pairs)
        if all_rel_logits:
            rel_logits = torch.cat(all_rel_logits, dim=0).numpy()
            rel_true = torch.cat(all_rel_true, dim=0).numpy().astype(int)
            rel_pred = rel_logits.argmax(axis=1)

            print(
                "\n=== Relation metrics (multiclass over candidate pairs) ==="
            )
            labels = np.arange(len(self.relations))
            print(
                classification_report(
                    y_true=rel_true,
                    y_pred=rel_pred,
                    labels=labels,
                    target_names=list(self.relations),
                    zero_division=0,
                )
            )
        else:
            print("\n(No relation pairs produced on this split.)")


class ClassificationHead(nn.Module):
    """Define a classification head for end-to-end models."""

    def __init__(
        self, input_size: int, n_entities: int, n_classes: int
    ) -> None:
        """Initialize the classification head.

        :param input_size: number of input features
        :param n_entities: number of output entities
        :param n_classes: number of output entity classes
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

    def initialize_classifier_bias(self, entity_freqs: torch.Tensor) -> None:
        """Initialize classifier bias using log odds from entity frequencies."""
        last: nn.Linear = self.entity_classifier[-1]
        with torch.no_grad():
            log_odds = torch.log(entity_freqs / (1 - entity_freqs))
            last.bias.copy_(log_odds.to(last.weight.device))

    def forward(self, input: Tensor) -> tuple[Tensor, Tensor]:
        entity_logits = self.entity_classifier(input)
        class_logits = self.class_classifier(input)

        return entity_logits, class_logits


class NERCTagger(Model):
    def __init__(
        self,
        config: None | ModelConfig = None,
    ) -> None:
        super().__init__(config)

        self.num_labels = len(config.classes)
        self.classifier = nn.Linear(
            self.hidden_block_output_size, self.num_labels
        )

    def forward(self, input_data: dict) -> torch.Tensor:
        x = self.dropout(self.base_model(**input_data).last_hidden_state)
        x = self.hidden(x)
        x = self.classifier(x)

        return x

    def train_model(
        self,
        train_data: data.DatasetConfig,
        val_data: data.DatasetConfig | None = None,
        save_checkpoint: bool = True,
        output_loss: bool = True,
    ) -> float | None:
        """Generic training loop for all models"""
        self.config.classes = train_data.classes.tolist()

        super().train_model(
            train_data=train_data,
            val_data=val_data,
            save_checkpoint=save_checkpoint,
            output_loss=output_loss,
        )

    @property
    def loss_fn(self, train_data: data.DatasetConfig) -> nn.Module:
        return nn.CrossEntropyLoss(
            weight=train_data.class_weights.to(self.device),
            ignore_index=train_data.null_index,
        )

    def predict(self, inputs: str | list[str]) -> Iterator[list[Token]]:
        dict_tagger = DictTagger(
            {
                "Enzyme": self.config.enzymes_list,
                "Bacteria": self.config.species_list,
                "Strains": self.config.strains_list,
            }
        )

        return (
            list(dict_tagger.tag(merge_off_tokens(sequence)))
            for sequence in self.get_predictions(inputs)
        )

    def logits_to_tags(
        self, logits: Float[Tensor, "length labels"]
    ) -> list[str]:
        return [self.config.classes[pos.argmax()] for pos in logits]

    def get_predictions(self, inputs: str | list[str]) -> Iterator[list[Token]]:
        self.eval()
        if isinstance(inputs, str):
            inputs = [inputs]

        stride = 50
        tokenized = split_and_tokenize(
            tokenizer=self.tokenizer, inputs=inputs, stride=stride
        )

        with torch.no_grad():
            predictions = self(
                {
                    k: torch.tensor(tokenized[k], device=self.device)
                    for k in ("input_ids", "attention_mask")
                }
            )

        get_cased = partial(
            tokenize_cased, tokenizer=self.tokenizer, clssep=True
        )
        sequences: Iterator[list[str]] = itertools.chain(
            *(map(get_cased, inputs))
        )

        probs, indices = torch.max(torch.softmax(predictions, dim=-1), dim=-1)

        tags = (
            [self.config.classes[ix] for ix in sample] for sample in indices
        )

        tagged_tokens: Iterator[list[Token]] = (
            [
                Token(s, off, pred, prob=prob.data.item())
                for s, off, pred, prob in zip(tokens, offsets, ts, probs)
            ]
            for tokens, offsets, ts, probs in zip(
                sequences, tokenized["offset_mapping"], tags, probs
            )
        )

        return merge_predictions(
            tagged_tokens, tokenized["overflow_to_sample_mapping"], stride
        )
