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
from jaxtyping import Bool, Float, Int16, Int64, Integer
from sklearn.metrics import classification_report
from torch import Tensor
from torch.autograd.profiler import record_function
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
torch.set_float32_matmul_precision("medium")

mconfig = machine_config()
cuda_embeddings_cache = Cache(maxsize=mconfig.cuda_embeddings_cache_size)
cpu_embeddings_cache = Cache(maxsize=mconfig.cpu_embeddings_cache_size)


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
        for seq in doc["doc_id"][0]:
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

        self.checkpoint = "checkpoint.pt"
        self.best_score: float = 0.0
        self.best_model_state: dict[str, Any]

    def build_layers(self, embedding_size: int) -> None:
        # Common layers setup
        self.dropout = (
            nn.Dropout(self.config.dropout)
            if self.config.dropout
            else nn.Identity()
        )

        self.hidden_layers = nn.ModuleList()
        in_features = embedding_size

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

        self.hidden_block_output_size = in_features

        def hidden_forward(x):
            for layer in self.hidden_layers:
                x = layer(x)
            return x

        self.hidden = hidden_forward

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for all compatible modules."""
        if hasattr(self.base_model, "gradient_checkpointing_enable") and any(
            param.requires_grad for param in self.base_model.parameters()
        ):
            self.base_model.gradient_checkpointing_enable()

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
        save_checkpoint: bool = False,
        output_loss: bool = True,
    ) -> float | None:
        """Generic training loop for all models"""
        optimizer, scheduler = self._setup_training()

        self.stop_counter = 0

        for epoch in trange(
            self.config.num_epochs,
            dynamic_ncols=True,
            position=0,
            desc="Epochs",
            leave=True,
        ):
            self.train()
            batch_ent_loss: float = 0.0
            batch_rel_loss = 0.0
            n_batches = 0

            for batch in tqdm(
                train_data,
                dynamic_ncols=True,
                position=1,
                desc="Batches",
                leave=False,
            ):
                optimizer.zero_grad()
                with torch.autocast(device_type=self.device):
                    ent_loss, rel_loss = self.compute_batch_losses(batch)
                loss = self.ent_scale * ent_loss + rel_loss
                batch_ent_loss += ent_loss
                batch_rel_loss += rel_loss
                n_batches += 1

                # self.scaler.scale(loss).backward()
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

                # del loss
                del rel_loss, ent_loss, loss
                torch.cuda.empty_cache()

            batch_loss = batch_rel_loss + batch_ent_loss
            tqdm.write(
                f"Average (entity) training loss: {batch_ent_loss.item() / n_batches:.4f}"
            )
            tqdm.write(
                f"Average (relation) training loss: {batch_rel_loss.item() / n_batches:.4f}"
            )
            tqdm.write(
                f"Average training loss: {batch_loss.item() / n_batches:.4f}"
            )

            if val_data is not None:
                val_loss = self.validate_model(val_data=val_data)

                if self.config.lr_scheduler == "reduce_on_plateau":
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

                tqdm.write(f"Average validation loss: {val_loss:.5f}")

                early_stop = self.early_stop(
                    val_loss, save_checkpoint=save_checkpoint
                )
                if early_stop:
                    if save_checkpoint:
                        print(
                            "Model converged. Loading the best epoch's parameters."
                        )
                        self.load_state_dict(self.best_model_state)
                    break

            tqdm.write("-" * 50)

        if val_data is not None and output_loss:
            return self.best_score
        return None

    def early_stop(
        self, metric: float, save_checkpoint: bool, goal: str = "min"
    ) -> bool:
        """Stop training after `self.config.patience` epochs have passed
        without improvement to `metric` according to the `goal`. Most likely
        we will want to minimize validation loss.

        If `save_checkpoint` is True, store the best model state in
        `self.best_model_state`.
        """
        if not self.best_score:
            self.best_score = metric

        if (goal == "min" and metric <= self.best_score) or (
            goal == "max" and metric >= self.best_score
        ):
            self.best_score = metric
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
        self,
        val_data: DataLoader,
    ) -> float:
        self.eval()

        batch_ent_loss = 0.0
        batch_rel_loss = 0.0
        n_batches = 0

        with torch.inference_mode(), torch.autocast(device_type=self.device):
            for batch in tqdm(
                val_data,
                dynamic_ncols=True,
                position=2,
                desc="Validation",
                leave=False,
            ):
                ent_loss, rel_loss = self.compute_batch_losses(batch)
                batch_ent_loss += ent_loss
                batch_rel_loss += rel_loss
                del rel_loss, ent_loss
                n_batches += 1
            batch_loss = batch_ent_loss + batch_rel_loss
            loss = batch_loss / n_batches

        tqdm.write(
            "Average (entity) validation loss: "
            f"{batch_ent_loss.item() / n_batches:.4f}"
        )
        tqdm.write(
            "Average (relation) validation loss: "
            f"{batch_rel_loss.item() / n_batches:.4f}"
        )

        return loss.item()

    def ids_to_tokens(
        self,
        ids: Iterable[int],
    ) -> list[str]:
        return self.tokenizer.convert_ids_to_tokens(ids)

    def save_config(self, path: str) -> None:
        save_model_config(self.config.model_dump(), path)


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
        self.classes = tuple(classes.keys())

        self.entities = tuple(itertools.chain.from_iterable(classes.values()))
        self.entity_to_class = {
            entity: cl for cl, ents in classes.items() for entity in ents
        }

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
            self.num_of_entities, self.num_of_classes, device=self.device
        )
        self.entity_to_index = {
            eid: idx for idx, eid in enumerate(self.entities)
        }

        for entity_id, class_id in self.entity_to_class.items():
            ent_idx = self.entity_to_index[entity_id]
            class_idx = self.classes.index(class_id)
            class_matrix[ent_idx, class_idx] = 1
        self.register_buffer("class_matrix", class_matrix)

    def initialize_classifier_bias(self, entity_freqs: torch.Tensor) -> None:
        """Initialize classifier bias using log odds from entity frequencies."""
        with torch.no_grad():
            log_odds = torch.log(entity_freqs / (1 - entity_freqs))
            self.entity_classifier.bias.copy_(log_odds.to(self.device))

    @property
    def loss_fn(self) -> nn.Module:
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
    ) -> Float[Tensor, "tokens embedding"]:
        inputs: list[None | Tensor] = [None] * len(batch)
        missing: list[Tensor] = []

        for ix, item in enumerate(batch):
            doc_id: int = item["id"].item()
            cached: Tensor | None = cuda_embeddings_cache.get(doc_id)
            if cached is not None:
                inputs[ix] = cached
            else:
                cpu_cached = cpu_embeddings_cache.get(doc_id)
                if cpu_cached is not None:
                    inputs[ix] = cpu_cached.to(self.device, non_blocking=True)
                else:
                    missing.append((ix, item))

        if missing:
            with torch.no_grad():
                batched_inputs = self.batch_input_tensors(
                    [item for _, item in missing]
                )
                with torch.autocast(device_type=self.device):
                    output = self.base_model(
                        input_ids=batched_inputs["input_ids"].to(
                            self.device, dtype=torch.int, non_blocking=True
                        ),
                        attention_mask=batched_inputs["attention_mask"].to(
                            self.device, non_blocking=True
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
                )
                masks = torch.stack(
                    tuple(
                        itertools.islice(
                            masks_iter, number_of_sequences_for_item
                        )
                    )
                )
                inputs[ix] = aggregate_embeddings(outs, masks)
                if not cuda_embeddings_cache.full():
                    cuda_embeddings_cache.set(item["id"].item(), outs)
                elif not cpu_embeddings_cache.full():
                    cpu_embeddings_cache.set(item["id"].item(), outs.cpu())

        return torch.concat(inputs)

    def compute_batch_losses[T: (Tensor, tuple[Tensor, ...])](
        self, batch: Sequence[Mapping[str, BatchEncoding | Tensor]]
    ) -> T:
        raise NotImplementedError


class BiaffineRelationClassifier(nn.Module):
    def __init__(self, hidden_size: int, num_relations: int):
        super().__init__()
        self.bilinear = nn.Parameter(
            torch.randn(num_relations, hidden_size, hidden_size)
        )
        nn.init.xavier_uniform_(self.bilinear)
        self.linear = nn.Linear(hidden_size * 2, num_relations)
        self.bias = nn.Parameter(torch.zeros(num_relations))

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        # x, y: [B, D]
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
        self.classifier = ClassificationHead(
            input_size=self.hidden_block_output_size,
            n_entities=self.num_of_entities,
            n_classes=self.num_of_classes,
        )

        self.relations = ("HasEnzyme", "HasSpecies", "none")
        self.num_relations = len(self.relations)
        self.relation_classifier = BiaffineRelationClassifier(
            hidden_size=self.hidden_block_output_size,
            num_relations=len(self.relations),
        )

        self.entity_logits_pooling = "logsumexp"
        self.entity_threshold = nn.Parameter(torch.tensor(0.7))
        self.evaluation = False
        self.ent_scale = self.config.entity_loss_scaling_factor
        self.relation_label_smoothing = self.config.relation_label_smoothing

    # @torch.compile
    def ground_truth(
        self,
        batch: Sequence[Mapping[str, BatchEncoding | Tensor]],
    ) -> tuple[
        Float[Tensor, "batch doc entities"],
        Float[Tensor, "batch doc classes"],
        list[IndexedRelation],
    ]:
        """Get ground truth for each document in the batch

        :param: Batch of documents.
        :return: Tuple containing:
            - Multi-hot encoded tensor, where each position of dim 2
              specifies whether the entity corresponding to that index occurs in
              the particular document along dim 1.
            - Idem for class labels
            - Tuple
        """
        entity_targets = torch.stack(
            tuple(doc["entities"] for doc in batch)
        ).to(self.device)

        class_targets = (
            entity_targets.to(dtype=self.class_matrix.dtype) @ self.class_matrix
        ).clamp(max=1)

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

        return entity_targets.float(), class_targets.float(), relation_targets

    def _dummy_relation_logits(self) -> Tensor:
        dummy_input1 = torch.zeros(
            (1, self.hidden_block_output_size),
            device=self.device,
            requires_grad=True,
        )
        dummy_input2 = torch.zeros(
            (1, self.hidden_block_output_size),
            device=self.device,
            requires_grad=True,
        )

        dummy = self.relation_classifier(dummy_input1, dummy_input2)

        return dummy

    def align_relation_predictions(
        self,
        true_relations: Sequence[IndexedRelation],
        rel_meta: dict[str, Tensor],
        rel_logits: Float[Tensor, "relation logits"] | None,
    ) -> (
        tuple[Int64[Tensor, " relation"], Float[Tensor, "relation logits"]]
        | None
    ):
        pool_fn = get_pool_fn(self.entity_logits_pooling)
        if not rel_meta:
            rel_meta = {
                "sequence": torch.empty(
                    0, dtype=torch.long, device=self.device
                ),
                "arg_pred_i": torch.empty(
                    0, dtype=torch.long, device=self.device
                ),
                "arg_pred_j": torch.empty(
                    0, dtype=torch.long, device=self.device
                ),
            }

        # Build a lookup from (doc_id, frozenset({arg1, arg2})) to predicted index list
        rel_lookup = defaultdict(list)

        for idx, (doc_ix, i, j) in enumerate(
            zip(
                rel_meta["sequence"].tolist(),
                rel_meta["arg_pred_i"].tolist(),
                rel_meta["arg_pred_j"].tolist(),
            )
        ):
            rel_lookup[(doc_ix, i, j)].append(idx)

        preds = []
        targets: list[Integer[Tensor, ""]] = []
        matched = 0.0
        doc_rel_keys = []

        for truerel in true_relations:
            try:
                subj_ix = self.entity_to_index[truerel.subject]
                obj_ix = self.entity_to_index[truerel.object]
            except KeyError:
                continue
            else:
                targets.append(truerel.label)
                rel_key = (truerel.docix, subj_ix, obj_ix)
                doc_rel_keys.append(rel_key)
                match_indices = rel_lookup.get(rel_key)
                if match_indices and rel_logits is not None:
                    logits_to_pool = rel_logits[match_indices]
                    pooled = pool_fn(logits_to_pool).unsqueeze(0)
                    matched += 1
                else:
                    pooled = self._dummy_relation_logits()

                preds.append(pooled)

        if self.training:
            # Penalize predicted entities that are not in gold entities
            for _ in range(len(set(rel_lookup) - set(doc_rel_keys))):
                preds.append(self._dummy_relation_logits())
                targets.append(
                    torch.tensor(
                        self.num_relations - 1,
                        device=self.device,
                    )
                )

        if targets:
            return (
                torch.tensor(targets, dtype=torch.int64, device=self.device),
                torch.cat(preds, dim=0),
            )
        else:
            return None

    @record_function("compute_relation_loss")
    def compute_relation_loss(
        self,
        true_relations: Sequence[IndexedRelation],
        rel_meta: dict[str, Tensor],
        rel_logits: Float[Tensor, "relation logits"] | None,
    ) -> Float[Tensor, ""]:
        target_preds = self.align_relation_predictions(
            true_relations=true_relations,
            rel_meta=rel_meta,
            rel_logits=rel_logits,
        )
        if target_preds is None:
            return torch.tensor(
                0.0, device=self.device, dtype=torch.float16, requires_grad=True
            )

        targets, preds = target_preds
        loss_fn = torch.nn.CrossEntropyLoss(
            reduction="mean", label_smoothing=self.relation_label_smoothing
        )
        loss = loss_fn(preds, targets)
        return loss

    # @torch.compile
    def compute_loss(
        self,
        predictions: tuple[Tensor, Tensor],
        targets: tuple[Tensor, Tensor],
        class_scale: float = 1,
    ) -> Float[Tensor, ""]:
        entity_loss = self.loss_fn(
            predictions[0].view(-1).float(), targets[0].view(-1).float()
        )
        class_loss = self.loss_fn(
            predictions[1].view(-1).float(), targets[1].view(-1).float()
        )
        return entity_loss + class_scale * class_loss

    def pool_logits(
        self,
        batch: Sequence[dict[str, Any]],
        entity_logits: BatchedLogits,
        class_logits: BatchedLogits,
    ) -> tuple[BatchedLogits, BatchedLogits]:
        """Pool logits corresponding to each document in a batch."""
        pool_fn = get_pool_fn(self.entity_logits_pooling)

        doc_ids = torch.concat(
            tuple(doc["doc_id"].squeeze(dim=0) for doc in batch)
        )

        ent_class_logits = tuple(
            torch.stack(
                [pool_fn(logit_t[doc_ids == i]) for i in range(len(batch))]
            )
            for logit_t in (entity_logits, class_logits)
        )

        return ent_class_logits

    def get_batch_logits(
        self, batch: Sequence[dict[str, Tensor | BatchEncoding]]
    ) -> tuple[
        Float[Tensor, "sequence entities"],
        Float[Tensor, "sequence classes"],
        tuple[dict[str, Tensor], Float[Tensor, "pairs relations"]] | None,
    ]:
        inputs = self.get_token_embeddings(batch)
        entities_in_batch = get_batch_entities(batch)

        entity_logits, class_logits, relation_index_logits = self(
            inputs, entities_in_batch
        )
        entity_pooled_logits, class_pooled_logits = self.pool_logits(
            batch, entity_logits, class_logits
        )

        return (
            entity_pooled_logits,
            class_pooled_logits,
            relation_index_logits,
        )

    def compute_batch_losses(
        self, batch: Sequence[dict[str, Tensor | BatchEncoding]]
    ) -> tuple[Float[Tensor, ""], Float[Tensor, ""]]:
        """Compute loss for a batch."""
        with torch.autocast(device_type=self.device):
            entity_logits, class_logits, relation_index_logits = (
                self.get_batch_logits(batch)
            )

            ent_true, class_true, rel_true = self.ground_truth(batch)

            ent_loss = self.compute_loss(
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

        entity_truth: Float[Tensor, "batch doc entities"]
        class_truth: Float[Tensor, "batch doc classes"]
        rel_truth: list[IndexedRelation]
        entity_truth, class_truth, rel_truth = self.ground_truth(batch)

        if rel_truth:
            if relation_index_logits:
                rel_meta: dict[str, Tensor]
                rel_logits: Float[Tensor, "pairs relations"]
                rel_meta, rel_logits = relation_index_logits
                target_pred = self.align_relation_predictions(
                    true_relations=rel_truth,
                    rel_meta=rel_meta,
                    rel_logits=rel_logits,
                )
                if target_pred is not None:
                    target, pred = target_pred
                    relations_true = target.numpy(force=True)
                    relations_pred = pred.numpy(force=True)
            else:
                relations_true = np.array([rel.label for rel in rel_truth])
                relations_pred = np.array([0] * len(relations_true))
        else:
            relations_true = np.array([])
            relations_pred = np.array([])

        try:
            relations_pred = relations_pred.argmax(axis=-1)
        except ValueError:
            # raised when relations_pred is an empty array
            pass

        return {
            "entities": {
                "true": entity_truth.numpy(force=True).squeeze(),
                "pred": torch.sigmoid(entity_logits)
                .squeeze()
                .round()
                .numpy(force=True),
            },
            "classes": {
                "true": class_truth.numpy(force=True).squeeze(),
                "pred": torch.sigmoid(class_logits)
                .squeeze()
                .round()
                .numpy(force=True),
            },
            "relations": {
                "true": relations_true,
                "pred": relations_pred,
            },
        }

    def _compute_relations_vectorized(
        self,
        entity_positions: Int64[Tensor, "n_entities 2"],
        entity_reprs: Float[Tensor, "n_entities features"],
        max_indices: Int64[Tensor, "sequence token"],
    ) -> tuple[dict[str, Tensor], Float[Tensor, "n_pairs relations"]]:
        """
        Compute relation logits for all valid entity pairs.
        Returns:
            - dict of raw tensors: {
                "sequence": LongTensor[n_pairs],
                "arg_pred_i": LongTensor[n_pairs],
                "arg_pred_j": LongTensor[n_pairs],
            }
            - logits: FloatTensor[n_pairs, n_relations]
        """
        sequence_ids = entity_positions[:, 0]
        token_positions = entity_positions[:, 1]
        entity_preds = max_indices[sequence_ids, token_positions]

        # Prepare output buffers
        seq_batch = []
        arg_pred_i = []
        arg_pred_j = []
        reprs_i = []
        reprs_j = []

        for seq_id in torch.unique_consecutive(sequence_ids):
            indices = torch.where(sequence_ids == seq_id)[0]

            if len(indices) < 2:
                continue

            local_pos = token_positions[indices]
            local_preds = entity_preds[indices]
            unique_local_preds = torch.unique(local_preds)
            local_reprs = entity_reprs[indices]

            # torch.unique sorts its input, so this will ensure that entities
            # are ranked by their indices
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
                torch.arange(len(grouped_entity_positions), device=self.device),
                r=2,
            )
            if len(pairs) == 0:
                continue

            i, j = pairs[:, 0], pairs[:, 1]
            pred_i = unique_local_preds[i]
            pred_j = unique_local_preds[j]

            n_pairs = len(i)
            seq_batch.append(seq_id.repeat(n_pairs))
            arg_pred_i.append(pred_i)
            arg_pred_j.append(pred_j)
            reprs_i.append(pooled_reprs[i])
            reprs_j.append(pooled_reprs[j])

        if reprs_i:
            all_repr_i = torch.cat(reprs_i, dim=0)
            all_repr_j = torch.cat(reprs_j, dim=0)
            logits = self.relation_classifier(all_repr_i, all_repr_j)

            meta = {
                "sequence": torch.cat(seq_batch),
                "arg_pred_i": torch.cat(arg_pred_i),
                "arg_pred_j": torch.cat(arg_pred_j),
            }
        else:
            logits = self._dummy_relation_logits()
            meta = {
                "sequence": torch.empty(
                    0, dtype=torch.long, device=self.device
                ),
                "arg_pred_i": torch.empty(
                    0, dtype=torch.long, device=self.device
                ),
                "arg_pred_j": torch.empty(
                    0, dtype=torch.long, device=self.device
                ),
            }

        return meta, logits

    @record_function("forward")
    def forward(
        self,
        input_data: Float[Tensor, "sequence token embedding"],
        entities_in_batch: tuple[Int16[Tensor, " entities"], ...],
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
            - Entity logits pooled across the batch.
            - Class logits pooled across the batch.
            - Tuple containing:
                - Index of entity A, where dim=-1 corresponds to the entity
                  selected in entity_index
                - Index of entity B
                - Relation type logits
        """
        with torch.autocast(device_type=self.device):
            hidden_output: Float[Tensor, "sequence token features"] = (
                self.hidden(input_data)
            )
            entity_logits, class_logits = self.classifier(hidden_output)

            # Find entity positions
            entity_probs: Float[Tensor, "sequence token ent_probs"] = (
                torch.sigmoid(entity_logits)
            )
            threshold = self.entity_threshold.clamp(min=0.01, max=0.999)

            max_probs: Float[Tensor, "sequence token"]
            max_probs, max_indices = entity_probs.max(dim=-1)
            hard_entity_mask: Bool[Tensor, "sequence token"]
            if not self.evaluation:
                # More efficient entity mask computation
                hard_entity_mask = max_probs > threshold
                # Apply entity filtering more efficiently
                # for seq_idx, seq_entities in enumerate(entities_in_batch):
                #     if seq_idx < len(max_indices):
                #         # Create mask for valid entities in this sequence
                #         valid_entities = torch.isin(
                #             max_indices[seq_idx], seq_entities
                #         )
                #         hard_entity_mask[seq_idx] = (
                #             hard_entity_mask[seq_idx] & valid_entities
                #         )
            else:
                # Mask where:
                # - some entity was confidently above threshold
                # - that entity is the most likely for the token
                hard_entity_mask = max_probs > threshold
            if not hard_entity_mask.any():
                return (
                    torch.logsumexp(entity_logits, dim=1),
                    torch.logsumexp(class_logits, dim=1),
                    None,
                )
            # Select the predicted entity representations
            entity_positions: Int64[Tensor, "position seqdim_tokendim"] = (
                hard_entity_mask.nonzero(as_tuple=False)
            )
            if entity_positions.numel() == 0:
                return (
                    torch.logsumexp(entity_logits, dim=1),
                    torch.logsumexp(class_logits, dim=1),
                    None,
                )

            entity_reprs = hidden_output[
                entity_positions[:, 0],  # batch
                entity_positions[:, 1],  # token
            ]

            # Return early if not enough entities to relate
            if len(entity_reprs) < 2:
                return (
                    torch.logsumexp(entity_logits, dim=1),
                    torch.logsumexp(class_logits, dim=1),
                    None,
                )

            # Efficient pairwise relation classification
            rel_pair_indices, relation_logits = (
                self._compute_relations_vectorized(
                    entity_positions, entity_reprs, max_indices
                )
            )

        return (
            torch.logsumexp(entity_logits, dim=1),
            torch.logsumexp(class_logits, dim=1),
            (rel_pair_indices, relation_logits),
        )

    def evaluate_model(
        self,
        test_data: DataLoader,
    ) -> None:
        """Evaluate the end-to-end model and output a classification report."""
        self.eval()

        result_dict: dict[str, dict[str, np.ndarray]] = {}

        with torch.no_grad(), torch.autocast(device_type=self.device):
            for batch in tqdm(test_data):
                curdict = self.compute_batch_true_x_pred(batch)
                for category, res in curdict.items():
                    for trueorpred, values in res.items():
                        result_dict.setdefault(category, {})[trueorpred] = (
                            values
                        )

        for category, res in result_dict.items():
            if res["true"].size:
                match category:
                    case "entities":
                        labels = np.array(tuple(self.entity_to_index.values()))
                        target_names = np.array(
                            tuple(self.entity_to_index.keys())
                        )
                    case "classes":
                        labels = np.arange(len(self.classes))
                        target_names = np.array(self.classes)
                    case "relations":
                        labels = np.arange(len(self.relations))
                        target_names = np.array(self.relations)

                print(f"\n{category}")
                y_true = np.vstack(res["true"])
                y_pred = res["pred"]
                if np.isscalar(y_pred) or (
                    hasattr(y_pred, "ndim") and y_pred.ndim == 0
                ):
                    y_pred = np.array([y_pred])
                else:
                    y_pred = np.vstack(y_pred)
                print(
                    classification_report(
                        y_true=np.vstack(y_true),
                        y_pred=y_pred,
                        zero_division=0,
                        labels=labels,
                        target_names=target_names,
                    )
                )


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
                out_features=2048,
                bias=True,
            ),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2048, n_entities),
        )
        # self.entity_classifier = nn.Linear(input_size, n_entities)
        self.class_classifier = nn.Linear(input_size, n_classes)

    def initialize_classifier_bias(self, entity_freqs: torch.Tensor) -> None:
        """Initialize classifier bias using log odds from entity frequencies."""
        with torch.no_grad():
            log_odds = torch.log(entity_freqs / (1 - entity_freqs))
            self.entity_classifier.bias.copy_(log_odds.to(self.device))

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
        save_checkpoint: bool = False,
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
