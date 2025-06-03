import itertools
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from copy import deepcopy
from functools import partial
from typing import Any

import numpy
import torch
import torch.nn as nn
import transformers
from cacheout import Cache
from d3text import data
from d3text.utils import (
    Token,
    merge_off_tokens,
    merge_predictions,
    split_and_tokenize,
    tokenize_cased,
)
from jaxtyping import Float, Integer, UInt8
from sklearn.metrics import classification_report
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import BatchEncoding

from .config import (
    ModelConfig,
    embedding_dims,
    optimizers,
    save_model_config,
    schedulers,
)
from .dict_tagger import DictTagger

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("medium")


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
        self.best_score: float
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
        max_mem_allocated = 0.0

        for epoch in trange(
            self.config.num_epochs,
            dynamic_ncols=True,
            position=0,
            desc="Epochs",
            leave=True,
        ):
            self.train()
            batch_losses: float = 0.0
            n_batches = 0

            for batch in tqdm(
                train_data,
                dynamic_ncols=True,
                position=1,
                desc="Batches",
                leave=False,
            ):
                torch.cuda.reset_peak_memory_stats()
                optimizer.zero_grad()
                predictions = self.compute_batch(batch)
                loss = self.compute_loss(
                    predictions=predictions, targets=self.ground_truth(batch)
                )
                del predictions
                batch_losses += loss.item()
                n_batches += 1

                if self.device == "cuda":
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                mem_allocated = torch.cuda.max_memory_allocated() / 1e6
                if mem_allocated > max_mem_allocated:
                    max_mem_allocated = mem_allocated
                    tqdm.write(
                        f"Maximum memory allocated: {max_mem_allocated:.2f} MB"
                    )

                del loss
                torch.cuda.empty_cache()

            tqdm.write(f"Average training loss: {batch_losses / n_batches:.2e}")

            if val_data is not None:
                val_loss = self.validate_model(val_data=val_data)

                if self.config.lr_scheduler == "reduce_on_plateau":
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

                tqdm.write(f"Average validation loss: {val_loss:.5f}")

                if self.early_stop(val_loss, save_checkpoint=save_checkpoint):
                    if save_checkpoint:
                        print(
                            "Model converged. Loading the best epoch's parameters."
                        )
                        self.load_state_dict(self.best_model_state)
                    break

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
        if not hasattr(self, "best_score"):
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

        batch_loss: float = 0.0
        n_batches = 0

        with torch.no_grad():
            batches = tqdm(
                val_data,
                dynamic_ncols=True,
                position=2,
                desc="Validation",
                leave=False,
            )
            for batch in batches:
                curr: Float[Tensor, ""] = self.compute_loss(
                    predictions=self.compute_batch(batch),
                    targets=self.ground_truth(batch),
                )
                batch_loss += curr.item()
                n_batches += 1
            loss: float = batch_loss / n_batches

        return loss

    def ids_to_tokens(
        self,
        ids: Iterable[int],
    ) -> list[str]:
        return self.tokenizer.convert_ids_to_tokens(ids)

    def evaluate_model(
        self,
        test_data: DataLoader,
        verbose: bool = False,
        output_sequence: bool = False,
        output_dict: bool = False,
    ) -> tuple[list[dict], str] | dict:
        raise NotImplementedError

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

        self.classifier = ClassificationHead(
            input_size=self.hidden_block_output_size,
            n_entities=self.num_of_entities,
            n_classes=self.num_of_classes,
        )


class ETEBrendaModel(BrendaClassificationModel):
    def __init__(
        self, classes: Mapping[str, set[str]], config: None | ModelConfig = None
    ) -> None:
        super().__init__(
            classes,
            config,
        )

        self.base_model = transformers.AutoModel.from_pretrained(
            config.base_model
        )
        self._base_output_cache = Cache(maxsize=2000)

        for param in self.base_model.parameters():
            param.requires_grad = False

        if self.device == "cuda":
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

        self.entity_logits_pooling = "logsumexp"

        self.entity_logits_pooling = "logsumexp"
        self.entity_logit_scale = nn.Parameter(torch.tensor(2.0))

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

    def ground_truth(
        self,
        batch: Sequence[Mapping[str, BatchEncoding | Tensor]],
    ) -> tuple[
        Float[Tensor, "batch doc entities"], Float[Tensor, "batch doc classes"]
    ]:
        """Get ground truth for all entity labels per document.

        :param: Batch of documents.
        :return: Multi-hot encoded tensor, where each position of dim 1
            specifies whether the entity corresponding to that index occurs in
            the particular document along dim 0.
        """
        entity_targets = torch.stack(
            tuple(doc["entities"] for doc in batch)
        ).to(self.device)

        class_targets = (
            entity_targets.to(dtype=self.class_matrix.dtype) @ self.class_matrix
        ).clamp(max=1)

        return entity_targets.float(), class_targets.float()

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

    def compute_batch(
        self, batch: Sequence[dict[str, Tensor | BatchEncoding]]
    ) -> tuple[Tensor, Tensor]:
        """Compute loss for a batch."""
        with torch.autocast(device_type=self.device):
            inputs = [None] * len(batch)
            missing = []

            for ix, item in enumerate(batch):
                doc_id = item["id"].item()
                cached = self._base_output_cache.get(doc_id)
                if cached is not None:
                    inputs[ix] = cached
                else:
                    missing.append((ix, item))

            if missing:
                with torch.no_grad():
                    batched_inputs = self.batch_input_tensors(
                        [item for _, item in missing]
                    )
                    output = self.base_model(
                        input_ids=batched_inputs["input_ids"].to(
                            self.device, dtype=torch.int
                        ),
                        attention_mask=batched_inputs["attention_mask"].to(
                            self.device
                        ),
                    ).last_hidden_state

                out_iter = iter(output)
                for ix, item in missing:
                    outs = torch.stack(
                        tuple(
                            itertools.islice(out_iter, item["doc_id"].shape[-1])
                        )
                    )
                    inputs[ix] = outs
                    self._base_output_cache.set(item["id"].item(), outs.cpu())

            inputs = torch.concat(tuple(t.to(self.device) for t in inputs))

            entity_logits, class_logits = self(inputs)

            pool_fn = {
                "max": lambda x: torch.amax(dim=0),
                "mean": lambda x: torch.mean(dim=0),
                "logsumexp": lambda x: torch.logsumexp(x, dim=0),
            }[self.entity_logits_pooling]

            doc_ids: UInt8[Tensor, " sequence"] = torch.concat(
                tuple(doc["doc_id"].squeeze(dim=0) for doc in batch)
            )

            doc_entity_logits = [
                pool_fn(entity_logits[doc_ids == i]) for i in range(len(batch))
            ]
            doc_class_logits = [
                pool_fn(class_logits[doc_ids == i]) for i in range(len(batch))
            ]

        # collect all the entity logits across the batch
        return torch.stack(doc_entity_logits), torch.stack(doc_class_logits)

    def forward(self, input_data: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass

        :return: entity and class logits
        """
        with torch.autocast(device_type=self.device):
            x = self.hidden(input_data)
            # pool across tokens in each sequence
            pooled = torch.logsumexp(x, dim=1)
            entity_logits, class_logits = self.classifier(pooled)

        return entity_logits, class_logits

    def evaluate_model(
        self,
        test_data: DataLoader,
        output_sequence: bool = False,
        output_dict: bool = False,
    ) -> None:
        """Evaluate the end-to-end model.

        For entity identification, return a classification report for the class
        judgment.
        """
        self.eval()

        ent_preds, ent_gts = [], []
        class_preds, class_gts = [], []

        with torch.no_grad():
            for batch in tqdm(test_data):
                entity_logits, class_logits = self.compute_batch(batch)
                gt_entities, gt_classes = self.ground_truth(batch)

                tqdm.write(
                    f"Raw entity_logits[0] max: {entity_logits[0].max().item():.3f}"
                )
                sig = torch.sigmoid(entity_logits[0])
                tqdm.write(
                    "Sigmoid entity_logits[0] max: "
                    f"{sig.max().item():3f} at {sig.argmax()}"
                )

                ent_preds.append(
                    torch.sigmoid(entity_logits).squeeze().round().cpu().numpy()
                )
                ent_gts.append(gt_entities.squeeze().cpu().numpy())

                class_pred = (
                    torch.sigmoid(class_logits).squeeze().round().cpu().numpy()
                )
                class_preds.append(class_pred)
                gt_classes_squeezed = gt_classes.squeeze().cpu().numpy()
                tqdm.write(f"{class_pred}\t{gt_classes_squeezed}")
                class_gts.append(gt_classes_squeezed)

        ent_preds = numpy.vstack(ent_preds).astype(int)
        ent_gts = numpy.vstack(ent_gts).astype(int)

        print(
            classification_report(
                ent_gts,
                ent_preds,
                zero_division=0,
                target_names=self.entities,
            )
        )

        class_preds = numpy.vstack(class_preds).astype(int)
        class_gts = numpy.vstack(class_gts).astype(int)
        print(
            classification_report(
                class_gts,
                class_preds,
                zero_division=0,
                target_names=self.classes,
            )
        )


class ETEBrendaClassifier(ETEBrendaModel):
    """Classification-only model taking text embeddings as input."""

    def __init__(
        self, classes: Mapping[str, set[int]], config: None | ModelConfig = None
    ) -> None:
        super().__init__(classes, config)

    def compute_batch(
        self, batch: Sequence[Mapping[str, Tensor | BatchEncoding]]
    ) -> tuple[Tensor, Tensor]:
        """Compute loss for a batch."""
        doc_indices = tuple(
            itertools.accumulate(
                batch,
                (
                    lambda acc, doc: (
                        acc[1],
                        acc[1] + len(doc["sequence"]),
                    )
                ),
                initial=(0, 0),
            )
        )[1:]

        # Concatenate the input tensors across the batch
        inputs = torch.concat(
            tuple(doc["sequence"].squeeze(dim=0) for doc in batch)
        ).to(self.device)

        entity_logits, class_logits = self(inputs)

        pool_fn = {
            "max": lambda x: torch.amax(x, dim=0).amax(dim=0),
            "mean": lambda x: torch.mean(x, dim=0).mean(dim=0),
            "logsumexp": lambda x: torch.logsumexp(
                torch.logsumexp(x, dim=0), dim=0
            ),
        }[self.entity_logits_pooling]

        doc_entity_logits = [
            pool_fn(entity_logits[start:end]) for start, end in doc_indices
        ]
        doc_class_logits = [
            pool_fn(class_logits[start:end]) for start, end in doc_indices
        ]
        del entity_logits, class_logits

        # collect all the entity logits across the batch
        return torch.stack(doc_entity_logits), torch.stack(doc_class_logits)

    def forward(self, input: Tensor) -> tuple[Tensor, Tensor]:
        with torch.autocast(device_type=self.device):
            hidden_output = self.hidden(input)
            entity_logits, class_logits = self.classifier(hidden_output)

            return entity_logits, class_logits


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
        self.entity_classifier = nn.Linear(input_size, n_entities)
        self.class_classifier = nn.Linear(input_size, n_classes)
        self.entity_logit_scale = nn.Parameter(torch.tensor(2.0))

    def initialize_classifier_bias(self, entity_freqs: torch.Tensor) -> None:
        """Initialize classifier bias using log odds from entity frequencies."""
        with torch.no_grad():
            log_odds = torch.log(entity_freqs / (1 - entity_freqs))
            self.entity_classifier.bias.copy_(log_odds.to(self.device))

    def forward(self, input: Tensor) -> tuple[Tensor, Tensor]:
        clamped_scale = torch.clamp(self.entity_logit_scale, min=0.1, max=10.0)
        entity_logits = clamped_scale * self.entity_classifier(input)
        class_logits = self.class_classifier(input)

        return entity_logits, class_logits


class NERCHead(nn.Module):
    """Define a named-entity recognition classification head.

    The forward method returns class logits for each entity type.
    """

    def __init__(self, input_size: int, n_classes: int) -> None:
        """Initialize the classification head.

        :param input_size: number of input features
        :param n_classes: number of output entity classes
        """
        super().__init__()
        self.class_classifier = nn.Linear(input_size, n_classes)

    def forward(self, input: Tensor):
        pass


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
