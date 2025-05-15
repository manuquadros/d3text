import itertools
import os
from collections.abc import (
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from copy import deepcopy
from functools import partial
from typing import Any

import numpy
import torch
import torch.nn as nn
import transformers
from jaxtyping import Float, UInt8
from seqeval.metrics import classification_report
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from d3text import data
from d3text.utils import (
    Token,
    merge_off_tokens,
    merge_predictions,
    merge_tokens,
    split_and_tokenize,
    tokenize_cased,
)

from .config import ModelConfig, optimizers, save_model_config, schedulers
from .dict_tagger import DictTagger

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class Model(torch.nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()

        self.config = config if config is not None else ModelConfig()

        self.base_model = transformers.AutoModel.from_pretrained(
            self.config.base_model
        )
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.config.base_model, clean_up_tokenization_spaces=False
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.checkpoint = "checkpoint.pt"
        self.best_score: float
        self.best_model_state: dict[str, Any]

        # Common layers setup
        self.dropout = (
            nn.Dropout(self.config.dropout)
            if self.config.dropout
            else nn.Identity()
        )

        self.hidden = nn.Sequential()
        in_features = self.base_model.config.hidden_size

        for layer_size in self.config.hidden_layers:
            self.hidden.append(nn.Linear(in_features, layer_size))
            self.hidden.append(self.dropout)

            match self.config.normalization:
                case "layer":
                    self.hidden.append(nn.LayerNorm(layer_size))
                case "batch":
                    self.hidden.append(PermutationBatchNorm1d(layer_size))
                case _:
                    pass

            in_features = layer_size

        self.hidden_block_output_size = in_features

    def get_loss_function(self, train_data: data.DatasetConfig) -> nn.Module:
        """Return the appropriate loss function for this model type"""
        raise NotImplementedError

    def compute_batch(
        self,
        batch: Any,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler,
        loss_fn: nn.Module,
    ) -> float:
        """Compute loss for a batch and perform optimization step.
        Returns the loss value for this batch."""
        raise NotImplementedError

    def train_model(
        self,
        train_data: DataLoader,
        val_data: data.DatasetConfig | None = None,
        save_checkpoint: bool = False,
        output_loss: bool = True,
    ) -> float | None:
        """Generic training loop for all models"""
        optimizer = optimizers[self.config.optimizer](
            self.parameters(), lr=self.config.lr
        )
        scaler = torch.amp.GradScaler(self.device)
        loss_fn = self.get_loss_function(train_data)

        match self.config.lr_scheduler:
            case "exponential":
                scheduler = schedulers["exponential"](optimizer, gamma=0.95)
            case "reduce_on_plateau":
                scheduler = schedulers["reduce_on_plateau"](
                    optimizer, min_lr=0.0001, patience=2, factor=0.5
                )

        epoch_val_losses: list[float] = []
        self.stop_counter: float = 0

        for epoch in range(self.config.num_epochs):
            self.train()
            batch_losses = []

            print(f"\nEpoch {epoch + 1}")
            for batch in tqdm(train_data):
                loss = self.compute_batch(batch, optimizer, scaler, loss_fn)
                batch_losses.append(loss)

                if self.device == "cuda":
                    torch.cuda.empty_cache()

            avg_batch_loss = numpy.mean(batch_losses)
            print(f"Average training loss on this epoch: {avg_batch_loss:.5f}")

            if val_data is not None:
                val_loss = self.validate_model(loss_fn, val_data)

                if self.config.lr_scheduler == "reduce_on_plateau":
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

                print(f"Average validation loss on this epoch: {val_loss:.5f}")

                if self.early_stop(val_loss, save_checkpoint=save_checkpoint):
                    epoch_val_losses.append(self.best_score)
                    if save_checkpoint:
                        print(
                            "Model converged. Loading the best epoch's parameters."
                        )
                        self.load_state_dict(self.best_model_state)
                    break

        if val_data is not None and output_loss:
            return numpy.mean(epoch_val_losses)
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
        try:
            current: float = self.best_score
        except AttributeError:
            self.best_score = metric
        else:
            if (goal == "min" and metric <= current) or (
                goal == "max" and metric >= current
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
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        val_data: data.DatasetConfig,
    ) -> float:
        self.eval()

        loss = 0.0
        with torch.no_grad():
            for batch in val_data.data:
                inputs, labels = (
                    batch["sequence"],
                    batch["nerc_tags"].to(self.device),
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self(inputs)
                loss += loss_fn(
                    outputs.view(-1, self.num_labels), labels.view(-1)
                ).item()

                if self.device == "cuda":
                    del inputs, labels, outputs
                    torch.cuda.empty_cache()

        loss = loss / len(val_data.data)

        return loss

    def ids_to_tokens(
        self,
        ids: Iterable[int],
    ) -> list[str]:
        return self.tokenizer.convert_ids_to_tokens(ids)

    def evaluate_model(
        self,
        test_data: data.DatasetConfig,
        verbose: bool = False,
        output_sequence: bool = False,
        output_dict: bool = False,
    ) -> tuple[list[dict], str] | dict:
        self.eval()

        if verbose:
            print("-" * 40)
            print(self.config)
            print("Evaluation:")

        tagged: list[dict[str, list[str]]] = []

        with torch.no_grad():
            for batch in tqdm(test_data.data):
                inputs = {
                    k: v.to(self.device) for k, v in batch["sequence"].items()
                }
                labels = (
                    [test_data.classes[idx] for idx in sample]
                    for sample in batch["nerc_tags"]
                )
                prediction = self.forward(inputs)
                tags = map(self.logits_to_tags, prediction.to("cpu"))
                tokens = map(self.ids_to_tokens, inputs["input_ids"].to("cpu"))

                tagged.extend(
                    merge_tokens(*ttl) for ttl in zip(tokens, tags, labels)
                )

                if self.device == "cuda":
                    del inputs, prediction
                    torch.cuda.empty_cache()

        report = classification_report(
            [sample["gold_labels"] for sample in tagged],
            [sample["predicted"] for sample in tagged],
            output_dict=output_dict,
        )

        return (tagged, report) if output_sequence else report

    def save_config(self, path: str) -> None:
        save_model_config(self.config.model_dump(), path)


class PermutationBatchNorm1d(nn.BatchNorm1d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = torch.permute(input, (0, 2, 1))
        out = torch.permute(super().forward(input), (0, 2, 1))
        return out


class ETEBrendaModel(Model):
    def __init__(
        self, classes: Mapping[str, set[int]], config: None | ModelConfig = None
    ) -> None:
        super().__init__(config)
        self.classes = tuple(classes.keys())
        self.class_classifier = nn.Linear(
            self.hidden_block_output_size, len(classes) + 1
        )
        self.entclassifiers = {
            class_: nn.Linear(self.hidden_block_output_size, len(entities) + 1)
            for class_, entities in classes.items()
        }

    def get_loss_function(self, train_data: data.DatasetConfig) -> nn.Module:
        return nn.BCEWithLogitsLoss()

    def compute_batch(
        self,
        batch: Sequence[
            dict[str, transformers.BatchEncoding | UInt8[Tensor, " indexes"]]
        ],
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler,
        loss_fn: nn.Module,
    ) -> float:
        """Compute loss for a batch."""
        doc_indices = itertools.accumulate(
            batch,
            (
                lambda acc, doc: (
                    acc[1],
                    acc[1] + len(doc["sequence"]["input_ids"]),
                )
            ),
            initial=(0, 0),
        )

        # Concatenate the input tensors across the batch
        inputs = {
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
        inputs = {
            k: v.to(self.device, non_blocking=True) for k, v in inputs.items()
        }

        optimizer.zero_grad()

        with torch.autocast(device_type=self.device):
            outputs = self(inputs)
            doc_outputs = (outputs[start:end] for start, end in doc_indices)

            # collect all the entity probabilities across the batch
            entoutputs = {
                cl: torch.stack(
                    tuple(
                        torch.max(
                            torch.stack(tuple(sample[cl] for sample in doc)),
                            dim=0,
                        ).values
                        for doc in doc_outputs
                    )
                )
                for cl in self.classes
            }
            entgold = {
                cl: torch.stack(tuple(doc[cl] for doc in batch))
                for cl in self.classes
            }

            loss = numpy.mean(
                tuple(
                    loss_fn(entoutputs[cl].view(-1), entgold[cl].view(-1))
                    for cl in self.classes
                )
            )

            if self.device == "cuda":
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            return loss.item()

    def forward(self, input_data: dict) -> torch.Tensor:
        """Forward pass

        :return: Dictionary with keys for each entity type
        """
        base_output = self.dropout(
            self.base_model(**input_data).last_hidden_state
        )
        x = self.hidden(base_output)
        entity_classification = self.class_classifier(x)
        entity_classification_max = entity_classification.argmax(dim=-1)

        # use mask indexing to take the tensors whose argmax(dim=-1)
        # match a each classifier to route the tokens accordingly
        entity_identification = entity_classification.clone()
        for ix, cl in enumerate(self.classes):
            print(f"Identifying {cl} in the batch")
            entity_classifier = self.entclassifiers[cl]
            mask = entity_classification_max == ix
            entity_identification[mask] = entity_classifier(x[mask])

        return entity_identification


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

    def get_loss_function(self, train_data: data.DatasetConfig) -> nn.Module:
        return nn.CrossEntropyLoss(
            weight=train_data.class_weights.to(self.device),
            ignore_index=train_data.null_index,
        )

    def compute_batch(
        self,
        batch: Any,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler,
        loss_fn: nn.Module,
    ) -> float:
        inputs, labels = batch["sequence"], batch["nerc_tags"].to(self.device)
        inputs = {
            k: v.to(self.device, non_blocking=True) for k, v in inputs.items()
        }

        optimizer.zero_grad()

        with torch.autocast(device_type=self.device):
            outputs = self(inputs)
            loss = loss_fn(
                outputs.view(-1, self.num_labels),
                labels.view(-1),
            )

            if self.device == "cuda":
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            return loss.item()

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
