import itertools
import os
from collections.abc import Callable, Iterable
from copy import deepcopy
from typing import Any

import numpy
import torch
import torch.nn as nn
import transformers
from seqeval.metrics import classification_report
from tqdm import tqdm

from config import save_model_config
from entities import data
from entities.utils import ModelConfig, Token, merge_tokens

os.environ["TOKENIZERS_PARALLELISM"] = "true"


optimizers = {
    "adam": torch.optim.Adam,
    "adamW": torch.optim.AdamW,
    "nadam": torch.optim.NAdam,
}
lrs = (0.01, 0.001, 0.002, 0.0003)
schedulers = {
    "reduce_on_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "exponential": torch.optim.lr_scheduler.ExponentialLR,
}
hidden_size = (2048, 1024, 512, 256, 128, 64)
hidden_layers = range(1, 4)
dropout = (0, 0.1, 0.2)
normalization = ("layer",)
batch_size = (64, 32, 16, 8)


def model_configs():
    hyps = itertools.product(
        optimizers,
        lrs,
        schedulers.keys(),
        dropout,
        hidden_layers,
        hidden_size,
        normalization,
        batch_size,
    )
    for config in hyps:
        yield ModelConfig(*config)


class Model(torch.nn.Module):
    def __init__(self, config: None | ModelConfig) -> None:
        super().__init__()
        self.base_model = transformers.AutoModel.from_pretrained(
            config.base_model
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            config.base_model
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.config = config if config is not None else ModelConfig()
        self.checkpoint = "checkpoint.pt"
        self.best_score: float
        self.best_model_state: dict[str, Any]

    def train_model(
        self,
        train_data: data.DatasetConfig,
        val_data: data.DatasetConfig | None = None,
        save_checkpoint: bool = False,
        output_loss: bool = True,
    ) -> float | None:
        self.classes = train_data.classes

        optimizer = optimizers[self.config.optimizer](
            self.parameters(), lr=self.config.lr
        )

        scaler = torch.amp.GradScaler(self.device)

        match self.config.lr_scheduler:
            case "exponential":
                scheduler = schedulers["exponential"](optimizer, gamma=0.95)
            case "reduce_on_plateau":
                scheduler = schedulers["reduce_on_plateau"](
                    optimizer, min_lr=0.0001, patience=2, factor=0.5
                )

        loss_fn = nn.CrossEntropyLoss(
            weight=train_data.class_weights.to(self.device),
            ignore_index=train_data.null_index,
        )

        epoch_val_losses: list[float] = []
        self.stop_counter: float = 0

        for epoch in range(self.config.num_epochs):
            self.train()
            batch_losses = []

            print(f"\nEpoch {epoch + 1}")
            for i, batch in tqdm(enumerate(train_data.data)):
                inputs, labels = (
                    batch["sequence"],
                    batch["nerc_tags"].to(self.device),
                )
                inputs = {
                    k: v.to(self.device, non_blocking=True)
                    for k, v in inputs.items()
                }

                optimizer.zero_grad()

                with torch.autocast(device_type=self.device):
                    outputs = self(inputs)
                    loss = loss_fn(
                        outputs.view(-1, self.config.num_labels),
                        labels.view(-1),
                    )

                if self.device == "cuda":
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                batch_losses.append(loss.item())

                if self.device == "cuda":
                    del inputs, outputs, labels, loss
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
                    outputs.view(-1, self.config.num_labels), labels.view(-1)
                ).item()

                if self.device == "cuda":
                    del inputs, labels, outputs
                    torch.cuda.empty_cache()

        loss = loss / len(val_data.data)

        return loss

    def predict(self, inputs: str | list[str]) -> list[list[Token]]:
        self.eval()

        if isinstance(inputs, str):
            inputs = [inputs]

        tokenized = self.tokenizer(
            inputs,
            padding=True,
            return_offsets_mapping=True,
            return_token_type_ids=False,
        )

        with torch.no_grad():
            predictions = self(
                {
                    k: torch.tensor(tokenized[k], device=self.device)
                    for k in ("input_ids", "attention_mask")
                }
            )

        sequences = map(self.ids_to_tokens, tokenized["input_ids"])
        tags = map(self.logits_to_tags, predictions)

        return [
            list(zip(tokens, offsets, ts))
            for tokens, offsets, ts in zip(
                sequences, tokenized["offset_mapping"], tags
            )
        ]

    def logits_to_tags(self, logits: torch.Tensor) -> list[str]:
        return [self.classes[pos.argmax()] for pos in logits]

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


class NERCTagger(Model):
    def __init__(
        self,
        config: None | ModelConfig = None,
    ) -> None:
        if config is None:
            config = ModelConfig()

        super().__init__(config)

        for param in self.base_model.parameters():
            param.requires_grad = False

        self.dropout = (
            nn.Dropout(self.config.dropout)
            if self.config.dropout
            else nn.Identity()
        )

        self.hidden = nn.Sequential()
        in_features = self.base_model.config.hidden_size

        for n in range(0, self.config.hidden_layers):
            out_features = max(32, self.config.hidden_size // (2**n))
            self.hidden.append(nn.Linear(in_features, out_features))
            self.hidden.append(self.dropout)

            match self.config.normalization:
                case "layer":
                    self.hidden.append(nn.LayerNorm(out_features))
                case "batch":
                    self.hidden.append(PermutationBatchNorm1d(out_features))
                case _:
                    pass

            in_features = out_features

        self.classifier = nn.Linear(in_features, self.config.num_labels)

    def forward(self, input_data: dict) -> torch.Tensor:
        x = self.dropout(self.base_model(**input_data).last_hidden_state)
        x = self.hidden(x)
        x = self.classifier(x)

        return x
