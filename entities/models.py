import dataclasses
import itertools
import os
from collections.abc import Callable

import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import transformers
from seqeval.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

from entities import utils

os.environ["TOKENIZERS_PARALLELISM"] = "true"


@dataclasses.dataclass
class ModelConfig:
    optimizer: str = "adam"
    lr: float = 0.0003
    lr_scheduler: str = ""
    dropout: float = 0
    hidden_layer: int = 0
    layer_norm: bool = False
    batch_size: int = 64
    num_epochs: int = 100
    patience: int = 6


optimizers = {
    "adam": torch.optim.Adam,
    "adamW": torch.optim.AdamW,
    "nadam": torch.optim.NAdam,
}
lrs = [0.01, 0.001, 0.002]
schedulers = {
    "reduce_on_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "exponential": torch.optim.lr_scheduler.ExponentialLR,
}
hidden_layer = [256, 192, 78, 0]
dropout = [0, 0.1, 0.2]


def model_configs():
    hyps = itertools.product(
        optimizers, lrs, schedulers.keys(), dropout, hidden_layer, [True, False]
    )
    for config in hyps:
        yield ModelConfig(*config)


class Model(torch.nn.Module):
    def __init__(
        self, model_id: str, dataset: datasets.Dataset, config: None | ModelConfig
    ) -> None:
        super().__init__()
        self.base_model = transformers.AutoModel.from_pretrained(model_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.config = config if config is not None else ModelConfig()
        self.checkpoint = "checkpoint.pt"

        self.train_data = dataset.train
        self.val_data = dataset.validation
        self.test_data = dataset.test
        self.classes = dataset.classes
        self.num_labels = len(self.classes)
        self.null_index = dataset.null_index
        self.tokenizer = dataset.tokenizer

    def train_model(
        self,
        shuffle: bool = True,
    ) -> None:
        optimizer = optimizers[self.config.optimizer](
            self.parameters(), lr=self.config.lr
        )

        print(self.config)

        scaler = torch.cuda.amp.GradScaler()

        match self.config.lr_scheduler:
            case "exponential":
                scheduler = schedulers["exponential"](optimizer, 0.95)
            case "reduce_on_plateau":
                scheduler = schedulers["reduce_on_plateau"](
                    optimizer, min_lr=0.0001, patience=5
                )

        loss_fn = nn.CrossEntropyLoss(ignore_index=self.null_index)

        for epoch in range(self.config.num_epochs):
            self.train()
            running_loss = 0.0

            print(f"Epoch {epoch + 1}")
            for i, data in tqdm(enumerate(self.train_data)):
                inputs, labels = data["sequence"], data["nerc_tags"]

                optimizer.zero_grad()

                with torch.autocast(device_type=self.device):
                    outputs = self.forward(inputs)
                    loss = loss_fn(outputs.view(-1, self.num_labels), labels.view(-1))

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()

            val_loss = self.validate_model(loss_fn)

            if self.config.lr_scheduler == "reduce_on_plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

            print(
                f"\nAverage training loss on this epoch: "
                f"{running_loss / len(self.train_data):.4f}"
                f"\nAverage validation loss on this epoch: {val_loss:.4f}"
            )

            if self.early_stop(val_loss):
                print("Model converged. Loading the best epoch's parameters.")
                self.load_state_dict(torch.load(self.checkpoint))
                break

        utils.log_model("models.csv", self.config, self.best_score)
        print("Training complete")

    def early_stop(self, val_loss: float) -> bool:
        try:
            current = self.best_score
        except AttributeError:
            self.save_checkpoint(val_loss)
        else:
            if val_loss < current:
                self.save_checkpoint(val_loss)
            else:
                self.stop_counter += 1
        finally:
            if self.stop_counter > self.config.patience:
                return True
            else:
                return False

    def save_checkpoint(self, val_loss: float) -> None:
        self.best_score = val_loss
        self.stop_counter = 0
        torch.save(self.state_dict(), self.checkpoint)

    def validate_model(
        self, loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> float:
        self.eval()

        loss = 0.0
        with torch.no_grad():
            for batch in self.val_data:
                inputs, labels = batch["sequence"], batch["nerc_tags"]
                outputs = self(inputs)
                loss += loss_fn(
                    outputs.view(-1, self.num_labels), labels.view(-1)
                ).item()

        loss = loss / len(self.val_data)

        return loss

    def evaluate_model(self) -> tuple[list[dict], str]:
        self.eval()
        test_data = DataLoader(self._data["test"])

        print("-" * 40)
        print("Evaluation:")

        tagged = []

        with torch.no_grad():
            for sample in tqdm(test_data):
                inputs = sample["sequence"]
                labels = (
                    self._label_encoder.classes_[idx] for idx in sample["nerc_tags"][0]
                )
                prediction = self.forward(inputs)
                tags = (
                    self._label_encoder.classes_[pos.argmax()] for pos in prediction[0]
                )
                tokens = self._tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
                tagged.append(utils.merge_tokens(tokens, tags, labels))

        return tagged, classification_report(
            [sample["true"] for sample in tagged],
            [sample["predicted"] for sample in tagged],
        )


class NERCTagger(Model):
    def __init__(
        self,
        dataset: datasets.DatasetDict,
        model_id: str = "michiyasunaga/BioLinkBERT-base",
        config: None | ModelConfig = None,
    ) -> None:
        super().__init__(model_id, dataset, config)

        for param in self.base_model.parameters():
            param.requires_grad = False

        self.dropout = (
            nn.Dropout(self.config.dropout) if self.config.layer_norm else nn.Identity()
        )

        self.layer_norm = (
            nn.LayerNorm(self.config.hidden_layer)
            if self.config.layer_norm
            else nn.Identity()
        )

        if self.config.hidden_layer:
            self.linear = nn.Linear(
                self.base_model.config.hidden_size, self.config.hidden_layer
            )
            self.classifier = nn.Linear(self.config.hidden_layer, self.num_labels)
            self.head = nn.Sequential(
                self.linear, self.dropout, self.layer_norm, self.classifier
            )
        else:
            self.head = nn.Linear(self.base_model.config.hidden_size, self.num_labels)

    def forward(self, input_data: dict) -> torch.Tensor:
        x = self.dropout(self.base_model(**input_data).last_hidden_state)
        x = self.head(x)

        return x
