import itertools
import os
from collections.abc import Callable

import datasets
import torch
import torch.nn as nn
import torch.optim as optim
import transformers
from seqeval.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

from entities import utils

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
hidden_size = (2048, 1024, 512, 256, 128, 64, 32)
hidden_layers = range(1, 6)
dropout = (0, 0.1, 0.2, 0.3)
normalization = ("batch", "layer", "none")
batch_size = (8, 16, 32, 64)


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
    def __init__(
        self, model_id: str, dataset: datasets.Dataset, config: None | ModelConfig
    ) -> None:
        super().__init__()
        self.base_model = transformers.AutoModel.from_pretrained(model_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.config = config if config is not None else ModelConfig()
        self.checkpoint = "checkpoint.pt"

        self.train_data = DataLoader(
            dataset.train, batch_size=self.config.batch_size, shuffle=True
        )
        self.val_data = DataLoader(
            dataset.validation, batch_size=self.config.batch_size
        )
        self.test_data = DataLoader(dataset.test, batch_size=self.config.batch_size)
        self.classes = dataset.classes
        self.class_weights = dataset.class_weights.to(self.device)
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

        loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights, ignore_index=self.null_index
        )

        for epoch in range(self.config.num_epochs):
            self.train()
            running_loss = 0.0

            print(f"Epoch {epoch + 1}")
            for i, batch in tqdm(enumerate(self.train_data)):
                inputs, labels = batch["sequence"], batch["nerc_tags"].to(self.device)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                optimizer.zero_grad()

                with torch.autocast(device_type=self.device):
                    outputs = self(inputs)
                    loss = loss_fn(outputs.view(-1, self.num_labels), labels.view(-1))

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()

                del inputs, outputs, labels, loss
                torch.cuda.empty_cache()

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

    def early_stop(self, metric: float, goal: str = "min") -> bool:
        try:
            current = self.best_score
        except AttributeError:
            self.save_checkpoint(metric)
        else:
            if (goal == "min" and metric <= current) or (
                goal == "max" and metric >= current
            ):
                self.save_checkpoint(metric)
            else:
                self.stop_counter += 1

        if self.stop_counter > self.config.patience:
            return True
        else:
            return False

    def save_checkpoint(self, metric: float) -> None:
        self.best_score = metric
        self.stop_counter = 0
        torch.save(self.state_dict(), self.checkpoint)

    def validate_model(
        self, loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> float:
        self.eval()

        loss = 0.0
        with torch.no_grad():
            for batch in self.val_data:
                inputs, labels = batch["sequence"], batch["nerc_tags"].to(self.device)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self(inputs)
                loss += loss_fn(
                    outputs.view(-1, self.num_labels), labels.view(-1)
                ).item()

                del inputs, labels, outputs
                torch.cuda.empty_cache()

        loss = loss / len(self.val_data)

        return loss

    def evaluate_model(self) -> tuple[list[dict], str]:
        self.eval()

        print("-" * 40)
        print(self.config)
        print("Evaluation:")

        tagged = []

        with torch.no_grad():
            for batch in tqdm(self.test_data):
                inputs = {k: v.to(self.device) for k, v in batch["sequence"].items()}
                labels = (
                    [self.classes[idx] for idx in sample]
                    for sample in batch["nerc_tags"]
                )
                prediction = self.forward(inputs)
                tags = (
                    (self.classes[pos.argmax()] for pos in sample)
                    for sample in prediction.to("cpu")
                )
                del prediction
                tokens = (
                    self.tokenizer.convert_ids_to_tokens(sample)
                    for sample in inputs["input_ids"].to("cpu")
                )
                tagged.extend(
                    utils.merge_tokens(*ttl) for ttl in zip(tokens, tags, labels)
                )
                del inputs
                torch.cuda.empty_cache()

        return tagged, classification_report(
            [sample["true"] for sample in tagged],
            [sample["predicted"] for sample in tagged],
        )

class PermutationBatchNorm1d(nn.BatchNorm1d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = torch.permute(input, (0, 2, 1))
        out = torch.permute(super().forward(input), (0, 2, 1))
        return out
    

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
