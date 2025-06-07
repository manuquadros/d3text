import itertools
import random
from collections.abc import Iterable

import tomlkit
import torch
from pydantic import (
    BaseModel,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)

optimizers = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "nadam": torch.optim.NAdam,
}
schedulers = {
    "reduce_on_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "exponential": torch.optim.lr_scheduler.ExponentialLR,
}
encodings = {
    "michiyasunaga/BioLinkBERT-base": "biolinkbert-base-zstd-22-encodings.hdf5",
    "prajjwal1/bert-mini": "prajjwal1_bert_mini-zstd-22-encodings.hdf5",
}
embedding_dims = {
    "michiyasunaga/BioLinkBERT-base": 768,
    "prajjwal1/bert-mini": 256,
}


class ModelConfig(BaseModel):
    classes: list[str] = []
    optimizer: str = "adam"
    lr: PositiveFloat = 0.0003
    lr_scheduler: str = ""
    dropout: NonNegativeFloat = 0
    hidden_layers: list[NonNegativeInt] = [32]
    normalization: str = "layer"
    batch_size: PositiveInt = 32
    num_epochs: PositiveInt = 100
    patience: NonNegativeInt = 5
    base_model: str = "michiyasunaga/BioLinkBERT-base"
    base_layers_to_unfreeze: NonNegativeInt = 0
    model_class: str = "ETEBrendaModel"


class ETEModelConfig(ModelConfig):
    entity_layers: list[NonNegativeInt]
    class_layers: list[NonNegativeInt]


def model_configs(model_class: str) -> Iterable[ModelConfig]:
    hypspace = {
        "optimizers": optimizers.keys(),
        "lrs": (0.01, 0.001, 0.002, 0.0003),
        "schedulers": schedulers.keys(),
        "hidden_size": (2048, 1024, 512, 256, 128, 64),
        "hidden_layers": range(1, 4),
        "dropout": (0, 0.1, 0.2),
        "normalization": ("layer",),
        "batch_size": (64, 32, 16, 8),
    }

    for cell in itertools.product(*hypspace.values()):
        config = dict(zip(hypspace.keys(), cell))
        print(config)
        yield ModelConfig(**config)


def load_model_config(path: str) -> ModelConfig:
    with open(path, "r") as config_file:
        model_config = ModelConfig(**tomlkit.load(config_file))

    return model_config


def load_tuning_config(path: str) -> list[ModelConfig]:
    with open(path, "r") as config_file:
        cfg = tomlkit.load(config_file)

    layer_sizes = cfg["hidden_layers"]
    cfg["hidden_layers"] = random.choices(
        tuple(
            itertools.chain(
                itertools.permutations(layer_sizes, 2),
                itertools.permutations(layer_sizes, 1),
            )
        ),
        k=10,
    )

    cfgs = tuple(
        ModelConfig(**dict(zip(cfg.keys(), cell)))
        for cell in itertools.product(*cfg.values())
    )

    return random.sample(cfgs, k=len(cfgs))


def save_model_config(config: dict, path: str) -> None:
    with open(path, "w") as config_file:
        tomlkit.dump(config, config_file)
