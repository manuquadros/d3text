import itertools
import pathlib
import random
from collections.abc import Iterable
from typing import Literal

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

Float32MatmulPrecision = Literal["highest", "high", "medium"]
RelationLossWeighting = Literal["unweighted", "balanced", "focal"]


class ModelConfig(BaseModel):
    model_class: str = "ETEBrendaModel"
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
    entity_loss_scaling_factor: PositiveFloat = 1.0
    relation_label_smoothing: NonNegativeFloat = 0.0
    relation_loss_weighting: RelationLossWeighting = "unweighted"
    relation_focal_gamma: NonNegativeFloat = 2.0
    common_hidden_block: bool = True
    ramp_epochs: int = 0
    separate_predicate_layer: bool = False
    consistency_weight: float = 0.1
    entity_logits_pooling: Literal["logsumexp", "logmeanexp", "max", "mean"] = (
        "logsumexp"
    )


class MachineConfig(BaseModel):
    """Per-machine settings, read from the repo-root ``config.toml``.

    The runtime fields are process-global torch/allocator settings, applied by
    ``d3text.runtime.configure()`` at script start-up rather than at import.
    See ``config.toml.example``.
    """

    cpu_embeddings_cache_size: NonNegativeInt
    float32_matmul_precision: Float32MatmulPrecision = "medium"
    cudnn_allow_tf32: bool = True
    expandable_segments: bool = True
    tokenizers_parallelism: bool = True


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
        # `ModelConfig(**config)` trips mypy (**dict[str, object]); this is
        # equivalent here (extra keys ignored) and type-clean.
        yield ModelConfig.model_validate(config)


def load_model_config(path: str) -> ModelConfig:
    with open(path, "r") as config_file:
        model_config = ModelConfig(**tomlkit.load(config_file))

    return model_config


def machine_config() -> MachineConfig:
    """Load the repo-root ``config.toml``.

    Falls back to a zero-cache default when the file is absent (e.g. a fresh
    checkout or CI) so that importing ``d3text.models`` never fails on a
    missing, uncommitted config. See ``config.toml.example``.
    """
    path = pathlib.Path(__file__).parent.parent.parent.parent / "config.toml"
    try:
        with path.open("r") as config:
            return MachineConfig(**tomlkit.load(config))
    except FileNotFoundError:
        return MachineConfig(cpu_embeddings_cache_size=0)


def load_tuning_config(path: str) -> list[ModelConfig]:
    with open(path, "r") as config_file:
        cfg = tomlkit.load(config_file)

    layer_sizes = cfg["hidden_layers"]
    cfg["hidden_layers"] = random.choices(
        tuple(
            itertools.chain(
                itertools.combinations_with_replacement(layer_sizes, 1),
            )
        ),
        k=100,
    )

    cfgs = tuple(
        ModelConfig(**dict(zip(cfg.keys(), cell)))
        for cell in itertools.product(*cfg.values())
    )

    return random.sample(cfgs, k=250)


def save_model_config(config: dict, path: str) -> None:
    with open(path, "w") as config_file:
        tomlkit.dump(config, config_file)
