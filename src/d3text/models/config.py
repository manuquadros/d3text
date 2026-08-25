import itertools
import pathlib
import random
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
Pooling = Literal["logsumexp", "logmeanexp", "max", "mean"]

# What the class head pools a column with, by class name. `logsumexp` is a
# smooth max, so one strong token carries the document -- right for a class
# present in most of them. It is also `max + log(T)` to within a bounded
# correction, and these documents run to ~8,000 tokens, so it adds about nine
# nats of length bias to every column alike; a class absent from most documents
# answers that by going uniformly dead, which is what it did (document recall
# 0.02 for bacteria and 0.03 for strains, against 0.82 and 0.38 under
# `logmeanexp`). `logmeanexp` is `logsumexp - log(T)` and subtracts precisely
# that term. The split follows measured prevalence: enzymes 95% positive,
# other_organisms 78%, strains 25%, bacteria 17%.
CLASS_LOGITS_POOLING: dict[str, Pooling] = {
    "enzymes": "logsumexp",
    "other_organisms": "logsumexp",
    "bacteria": "logmeanexp",
    "strains": "logmeanexp",
}

# What a column the map does not name is pooled with -- a class from another
# schema, and the head's own OOS column. The historical uniform setting, so a
# schema gaining a type leaves the types already in the map alone and a caller
# who never heard of this setting gets what it used to do.
UNMAPPED_CLASS_POOLING: Pooling = "logsumexp"

# How many configurations one `pdm run tuning` sweep draws from the grid.
SWEEP_SIZE = 250


class ModelConfig(BaseModel):
    model_class: str = "ETEBrendaModel"
    optimizer: str = "adam"
    lr: PositiveFloat = 0.0003
    lr_scheduler: str = ""
    dropout: NonNegativeFloat = 0
    hidden_layers: list[NonNegativeInt] = [32]
    normalization: str = "layer"
    batch_size: PositiveInt = 32
    # Batch by padded chunk budget rather than document count, bounding peak
    # VRAM instead of batch size. 0 is off, and keeps the fixed count; TOML has
    # no null, so a sentinel rather than None (`save_model_config` round-trips
    # every field through tomlkit, which cannot serialise one).
    batch_max_chunks: NonNegativeInt = 0
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
    entity_logits_pooling: Pooling = "logsumexp"
    # Per column for the class head, or one name for all of them. The entity
    # head keeps `entity_logits_pooling`: it has one column per training-split
    # entity ID rather than one per type, so a per-class map does not describe
    # it. A name the map omits falls back to `UNMAPPED_CLASS_POOLING`.
    class_logits_pooling: Pooling | dict[str, Pooling] = CLASS_LOGITS_POOLING
    entity_entropy_threshold: NonNegativeFloat = 0.8
    biaffine_hidden_size: PositiveInt = 32


class MachineConfig(BaseModel):
    """Per-machine settings, read from the repo-root ``config.toml``.

    The runtime fields are process-global torch/allocator settings, applied by
    ``d3text.runtime.configure()`` at script start-up rather than at import.
    See ``config.toml.example``.
    """

    cpu_embeddings_cache_size: NonNegativeInt
    embeddings_store: str | None = None
    float32_matmul_precision: Float32MatmulPrecision = "medium"
    cudnn_allow_tf32: bool = True
    expandable_segments: bool = True
    tokenizers_parallelism: bool = True


class ETEModelConfig(ModelConfig):
    entity_layers: list[NonNegativeInt]
    class_layers: list[NonNegativeInt]


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


def load_tuning_config(
    path: str, rng: random.Random | None = None
) -> list[ModelConfig]:
    """Draw a random subset of the hyperparameter grid described by ``path``.

    ``rng`` is injectable so that a sweep can be replayed exactly; the default
    draws from a fresh ``Random``, which leaves successive sweeps independent
    of each other without reading or advancing the process-global ``random``
    state.
    """
    generator = random.Random() if rng is None else rng

    with open(path, "r") as config_file:
        # `unwrap()` to plain Python types. tomlkit's Integer/Float/String/Array
        # subclass their builtins, so pydantic takes them, but `bool` cannot be
        # subclassed -- a TOML bool inside an array arrives as `tomlkit.Bool`
        # and every ModelConfig with a bool field fails to validate.
        cfg = tomlkit.load(config_file).unwrap()

    layer_sizes = cfg["hidden_layers"]
    cfg["hidden_layers"] = generator.choices(
        tuple(itertools.combinations_with_replacement(layer_sizes, 1)),
        k=100,
    )

    cfgs = tuple(
        ModelConfig(**dict(zip(cfg.keys(), cell)))
        for cell in itertools.product(*cfg.values())
    )

    # A grid smaller than the sweep is a legitimate config, not an error, so
    # take it whole rather than letting `sample` raise on the population size.
    return generator.sample(cfgs, k=min(SWEEP_SIZE, len(cfgs)))


def save_model_config(config: dict, path: str) -> None:
    with open(path, "w") as config_file:
        tomlkit.dump(config, config_file)
