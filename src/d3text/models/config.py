import itertools
import pathlib
import random
from typing import Literal

import tomlkit
import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
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
TokenLossWeighting = Literal["unweighted", "balanced", "focal"]

# How many configurations one `pdm run tuning` sweep draws from the grid.
SWEEP_SIZE = 250


class ModelConfig(BaseModel):
    # Forbid rather than ignore: a field with no reader (nothing outside
    # this file names it) must fail loudly at load time, not be silently
    # dropped while every config that still carries it looks accepted.
    model_config = ConfigDict(extra="forbid")

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
    relation_label_smoothing: NonNegativeFloat = 0.0
    relation_loss_weighting: RelationLossWeighting = "unweighted"
    relation_focal_gamma: NonNegativeFloat = 2.0
    common_hidden_block: bool = True
    # Epochs over which `ETEBrendaModel` ramps its relation loss up to full
    # weight; no other objective in any model rides this schedule.
    ramp_epochs: int = 0
    separate_predicate_layer: bool = False
    consistency_weight: float = 0.1
    # Pools both heads. `logmeanexp` is `logsumexp - log(T)`: `logsumexp` is a
    # smooth max, but it is also `max + log(T)` to within a bounded correction,
    # so on the ~8,000-token documents here it added about nine nats of length
    # bias to every column alike. A class absent from most documents cannot be
    # made negative under that without pushing all its tokens far down, and the
    # cheapest answer to the pooled objective is a channel that never fires.
    # That was measured at `--limit 500`, where it is stark: document recall
    # 0.114 for strains and 0.143 for bacteria, against 0.494 and 0.755 under
    # `logmeanexp`, which subtracts precisely that term and nothing else.
    #
    # On the whole training split the collapse does not reproduce -- `logsumexp`
    # reaches 0.829 and 0.925 there -- and the two poolings tie to within noise
    # on every class, by F1 and by average precision alike. `logmeanexp` is the
    # default because it is marginally ahead on validation and because a head
    # whose document logit does not grow with document length is the one to
    # prefer when nothing separates them, not because the alternative fails.
    # The price is that a lone mention no longer carries a long document, which
    # is what the smooth max was for.
    entity_logits_pooling: Literal["logsumexp", "logmeanexp", "max", "mean"] = (
        "logmeanexp"
    )
    # Entropy cutoff (in nats, so bounded by log(num_entities)) on the entity
    # softmax below which a token is proposed as a relation argument. It sets
    # how many candidate pairs the relation head ever sees, and so how much gold
    # it can never recover: raising it proposes more pairs, most of them `none`.
    entity_entropy_threshold: NonNegativeFloat = 0.8
    biaffine_hidden_size: PositiveInt = 32
    # Path to a `precompute-token-labels` store. Non-empty builds the
    # token-level span tagger head and adds its masked cross-entropy to the
    # document-level losses (which stay: they carry the gold links never named
    # in the text, which no token supervision reaches). Empty — the default,
    # and TOML's spelling of null — keeps the model exactly as before, tagger
    # head and all: old configs and old checkpoints are untouched.
    token_labels_store: str = ""
    # The span tagger's `OUTSIDE` column is ~91% of kept tokens (label_audit.json
    # from the FEAT-06 tagger arm), so a plain argmax over a plainly-averaged
    # cross-entropy defaults toward predicting it — the same imbalance
    # `relation_loss_weighting` exists to counter on the relation head, mirrored
    # here with the same three-way choice. `unweighted` — the default — is
    # byte-identical to the previous behaviour; a config with no
    # `token_labels_store` never reads either field.
    token_loss_weighting: TokenLossWeighting = "unweighted"
    token_focal_gamma: NonNegativeFloat = 2.0
    # A document-level class negative is asserted even for a class whose text
    # names an entity of that type — BRENDA links only what an enzyme record
    # needs, not everything mentioned. `False` — the default — keeps the hard
    # 0 target, as before. `True` abstains that (document, class) negative
    # wherever `token_labels_store`'s dictionary matched the type anywhere in
    # the text, gold-linked or not, so it requires that store: there is
    # nothing to abstain against without it.
    class_negative_abstention: bool = False

    @model_validator(mode="after")
    def _class_negative_abstention_needs_a_label_store(self) -> "ModelConfig":
        if self.class_negative_abstention and not self.token_labels_store:
            msg = (
                "class_negative_abstention requires token_labels_store: "
                "the abstention mask is read from its dictionary matches"
            )
            raise ValueError(msg)
        return self


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
