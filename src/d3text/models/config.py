import itertools
import math
import pathlib
import random
import warnings
from collections.abc import Collection, Iterator
from typing import Annotated, Any, Literal

import tomlkit
import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    ValidationError,
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
Float32MatmulPrecision = Literal["highest", "high", "medium"]
# Both are read through a `match` whose unmatched arm is a no-op, so an
# unvalidated typo would silently train with neither. `""` is TOML's null.
LRSchedulerName = Literal["", "reduce_on_plateau", "exponential"]
Normalization = Literal["layer", "batch", "none"]
RelationLossWeighting = Literal["unweighted", "balanced", "focal"]
TokenLossWeighting = Literal["unweighted", "balanced", "focal"]

# How many configurations one `pdm run tuning` sweep draws from the grid.
SWEEP_SIZE = 250
MAX_HIDDEN_LAYERS = 3

# The key `cpu_embeddings_cache_mb` replaced, and a rough per-document cost
# to translate an old document count into memory in the error message.
DOCUMENT_BUDGET_KEY = "cpu_embeddings_cache_size"
MB_PER_CACHED_DOCUMENT = 15

# The run-config key `token_supervision` replaced. Old run configs, and those
# saved beside checkpoints, still carry it.
LABEL_STORE_PATH_KEY = "token_labels_store"

MACHINE_CONFIG_PATH = (
    pathlib.Path(__file__).parent.parent.parent.parent / "config.toml"
)


class ModelConfig(BaseModel):
    """One run's training configuration; see the configuration reference."""

    # A misspelt key must fail at load, not be dropped while the config
    # looks accepted.
    model_config = ConfigDict(extra="forbid")

    model_class: str = "ETEBrendaModel"
    seed: int = 42
    optimizer: str = "adam"
    lr: PositiveFloat = 0.0003
    lr_scheduler: LRSchedulerName = ""
    dropout: Annotated[float, Field(ge=0.0, le=1.0)] = 0
    hidden_layers: list[NonNegativeInt] = [32]
    normalization: Normalization = "layer"
    batch_size: PositiveInt = 32
    # Here and in the `*_lr` fields, 0 means "unset": TOML has no null, and
    # `save_model_config` round-trips every field through tomlkit.
    batch_max_chunks: NonNegativeInt = 0
    num_epochs: PositiveInt = 100
    patience: NonNegativeInt = 2
    selection_metrics: list[str] = []
    base_model: str = "michiyasunaga/BioLinkBERT-base"
    relation_label_smoothing: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    relation_loss_weighting: RelationLossWeighting = "unweighted"
    relation_focal_gamma: NonNegativeFloat = 2.0
    common_hidden_block: bool = True
    gradient_checkpointing: bool = False
    unfrozen_top_layers: NonNegativeInt = 0
    base_model_lr: NonNegativeFloat = 0.0
    class_head_lr: NonNegativeFloat = 0.0
    # Non-negative: the ramp divides by it, and a negative value inverted
    # the schedule instead of raising.
    ramp_epochs: NonNegativeInt = 0
    separate_predicate_layer: bool = False
    entity_logits_pooling: Literal["logsumexp", "logmeanexp", "max", "mean"] = (
        "logmeanexp"
    )
    biaffine_hidden_size: PositiveInt = 32
    token_supervision: bool = False
    token_loss_weighting: TokenLossWeighting = "unweighted"
    token_focal_gamma: NonNegativeFloat = 2.0
    token_ambiguous_downweight: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    class_negative_abstention: bool = False
    class_negative_abstention_min_chars: NonNegativeInt = 8
    class_negative_abstention_min_chars_by_class: dict[str, NonNegativeInt] = {}
    class_negative_downweight: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0

    @model_validator(mode="before")
    @classmethod
    def _migrate_the_label_store_path(cls, data: Any) -> Any:
        """Turn an old run config's store path into `token_supervision`.

        A non-empty path meant "train with token supervision", which is what
        it becomes; the path itself is now read from the machine config, so a
        path that disagrees with it — or that it lacks — is warned about
        rather than silently swapped for another store. An empty one meant
        the default and is dropped.
        """
        if not isinstance(data, dict) or LABEL_STORE_PATH_KEY not in data:
            return data

        # A copy, as `MachineConfig._migrate_the_document_budget` takes one.
        data = dict(data)
        old_path = data.pop(LABEL_STORE_PATH_KEY)
        if not old_path:
            return data
        if not data.get("token_supervision", True):
            msg = (
                f"{LABEL_STORE_PATH_KEY} = {old_path!r} asks for token "
                "supervision and token_supervision = false refuses it; drop "
                f"{LABEL_STORE_PATH_KEY}, whose path now lives in the "
                f"[{LABEL_STORE_PATH_KEY}] table of {MACHINE_CONFIG_PATH}"
            )
            raise ValueError(msg)

        data["token_supervision"] = True
        base_model = data.get(
            "base_model", cls.model_fields["base_model"].default
        )
        configured = machine_config().token_labels_store.get(base_model)
        # RuntimeWarning rather than DeprecationWarning, for the reason
        # `_migrate_the_document_budget` gives.
        if configured is None:
            warnings.warn(
                f"{LABEL_STORE_PATH_KEY} = {old_path!r} is now "
                f"token_supervision = true, and the path moved to the "
                f"[{LABEL_STORE_PATH_KEY}] table of {MACHINE_CONFIG_PATH}, "
                f"which has no entry for {base_model!r}: add "
                f"{base_model!r} = {old_path!r} there",
                RuntimeWarning,
                stacklevel=2,
            )
        elif not _same_file(configured, old_path):
            warnings.warn(
                f"{LABEL_STORE_PATH_KEY} = {old_path!r} is now "
                f"token_supervision = true, and the store read is the one "
                f"[{LABEL_STORE_PATH_KEY}] in {MACHINE_CONFIG_PATH} names for "
                f"{base_model!r}: {configured!r}, not the path this config "
                "gave",
                RuntimeWarning,
                stacklevel=2,
            )
        return data

    @model_validator(mode="after")
    def _class_negative_abstention_needs_a_label_store(self) -> "ModelConfig":
        if self.class_negative_abstention and not self.token_supervision:
            msg = (
                "class_negative_abstention requires token_supervision: "
                "the abstention mask is read from its dictionary matches"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _ete_needs_a_label_store(self) -> "ModelConfig":
        if self.model_class == "ETEBrendaModel" and not self.token_supervision:
            msg = (
                "ETEBrendaModel requires token_supervision: a gold "
                "relation argument's representation is pooled from its own "
                "mention positions there, and with no store every gold "
                "argument is dropped rather than trained on"
            )
            raise ValueError(msg)
        return self


class MachineConfig(BaseModel):
    """Per-machine settings, read from the repo-root `config.toml`.

    The runtime fields are process-global torch and allocator settings, applied
    by `d3text.runtime.configure()` at script start-up rather than at import.
    See `config.toml.example`.
    """

    # A misspelt performance knob silently left at its default reads as a
    # slow machine, with nothing in any log to tell the two apart.
    model_config = ConfigDict(extra="forbid")

    cpu_embeddings_cache_mb: NonNegativeInt = 0
    # Store tables are keyed by base model: each store is built by one.
    embeddings_store: dict[str, str] = {}
    layer_boundary_store: dict[str, str] = {}
    encodings_store: dict[str, str] = {}
    token_labels_store: dict[str, str] = {}
    linking_corpora: str | None = None
    float32_matmul_precision: Float32MatmulPrecision = "medium"
    cudnn_allow_tf32: bool = True
    expandable_segments: bool = True
    tokenizers_parallelism: bool = True

    @model_validator(mode="before")
    @classmethod
    def _migrate_the_document_budget(cls, data: Any) -> Any:
        """Refuse a non-zero document count where megabytes are now expected.

        The rename changed the unit as well as the name, and no non-zero
        number means the same thing under both: silently reading 4000 as
        megabytes would cap the cache at 4 GB where the file asked for 58.
        `0` does mean the same thing in either unit, so it is migrated with a
        warning rather than refused — every `config.toml` the tracked scripts
        generated before the rename carries exactly that.
        """
        if not isinstance(data, dict) or DOCUMENT_BUDGET_KEY not in data:
            return data

        # A copy: `model_validate` hands the caller's own dict over, and
        # dropping a key out of it would be a side effect of reading a config.
        data = dict(data)
        documents = data.pop(DOCUMENT_BUDGET_KEY)
        if documents == 0:
            # RuntimeWarning, as the rest of the tree warns its operator:
            # DeprecationWarning is filtered out of a normal run, and the
            # reader of `config.toml` would never see this one.
            warnings.warn(
                f"{DOCUMENT_BUDGET_KEY} is now cpu_embeddings_cache_mb, in "
                "megabytes; 0 means the same in either unit, so the cache "
                "stays off and the key can be renamed at leisure",
                RuntimeWarning,
                stacklevel=2,
            )
            return data

        estimate = (
            f"about {documents * MB_PER_CACHED_DOCUMENT} MB"
            if isinstance(documents, int)
            else "far more than that many megabytes"
        )
        msg = (
            f"{DOCUMENT_BUDGET_KEY} is now cpu_embeddings_cache_mb, and the "
            "unit changed with the name: it counted documents, and a cached "
            f"document is one row per token — ~{MB_PER_CACHED_DOCUMENT} MB "
            f"on this corpus — so the {documents} configured here is "
            f"{estimate}, not {documents} MB. Set cpu_embeddings_cache_mb to "
            "a budget in megabytes (0 turns the cache off)."
        )
        raise ValueError(msg)


def load_model_config(path: str) -> ModelConfig:
    with open(path, "r") as config_file:
        model_config = ModelConfig(**tomlkit.load(config_file))

    return model_config


def machine_config() -> MachineConfig:
    """Load the repo-root `config.toml`.

    :return: the settings, falling back to a zero-cache default when the file
        is absent so that importing `d3text.models` never fails on a missing,
        uncommitted config.
    """
    path = MACHINE_CONFIG_PATH
    try:
        with path.open("r") as config:
            contents = tomlkit.load(config)
    except FileNotFoundError:
        return MachineConfig()
    try:
        return MachineConfig(**contents)
    except ValidationError as error:
        # The failure surfaces at import of `d3text.models.base`, far from the
        # file that caused it, and pydantic names only the field.
        error.add_note(f"while reading {path}")
        raise


def _same_file(first: str, second: str) -> bool:
    """Whether two configured paths name one file, relative ones from here."""
    return (
        pathlib.Path(first).expanduser().resolve()
        == pathlib.Path(second).expanduser().resolve()
    )


def _store_path(
    table: str, entries: dict[str, str], base_model: str, builder: str
) -> pathlib.Path:
    """`entries[base_model]` as a path, or an error naming what to add."""
    try:
        return pathlib.Path(entries[base_model]).expanduser()
    except KeyError:
        msg = (
            f"no [{table}] entry for {base_model!r} in {MACHINE_CONFIG_PATH}: "
            f'add {base_model!r} = "<path>" under [{table}], naming the '
            f"store `pdm run {builder}` wrote for that base model"
        )
        raise LookupError(msg) from None


def encodings_path(base_model: str) -> pathlib.Path:
    """The precomputed encodings HDF5 this machine holds for `base_model`.

    :param base_model: the Hugging Face id the encodings were tokenized with.
    :return: the path `MachineConfig.encodings_store` gives for it, with `~`
        expanded; a relative one is left relative to the working directory.
    :raises LookupError: if `config.toml` has no `[encodings_store]` entry
        for `base_model`.
    """
    return _store_path(
        "encodings_store",
        machine_config().encodings_store,
        base_model,
        "precompute-encodings",
    )


def token_labels_path(config: ModelConfig) -> pathlib.Path | None:
    """The token label store a run under `config` reads, if it reads one.

    :param config: the run's config; its `base_model` keys the lookup.
    :return: the path `MachineConfig.token_labels_store` gives for the base
        model, with `~` expanded, or `None` when `token_supervision` is off.
    :raises LookupError: if `token_supervision` is on and `config.toml` has
        no `[token_labels_store]` entry for the base model.
    """
    if not config.token_supervision:
        return None
    return _store_path(
        "token_labels_store",
        machine_config().token_labels_store,
        config.base_model,
        "precompute-token-labels",
    )


def load_tuning_config(
    path: str,
    rng: random.Random | None = None,
    excluded: Collection[ModelConfig] = (),
) -> Iterator[ModelConfig]:
    """Yield unique random configurations from the grid described by `path`.

    :param path: the sweep config to read.
    :param rng: injectable so a sweep can be replayed exactly; the default
        draws from a fresh `Random`, leaving successive sweeps independent
        without touching the process-global `random` state.
    :param excluded: configurations already attempted by an earlier run.
    :return: at most `SWEEP_SIZE` configurations, built as they are consumed.
    """
    generator = random.Random() if rng is None else rng

    with open(path, "r") as config_file:
        # `unwrap()` to plain Python types. tomlkit's Integer/Float/String/Array
        # subclass their builtins, so pydantic takes them, but `bool` cannot be
        # subclassed -- a TOML bool inside an array arrives as `tomlkit.Bool`
        # and every ModelConfig with a bool field fails to validate.
        cfg = tomlkit.load(config_file).unwrap()

    layer_sizes = sorted(set(cfg["hidden_layers"]), reverse=True)
    cfg["hidden_layers"] = [
        list(layers)
        for depth in range(1, MAX_HIDDEN_LAYERS + 1)
        for layers in itertools.combinations_with_replacement(
            layer_sizes, depth
        )
    ]

    keys = tuple(cfg)
    choices = tuple(cfg.values())
    grid_size = math.prod(len(values) for values in choices)
    excluded_keys = {config.model_dump_json() for config in excluded}

    # Sparse Fisher-Yates shuffle: draw grid indices without replacement while
    # storing only positions visited by this sweep, not the Cartesian product.
    swaps: dict[int, int] = {}
    yielded = 0
    for remaining in range(grid_size, 0, -1):
        pick = generator.randrange(remaining)
        index = swaps.get(pick, pick)
        swaps[pick] = swaps.get(remaining - 1, remaining - 1)

        offsets = []
        for values in reversed(choices):
            index, offset = divmod(index, len(values))
            offsets.append(offset)
        cell = [
            values[offset] for values, offset in zip(choices, reversed(offsets))
        ]
        config = ModelConfig(**dict(zip(keys, cell)))
        if config.model_dump_json() in excluded_keys:
            continue

        yield config
        yielded += 1
        if yielded == SWEEP_SIZE:
            return


def save_model_config(config: dict, path: str) -> None:
    with open(path, "w") as config_file:
        tomlkit.dump(config, config_file)
