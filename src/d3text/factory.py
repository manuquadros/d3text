"""Building a model from a config, and loading a checkpoint back into it.

The seam between a `ModelConfig` + a dataset and a ready-to-train `Model`.
`train`, `tune` and `evaluate` each used to spell this out themselves, and the
three copies had already drifted apart.

Lives above `d3text.models` rather than inside it: resolving a dataset into
constructor arguments needs `d3text.data`, and importing that pulls in the
whole BRENDA layer. Keeping that out of `d3text.models` keeps the model classes
importable — in tests, in notebooks — without the data layer coming along.
"""

import functools
from collections.abc import Callable

import torch
from jaxtyping import Float
from torch import Tensor

from .data.data import EntityRelationDataset
from .models.brenda import BrendaModel
from .models.config import ModelConfig
from .models.ner import NERClassificationModel

# What a config is allowed to name. The `Model` base class is too weak to stand
# here: it declares no `evaluate_model`, though every concrete model implements
# one, so a caller holding a `Model` cannot evaluate it without the type system
# objecting — correctly.
ConfigurableModel = BrendaModel | NERClassificationModel

# The names a `config.model_class` may carry, and what each builds. Two of them
# name the *same* class: `BrendaModel` does end-to-end extraction when it holds a
# `RelationExtractor` and entity linking alone when it does not, so the choice is
# a constructor argument, not a subclass. `ETEBrendaModel` used to be that
# subclass, and overrode almost every method of its parent to widen it.
MODEL_BUILDERS: dict[str, Callable[..., ConfigurableModel]] = {
    "BrendaClassificationModel": functools.partial(
        BrendaModel, extract_relations=False
    ),
    "ETEBrendaModel": functools.partial(BrendaModel, extract_relations=True),
    "NERClassificationModel": NERClassificationModel,
}


def build_model(
    config: ModelConfig,
    dataset: EntityRelationDataset,
    entity_freqs: Float[Tensor, " entities"] | None = None,
    class_freqs: Float[Tensor, " classes"] | None = None,
) -> ConfigurableModel:
    """The model `config.model_class` names, built against `dataset`.

    Resolved from an explicit registry rather than `getattr(models, name)`,
    which was wrong twice over: a name naming no model at all surfaced as an
    `AttributeError` only once the ~300 MB dataset had finished loading, and a
    name matching *any* attribute of the package — an import, a helper —
    resolved to it and failed later still.
    """
    try:
        build = MODEL_BUILDERS[config.model_class]
    except KeyError:
        known = ", ".join(sorted(MODEL_BUILDERS))
        msg = (
            f"config names no such model: {config.model_class!r}. "
            f"Expected one of: {known}."
        )
        raise ValueError(msg) from None

    return build(
        schema=dataset.schema,
        class_matrix=dataset.class_matrix,
        config=config,
        entity_index=dataset.entity_index,
        entity_freqs=entity_freqs,
        class_freqs=class_freqs,
    )


def fix_keys_hook(
    module: torch.nn.Module,
    state_dict: dict,
    prefix: str,
    local_metadata: dict,
    strict: bool,
    missing_keys: list,
    unexpected_keys: list,
    error_msgs: list,
) -> None:
    """Strip the ``_orig_mod.`` that `torch.compile` prepends to every key.

    `train` now compiles the model in place, so the checkpoints it writes are
    keyed against the model itself and this is a no-op on them. It stays for
    the ones written while `train` wrapped the model instead: those are keyed
    against the wrapper, and `evaluate` loads them into an uncompiled model.

    Must edit `state_dict` **in place**: torch slices each child module's state
    dict out of this very object after the hook returns, so a fresh dict would
    be built and dropped on the floor.
    """
    renamed = {
        key.replace("_orig_mod.", ""): value
        for key, value in state_dict.items()
    }
    state_dict.clear()
    state_dict.update(renamed)


def model_size_mb(module: torch.nn.Module) -> float:
    """The resident size of `module`'s parameters and buffers, in MiB.

    Piotr Bialecki @ https://discuss.pytorch.org/t/finding-model-size/130275/2
    """
    param_size = sum(
        param.nelement() * param.element_size() for param in module.parameters()
    )
    buffer_size = sum(
        buffer.nelement() * buffer.element_size() for buffer in module.buffers()
    )
    return (param_size + buffer_size) / 1024**2


def model_metrics(module: torch.nn.Module) -> dict[str, float]:
    """The built model's size, keyed for a tracking run.

    The trainable count is the one that moves between configurations: the base
    transformer is frozen, so the head geometry and `base_layers_to_unfreeze`
    are all that change it. A run whose trainable count is the *whole* model
    has silently trained the encoder, which is visible here and nowhere else
    short of reading the checkpoint.
    """
    total = sum(param.numel() for param in module.parameters())
    trainable = sum(
        param.numel() for param in module.parameters() if param.requires_grad
    )

    return {
        "model/size_mb": model_size_mb(module),
        "model/parameters": float(total),
        "model/trainable_parameters": float(trainable),
        "model/trainable_fraction": trainable / total if total else 0.0,
    }


def dataset_metrics(dataset: EntityRelationDataset) -> dict[str, float]:
    """Split sizes and head geometry, keyed for a tracking run.

    Metrics rather than params so the run table sorts on them numerically: the
    first question asked of a surprising loss curve is whether that run saw the
    whole corpus or a `--limit` slice of it, and a param sorts as a string.

    The document counts are what each split *planned* to hold: this runs at
    setup, before anything has been read, so it cannot know how many documents
    the encodings file actually backs. `coverage_metrics` logs that from the
    pass that does know, under the same `dataset/` prefix.

    Batch counts are deliberately absent. `TokenBudgetBatchSampler` declares no
    `__len__` — how many batches a budget yields depends on the order the inner
    sampler draws — so `len(loader)` raises for exactly the configuration whose
    batch count would be most worth knowing. `run_epoch` counts batches as it
    goes and the per-epoch rate metrics carry the total instead.
    """
    entities, classes = dataset.class_matrix.shape
    metrics = {
        "dataset/entities": float(entities),
        "dataset/classes": float(classes),
    }
    for split, rows in dataset.data.items():
        metrics[f"dataset/{split}_documents"] = float(len(rows))

    return metrics
