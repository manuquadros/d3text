"""Building a model from a config, and loading a checkpoint back into it.

The seam between a `ModelConfig` + a dataset and a ready-to-train `Model`.
`train`, `tune` and `evaluate` each used to spell this out themselves, and the
three copies had already drifted apart.

Lives above `d3text.models` rather than inside it: resolving a dataset into
constructor arguments needs `d3text.data`, and importing that pulls in the
whole BRENDA layer. Keeping that out of `d3text.models` keeps the model classes
importable — in tests, in notebooks — without the data layer coming along.
"""

import torch
from jaxtyping import Float
from torch import Tensor

from .data.data import EntityRelationDataset
from .models.config import ModelConfig
from .models.models import (
    BrendaClassificationModel,
    ETEBrendaModel,
    NERClassificationModel,
)

# What a config is allowed to name. The `Model` base class is too weak to stand
# here: it declares neither `compute_batch_losses` nor `evaluate_model`, though
# every concrete model implements both, so a caller holding a `Model` cannot
# train or evaluate it without the type system objecting — correctly.
# `ETEBrendaModel` subclasses `BrendaClassificationModel`, so it is covered.
ConfigurableModel = BrendaClassificationModel | NERClassificationModel

MODEL_CLASSES: dict[str, type[ConfigurableModel]] = {
    "BrendaClassificationModel": BrendaClassificationModel,
    "ETEBrendaModel": ETEBrendaModel,
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
        model_class = MODEL_CLASSES[config.model_class]
    except KeyError:
        known = ", ".join(sorted(MODEL_CLASSES))
        msg = (
            f"config names no such model: {config.model_class!r}. "
            f"Expected one of: {known}."
        )
        raise ValueError(msg) from None

    return model_class(
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

    `train` compiles the model before saving it, so its checkpoints are keyed
    against the compiled wrapper; `evaluate` loads them into an uncompiled one.

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
