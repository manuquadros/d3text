"""Building a model from a config, and loading a checkpoint back into it.

The seam between a `ModelConfig` and a ready-to-train `Model`, plus the run
context a tracking run records about the built splits. Lives above
`d3text.models` rather than inside it because `dataset_metrics` reads an
`EntityRelationDataset`, and the model classes must stay importable without
the BRENDA data layer coming along.
"""

import torch
from jaxtyping import Float
from torch import Tensor

from .data.data import EntityRelationDataset
from .models.config import ModelConfig
from .models.entity_linking import BrendaClassificationModel
from .models.ete import ETEBrendaModel
from .models.ner import NERClassificationModel
from .schema import Schema

# What a config is allowed to name. The `Model` base class is too weak to stand
# here: it declares neither `compute_batch_losses` nor `evaluate_model`, though
# every concrete model implements both, so a caller holding a `Model` cannot
# train or evaluate it without the type system objecting — correctly.
# `ETEBrendaModel` composes a `BrendaClassificationModel` rather than
# subclassing it, so it is named here in its own right.
ConfigurableModel = (
    BrendaClassificationModel | ETEBrendaModel | NERClassificationModel
)

MODEL_CLASSES: dict[str, type[ConfigurableModel]] = {
    "BrendaClassificationModel": BrendaClassificationModel,
    "ETEBrendaModel": ETEBrendaModel,
    "NERClassificationModel": NERClassificationModel,
}


def build_model(
    config: ModelConfig,
    schema: Schema,
    class_freqs: Float[Tensor, " classes"] | None = None,
) -> ConfigurableModel:
    """The model `config.model_class` names, built under `schema`.

    Resolved from an explicit registry rather than `getattr(models, name)`,
    which resolved *any* attribute of the package and so failed late or not
    at all.

    :param config: names the model class and its hyperparameters.
    :param schema: the schema the corpus is indexed under. Its `class_names`
        become the class head's column order, and `ETEBrendaModel` reads its
        relation types off it rather than hardcoding them.
    :param class_freqs: class label frequencies, to seed the head's bias.
    :return: the built model.
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
        schema=schema,
        config=config,
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
    """Strip the `_orig_mod.` that `torch.compile` prepends to every key.

    A no-op on checkpoints `train` writes now that it compiles in place; it
    stays for the ones written while `train` wrapped the model instead. Must
    edit `state_dict` **in place**: torch slices each child module's state dict
    out of this very object after the hook returns.
    """
    renamed = {
        key.replace("_orig_mod.", ""): value
        for key, value in state_dict.items()
    }
    state_dict.clear()
    state_dict.update(renamed)


def model_size_mb(module: torch.nn.Module) -> float:
    """The resident size of `module`'s parameters and buffers, in MiB.

    :param module: the model to measure.
    :return: its size in MiB.
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

    The trainable count is the one that moves between configurations, since the
    base transformer is frozen — a run whose trainable count is the whole model
    has silently trained the encoder.

    :param module: the model to measure.
    :return: the metrics, under their tracking keys.
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

    Metrics rather than params so a run table sorts on them numerically. The
    document counts are what each split *planned* to hold, since this runs
    before anything has been read; `coverage_metrics` logs what was actually
    scored. Batch counts are absent because `TokenBudgetBatchSampler` declares
    no `__len__`.

    :param dataset: the built splits.
    :return: the metrics, under their tracking keys.
    """
    metrics = {"dataset/classes": float(len(dataset.class_map))}
    for split, rows in dataset.data.items():
        metrics[f"dataset/{split}_documents"] = float(len(rows))

    return metrics
