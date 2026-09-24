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

from .data.data import BrendaDataset, EntityRelationDataset
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
    state_dict: dict[str, Tensor],
    prefix: str,
    local_metadata: dict[str, object],
    strict: bool,
    missing_keys: list[str],
    unexpected_keys: list[str],
    error_msgs: list[str],
) -> None:
    """Strip the `_orig_mod.` that `torch.compile` prepends to every key.

    A no-op on checkpoints `train` writes now that it compiles in place; it
    stays for the ones written while `train` wrapped the model instead. Also
    drops every key ending in `_neg_inf`, the padding-fill constant older
    checkpoints carried as a buffer on every `Model` (including ones nested
    under a composing model, e.g. `two_head._neg_inf`), which strict loading
    would otherwise reject as unexpected. Must edit `state_dict` **in
    place**: torch slices each child module's state dict out of this very
    object after the hook returns.
    """
    renamed = {
        key.replace("_orig_mod.", ""): value
        for key, value in state_dict.items()
        if key.rsplit(".", 1)[-1] != "_neg_inf"
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


def _split_entities_by_type(
    split: BrendaDataset, schema: Schema
) -> dict[str, set[str]]:
    """The distinct entity IDs `split` names, grouped by type.

    Relation arguments are the only entity IDs `BrendaDataset.data` still
    carries — the raw per-type mention columns are dropped when a split is
    built — so a document naming an entity in no relation contributes nothing
    here.

    :param split: the built split to read.
    :param schema: types each entity ID by its declared prefix.
    :return: entity type name -> the distinct IDs of that type `split` names.
    :raises KeyError: if an entity ID wears a prefix `schema` never declared.
        `filter_relations` drops such arguments before a split is built, so
        this only fires if that invariant breaks.
    """
    entities: dict[str, set[str]] = {}
    for relations in split.data["relations"]:
        for pairs in relations:
            for pair in pairs:
                for entity_id in pair:
                    type_name = schema.type_of(entity_id).name
                    entities.setdefault(type_name, set()).add(entity_id)

    return entities


def dataset_metrics(
    dataset: EntityRelationDataset, schema: Schema
) -> dict[str, float]:
    """Split sizes, head geometry and entity novelty, keyed for a tracking run.

    Metrics rather than params so a run table sorts on them numerically. The
    document counts are what each split *planned* to hold, since this runs
    before anything has been read; `coverage_metrics` logs what was actually
    scored. Batch counts are absent because `TokenBudgetBatchSampler` declares
    no `__len__`.

    `dataset/{split}_{type}_unseen_rate` is the share of that split's distinct
    entities, by type, absent from `dataset.class_map` — the training split's
    vocabulary whether it was derived from a loaded training split or read
    back off a checkpoint. A type with no entity in the split is omitted
    rather than divided by zero; a type with an empty (or missing)
    `class_map` entry still reports, at rate 1.0, since every one of its
    split entities is then unseen.

    :param dataset: the built splits.
    :param schema: types each entity ID by its declared prefix, rather than
        re-deriving the prefix map from `dataset.class_map`, which has no
        entry — and so no prefix — for a type absent from the training split.
    :return: the metrics, under their tracking keys.
    :raises KeyError: if a split names an entity ID wearing a prefix `schema`
        never declared. `filter_relations` keeps `BrendaDataset` free of such
        IDs, so this only fires if that invariant breaks.
    """
    metrics = {"dataset/classes": float(len(dataset.class_map))}
    for split, rows in dataset.data.items():
        metrics[f"dataset/{split}_documents"] = float(len(rows))
        for type_name, entity_ids in _split_entities_by_type(
            rows, schema
        ).items():
            seen = dataset.class_map.get(type_name, set())
            unseen = entity_ids - seen
            metrics[f"dataset/{split}_{type_name}_unseen_rate"] = len(
                unseen
            ) / len(entity_ids)

    return metrics
