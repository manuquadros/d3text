"""Building a model from a config, and loading a checkpoint back into it.

`train`, `tune` and `evaluate` each used to construct the model themselves with
`getattr(models, config.model_class)`. That resolved any attribute of the
package, checked nothing, and — because it ran after the dataset had loaded —
reported a misspelled class name only minutes into a run.
"""

import dataclasses

import pytest
import torch
from torch import nn

from d3text import factory
from d3text.data.data import EntityRelationDataset
from d3text.models.config import ModelConfig
from d3text.models.models import (
    BrendaClassificationModel,
    ETEBrendaModel,
    NERClassificationModel,
)
from d3text.schema import RelationType

MODEL_NAMES = [
    "BrendaClassificationModel",
    "ETEBrendaModel",
    "NERClassificationModel",
]


@pytest.fixture
def dataset(tiny_schema):
    """The fields `build_model` reads off a dataset. The splits are not among
    them, so they stay empty."""
    return EntityRelationDataset(
        data={},
        schema=tiny_schema,
        entity_index={"enz1": 0, "bac1": 1},
        class_map={"enzymes": {"enz1"}, "bacteria": {"bac1"}},
        class_matrix=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    )


def config_for(name: str, **overrides) -> ModelConfig:
    return ModelConfig(
        model_class=name,
        base_model="prajjwal1/bert-mini",
        hidden_layers=[8],
        **overrides,
    )


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_documented_model_can_be_built(name, dataset, patch_base_model):
    """A model class the factory cannot reach is unreachable from every config,
    however correct the class itself is."""
    model = factory.build_model(config_for(name), dataset)

    assert type(model).__name__ == name


def test_the_built_model_is_wired_to_the_dataset(dataset, patch_base_model):
    """The entity head must be as wide as the dataset's entity index (plus the
    UNK column), or nothing downstream lines up."""
    model = factory.build_model(config_for("ETEBrendaModel"), dataset)

    assert isinstance(model, ETEBrendaModel)
    assert model.entities == ["enz1", "bac1", "UNK"]
    assert model.classes == ["enzymes", "bacteria", "OOS"]


@pytest.mark.parametrize(
    "name", ["BrendaClassificationModel", "NERClassificationModel"]
)
def test_the_class_head_takes_its_columns_from_the_schema(
    name, dataset, patch_base_model
):
    """The class head's column order is the schema's, not the `class_map`'s.

    Both are built from the same schema by the adapter, so they agree in
    practice — but a mapping's key order is an accident of how it was
    populated, and reading the columns off it makes every class label depend on
    that accident. Here the map is handed over reversed, and the head must not
    follow it.
    """
    dataset.class_map = dict(reversed(list(dataset.class_map.items())))

    model = factory.build_model(config_for(name), dataset)

    assert model.classes == ["enzymes", "bacteria", "OOS"]
    assert model.known_classes == ["enzymes", "bacteria"]


def test_the_relation_head_takes_its_columns_from_the_schema(
    dataset, patch_base_model
):
    """The relation head's columns are the schema's relation types, in its
    order. They were a tuple hardcoded in the model, duplicating the schema: a
    corpus declaring other relations still got BRENDA's, so the argmax of the
    corpus' label vector indexed some other relation's column.
    """
    schema = dataclasses.replace(
        dataset.schema,
        relation_types=(
            RelationType(
                name="Inhibits",
                subject_types=("enzymes",),
                object_types=("bacteria",),
            ),
            RelationType(name="none", is_none=True),
        ),
    )
    dataset.schema = schema

    model = factory.build_model(config_for("ETEBrendaModel"), dataset)

    assert isinstance(model, ETEBrendaModel)
    assert model.relations == ("Inhibits", "none")
    assert model.num_relations == 2
    assert model.relations_none_index == 1
    assert model.relation_classifier.bilinear.shape[0] == 2


@pytest.mark.parametrize("separate", [True, False])
def test_the_predicate_layer_the_config_asks_for_reaches_the_relation_head(
    separate, dataset, patch_base_model
):
    """The relation head can give the object of a pair its own projection
    instead of aliasing the subject's. The flag lived in the config and in the
    head's constructor, but the model never carried it from one to the other, so
    every model trained with the shared projection whatever its config said —
    and a sweep over the flag compared an architecture against itself.

    Building the head directly cannot catch that; only building a model from a
    config can.
    """
    model = factory.build_model(
        config_for("ETEBrendaModel", separate_predicate_layer=separate), dataset
    )
    assert isinstance(model, ETEBrendaModel)

    head = model.relation_classifier
    subject_params = {id(p) for p in head.hidden_linear.parameters()}
    object_params = {id(p) for p in head.hidden_linear_y.parameters()}

    if separate:
        assert head.hidden_linear_y is not head.hidden_linear
        assert object_params.isdisjoint(subject_params)
    else:
        assert head.hidden_linear_y is head.hidden_linear
        assert object_params == subject_params


@pytest.mark.parametrize("width", [8, 16])
def test_the_relation_head_is_as_wide_as_the_config_asks(
    width, dataset, patch_base_model
):
    """The width of the biaffine projection was a literal inside the head, so
    the config could not tune it and every model got 32 regardless."""
    model = factory.build_model(
        config_for("ETEBrendaModel", biaffine_hidden_size=width), dataset
    )
    assert isinstance(model, ETEBrendaModel)

    head = model.relation_classifier
    assert tuple(head.bilinear.shape) == (model.num_relations, width, width)
    assert head.linear.in_features == 2 * width
    assert head.hidden_linear[0].out_features == width
    assert head.hidden_linear_y[0].out_features == width


def test_the_frequencies_reach_both_heads(dataset, patch_base_model):
    """`train` and `tune` seed each head's bias from the training frequencies;
    `evaluate` passes none and takes the default init. Sending a frequency to
    the wrong head, or to neither, mis-seeds every prediction — and would look
    like nothing at all from outside.
    """
    freqs = torch.tensor([0.5, 0.25])
    seeded = factory.build_model(
        config_for("BrendaClassificationModel"),
        dataset,
        entity_freqs=freqs,
        class_freqs=freqs,
    )
    assert isinstance(seeded, BrendaClassificationModel)

    log_odds = torch.logit(freqs)
    torch.testing.assert_close(
        seeded.classifier.entity_classifier[-1].bias[:2], log_odds
    )
    torch.testing.assert_close(
        seeded.classifier.class_classifier.bias[:2], log_odds
    )


def test_a_model_built_without_frequencies_is_not_seeded(
    dataset, patch_base_model
):
    """`evaluate` builds the model with no frequencies at all; it must still
    build, and must not pretend to a prior it was never given."""
    unseeded = factory.build_model(
        config_for("BrendaClassificationModel"), dataset
    )
    assert isinstance(unseeded, BrendaClassificationModel)

    seeded = factory.build_model(
        config_for("BrendaClassificationModel"),
        dataset,
        class_freqs=torch.tensor([0.5, 0.25]),
    )
    assert not torch.equal(
        unseeded.classifier.class_classifier.bias,
        seeded.classifier.class_classifier.bias,
    )


def test_an_unknown_model_class_is_rejected_before_the_dataset_loads(dataset):
    """The point of the registry. `getattr` raised an `AttributeError` naming
    only the missing attribute, after the ~300 MB dataset had been read."""
    with pytest.raises(ValueError, match="names no such model") as excinfo:
        factory.build_model(config_for("NERClassicationModel"), dataset)

    # The message has to be actionable: this exact typo shipped in the repo's
    # own tuning grid, and `AttributeError` gave no hint what to write instead.
    for name in MODEL_NAMES:
        assert name in str(excinfo.value)


def test_a_model_class_naming_any_other_attribute_is_rejected(dataset):
    """`getattr(models, "torch")` resolved happily and failed later, somewhere
    else. A registry only knows about models."""
    with pytest.raises(ValueError, match="names no such model"):
        factory.build_model(config_for("torch"), dataset)


def test_the_registry_holds_exactly_the_documented_models():
    assert sorted(factory.MODEL_CLASSES) == sorted(MODEL_NAMES)
    assert factory.MODEL_CLASSES["NERClassificationModel"] is (
        NERClassificationModel
    )


class _Checkpointed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)


def compiled_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """What `torch.save(model.state_dict())` writes for a compiled model: every
    key behind the wrapper `torch.compile` put in front of it."""
    return {
        f"_orig_mod.{key}": value for key, value in module.state_dict().items()
    }


def test_a_compiled_checkpoint_loads_into_an_uncompiled_model():
    """`train` compiles before saving, `evaluate` does not compile before
    loading. Without the hook the keys match nothing at all."""
    trained = _Checkpointed()
    checkpoint = compiled_state_dict(trained)

    evaluated = _Checkpointed()
    evaluated.register_load_state_dict_pre_hook(factory.fix_keys_hook)
    evaluated.load_state_dict(checkpoint)

    torch.testing.assert_close(evaluated.linear.weight, trained.linear.weight)
    torch.testing.assert_close(evaluated.linear.bias, trained.linear.bias)


def test_the_checkpoint_is_unloadable_without_the_hook():
    """Proves the hook above is doing the work, rather than the checkpoint
    happening to load anyway."""
    checkpoint = compiled_state_dict(_Checkpointed())

    with pytest.raises(RuntimeError, match="Unexpected key"):
        _Checkpointed().load_state_dict(checkpoint)


def test_an_uncompiled_checkpoint_still_loads():
    """The hook must be a no-op on a checkpoint that was never compiled."""
    trained = _Checkpointed()

    evaluated = _Checkpointed()
    evaluated.register_load_state_dict_pre_hook(factory.fix_keys_hook)
    evaluated.load_state_dict(trained.state_dict())

    torch.testing.assert_close(evaluated.linear.weight, trained.linear.weight)


def test_the_hook_rewrites_the_state_dict_in_place():
    """torch slices each child module's state dict out of this very object once
    the hook returns, so a hook that built a fresh dict would be ignored."""
    state_dict = {"_orig_mod.linear.weight": torch.ones(2, 2)}
    original = state_dict

    factory.fix_keys_hook(nn.Linear(2, 2), state_dict, "", {}, True, [], [], [])

    assert state_dict is original
    assert list(state_dict) == ["linear.weight"]


def test_model_size_counts_parameters_and_buffers():
    module = nn.Linear(100, 100)  # 100 * 100 + 100 float32 parameters
    module.register_buffer("running", torch.zeros(256, dtype=torch.float32))

    expected = ((100 * 100 + 100) + 256) * 4 / 1024**2

    assert factory.model_size_mb(module) == pytest.approx(expected)


def test_an_empty_module_has_no_size():
    assert factory.model_size_mb(nn.Module()) == 0.0
