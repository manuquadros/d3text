"""Building a model from a config, and loading a checkpoint back into it.

`train`, `tune` and `evaluate` each used to construct the model themselves with
`getattr(models, config.model_class)`. That resolved any attribute of the
package, checked nothing, and — because it ran after the dataset had loaded —
reported a misspelled class name only minutes into a run.
"""

import inspect

import pytest
import torch
from torch import nn

from d3text import factory
from d3text.data.data import EntityRelationDataset
from d3text.models.config import ModelConfig
from d3text.models.entity_linking import BrendaClassificationModel
from d3text.models.ete import ETEBrendaModel
from d3text.models.ner import NERClassificationModel
from d3text.schema import EntityType, RelationType, Schema

MODEL_NAMES = [
    "BrendaClassificationModel",
    "ETEBrendaModel",
    "NERClassificationModel",
]

# `ETEBrendaModel` needs relation types to build at all (its `none` column is
# found on the schema), so this carries one relation even though the other two
# model classes never look at it.
SCHEMA = Schema(
    entity_types=(
        EntityType(name="enzymes", prefix="enz"),
        EntityType(name="bacteria", prefix="bac"),
    ),
    relation_types=(
        RelationType(
            name="HasEnzyme", subject_types=("bacteria",), object_type="enzymes"
        ),
        RelationType(name="none", is_none=True),
    ),
)


@pytest.fixture
def dataset():
    """What `dataset_metrics` reads. The splits are not among it, so they stay
    empty."""
    return EntityRelationDataset(
        data={}, class_map={"enzymes": {"enz1"}, "bacteria": {"bac1"}}
    )


def config_for(name: str, token_labels_store: str = "") -> ModelConfig:
    return ModelConfig(
        model_class=name,
        base_model="prajjwal1/bert-mini",
        hidden_layers=[8],
        token_labels_store=token_labels_store,
    )


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_documented_model_can_be_built(
    name, patch_base_model, empty_token_label_store
):
    """A model class the factory cannot reach is unreachable from every config,
    however correct the class itself is."""
    store = str(empty_token_label_store) if name == "ETEBrendaModel" else ""
    model = factory.build_model(config_for(name, store), SCHEMA)

    assert type(model).__name__ == name


def test_the_built_model_is_wired_to_the_schema(
    patch_base_model, empty_token_label_store
):
    """The class head's columns come from the schema alone, so nothing about a
    built model's geometry follows the corpus any more."""
    model = factory.build_model(
        config_for("ETEBrendaModel", str(empty_token_label_store)), SCHEMA
    )

    assert isinstance(model, ETEBrendaModel)
    assert model.classes == ["enzymes", "bacteria", "OOS"]


def test_build_model_takes_neither_a_dataset_nor_entity_frequencies():
    """It read a dataset only for the entity head's index and class matrix,
    and `entity_freqs` only to seed that head's bias. Either left in the
    signature would be an argument every caller computes and nothing reads —
    `entity_freqs` costs a pass over the training split to produce."""
    assert list(inspect.signature(factory.build_model).parameters) == [
        "config",
        "schema",
        "class_freqs",
    ]


def test_the_class_frequencies_reach_the_class_head(patch_base_model):
    """`train` and `tune` seed the head's bias from the training frequencies;
    `evaluate` passes none and takes the default init. A frequency that
    reaches neither head mis-seeds every prediction — and would look like
    nothing at all from outside.
    """
    freqs = torch.tensor([0.5, 0.25])
    seeded = factory.build_model(
        config_for("BrendaClassificationModel"), SCHEMA, class_freqs=freqs
    )
    assert isinstance(seeded, BrendaClassificationModel)

    torch.testing.assert_close(
        seeded.classifier.class_classifier.bias[:2], torch.logit(freqs)
    )


def test_a_model_built_without_frequencies_is_not_seeded(patch_base_model):
    """`evaluate` builds the model with no frequencies at all; it must still
    build, and must not pretend to a prior it was never given."""
    unseeded = factory.build_model(
        config_for("BrendaClassificationModel"), SCHEMA
    )
    assert isinstance(unseeded, BrendaClassificationModel)

    seeded = factory.build_model(
        config_for("BrendaClassificationModel"),
        SCHEMA,
        class_freqs=torch.tensor([0.5, 0.25]),
    )
    assert not torch.equal(
        unseeded.classifier.class_classifier.bias,
        seeded.classifier.class_classifier.bias,
    )


def test_an_unknown_model_class_is_rejected():
    """The point of the registry. `getattr` raised an `AttributeError` naming
    only the missing attribute, and resolved any attribute of the package."""
    with pytest.raises(ValueError, match="names no such model") as excinfo:
        factory.build_model(config_for("NERClassicationModel"), SCHEMA)

    # The message has to be actionable: this exact typo shipped in the repo's
    # own tuning grid, and `AttributeError` gave no hint what to write instead.
    for name in MODEL_NAMES:
        assert name in str(excinfo.value)


def test_a_model_class_naming_any_other_attribute_is_rejected():
    """`getattr(models, "torch")` resolved happily and failed later, somewhere
    else. A registry only knows about models."""
    with pytest.raises(ValueError, match="names no such model"):
        factory.build_model(config_for("torch"), SCHEMA)


def test_the_dataset_metrics_no_longer_count_entity_columns(dataset):
    """`dataset/entities` was the entity head's width. With no such head the
    key would chart a number nothing sizes anything from."""
    metrics = factory.dataset_metrics(dataset)

    assert metrics == {"dataset/classes": 2.0}


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
