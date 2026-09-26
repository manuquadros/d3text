"""Building a model from a config, and loading a checkpoint back into it.

`train`, `tune` and `evaluate` each used to construct the model themselves with
`getattr(models, config.model_class)`. That resolved any attribute of the
package, checked nothing, and — because it ran after the dataset had loaded —
reported a misspelled class name only minutes into a run.
"""

import inspect

import pandas as pd
import pytest
import torch
from torch import nn

from d3text import factory
from d3text.data.data import BrendaDataset, EntityRelationDataset
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


def config_for(name: str, token_supervision: bool = False) -> ModelConfig:
    return ModelConfig(
        model_class=name,
        base_model="prajjwal1/bert-mini",
        hidden_layers=[8],
        token_supervision=token_supervision,
    )


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_documented_model_can_be_built(
    name, patch_base_model, empty_token_label_store
):
    """A model class the factory cannot reach is unreachable from every config,
    however correct the class itself is."""
    model = factory.build_model(
        config_for(name, token_supervision=name == "ETEBrendaModel"), SCHEMA
    )

    assert type(model).__name__ == name


def test_the_built_model_is_wired_to_the_schema(
    patch_base_model, empty_token_label_store
):
    """The class head's columns come from the schema alone, so nothing about a
    built model's geometry follows the corpus any more."""
    model = factory.build_model(
        config_for("ETEBrendaModel", token_supervision=True), SCHEMA
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
    metrics = factory.dataset_metrics(dataset, SCHEMA)

    assert metrics == {"dataset/classes": 2.0}


def test_dataset_classes_excludes_the_head_s_oos_column(
    dataset, patch_base_model
):
    """The glossary describes `dataset/classes` as class-head columns with
    `OOS` excluded — pinned against a built model's actual head width,
    rather than restating the schema's entity-type count under another
    name."""
    model = factory.build_model(config_for("BrendaClassificationModel"), SCHEMA)
    metrics = factory.dataset_metrics(dataset, SCHEMA)

    assert metrics["dataset/classes"] == len(model.classes) - 1


def _split(relations: list[list[dict[tuple[str, str], int]]]) -> BrendaDataset:
    """A `BrendaDataset` built straight from `relations`, one list per
    document. `encodings=None` skips `_drop_empty_documents` and the
    provenance check entirely, so no HDF5 file is needed."""
    frame = pd.DataFrame(
        {
            "pubmed_id": range(len(relations)),
            "relations": relations,
            "classes": [[] for _ in relations],
        }
    )
    return BrendaDataset(frame)


def test_dataset_metrics_reports_unseen_entity_rate_per_split_and_type():
    """Distinct entities of a type in a split, absent from the training
    vocabulary, over that split's distinct count of the type — checked
    against a hand-computed expectation on a fixture built to overlap only
    partially."""
    train_vocab = {"enzymes": {"enz1", "enz2"}, "bacteria": {"bac1"}}
    # val: enzymes {enz1 seen, enz3 unseen} -> 1/2; bacteria {bac1 seen,
    # bac2 unseen} -> 1/2.
    val = _split(
        [
            [{("bac1", "enz1"): 0}],
            [{("bac2", "enz3"): 0}],
        ]
    )
    # test: enzymes {enz1, enz2}, both seen -> 0/2; bacteria {bac1 seen,
    # bac2, bac3 unseen} -> 2/3.
    test = _split(
        [
            [{("bac1", "enz1"): 0}],
            [{("bac2", "enz2"): 0}],
            [{("bac3", "enz2"): 0}],
        ]
    )
    dataset = EntityRelationDataset(
        data={"val": val, "test": test}, class_map=train_vocab
    )

    metrics = factory.dataset_metrics(dataset, SCHEMA)

    assert metrics["dataset/val_enzymes_unseen_rate"] == pytest.approx(0.5)
    assert metrics["dataset/val_bacteria_unseen_rate"] == pytest.approx(0.5)
    assert metrics["dataset/test_enzymes_unseen_rate"] == pytest.approx(0.0)
    assert metrics["dataset/test_bacteria_unseen_rate"] == pytest.approx(2 / 3)


def test_dataset_metrics_omits_unseen_rate_for_a_type_with_no_split_entity():
    """Dividing by a split's distinct count of a type it names zero of would
    be a `ZeroDivisionError`; the key is left out instead, the same
    convention `test/relation_argument_set_size` uses."""
    dataset = EntityRelationDataset(
        data={"test": _split([[{("bac1", "enz1"): 0}]])},
        class_map={"enzymes": {"enz1"}, "bacteria": {"bac1"}},
    )

    metrics = factory.dataset_metrics(dataset, SCHEMA)

    assert "dataset/test_enzymes_unseen_rate" in metrics
    assert "dataset/test_strains_unseen_rate" not in metrics


def test_dataset_metrics_evaluate_style_dataset_gets_a_test_rate():
    """`evaluate.py` builds only the test split, under the checkpoint's
    recorded vocabulary — `class_map` here plays the training-vocabulary role
    even though no training split is loaded, so the rate must still land."""
    dataset = EntityRelationDataset(
        data={"test": _split([[{("bac9", "enz1"): 0}]])},
        class_map={"enzymes": {"enz1"}, "bacteria": {"bac1"}},
    )

    metrics = factory.dataset_metrics(dataset, SCHEMA)

    assert metrics["dataset/test_bacteria_unseen_rate"] == pytest.approx(1.0)
    assert metrics["dataset/test_enzymes_unseen_rate"] == pytest.approx(0.0)


def test_dataset_metrics_reports_all_unseen_for_a_type_absent_from_training():
    """A type can be in the schema but have no training-split member at all
    (an empty `--limit`-truncated vocabulary, or a type rare enough to miss
    the split) — `class_map["bacteria"]` is `set()`, not a missing key, so
    there is no training entity to read a prefix back off. Every split
    entity of that type is then unseen, and the rate must say `1.0` rather
    than be omitted, which would read as "no data" instead of "all unseen".
    """
    dataset = EntityRelationDataset(
        data={"test": _split([[{("bac9", "enz1"): 0}]])},
        class_map={"enzymes": {"enz1"}, "bacteria": set()},
    )

    metrics = factory.dataset_metrics(dataset, SCHEMA)

    assert metrics["dataset/test_bacteria_unseen_rate"] == pytest.approx(1.0)


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


def test_a_checkpoint_carrying_the_retired_padding_fill_still_loads():
    """`Model` used to register the padding fill as a persistent buffer, so
    every checkpoint written then carries `_neg_inf`; strict loading would
    reject it as unexpected now that nothing registers it."""
    trained = _Checkpointed()
    checkpoint = {"_neg_inf": torch.tensor(-1e9), **trained.state_dict()}

    evaluated = _Checkpointed()
    evaluated.register_load_state_dict_pre_hook(factory.fix_keys_hook)
    evaluated.load_state_dict(checkpoint)

    torch.testing.assert_close(evaluated.linear.weight, trained.linear.weight)


class _NestedCheckpointed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.two_head = _Checkpointed()


def test_a_nested_retired_padding_fill_still_loads():
    """Before 671a216 every `Model` registered `_neg_inf`, so a composed
    submodule (e.g. `ETEBrendaModel.two_head`) carried it under its own
    prefix, not just at the root; the hook must drop `two_head._neg_inf`
    too, not only the bare root-level key."""
    trained = _NestedCheckpointed()
    checkpoint = {
        "two_head._neg_inf": torch.tensor(-1e9),
        **trained.state_dict(),
    }

    evaluated = _NestedCheckpointed()
    evaluated.register_load_state_dict_pre_hook(factory.fix_keys_hook)
    evaluated.load_state_dict(checkpoint)

    torch.testing.assert_close(
        evaluated.two_head.linear.weight, trained.two_head.linear.weight
    )


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
