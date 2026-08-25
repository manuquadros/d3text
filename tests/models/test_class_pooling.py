"""The class head pools per column, and the entity head does not.

One setting, `entity_logits_pooling`, used to pool both heads. `logsumexp` is
`max + log(T)` to within a bounded correction, so on the ~8,000-token documents
this corpus carries it hands every class column about nine nats of length bias;
a class that is absent from most documents answers that by predicting nothing
at all, and a dead channel is invisible in the pooled loss the training loop
prints. `class_logits_pooling` gives each class the pooling its prevalence
calls for.

The assertions turn on an exact identity rather than a fitted number:
concatenating a document with a verbatim copy of itself doubles every
`exp(logit)` sum, so `logsumexp` gains exactly `log 2` while `logmeanexp`,
which subtracts `log(T)`, gains exactly nothing. That separates the two
poolings per column with no training, no data, and no GPU.
"""

import math

import pytest
import torch

from d3text.models.config import (
    CLASS_LOGITS_POOLING,
    UNMAPPED_CLASS_POOLING,
    ModelConfig,
    load_model_config,
    save_model_config,
)
from d3text.models.models import BrendaClassificationModel

pytestmark = pytest.mark.slow

TOKENS = 12

# `forward` runs under autocast, so the pooled logits come back in bfloat16 and
# `log 2` is only good to about three decimals there. The two poolings are
# separated by 0.69, so a tolerance an order of magnitude below that still
# tells them apart; the exact identity is checked in float32 against
# `_pool_class_logits` below.
ATOL = 0.05


@pytest.fixture(autouse=True)
def _offline(patch_base_model):
    """Inject the tiny random BERT for every test here: the class head is what
    is under test, and the frozen encoder is never run."""


def build(classes, **config_kwargs):
    """A real `BrendaClassificationModel` over `classes`, on a tiny random BERT.

    One entity per class, which is all the class head's geometry needs: the
    columns under test are the class ones.
    """
    names = list(classes)
    entity_index = {f"e{index}": index for index in range(len(names))}
    model = BrendaClassificationModel(
        classes={name: {f"e{index}"} for index, name in enumerate(names)},
        class_matrix=torch.eye(len(names)),
        entity_index=entity_index,
        config=ModelConfig(
            base_model="prajjwal1/bert-mini",
            hidden_layers=[8],
            **config_kwargs,
        ),
        device="cpu",
    )
    model.eval()
    return model


def length_gain(model):
    """How each pooled logit moves when the document is duplicated token for
    token: `log 2` under a length-biased pooling, `0` under an invariant one.

    Run through `forward`, not through the pooling helper, so that a head wired
    to the wrong pooling fails here even though the helper is correct.
    """
    embeddings = torch.randn(1, TOKENS, 256)
    doubled = embeddings.repeat(1, 2, 1)
    with torch.no_grad():
        entity_once, class_once = model(
            embeddings, torch.ones(1, TOKENS, dtype=torch.bool)
        )
        entity_twice, class_twice = model(
            doubled, torch.ones(1, 2 * TOKENS, dtype=torch.bool)
        )
    return (
        (entity_twice - entity_once).float(),
        (class_twice - class_once).float(),
    )


def test_pool_class_logits_gives_each_column_its_own_pooling():
    """The identity, exactly, away from autocast: duplicating the token
    dimension doubles every `exp(logit)` sum, so a `logsumexp` column gains
    `log 2` and a `logmeanexp` column gains nothing."""
    model = build(["enzymes", "bacteria"])
    logits = torch.randn(1, TOKENS, len(model.classes))
    doubled = logits.repeat(1, 2, 1)

    gain = model._pool_class_logits(doubled) - model._pool_class_logits(logits)
    expected = torch.tensor([[math.log(2), 0.0, math.log(2)]])
    assert torch.allclose(gain, expected, atol=1e-5)


def test_class_columns_are_pooled_by_class_and_entities_are_not():
    """The shipped default: `enzymes` keeps the smooth max that lets one
    mention carry a document, `bacteria` gets the length-invariant pooling that
    kept its channel from going dead — in the same forward pass, and with the
    entity head still pooled globally by `entity_logits_pooling`."""
    model = build(["enzymes", "bacteria"])
    assert model.class_pooling == ("logsumexp", "logmeanexp", "logsumexp")

    entity_gain, class_gain = length_gain(model)
    assert torch.allclose(
        class_gain[:, 0], torch.full((1,), math.log(2)), atol=ATOL
    )
    assert torch.allclose(class_gain[:, 1], torch.zeros(1), atol=ATOL)
    assert torch.allclose(
        entity_gain, torch.full_like(entity_gain, math.log(2)), atol=ATOL
    )


def test_class_pooling_follows_column_order_not_map_order():
    """The map is keyed by name and the head is positional, so the resolution
    has to read `self.classes`. Declaring the same two classes the other way
    round must move the poolings with them."""
    model = build(["bacteria", "enzymes"])
    assert model.class_pooling == ("logmeanexp", "logsumexp", "logsumexp")

    _, class_gain = length_gain(model)
    assert torch.allclose(class_gain[:, 0], torch.zeros(1), atol=ATOL)
    assert torch.allclose(
        class_gain[:, 1], torch.full((1,), math.log(2)), atol=ATOL
    )


def test_an_unmapped_class_falls_back_rather_than_raising(caplog):
    """A schema the map has not heard of must not be a `KeyError` partway
    through an epoch. It gets the historical uniform setting, and says so."""
    with caplog.at_level("WARNING"):
        model = build(["widgets", "bacteria"])

    assert model.class_pooling[0] == UNMAPPED_CLASS_POOLING
    assert "widgets" in caplog.text
    # OOS is a property of the head, not a class the map should name.
    assert "OOS" not in caplog.text


def test_a_scalar_setting_pools_every_class_alike():
    """The pre-existing shape of the knob still works: one name pools the whole
    head, which is what every config written before the map did."""
    model = build(["enzymes", "bacteria"], class_logits_pooling="logmeanexp")
    assert set(model.class_pooling) == {"logmeanexp"}

    entity_gain, class_gain = length_gain(model)
    assert torch.allclose(class_gain, torch.zeros_like(class_gain), atol=ATOL)
    assert torch.allclose(
        entity_gain, torch.full_like(entity_gain, math.log(2)), atol=ATOL
    )


def test_entity_pooling_no_longer_reaches_the_class_head():
    """The two heads are separable: setting `entity_logits_pooling` alone
    leaves the class columns on their own map."""
    model = build(["enzymes", "bacteria"], entity_logits_pooling="logmeanexp")
    entity_gain, class_gain = length_gain(model)
    assert torch.allclose(entity_gain, torch.zeros_like(entity_gain), atol=ATOL)
    assert torch.allclose(
        class_gain[:, 0], torch.full((1,), math.log(2)), atol=ATOL
    )


@pytest.mark.parametrize(
    "pooling", [CLASS_LOGITS_POOLING, "max", {"bacteria": "mean"}]
)
def test_the_setting_round_trips_through_toml(tmp_path, pooling):
    """`save_model_config` dumps every field through tomlkit and `train` reads
    a config back the same way, so a mapping field has to survive both."""
    path = tmp_path / "config.toml"
    config = ModelConfig(class_logits_pooling=pooling)
    save_model_config(config.model_dump(), str(path))
    assert load_model_config(str(path)) == config


def test_a_config_written_before_the_split_still_loads(tmp_path):
    """An old config names only `entity_logits_pooling`. It must validate, and
    the class head must pick up the per-class default rather than inheriting a
    setting that no longer describes it."""
    path = tmp_path / "old.toml"
    path.write_text('entity_logits_pooling = "logsumexp"\n')
    config = load_model_config(str(path))
    assert config.class_logits_pooling == CLASS_LOGITS_POOLING
