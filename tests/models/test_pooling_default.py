"""The shipped pooling default is length-invariant, on the class head.

`logsumexp` grows with `log(T)`, so on long documents never firing is the
cheapest answer. Duplicating a document adds exactly `log 2` under
`logsumexp` and nothing under `logmeanexp`, an exact identity to assert on.
"""

import math

import pytest
import torch

from d3text.models.config import ModelConfig
from d3text.models.entity_linking import BrendaClassificationModel
from d3text.schema import EntityType, Schema

pytestmark = pytest.mark.slow

TOKENS = 12

# bf16 logits hold `log 2` to about three decimals; a tolerance well below
# `log 2` still separates the two poolings.
ATOL = 0.05


@pytest.fixture(autouse=True)
def _offline(patch_base_model):
    """Inject the tiny random BERT: the heads are what is under test, and the
    frozen encoder is never run."""


def build(classes, **config_kwargs):
    """A real `BrendaClassificationModel` over `classes`, on a tiny random BERT.

    One column per class, which is all the geometry needs here.
    """
    names = list(classes)
    schema = Schema(
        entity_types=tuple(
            EntityType(name=name, prefix=f"e{index}")
            for index, name in enumerate(names)
        )
    )
    model = BrendaClassificationModel(
        schema=schema,
        config=ModelConfig(
            model_class="BrendaClassificationModel",
            base_model="prajjwal1/bert-mini",
            hidden_layers=[8],
            **config_kwargs,
        ),
        device="cpu",
    )
    model.eval()
    return model


def length_gain(model):
    """How each pooled logit moves when the document is duplicated.

    `log 2` under a length-biased pooling, `0` under an invariant one. Run
    through `forward` rather than the pooling helper, so a head wired to the
    wrong pooling fails here even though the helper is correct.
    """
    embeddings = torch.randn(1, TOKENS, 256)
    doubled = embeddings.repeat(1, 2, 1)
    with torch.no_grad():
        (class_once, _) = model(
            embeddings, torch.ones(1, TOKENS, dtype=torch.bool)
        )
        (class_twice, _) = model(
            doubled, torch.ones(1, 2 * TOKENS, dtype=torch.bool)
        )
    return (class_twice - class_once).float()


def test_the_default_pooling_does_not_reward_a_longer_document():
    """Duplicating a document token for token must not move the class head's
    logits. Under `logsumexp` every column would gain `log 2` instead, and the
    low-prevalence class channels go dead at document length."""
    model = build(["enzymes", "bacteria"])
    assert model.entity_logits_pooling == "logmeanexp"

    class_gain = length_gain(model)

    assert torch.allclose(class_gain, torch.zeros_like(class_gain), atol=ATOL)


def test_logsumexp_is_still_available_and_still_length_biased():
    """The mode itself is not removed — only the default moved. This is the
    behaviour the default used to have, and the contrast that makes the test
    above mean something."""
    model = build(["enzymes", "bacteria"], entity_logits_pooling="logsumexp")

    class_gain = length_gain(model)

    assert torch.allclose(
        class_gain, torch.full_like(class_gain, math.log(2)), atol=ATOL
    )
