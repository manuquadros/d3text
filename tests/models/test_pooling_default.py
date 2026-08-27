"""The shipped pooling default is length-invariant, on both heads.

`logsumexp` is a smooth max — one strong token carries the document — but it is
also `max + log(T)` to within a bounded correction, so on the ~8,000-token
documents this corpus carries it hands every column about nine nats of length
bias. A class absent from most documents cannot be made negative under that
without pushing all its tokens far down, and the cheapest answer to the pooled
objective is a channel that never fires: measured document recall 0.114 for
strains and 0.143 for bacteria, against 0.494 and 0.755 once the bias term is
gone. A dead channel is invisible in the pooled loss the training loop prints,
which is why this needs a test rather than a watchful reader.

Those four numbers are `--limit 500` runs, and the collapse they describe does
not reproduce on the whole training split: there `logsumexp` reaches 0.829 and
0.925 and the two poolings tie to within noise
(`design/tickets/DEC-03.md`, the 2026-08-26 amendment). What this file pins is
the wiring, which the correction does not touch -- the shipped default is
length-invariant and both heads pool with it -- but read the motivation above
as the argument that selected the default under measurement that has since
narrowed, not as a description of what the shipped alternative would do.

The assertion turns on an exact identity rather than a fitted number:
concatenating a document with a verbatim copy of itself doubles every
`exp(logit)` sum, so `logsumexp` gains exactly `log 2` while `logmeanexp`,
which subtracts `log(T)`, gains exactly nothing. That separates the two with no
training, no data and no GPU.
"""

import math

import pytest
import torch

from d3text.models.config import ModelConfig
from d3text.models.entity_linking import BrendaClassificationModel

pytestmark = pytest.mark.slow

TOKENS = 12

# `forward` runs under autocast, so the pooled logits come back in bfloat16 and
# `log 2` is only good to about three decimals there. The two poolings are
# separated by 0.69, so a tolerance an order of magnitude below that still
# tells them apart.
ATOL = 0.05


@pytest.fixture(autouse=True)
def _offline(patch_base_model):
    """Inject the tiny random BERT: the heads are what is under test, and the
    frozen encoder is never run."""


def build(classes, **config_kwargs):
    """A real `BrendaClassificationModel` over `classes`, on a tiny random BERT.

    One entity per class, which is all the geometry needs here.
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

    Run through `forward`, not through the pooling helper, so a head wired to
    the wrong pooling fails here even though the helper is correct.
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


def test_the_default_pooling_does_not_reward_a_longer_document():
    """Duplicating a document token for token must not move either head's
    logits. Under `logsumexp` every column would gain `log 2` instead, and the
    low-prevalence class channels go dead at document length."""
    model = build(["enzymes", "bacteria"])
    assert model.entity_logits_pooling == "logmeanexp"

    entity_gain, class_gain = length_gain(model)

    assert torch.allclose(class_gain, torch.zeros_like(class_gain), atol=ATOL)
    assert torch.allclose(entity_gain, torch.zeros_like(entity_gain), atol=ATOL)


def test_logsumexp_is_still_available_and_still_length_biased():
    """The mode itself is not removed — only the default moved. This is the
    behaviour the default used to have, and the contrast that makes the test
    above mean something."""
    model = build(["enzymes", "bacteria"], entity_logits_pooling="logsumexp")

    entity_gain, class_gain = length_gain(model)

    assert torch.allclose(
        class_gain, torch.full_like(class_gain, math.log(2)), atol=ATOL
    )
    assert torch.allclose(
        entity_gain, torch.full_like(entity_gain, math.log(2)), atol=ATOL
    )
