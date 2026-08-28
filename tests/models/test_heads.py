"""Pure unit tests for `d3text.models.heads`: the classification head, the
biaffine relation classifier, and the sentinel-aware bias initializer they
both build on.
"""

import math

import pytest
import torch

from d3text.models.heads import (
    BiaffineRelationClassifier,
    ClassificationHead,
    initialize_classifier_bias,
)


# --------------------------------------------------------------------------- #
# initialize_classifier_bias                                                   #
# --------------------------------------------------------------------------- #
def test_initialize_classifier_bias_sets_logits_and_unk_tail():
    linear = torch.nn.Linear(4, 3)
    initialize_classifier_bias(
        linear, torch.tensor([0.5, 0.1])
    )  # unk_prior=0.1
    bias = linear.bias.detach()
    assert bias[0].item() == pytest.approx(0.0, abs=1e-5)  # logit(0.5)
    logit_01 = math.log(0.1) - math.log1p(-0.1)
    assert bias[1].item() == pytest.approx(logit_01, abs=1e-4)
    assert bias[2].item() == pytest.approx(logit_01, abs=1e-4)  # UNK tail slot


def test_initialize_classifier_bias_rejects_wrong_length():
    with pytest.raises(ValueError):
        # 3 freqs but out_features-1 == 2
        initialize_classifier_bias(
            torch.nn.Linear(4, 3), torch.tensor([0.5, 0.1, 0.2])
        )


def test_initialize_classifier_bias_seeds_the_sentinel_by_index():
    """The frequencies fill the supervised columns *around* the sentinel, which
    is seeded from the prior — so moving the sentinel off the tail moves both.
    """
    linear = torch.nn.Linear(4, 3)
    initialize_classifier_bias(
        linear, torch.tensor([0.5, 0.1]), sentinel_index=0
    )
    bias = linear.bias.detach()
    logit_01 = math.log(0.1) - math.log1p(-0.1)
    assert bias[0].item() == pytest.approx(logit_01, abs=1e-4)  # sentinel prior
    assert bias[1].item() == pytest.approx(0.0, abs=1e-5)  # logit(0.5)
    assert bias[2].item() == pytest.approx(logit_01, abs=1e-4)  # logit(0.1)


def test_initialize_classifier_bias_without_sentinel_fills_every_column():
    linear = torch.nn.Linear(4, 2)
    initialize_classifier_bias(
        linear, torch.tensor([0.5, 0.5]), sentinel_index=None
    )
    assert linear.bias.detach().tolist() == pytest.approx([0.0, 0.0], abs=1e-5)


# --------------------------------------------------------------------------- #
# ClassificationHead                                                           #
# --------------------------------------------------------------------------- #
def test_classification_head_returns_entity_and_class_logits():
    head = ClassificationHead(input_size=8, n_entities=5, n_classes=3)
    entity_logits, class_logits = head(torch.randn(2, 8))
    assert tuple(entity_logits.shape) == (2, 5)
    assert tuple(class_logits.shape) == (2, 3)


def test_classification_head_rejects_bad_entity_freqs():
    with pytest.raises(ValueError):
        # entity_freqs length must be n_entities - 1 == 4
        ClassificationHead(
            input_size=8, n_entities=5, n_classes=3, entity_freqs=torch.rand(3)
        )


# --------------------------------------------------------------------------- #
# BiaffineRelationClassifier.forward                                           #
# --------------------------------------------------------------------------- #
def test_biaffine_forward_shape_and_gradient():
    model = BiaffineRelationClassifier(hidden_size=8, num_relations=3)
    out = model(torch.randn(4, 8), torch.randn(4, 8))
    assert tuple(out.shape) == (4, 3)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert model.bilinear.grad is not None


def test_biaffine_hidden_size_sets_the_bilinear_width():
    """The internal projection width is injectable, not a hardcoded 32: the
    bilinear parameter is (num_relations, width, width)."""
    model = BiaffineRelationClassifier(
        hidden_size=8, num_relations=3, biaff_hidden_size=16
    )
    assert tuple(model.bilinear.shape) == (3, 16, 16)
