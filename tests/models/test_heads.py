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
    PermutationBatchNorm1d,
    initialize_classifier_bias,
)


# --------------------------------------------------------------------------- #
# initialize_classifier_bias                                                   #
# --------------------------------------------------------------------------- #
def test_initialize_classifier_bias_sets_logits_and_sentinel_tail():
    linear = torch.nn.Linear(4, 3)
    initialize_classifier_bias(
        linear, torch.tensor([0.5, 0.1])
    )  # sentinel_prior=0.1
    bias = linear.bias.detach()
    assert bias[0].item() == pytest.approx(0.0, abs=1e-5)  # logit(0.5)
    logit_01 = math.log(0.1) - math.log1p(-0.1)
    assert bias[1].item() == pytest.approx(logit_01, abs=1e-4)
    assert bias[2].item() == pytest.approx(logit_01, abs=1e-4)  # sentinel slot


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
def test_classification_head_returns_class_logits_alone():
    """One tensor, not a pair: the entity head is gone, and a caller
    unpacking two would silently take the class logits' first row."""
    head = ClassificationHead(input_size=8, n_classes=3)

    class_logits = head(torch.randn(2, 8))

    assert torch.is_tensor(class_logits)
    assert tuple(class_logits.shape) == (2, 3)


def test_the_classification_head_holds_only_the_class_layer():
    """Its whole parameter set, so an entity layer left behind — even an
    unused one — is caught here rather than as a checkpoint key nobody
    reads."""
    head = ClassificationHead(input_size=8, n_classes=3)

    assert {name for name, _ in head.named_parameters()} == {
        "class_classifier.weight",
        "class_classifier.bias",
    }


def test_classification_head_rejects_bad_class_freqs():
    with pytest.raises(ValueError):
        # class_freqs length must be n_classes - 1 == 2
        ClassificationHead(input_size=8, n_classes=3, class_freqs=torch.rand(3))


def test_the_class_bias_is_seeded_from_the_class_frequencies():
    """The head's own wiring of `initialize_classifier_bias`: the `OOS`
    column takes the 0.9 prior and the rest take their frequencies' log
    odds, so a head seeded through the wrong argument reads differently
    here."""
    head = ClassificationHead(
        input_size=8, n_classes=3, class_freqs=torch.tensor([0.5, 0.1])
    )

    bias = head.class_classifier.bias.detach()

    assert bias[0].item() == pytest.approx(0.0, abs=1e-5)
    assert bias[1].item() == pytest.approx(
        math.log(0.1) - math.log1p(-0.1), abs=1e-4
    )
    assert bias[2].item() == pytest.approx(
        math.log(0.9) - math.log1p(-0.9), abs=1e-4
    )


# --------------------------------------------------------------------------- #
# BiaffineRelationClassifier.forward                                           #
# --------------------------------------------------------------------------- #
def test_biaffine_forward_shape_and_gradient():
    model = BiaffineRelationClassifier(
        hidden_size=8,
        num_relations=3,
        separate_predicate_layer=False,
        biaff_hidden_size=32,
    )
    out = model(torch.randn(4, 8), torch.randn(4, 8))
    assert tuple(out.shape) == (4, 3)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert model.bilinear.grad is not None


def test_biaffine_hidden_size_sets_the_bilinear_width():
    """The internal projection width is injectable, not a hardcoded 32: the
    bilinear parameter is (num_relations, width, width)."""
    model = BiaffineRelationClassifier(
        hidden_size=8,
        num_relations=3,
        separate_predicate_layer=False,
        biaff_hidden_size=16,
    )
    assert tuple(model.bilinear.shape) == (3, 16, 16)


# --------------------------------------------------------------------------- #
# PermutationBatchNorm1d                                                      #
# --------------------------------------------------------------------------- #
def test_permutation_batch_norm_ignores_appended_padding():
    """Appending zero-padding positions to a batch, with the matching mask
    entries False, must not change the real positions' output, nor the
    running statistics: both come from `mask`'s real positions only, not
    every `document * token` position of the padded block.

    A `PermutationBatchNorm1d` that averaged over every position instead
    (the un-masked `nn.BatchNorm1d` route) would shift the real positions'
    mean and shrink their variance as more padding is appended, and its
    `forward` takes no `mask` at all — this fails loudly (`TypeError`) on
    that version rather than silently comparing against a value it never
    computed.
    """
    features = 4
    real = torch.randn(2, 5, features)
    real_mask = torch.ones(2, 5, dtype=torch.bool)

    padded = torch.cat([real, torch.zeros(2, 20, features)], dim=1)
    padded_mask = torch.cat(
        [real_mask, torch.zeros(2, 20, dtype=torch.bool)], dim=1
    )

    norm_unpadded = PermutationBatchNorm1d(features)
    norm_padded = PermutationBatchNorm1d(features)
    norm_unpadded.train()
    norm_padded.train()

    out_unpadded = norm_unpadded(real, real_mask)
    out_padded = norm_padded(padded, padded_mask)

    assert torch.allclose(
        out_padded[padded_mask], out_unpadded[real_mask], atol=1e-6
    )
    assert torch.allclose(
        norm_padded.running_mean, norm_unpadded.running_mean, atol=1e-6
    )
    assert torch.allclose(
        norm_padded.running_var, norm_unpadded.running_var, atol=1e-6
    )


def test_permutation_batch_norm_rejects_an_all_padding_batch():
    norm = PermutationBatchNorm1d(4)
    with pytest.raises(ValueError):
        norm(torch.randn(2, 5, 4), torch.zeros(2, 5, dtype=torch.bool))
