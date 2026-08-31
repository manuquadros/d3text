"""Property-based tests for the relation-loss weighting helpers in
`d3text.models.base`: `balanced_class_weights` and `focal_cross_entropy`.

`tests/models/test_base.py` pins each at a couple of hand-picked batches (a
`[2, 2, 2, 0]` target vector, an "easy" vs. "hard" logit pair). What the two
functions' docstrings actually claim are properties over any batch of
candidate pairs — the weighting stays finite when a class is absent, `gamma ==
0` reproduces plain cross-entropy exactly, and the class weight is the exact
inverse-frequency ratio for every class that *is* present — so this file
generates the batches instead.

Marked `slow`: `@given` draws many examples per test.
"""

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from d3text.models.base import balanced_class_weights, focal_cross_entropy

pytestmark = pytest.mark.slow

_NUM_CLASSES = st.integers(min_value=2, max_value=6)


@st.composite
def _targets(draw: st.DrawFn) -> tuple[torch.Tensor, int]:
    num_classes = draw(_NUM_CLASSES)
    values = draw(
        st.lists(
            st.integers(min_value=0, max_value=num_classes - 1),
            min_size=0,
            max_size=20,
        )
    )
    return torch.tensor(values, dtype=torch.int64), num_classes


@st.composite
def _classification_batch(draw: st.DrawFn) -> tuple[torch.Tensor, torch.Tensor]:
    num_classes = draw(_NUM_CLASSES)
    rows = draw(st.integers(min_value=1, max_value=10))
    logits = draw(
        st.lists(
            st.floats(
                min_value=-20.0,
                max_value=20.0,
                allow_nan=False,
                allow_infinity=False,
                width=32,
            ),
            min_size=rows * num_classes,
            max_size=rows * num_classes,
        )
    )
    preds = torch.tensor(logits, dtype=torch.float32).reshape(rows, num_classes)
    targets = torch.tensor(
        draw(
            st.lists(
                st.integers(min_value=0, max_value=num_classes - 1),
                min_size=rows,
                max_size=rows,
            )
        ),
        dtype=torch.int64,
    )
    return preds, targets


# --------------------------------------------------------------------------- #
# balanced_class_weights                                                      #
# --------------------------------------------------------------------------- #
@given(data=_targets())
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_balanced_class_weights_are_always_finite(data):
    """The property `test_balanced_class_weights_stay_finite_when_a_class_is_
    absent` pins at one distribution: no class, however rare or entirely
    missing, may divide the weight tensor by zero."""
    targets, num_classes = data

    weights = balanced_class_weights(targets, num_classes)

    assert torch.isfinite(weights).all()
    assert weights.shape == (num_classes,)


@given(data=_targets().filter(lambda pair: pair[0].numel() > 0))
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_present_classes_get_the_exact_inverse_frequency_weight(data):
    """`weight[c] = numel / (num_classes * count[c])`, so
    `weight[c] * count[c]` is the same constant for every class that actually
    occurs in the batch -- a scaling bug (`numel` and `num_classes` swapped,
    say) would break this for anything but the one balanced case a hand-picked
    example happens to hit."""
    targets, num_classes = data

    weights = balanced_class_weights(targets, num_classes)
    counts = torch.bincount(targets, minlength=num_classes)
    expected_product = targets.numel() / num_classes

    for class_id in torch.unique(targets).tolist():
        torch.testing.assert_close(
            (weights[class_id] * counts[class_id]).item(),
            expected_product,
        )


# --------------------------------------------------------------------------- #
# focal_cross_entropy                                                         #
# --------------------------------------------------------------------------- #
@given(batch=_classification_batch())
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_zero_gamma_always_reproduces_plain_cross_entropy(batch):
    preds, targets = batch

    torch.testing.assert_close(
        focal_cross_entropy(preds, targets, gamma=0.0),
        torch.nn.functional.cross_entropy(preds, targets),
    )


@given(
    batch=_classification_batch(),
    gamma=st.floats(
        min_value=0.0, max_value=8.0, allow_nan=False, allow_infinity=False
    ),
)
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_the_loss_is_always_finite_and_non_negative(batch, gamma):
    """The degenerate batch the docstring calls out by name: every pair
    already scored confidently, so the modulation mass vanishes along with
    the numerator instead of the ratio exploding to nan/inf."""
    preds, targets = batch

    loss = focal_cross_entropy(preds, targets, gamma=gamma)

    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
