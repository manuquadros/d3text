"""Property-based tests for the relation-loss weighting helpers.

`test_base.py` pins `balanced_class_weights` and `focal_cross_entropy` at a
couple of hand-picked batches. What they actually claim are properties over any
batch — finite when a class is absent, `gamma == 0` reproducing plain
cross-entropy exactly, the weight the exact inverse-frequency ratio for every
class present — so this generates the batches instead. Marked `slow`.
"""

import pytest
import torch
from hypothesis import HealthCheck, example, given, settings
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
@example(data=(torch.tensor([0] * 15 + [1, 1], dtype=torch.int64), 2))
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_present_classes_get_the_exact_inverse_frequency_weight(data):
    """`weight[c] * count[c]` is one constant for every present class.

    A scaling bug passes only on a balanced example. Compared as tensors, so
    float32 is graded at float32 tolerance; the pinned example's product is
    one ULP off, a case the search finds only by luck.
    """
    targets, num_classes = data

    weights = balanced_class_weights(targets, num_classes)
    counts = torch.bincount(targets, minlength=num_classes)
    expected_product = targets.numel() / num_classes

    for class_id in torch.unique(targets).tolist():
        product = weights[class_id] * counts[class_id]
        torch.testing.assert_close(
            product, torch.full_like(product, expected_product)
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


@example(batch=(torch.tensor([[20.0, -20.0]]), torch.tensor([0])), gamma=2.0)
@given(
    batch=_classification_batch(),
    gamma=st.floats(
        min_value=0.0, max_value=8.0, allow_nan=False, allow_infinity=False
    ),
)
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_the_loss_is_always_finite_and_non_negative(batch, gamma):
    """Loss and gradient stay finite, and the loss non-negative, however
    confident the logits.

    A confident pair rounds `1 - p_t` to 0 in float32. A naive
    `log1p(-p_t)` modulation is then -inf: at a gamma nonzero in float32 the
    loss survives (`_weighted_mean` gives that row weight 0), but its
    gradient is NaN. `_log_one_minus_p_t` takes the log in log space and
    keeps it finite.
    """
    preds, targets = batch
    preds = preds.detach().clone().requires_grad_(True)

    loss = focal_cross_entropy(preds, targets, gamma=gamma)
    loss.backward()

    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
    assert preds.grad is not None
    assert torch.isfinite(preds.grad).all()
