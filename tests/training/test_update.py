"""Unit tests for the per-step weight update and its gradient telemetry.

Relocated from `tests/models/test_models.py`: the accumulators these drive
belong to `BatchUpdate`, not to the model it steps.
"""

import pytest
import torch
from d3text.training.update import GRAD_CLIP_NORM, BatchUpdate


@pytest.fixture
def update():
    model = torch.nn.Linear(4, 3)
    return BatchUpdate(
        model, torch.optim.SGD(model.parameters(), lr=0.1), "cpu"
    )


def test_grad_norm_metrics_are_absent_without_an_optimizer_step(update):
    """A validation pass never applies the update, so it must not report a
    gradient statistic for an epoch that computed no gradients."""
    assert update.grad_norm_metrics() == {}


def test_grad_norm_metrics_average_the_preclip_norms(update):
    for norm in (2.0, 0.5):
        update._record_grad_norm(torch.tensor(norm))

    metrics = update.grad_norm_metrics()
    assert metrics["training/grad_norm"] == pytest.approx(1.25)
    # Exactly one of the two exceeded the clip threshold.
    assert metrics["training/grad_clip_rate"] == pytest.approx(0.5)


def test_grad_clip_rate_saturates_when_every_step_clips(update):
    """A rate pinned at 1.0 is the signal that the clip, not the learning
    rate, is setting the step size."""
    for _ in range(3):
        update._record_grad_norm(torch.tensor(GRAD_CLIP_NORM * 10))

    assert update.grad_norm_metrics()["training/grad_clip_rate"] == 1.0


def test_resetting_grad_norms_drops_the_previous_epoch(update):
    update._record_grad_norm(torch.tensor(4.0))
    update.reset_grad_norms()

    assert update.grad_norm_metrics() == {}


def test_the_update_clips_over_the_model_it_was_given(update):
    """The clip covers every parameter of the model, so the norm it records is
    the whole model's — the frozen base model included."""
    loss = update.model(torch.ones(1, 4)).sum() * 1000

    update.zero_grad()
    update(loss)

    assert update.grad_norm_metrics()["training/grad_clip_rate"] == 1.0
    assert all(
        param.grad is not None and param.grad.norm() <= GRAD_CLIP_NORM + 1e-5
        for param in update.model.parameters()
    )


@pytest.mark.parametrize(
    "amp_dtype, enabled",
    [(torch.float16, True), (torch.bfloat16, False), (torch.float32, False)],
)
def test_loss_scaling_is_reserved_for_float16(amp_dtype, enabled):
    """Only fp16 gradients can fall into the subnormal range, so only fp16
    pays for the scaler — whose `step` syncs the host against the device."""
    model = torch.nn.Linear(4, 3)
    update = BatchUpdate(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        "cpu",
        amp_dtype=amp_dtype,
    )

    assert update.scaler.is_enabled() is enabled


def test_an_unscaled_update_still_steps_the_parameters():
    """A disabled scaler passes `scale`/`unscale_`/`step`/`update` through, so
    the bf16 path applies the same weight update the fp16 path does."""
    model = torch.nn.Linear(4, 3)
    update = BatchUpdate(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        "cpu",
        amp_dtype=torch.bfloat16,
    )
    before = model.weight.detach().clone()

    update.zero_grad()
    update(model(torch.ones(1, 4)).sum())

    assert not torch.equal(before, model.weight.detach())
    assert update.grad_norm_metrics()["training/grad_norm"] > 0
