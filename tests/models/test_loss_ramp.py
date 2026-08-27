"""Validation losses must not move with the loss-weight ramp.

With ``ramp_epochs > 0``, ``ETEBrendaModel.relation_loss_weight`` ramps the
relation loss from 0.1 to 1.0 over the first epochs -- and no other objective
in any model rides it. The ramp shapes the *training* gradient; the validation
totals feed ``Trainer._early_stop``, which compares them across epochs as one
series. Scored under the per-epoch ramp weights, an early epoch's total omits
most of the ramped objective and reads as spuriously low, so the best-model
snapshot pins to epoch 0 and every later epoch counts as no improvement.
Validation is therefore scored under the ramp's final (t = 1) weights, and
these tests hold ``run_epoch`` — the real one, not a script — to that.
"""

import pytest
import torch
from torch.utils.data import DataLoader

from d3text.models.base import Step
from d3text.models.config import ModelConfig
from d3text.models.entity_linking import BrendaClassificationModel
from d3text.models.ete import ETEBrendaModel
from d3text.training.trainer import Trainer
from d3text.training.update import BatchUpdate

RAMP_EPOCHS = 4


def _build(model_class, **config):
    return model_class(
        classes={"enzymes": {"enz1", "enz2"}, "bacteria": {"bac1"}},
        class_matrix=torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        entity_index={"enz1": 0, "enz2": 1, "bac1": 2},
        config=ModelConfig(
            base_model="prajjwal1/bert-mini",
            hidden_layers=[8],
            ramp_epochs=RAMP_EPOCHS,
            lr=0.1,
            **config,
        ),
        device="cpu",
    )


def _pin_batch_losses(monkeypatch, model, values: tuple[float, ...]) -> None:
    """Make every batch report the same per-objective losses.

    Each loss is anchored to a trainable parameter times zero, so the training
    step's ``backward`` has a graph to walk while the value never moves.
    """
    anchor = next(p for p in model.parameters() if p.requires_grad)

    def constant_losses(batch):
        return tuple(anchor.sum() * 0.0 + value for value in values)

    monkeypatch.setattr(model, "compute_batch_losses", constant_losses)


def _loader() -> DataLoader:
    return DataLoader([0], batch_size=1)


@pytest.mark.parametrize(
    "model_class, values",
    [
        (BrendaClassificationModel, (1.0, 1.0)),
        (ETEBrendaModel, (1.0, 1.0, 1.0)),
    ],
)
def test_validation_totals_do_not_move_with_the_ramp(
    patch_base_model, monkeypatch, model_class, values
):
    """Constant batch losses must yield constant validation totals, equal to
    the fixed-weight sum, at every point of the ramp."""
    model = _build(model_class)
    _pin_batch_losses(monkeypatch, model, values)
    update = BatchUpdate(
        model, torch.optim.SGD(model.parameters(), lr=0.0), "cpu"
    )
    model.eval()

    totals = []
    for epoch in range(RAMP_EPOCHS + 1):
        losses, denominator = model.run_epoch(
            data=_loader(), step=Step.VALIDATION, epoch=epoch, update=update
        )
        totals.append(sum(losses.values()) / denominator)

    assert totals == pytest.approx([sum(values)] * (RAMP_EPOCHS + 1))


@pytest.mark.parametrize(
    "model_class, values",
    [
        (ETEBrendaModel, (1.0, 1.0, 1.0)),
    ],
)
def test_training_totals_still_follow_the_ramp(
    patch_base_model, monkeypatch, model_class, values
):
    """The guard against fixing validation by unramping training: the same
    constant losses must total less at the ramp's start than at its end."""
    model = _build(model_class)
    _pin_batch_losses(monkeypatch, model, values)
    update = BatchUpdate(
        model, torch.optim.SGD(model.parameters(), lr=0.0), "cpu"
    )

    def training_total(epoch: int) -> float:
        losses, denominator = model.run_epoch(
            data=_loader(), step=Step.TRAINING, epoch=epoch, update=update
        )
        return sum(losses.values()) / denominator

    start, end = training_total(0), training_total(RAMP_EPOCHS)

    assert start < end
    assert end == pytest.approx(sum(values))


def test_entity_linking_training_totals_ignore_the_ramp(
    patch_base_model, monkeypatch
) -> None:
    """The reported defect: this model has no relation head, so neither of its
    losses may ride the relation schedule.

    It once shared one `(w_ent, w_rel)` helper with the end-to-end model and
    unpacked the ramping slot as its class weight, so with `ramp_epochs > 0`
    its class loss started at a tenth of its weight and reached full weight
    only at the end of a ramp nothing in this model was waiting for.
    """
    model = _build(BrendaClassificationModel)
    _pin_batch_losses(monkeypatch, model, (1.0, 1.0))
    update = BatchUpdate(
        model, torch.optim.SGD(model.parameters(), lr=0.0), "cpu"
    )

    def training_total(epoch: int) -> float:
        losses, denominator = model.run_epoch(
            data=_loader(), step=Step.TRAINING, epoch=epoch, update=update
        )
        return sum(losses.values()) / denominator

    assert training_total(0) == pytest.approx(2.0)
    assert training_total(0) == pytest.approx(training_total(RAMP_EPOCHS))


def test_best_epoch_is_not_pinned_to_the_ramp_floor(
    patch_base_model, monkeypatch
):
    """The trainer-level consequence: with constant per-objective validation
    losses, no epoch is better than any other, so the best must not sit at
    epoch 0 merely because the ramp deflated its total."""
    model = _build(ETEBrendaModel, num_epochs=6, patience=1)
    _pin_batch_losses(monkeypatch, model, (1.0, 1.0, 1.0))
    trainer = Trainer(model)

    trainer.fit(train_data=_loader(), val_data=_loader(), save_checkpoint=False)

    assert trainer.best_val_loss == pytest.approx(3.0)
    assert trainer.best_epoch > 0
