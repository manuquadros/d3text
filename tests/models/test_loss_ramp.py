"""Training losses follow the loss-weight ramp; selection does not."""

import pytest
import torch
from torch.utils.data import DataLoader

from d3text.models.config import ModelConfig
from d3text.models.entity_linking import BrendaClassificationModel
from d3text.models.ete import ETEBrendaModel
from d3text.models.model_types import BatchLosses
from d3text.schema import EntityType, RelationType, Schema
from d3text.training.trainer import Trainer
from d3text.training.update import BatchUpdate

RAMP_EPOCHS = 4

# `_build` builds both `BrendaClassificationModel` and `ETEBrendaModel`; the
# latter needs relation types on the schema to build at all.
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


def _build(model_class, token_supervision: bool = False, **config):
    return model_class(
        schema=SCHEMA,
        config=ModelConfig(
            model_class=model_class.__name__,
            base_model="prajjwal1/bert-mini",
            hidden_layers=[8],
            ramp_epochs=RAMP_EPOCHS,
            lr=0.1,
            token_supervision=token_supervision,
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
        return BatchLosses(*(anchor.sum() * 0.0 + value for value in values))

    monkeypatch.setattr(model, "compute_batch_losses", constant_losses)


def _loader() -> DataLoader:
    """One batch of one placeholder document.

    Its content is never read, but the batch still passes through the real
    `compute_losses`, whose `Sequence[BatchItem]` a bare collated tensor would
    not satisfy.
    """
    return DataLoader([[{}]], batch_size=1, collate_fn=lambda items: items[0])


@pytest.mark.parametrize(
    "model_class, values",
    [
        (ETEBrendaModel, (1.0, 1.0)),
    ],
)
def test_training_totals_still_follow_the_ramp(
    patch_base_model, monkeypatch, empty_token_label_store, model_class, values
):
    """The same constant losses must total less at the ramp's start than at
    its end."""
    model = _build(model_class, token_supervision=True)
    _pin_batch_losses(monkeypatch, model, values)
    update = BatchUpdate(
        model, torch.optim.SGD(model.parameters(), lr=0.0), "cpu"
    )

    def training_total(epoch: int) -> float:
        losses, denominator = model.run_epoch(
            data=_loader(), epoch=epoch, update=update
        )
        return sum(losses.values()) / denominator

    start, end = training_total(0), training_total(RAMP_EPOCHS)

    assert start < end
    assert end == pytest.approx(sum(values))


def test_two_head_training_totals_ignore_the_ramp(
    patch_base_model, monkeypatch
) -> None:
    """A model with no relation head may not ride the relation schedule.

    It once shared one `(w_ent, w_rel)` helper with the end-to-end model and
    unpacked the ramping slot as its class weight, so its class loss started at
    a tenth of its weight for a ramp nothing here was waiting for.
    """
    model = _build(BrendaClassificationModel)
    _pin_batch_losses(monkeypatch, model, (1.0,))
    update = BatchUpdate(
        model, torch.optim.SGD(model.parameters(), lr=0.0), "cpu"
    )

    def training_total(epoch: int) -> float:
        losses, denominator = model.run_epoch(
            data=_loader(), epoch=epoch, update=update
        )
        return sum(losses.values()) / denominator

    assert training_total(0) == pytest.approx(1.0)
    assert training_total(0) == pytest.approx(training_total(RAMP_EPOCHS))


def test_best_epoch_follows_the_selection_metric_through_the_ramp(
    patch_base_model, monkeypatch, empty_token_label_store
):
    """The trainer-level consequence, against a real model class: selection
    reads `evaluate_model`'s scores, not `run_epoch`'s losses, so a ramp that
    deflates the early-epoch total cannot pin the best epoch to it. Here the
    scripted score peaks mid-run regardless of the pinned constant losses,
    and that is what `fit` must restore.
    """
    model = _build(
        ETEBrendaModel,
        token_supervision=True,
        num_epochs=6,
        patience=1,
    )
    _pin_batch_losses(monkeypatch, model, (1.0, 1.0))
    scores = [0.1, 0.2, 0.9, 0.3, 0.2, 0.1]

    def scripted_evaluate_model(
        data, tau_cls=0.5, prefix="test", log_reports=True, step=None
    ):
        assert step is not None
        value = scores[step]
        return {
            f"{prefix}/{name}": value
            for name in ETEBrendaModel.default_selection_metrics
        }

    monkeypatch.setattr(model, "evaluate_model", scripted_evaluate_model)
    trainer = Trainer(model)

    trainer.fit(train_data=_loader(), val_data=_loader(), save_checkpoint=False)

    assert trainer.best_epoch == 2
    assert trainer.best_selection_score == pytest.approx(0.9)
