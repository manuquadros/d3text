"""Unit tests for the epoch loop and the per-model loss seam it drives.

`Model.run_epoch` walks the batches, steps the optimizer on TRAINING and
accumulates whatever losses the subclass names; each subclass says only what one
batch costs, via `compute_losses`. The two halves are tested apart: the loop
against a stub whose losses are known constants, and each subclass's
`compute_losses` against a stubbed `compute_batch_losses`, so the names and the
ramp weighting are pinned without running a transformer.

The batches are empty lists — a valid ``Sequence[BatchItem]``, which beartype
checks at runtime — because no test here reaches a real forward pass.
"""

import pytest
import torch
from torch.utils.data import DataLoader

from d3text.models.base import Model, Step
from d3text.models.entity_linking import BrendaClassificationModel
from d3text.models.ete import ETEBrendaModel
from d3text.models.ner import NERClassificationModel


class _RecordingOptimizer:
    def __init__(self):
        self.zero_grad_calls = 0

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.zero_grad_calls += 1


def empty_batches(n: int) -> DataLoader:
    """A DataLoader yielding `n` empty batches, with automatic batching off so
    each item arrives as the loop sees it."""
    return DataLoader([[] for _ in range(n)], batch_size=None)


def _looping_model(stub, batch_losses):
    """A bare `Model` whose `compute_losses` replays `batch_losses` — one dict
    of scalar losses per batch — and whose optimizer step is recorded, not run.
    """
    updates: list[tuple[float, ...]] = []
    replay = iter(batch_losses)

    model = stub(
        Model,
        optimizer=_RecordingOptimizer(),
        _update=lambda *losses: updates.append(
            tuple(loss.item() for loss in losses)
        ),
        compute_losses=lambda batch, epoch: {
            name: torch.tensor(value, requires_grad=True)
            for name, value in next(replay).items()
        },
    )
    return model, empty_batches(len(batch_losses)), updates


# --------------------------------------------------------------------------- #
# Model.run_epoch                                                              #
# --------------------------------------------------------------------------- #
def test_run_epoch_sums_each_named_loss_over_the_batches(stub):
    model, data, _ = _looping_model(
        stub,
        [
            {"entity": 1.0, "relation": 0.5},
            {"entity": 2.0, "relation": 0.25},
        ],
    )

    losses, denominator = model.run_epoch(
        data=data, step=Step.TRAINING, epoch=0
    )

    assert losses == {"entity": 3.0, "relation": 0.75}
    assert denominator == 2


def test_run_epoch_keeps_the_names_the_subclass_gave(stub):
    """The keys are the subclass's, not a fixed set: `print_epoch_stats` and the
    validation loss both report whatever the loop accumulated."""
    model, data, _ = _looping_model(stub, [{"class": 4.0}])

    losses, _ = model.run_epoch(data=data, step=Step.VALIDATION, epoch=0)

    assert list(losses) == ["class"]


def test_run_epoch_returns_plain_floats_not_graph_bound_tensors(stub):
    model, data, _ = _looping_model(stub, [{"class": 1.0}])

    losses, _ = model.run_epoch(data=data, step=Step.TRAINING, epoch=0)

    assert all(isinstance(loss, float) for loss in losses.values())


def test_training_zeroes_and_steps_once_per_batch(stub):
    model, data, updates = _looping_model(
        stub,
        [{"entity": 1.0, "relation": 0.5}, {"entity": 2.0, "relation": 3.0}],
    )

    model.run_epoch(data=data, step=Step.TRAINING, epoch=0)

    assert model.optimizer.zero_grad_calls == 2
    # Every named loss reaches `_update`, which sums them into the step.
    assert updates == [(1.0, 0.5), (2.0, 3.0)]


@pytest.mark.parametrize("step", [Step.VALIDATION, Step.TESTING])
def test_only_training_touches_the_optimizer(stub, step):
    model, data, updates = _looping_model(
        stub, [{"class": 1.0}, {"class": 2.0}]
    )

    losses, denominator = model.run_epoch(data=data, step=step, epoch=0)

    assert model.optimizer.zero_grad_calls == 0
    assert updates == []
    assert (losses, denominator) == ({"class": 3.0}, 2)


def test_epoch_start_hook_runs_once_before_the_batches(stub):
    seen: list[tuple[Step, int]] = []
    model, data, _ = _looping_model(stub, [{"class": 1.0}, {"class": 2.0}])
    object.__setattr__(
        model,
        "on_epoch_start",
        lambda step, epoch: seen.append((step, epoch)),
    )

    model.run_epoch(data=data, step=Step.TRAINING, epoch=3)

    assert seen == [(Step.TRAINING, 3)]


# --------------------------------------------------------------------------- #
# compute_losses: the names and the ramp weighting, per model                  #
# --------------------------------------------------------------------------- #
def _weighted(stub, cls, ramp_epochs, losses):
    """`cls` stubbed down to what `compute_losses` reads: the ramp schedule and
    a `compute_batch_losses` returning `losses` as scalar tensors."""
    return stub(
        cls,
        ramp_epochs=ramp_epochs,
        compute_batch_losses=lambda batch: (
            tuple(torch.tensor(loss) for loss in losses)
            if len(losses) > 1
            else torch.tensor(losses[0])
        ),
    )


def _as_floats(losses):
    return {name: loss.item() for name, loss in losses.items()}


def test_ner_reports_only_the_class_loss_unweighted(stub):
    model = _weighted(stub, NERClassificationModel, ramp_epochs=4, losses=[2.0])

    # No ramp applies: with no second head, the class loss is the whole
    # objective at every epoch.
    assert _as_floats(model.compute_losses(batch=[], epoch=0)) == {"class": 2.0}


def test_entity_linking_reports_entity_and_class(stub):
    model = _weighted(
        stub, BrendaClassificationModel, ramp_epochs=0, losses=[1.0, 2.0]
    )

    assert _as_floats(model.compute_losses(batch=[], epoch=0)) == {
        "entity": 1.0,
        "class": 2.0,
    }


def test_entity_linking_does_not_ramp_either_of_its_losses(stub):
    """Neither head here has anything to be held back for: with `ramp_epochs`
    set, both losses still train at full weight from the first epoch."""
    model = _weighted(
        stub, BrendaClassificationModel, ramp_epochs=4, losses=[1.0, 2.0]
    )

    over_the_ramp = [
        _as_floats(model.compute_losses(batch=[], epoch=epoch))
        for epoch in range(5)
    ]

    assert over_the_ramp == [{"entity": 1.0, "class": 2.0}] * 5


def test_ete_ramps_the_relation_loss_against_the_entity_losses(stub):
    model = _weighted(
        stub, ETEBrendaModel, ramp_epochs=4, losses=[1.0, 1.0, 1.0]
    )

    first = _as_floats(model.compute_losses(batch=[], epoch=0))
    last = _as_floats(model.compute_losses(batch=[], epoch=4))

    assert first == pytest.approx(
        {"entity": 1.0, "class": 1.0, "relation": 0.1}
    )
    assert last == pytest.approx({"entity": 1.0, "class": 1.0, "relation": 1.0})


def test_ete_without_a_ramp_leaves_every_loss_at_full_weight(stub):
    model = _weighted(
        stub, ETEBrendaModel, ramp_epochs=0, losses=[1.0, 2.0, 3.0]
    )

    assert _as_floats(model.compute_losses(batch=[], epoch=0)) == {
        "entity": 1.0,
        "class": 2.0,
        "relation": 3.0,
    }


def test_ete_announces_the_epochs_relation_weight(stub, capsys):
    model = _weighted(stub, ETEBrendaModel, ramp_epochs=4, losses=[1.0])

    model.on_epoch_start(step=Step.TRAINING, epoch=2)

    assert "w_rel=0.550" in capsys.readouterr().out


def test_run_epoch_drives_a_real_subclass_seam(stub):
    """The loop and the seam fit together: a model that defines only
    `compute_losses` trains without knowing anything about the loop."""
    stepped: list[int] = []
    model = stub(
        ETEBrendaModel,
        ramp_epochs=0,
        optimizer=_RecordingOptimizer(),
        _update=lambda *losses: stepped.append(len(losses)),
        compute_batch_losses=lambda batch: (
            torch.tensor(1.0),
            torch.tensor(2.0),
            torch.tensor(3.0),
        ),
    )

    losses, denominator = model.run_epoch(
        data=empty_batches(1), step=Step.TRAINING, epoch=0
    )

    assert losses == {"entity": 1.0, "class": 2.0, "relation": 3.0}
    assert denominator == 1
    assert stepped == [3]
