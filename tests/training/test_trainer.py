"""Unit tests for `Trainer`: the epochs, early stopping, and the checkpoint.

The model is a stub whose `run_epoch` replays a scripted validation loss per
epoch, so the run's *decisions* — when to stop, which epoch to keep, what to
write — are pinned without a transformer, a dataset, or a backward pass. The
loop it drives is tested against a real `Model` in
`tests/models/test_training_loop.py`; the one test here that steps a real
optimizer proves the two halves fit together.
"""

import math
import types

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from d3text.data.data import get_batch_loader
from d3text.models.base import Model, Step
from d3text.models.config import ModelConfig
from d3text.models.ner import NERClassificationModel
from d3text.training import Trainer


def config(**overrides):
    """The four fields `Trainer` reads off a `ModelConfig`."""
    return types.SimpleNamespace(
        **{
            "num_epochs": 10,
            "patience": 1,
            "optimizer": "adamw",
            "lr": 0.1,
            "lr_scheduler": "none",
            **overrides,
        }
    )


def scripted_model(stub, val_losses, ramp_epochs=0, **cfg):
    """A model whose validation epochs return `val_losses` in order and whose
    training epochs cost 0, recording every `run_epoch` call it is given.

    A single real parameter, so the optimizer has something to hold and each
    epoch can leave a distinct mark in the state dict: the weight is set to the
    epoch number, which is what makes "the checkpoint holds the *best* epoch"
    an assertion about the tensor and not just about a counter.
    """
    weight = nn.Parameter(torch.zeros(1))
    calls: list[tuple[Step, int]] = []
    replay = iter(val_losses)

    def run_epoch(data, step, epoch, optimization=None):
        calls.append((step, epoch))
        if step is not Step.TRAINING:
            return {"loss": next(replay)}, 1

        # Step it as the real loop would — the scheduler warns if it is stepped
        # before the optimizer ever is — and then stamp the epoch on the weight.
        optimization.zero_grad()
        optimization.update((weight * 0.0).squeeze())
        with torch.no_grad():
            weight.fill_(float(epoch))
        return {"loss": 0.0}, 1

    return stub(
        Model,
        config=config(**cfg),
        device="cpu",
        ramp_epochs=ramp_epochs,
        run_epoch=run_epoch,
        parameters=lambda: iter([weight]),
        state_dict=lambda: {"weight": weight.detach().clone()},
        load_state_dict=lambda state, strict=True: weight.data.copy_(
            state["weight"]
        ),
        train=lambda: None,
        eval=lambda: None,
        calls=calls,
        weight=weight,
    )


def batches(n: int = 1) -> DataLoader:
    return DataLoader([[] for _ in range(n)], batch_size=None)


def test_fit_runs_every_epoch_when_the_loss_keeps_improving(stub):
    model = scripted_model(stub, val_losses=[5.0, 4.0, 3.0], num_epochs=3)

    best = Trainer(model).fit(batches(), val_data=batches())

    assert best == 3.0
    assert model.calls == [
        (Step.TRAINING, 0),
        (Step.VALIDATION, 0),
        (Step.TRAINING, 1),
        (Step.VALIDATION, 1),
        (Step.TRAINING, 2),
        (Step.VALIDATION, 2),
    ]


def test_fit_stops_once_patience_is_exhausted(stub):
    model = scripted_model(stub, val_losses=[1.0, 2.0, 3.0], patience=1)

    trainer = Trainer(model)
    best = trainer.fit(batches(), val_data=batches())

    # Epoch 0 improves; 1 and 2 do not, and patience(1) tolerates only the
    # first of them.
    assert [step for step, _ in model.calls].count(Step.TRAINING) == 3
    assert (best, trainer.best_epoch) == (1.0, 0)


def test_the_ramp_epochs_do_not_count_against_patience(stub):
    """A model still holding an objective back at a fraction of its weight is
    not the model early stopping is there to judge, so a loss that rises through
    the ramp must not end the run."""
    model = scripted_model(
        stub,
        val_losses=[1.0, 2.0, 3.0, 4.0, 5.0],
        ramp_epochs=3,
        patience=1,
        num_epochs=5,
    )

    Trainer(model).fit(batches(), val_data=batches())

    # Epochs 1-3 all worsen, but sit inside the ramp: the counter is reset each
    # time, so the run only ends when epoch 4 worsens outside it.
    assert [step for step, _ in model.calls].count(Step.TRAINING) == 5


def test_without_validation_data_nothing_early_stops(stub):
    model = scripted_model(stub, val_losses=[], num_epochs=4)

    best = Trainer(model).fit(batches())

    assert best is None
    assert model.calls == [(Step.TRAINING, epoch) for epoch in range(4)]


def test_fit_leaves_the_model_holding_the_best_epochs_weights(stub):
    """Even without early stopping. A run that goes the distance would otherwise
    be evaluated on its last epoch and checkpointed on its best — two different
    models under one name."""
    model = scripted_model(stub, val_losses=[1.0, 9.0, 9.0], num_epochs=3)

    trainer = Trainer(model)
    trainer.fit(batches(), val_data=batches())

    assert trainer.best_epoch == 0
    assert model.weight.item() == 0.0


def test_save_writes_the_best_epochs_weights_not_the_last(stub, tmp_path):
    model = scripted_model(stub, val_losses=[9.0, 1.0, 9.0], num_epochs=3)
    checkpoint = tmp_path / "model.pt"

    trainer = Trainer(model)
    trainer.fit(batches(), val_data=batches())
    trainer.save(str(checkpoint))

    assert torch.load(checkpoint)["weight"].item() == 1.0


def test_save_without_a_checkpoint_writes_the_weights_the_run_ended_on(
    stub, tmp_path
):
    """What tuning leaves behind: it keeps the loss and throws the weights
    away, so there is no best epoch to write."""
    model = scripted_model(stub, val_losses=[9.0, 1.0], num_epochs=2)
    checkpoint = tmp_path / "model.pt"

    trainer = Trainer(model, save_checkpoint=False)
    trainer.fit(batches(), val_data=batches())
    trainer.save(str(checkpoint))

    assert trainer.best_val_loss == 1.0
    assert trainer.best_model_state is None
    assert torch.load(checkpoint)["weight"].item() == 1.0


def test_a_search_without_a_checkpoint_still_reports_the_best_loss(stub):
    model = scripted_model(stub, val_losses=[3.0, 1.0, 2.0], num_epochs=3)

    trainer = Trainer(model, save_checkpoint=False)

    assert trainer.fit(batches(), val_data=batches()) == 1.0


def test_the_plateau_scheduler_is_stepped_on_the_validation_loss(stub):
    """`ReduceLROnPlateau.step` takes the monitored metric, not an epoch — and
    it is no `LRScheduler`, so nothing but a type check tells the two apart."""
    model = scripted_model(
        stub, val_losses=[1.0], num_epochs=1, lr_scheduler="reduce_on_plateau"
    )

    trainer = Trainer(model)
    stepped: list[float] = []
    trainer.scheduler.step = lambda metric: stepped.append(metric)
    trainer.fit(batches(), val_data=batches())

    assert stepped == [1.0]


def test_the_epoch_scheduler_is_stepped_without_a_metric(stub):
    model = scripted_model(
        stub, val_losses=[1.0, 2.0], num_epochs=2, lr_scheduler="exponential"
    )

    trainer = Trainer(model)
    lrs = []
    trainer.fit(batches(), val_data=batches())
    lrs.append(trainer.optimizer.param_groups[0]["lr"])

    assert lrs[0] == pytest.approx(0.1 * 0.95**2)


def test_update_steps_the_optimizer_on_the_sum_of_the_losses(stub):
    """The `Trainer` is the `Optimization` the loop steps: the losses the model
    named reach a real backward pass and move a real parameter."""
    weight = nn.Parameter(torch.zeros(1))
    model = stub(
        Model, config=config(lr=1.0), device="cpu", parameters=lambda: [weight]
    )

    trainer = Trainer(model)
    trainer.zero_grad()
    trainer.update((weight * 2).squeeze(), (weight * 3).squeeze())

    # d(2w + 3w)/dw = 5, clipped to a unit norm, at lr 1.0.
    assert weight.item() == pytest.approx(-1.0)


@pytest.fixture
def tiny_ner(patch_base_model, tiny_schema):
    """A real `NERClassificationModel` over a tiny random BERT: the smallest
    model that can be trained for real, offline and on CPU."""
    return NERClassificationModel(
        schema=tiny_schema,
        config=ModelConfig(
            base_model="prajjwal1/bert-mini",
            hidden_layers=[8],
            num_epochs=2,
            patience=1,
        ),
        device="cpu",
    )


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason=(
        "The DataLoader's collate gives every field a leading singleton dim, so "
        "each document's `sequence` reaches `batch_input_tensors` as "
        "[1, n_chunks, token]. Concatenating those blocks returns 3-D where the "
        "annotation says 2-D, and cannot concatenate at all when two documents "
        "in a batch have different chunk counts (here: 2, 5 and 1). No batch the "
        "DataLoader yields survives, at any batch size."
    ),
)
def test_fit_trains_a_real_model_and_writes_a_loadable_checkpoint(
    tiny_ner, tiny_brenda, tmp_path
):
    """Everything the unit tests above stub out, once, for real: the batches
    reach the base model through the loader the pipeline actually uses, the
    losses reach the optimizer, the weights move, and what `save` writes loads
    back into a model built from the same config.

    The documents deliberately differ in chunk count, which is what the batching
    has to survive and currently does not.
    """
    data = get_batch_loader(dataset=tiny_brenda.present, batch_size=2)
    checkpoint = tmp_path / "model.pt"
    before = tiny_ner.classifier[-1].weight.detach().clone()

    trainer = Trainer(tiny_ner)
    best = trainer.fit(train_data=data, val_data=data)
    trainer.save(str(checkpoint))

    assert best is not None and math.isfinite(best)
    assert not torch.equal(before, tiny_ner.classifier[-1].weight)

    reloaded = torch.load(checkpoint, weights_only=True)
    assert tiny_ner.load_state_dict(reloaded, strict=True)
    assert reloaded.keys() == tiny_ner.state_dict().keys()
