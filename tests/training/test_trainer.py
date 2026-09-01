"""Unit tests for the epoch schedule.

CPU only: `Model.__init__` loads no base model, so a real `Model` subclass with
a single `Linear` is enough to drive a whole `fit`. The early-stopping and
snapshot tests live here rather than with the models because the comparison and
the best-epoch state are the `Trainer`'s.
"""

import types

import pytest
import torch
from beartype.roar import BeartypeCallHintParamViolation
from d3text import metric_docs, runtime
from d3text.models.config import ModelConfig
from d3text.models.base import Model, Step
from d3text.training.trainer import Trainer
from torch.utils.data import DataLoader


class _ScriptedModel(Model):
    """A real `Model` whose `run_epoch` trains one synthetic batch and reads
    its validation losses off a script, so the schedule is deterministic."""

    def __init__(self, val_losses: list[float], **config: object) -> None:
        super().__init__(config=ModelConfig(**config), device="cpu")
        self.head = torch.nn.Linear(4, 1)
        self.val_losses = val_losses
        self.seen: list[tuple[Step, int]] = []
        self.weights: dict[int, torch.Tensor] = {}

    def run_epoch(self, data, step, epoch, update):
        self.seen.append((step, epoch))
        if step == Step.TRAINING:
            update.zero_grad()
            loss = self.head(torch.ones(1, 4)).sum().square()
            update(loss)
            self.weights[epoch] = self.head.weight.detach().clone()
            return {"class": loss.detach().item()}, 1
        return {"class": self.val_losses[epoch]}, 1


def _loader() -> DataLoader:
    return DataLoader([0], batch_size=1)


def _scripted(**config: object) -> _ScriptedModel:
    return _ScriptedModel(
        [3.0, 1.0, 2.0, 2.5, 2.6, 2.7],
        num_epochs=6,
        patience=1,
        ramp_epochs=0,
        lr=0.1,
        **config,
    )


def test_the_model_no_longer_carries_the_training_loop():
    """The optimizer, the scheduler, the early-stop comparison and the
    best-epoch snapshot are the trainer's; a model built for evaluation must
    not drag them along."""
    model = _scripted()

    for attribute in (
        "optimizer",
        "scheduler",
        "scaler",
        "train_model",
        "early_stop",
        "_setup_training",
        "_update",
        "_cpu_state_dict",
        "validate_model",
        "best_val_loss",
        "best_model_state",
        "stop_counter",
    ):
        assert not hasattr(model, attribute), attribute


def test_the_trainer_owns_the_optimizer_the_config_names():
    model = _scripted(optimizer="adamw")
    trainer = Trainer(model)

    assert isinstance(trainer.optimizer, torch.optim.AdamW)
    assert trainer.scheduler is None
    assert trainer.optimizer.param_groups[0]["lr"] == model.config.lr
    assert [
        parameter
        for group in trainer.optimizer.param_groups
        for parameter in group["params"]
    ] == list(model.parameters())


def test_fit_steps_the_weights_through_the_trainers_update():
    model = _scripted()
    before = model.head.weight.detach().clone()

    Trainer(model).fit(train_data=_loader())

    assert not torch.equal(model.head.weight.detach(), before)


def test_fit_stops_early_and_restores_the_best_epoch():
    model = _scripted()
    trainer = Trainer(model)

    best = trainer.fit(train_data=_loader(), val_data=_loader())

    # 3.0, then the best at 1.0, then two epochs without improvement — one
    # more than `patience`.
    assert best is trainer.best_model_state
    assert trainer.best_val_loss == 1.0
    assert trainer.best_epoch == 1
    assert model.seen == [
        (step, epoch)
        for epoch in range(4)
        for step in (Step.TRAINING, Step.VALIDATION)
    ]
    assert trainer.best_model_state is not None
    assert torch.equal(
        model.head.weight.detach(), trainer.best_model_state["head.weight"]
    )


def test_fit_restores_the_best_epoch_when_the_epochs_run_out():
    """A run whose validation loss dips and then drifts back up without ever
    going `patience` epochs without improvement never trips early stopping, so
    it falls out of the loop still holding the last epoch's parameters — up to
    `patience` epochs past the snapshot the trainer was keeping."""
    model = _ScriptedModel(
        [3.0, 1.0, 2.0, 2.5],
        num_epochs=4,
        patience=2,
        ramp_epochs=0,
        lr=0.1,
    )
    trainer = Trainer(model)

    trainer.fit(train_data=_loader(), val_data=_loader())

    assert trainer.best_epoch == 1
    assert trainer.best_val_loss == 1.0
    assert len(model.seen) == 8  # the loop ran out rather than stopping early
    assert trainer.best_model_state is not None
    assert not torch.equal(model.weights[1], model.weights[3])
    assert torch.equal(model.head.weight.detach(), model.weights[1])
    assert torch.equal(
        model.head.weight.detach(), trainer.best_model_state["head.weight"]
    )


def test_fit_without_a_checkpoint_leaves_the_last_epoch_in_place():
    """`tune` trains with `save_checkpoint=False`: there is no snapshot to
    restore, so the weights stay where the last epoch left them."""
    model = _scripted()
    trainer = Trainer(model)
    best = trainer.fit(
        train_data=_loader(), val_data=_loader(), save_checkpoint=False
    )

    # `fit` hands back nothing rather than a copy of the live parameters: a
    # sweep would otherwise hold one full parameter set per trial.
    assert best is None
    assert trainer.best_model_state is None
    assert trainer.best_val_loss == 1.0


def test_fit_logs_the_epoch_accounting(monkeypatch):
    """The per-epoch and summary metrics a run list is scanned by."""
    logged: list[tuple[dict[str, float], int | None]] = []
    monkeypatch.setattr(
        "d3text.tracking.log_metrics",
        lambda metrics, step=None: logged.append((metrics, step)),
    )

    model = _scripted()
    trainer = Trainer(model)
    trainer.fit(train_data=_loader(), val_data=_loader())

    per_epoch: dict[int, dict[str, float]] = {}
    summary: dict[str, float] = {}
    for metrics, step in logged:
        if step is None:
            summary.update(metrics)
        else:
            per_epoch.setdefault(step, {}).update(metrics)

    assert summary == {
        "best_val_loss": 1.0,
        "best_epoch": 1.0,
        "epochs_run": 4.0,
        "epochs_after_best": 2.0,
        "stopped_early": 1.0,
    }
    assert per_epoch[3]["early_stopping/epochs_without_improvement"] == 2.0
    assert per_epoch[0]["learning_rate"] == model.config.lr
    for epoch in range(4):
        assert "training/grad_norm" in per_epoch[epoch]
        assert "training/grad_clip_rate" in per_epoch[epoch]
        assert "training/loss_total" in per_epoch[epoch]
        assert "validation/loss_total" in per_epoch[epoch]


def test_every_metric_fit_logs_is_documented(monkeypatch):
    """Nothing reaches the tracking server whose y-axis is recorded nowhere.

    Driving `fit` rather than listing names keeps this honest — a new key
    appears here the moment it is logged.
    """
    logged: list[dict[str, float]] = []
    monkeypatch.setattr(
        "d3text.tracking.log_metrics",
        lambda metrics, step=None: logged.append(dict(metrics)),
    )

    Trainer(_scripted()).fit(train_data=_loader(), val_data=_loader())

    names = {name for metrics in logged for name in metrics}
    assert names
    assert [
        name for name in sorted(names) if metric_docs.describe(name) is None
    ] == []


def test_fit_logs_epoch_accounting_without_validation_data(monkeypatch):
    """A run with no validation split has no `best_val_loss`, `best_epoch` or
    `epochs_after_best` to report, but it still ran a known number of epochs
    and never had a signal to early-stop on — both are meaningful without
    validation and must still reach the tracking layer."""
    logged: list[tuple[dict[str, float], int | None]] = []
    monkeypatch.setattr(
        "d3text.tracking.log_metrics",
        lambda metrics, step=None: logged.append((metrics, step)),
    )

    model = _scripted()
    trainer = Trainer(model)
    trainer.fit(train_data=_loader())

    summary: dict[str, float] = {}
    for metrics, step in logged:
        if step is None:
            summary.update(metrics)

    assert summary == {"epochs_run": 6.0, "stopped_early": 0.0}


def test_the_scheduler_steps_once_per_validated_epoch(monkeypatch):
    """`reduce_on_plateau` is stepped with the monitored loss, not the epoch;
    it is not an `LRScheduler` subclass and stepping it with an epoch would be
    silently accepted."""
    model = _scripted(lr_scheduler="reduce_on_plateau")
    trainer = Trainer(model)
    stepped: list[float] = []
    monkeypatch.setattr(
        trainer.scheduler, "step", lambda metric: stepped.append(metric)
    )

    trainer.fit(train_data=_loader(), val_data=_loader())

    assert stepped == [3.0, 1.0, 2.0, 2.5]


def test_a_ramped_run_stops_on_a_plateau_inside_the_ramp():
    """The patience counter is not reset through the loss-weight ramp.

    Validation is scored under the ramp's final weights, so an early-ramp
    epoch's total is comparable with a late one's.
    """
    model = _ScriptedModel(
        [1.0, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6],
        num_epochs=8,
        patience=1,
        ramp_epochs=4,
        lr=0.1,
    )
    trainer = Trainer(model)

    trainer.fit(train_data=_loader(), val_data=_loader())

    assert trainer.best_epoch == 0
    assert [epoch for step, epoch in model.seen if step is Step.VALIDATION] == [
        0,
        1,
        2,
    ]


# --------------------------------------------------------------------------- #
# Trainer._early_stop                                                          #
# --------------------------------------------------------------------------- #
def _early_stopper(stub, patience):
    return stub(
        Trainer,
        best_val_loss=float("inf"),
        stop_counter=0,
        config=types.SimpleNamespace(patience=patience),
    )


def test_early_stop_never_triggers_on_improvement(stub):
    t = _early_stopper(stub, patience=2)
    stops = [
        t._early_stop(v, epoch=e, save_checkpoint=False)
        for e, v in enumerate((5.0, 4.0, 3.0, 2.0))
    ]
    assert stops == [False, False, False, False]
    assert t.stop_counter == 0
    assert t.best_val_loss == 2.0


def test_early_stop_triggers_after_patience_exceeded(stub):
    t = _early_stopper(stub, patience=2)
    stops = [
        t._early_stop(v, epoch=e, save_checkpoint=False)
        for e, v in enumerate((1.0, 2.0, 3.0, 4.0))
    ]
    # improvement, then patience(2) tolerated increases, then stop
    assert stops == [False, False, False, True]
    assert t.best_val_loss == 1.0  # best preserved


def test_early_stop_records_the_epoch_that_produced_the_best_loss(stub):
    """`best_epoch` was initialised to -1 and never assigned, so a run that
    peaked at epoch 0 and then degraded reported having peaked at epoch -1."""
    t = _early_stopper(stub, patience=2)
    t.best_epoch = -1

    for epoch, val_loss in enumerate((1.0, 2.0, 3.0)):
        t._early_stop(val_loss, epoch=epoch, save_checkpoint=False)

    assert t.best_val_loss == 1.0
    assert t.best_epoch == 0


class _CheckpointableModel(Model):
    """The smallest real `Model`: `Model.__init__` loads no base model, so this
    has a genuine `state_dict` without a network download."""

    def __init__(self, device):
        super().__init__(config=ModelConfig(), device=device)
        self.head = torch.nn.Linear(4, 3)


def test_early_stop_snapshots_the_best_state_on_cpu(device):
    """The best-epoch snapshot must not sit on the GPU.

    `deepcopy(state_dict())` preserved each tensor's device, pinning a second
    resident copy of the whole model. The CPU variant is the semantics guard;
    the CUDA variant is the red.
    """
    model = _CheckpointableModel(device).to(device)
    trainer = Trainer(model)

    trainer._early_stop(1.0, epoch=0, save_checkpoint=True)

    assert trainer.best_model_state  # parameters and the _neg_inf buffer
    assert all(
        tensor.device.type == "cpu"
        for tensor in trainer.best_model_state.values()
    )
    # the live model has not moved
    assert model.head.weight.device.type == device


def test_early_stop_snapshot_does_not_alias_the_live_parameters(device):
    """`.to("cpu")` returns *self* for a tensor already there, so a CPU run
    would otherwise snapshot references that follow training."""
    model = _CheckpointableModel(device).to(device)
    trainer = Trainer(model)

    trainer._early_stop(1.0, epoch=0, save_checkpoint=True)
    snapshot = trainer.best_model_state["head.weight"].clone()

    with torch.no_grad():
        model.head.weight.add_(1.0)

    assert torch.equal(trainer.best_model_state["head.weight"], snapshot)


def test_early_stop_snapshot_still_reloads_strictly(device):
    """The convergence path in `fit`: the snapshot goes back in whole, and
    `load_state_dict` returns each tensor to the parameter's own device."""
    model = _CheckpointableModel(device).to(device)
    trainer = Trainer(model)

    trainer._early_stop(1.0, epoch=0, save_checkpoint=True)
    # to CPU explicitly: this is a both-ways guard on the reload, so it must
    # not red merely because the snapshot's own device changed.
    best = trainer.best_model_state["head.weight"].detach().cpu().clone()

    with torch.no_grad():
        model.head.weight.add_(1.0)
    model.load_state_dict(trainer.best_model_state, strict=True)

    assert model.head.weight.device.type == device
    assert torch.equal(model.head.weight.detach().cpu(), best)


@pytest.mark.parametrize("save_checkpoint", [True, False])
def test_run_epoch_is_handed_the_trainers_update(save_checkpoint):
    """The model computes the losses; the object that applies them is the
    trainer's, so the same model can be evaluated without one."""
    model = _scripted()
    trainer = Trainer(model)
    seen: list[object] = []

    original = model.run_epoch

    def spy(data, step, epoch, update):
        seen.append(update)
        return original(data, step, epoch, update)

    model.run_epoch = spy  # type: ignore[method-assign]
    trainer.fit(
        train_data=_loader(),
        val_data=_loader(),
        save_checkpoint=save_checkpoint,
    )

    assert seen and all(update is trainer.update for update in seen)


def test_trainer_rejects_a_torch_compiled_model():
    """The wrapper `torch.compile` returns is an `nn.Module` but not a `Model`.

    `train` compiles in place precisely so the trainer never has to accept one.
    The wrapper is returned on CPU too, and only executing the graph needs
    Triton.
    """
    model = _scripted()
    compiled = torch.compile(model, dynamic=True)

    with pytest.raises(BeartypeCallHintParamViolation):
        Trainer(compiled)


def test_trainer_rejects_a_module_that_is_no_model():
    """The widened annotation must still be worth checking — and beartype's
    import hook must still be instrumenting this module, or the test above
    passes for the wrong reason."""
    with pytest.raises(BeartypeCallHintParamViolation):
        Trainer(torch.nn.Linear(4, 1))


class _ForwardingModel(Model):
    """A `Model` that reaches its own forward the way the real ones do: the
    trainer calls `run_epoch`, and the forward is invoked as ``self(...)``
    several frames below it — never on whatever object the trainer holds."""

    head: torch.nn.Linear

    def __init__(self, **config: object) -> None:
        super().__init__(config=ModelConfig(**config), device="cpu")
        self.head = torch.nn.Linear(4, 1)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.head(batch)

    def run_epoch(self, data, step, epoch, update):
        loss = self(torch.ones(1, 4)).sum().square()
        if step == Step.TRAINING:
            update.zero_grad()
            update(loss)
        return {"class": loss.detach().item()}, 1


def test_compiling_puts_the_trainers_forward_on_the_compiled_path(monkeypatch):
    """Compiling has to reach the forward the trainer's call chain takes.

    `torch.compile`'s wrapper forwards methods bound to the *uncompiled* model,
    so `run_epoch`'s `self(...)` runs eager. GPU-free: `nn.Module.compile`
    looks `torch.compile` up at call time, so a stand-in records what was
    compiled.
    """
    entered: list[object] = []

    def recording_compile(target, **kwargs):
        def compiled(*args, **call_kwargs):
            entered.append(target)
            return target(*args, **call_kwargs)

        return compiled

    monkeypatch.setattr(runtime, "is_triton_compatible", lambda: True)
    monkeypatch.setattr(torch, "compile", recording_compile)

    model = _ForwardingModel(num_epochs=1, ramp_epochs=0, lr=0.1)
    assert runtime.compile_model(model) is True

    Trainer(model).fit(train_data=_loader(), save_checkpoint=False)

    assert entered == [model._call_impl]


@pytest.mark.parametrize(
    "amp_dtype, enabled",
    [(torch.float16, True), (torch.bfloat16, False)],
)
def test_the_scaler_follows_the_models_autocast_dtype(amp_dtype, enabled):
    """The trainer builds the update, so it is the one place that knows which
    dtype the forward autocasts to; a bf16 forward must not be scaled."""
    model = _scripted()
    model.amp_dtype = amp_dtype

    assert Trainer(model).update.scaler.is_enabled() is enabled
