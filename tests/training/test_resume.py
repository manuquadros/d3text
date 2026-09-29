"""Resuming an interrupted `Trainer.fit` from its per-epoch resume file.

CPU only, with a one-`Linear` `Model` like the other trainer tests. The model
draws from torch's global RNG each step and the loader shuffles off it, so a
resume that dropped the RNG state, the Adam moments or the plateau counter
would end on different parameters rather than fail loudly.
"""

import pytest
import torch
from d3text.models.base import Model
from d3text.models.config import ModelConfig
from d3text.training.trainer import ResumeFile, Trainer
from torch.utils.data import DataLoader

# The best epoch (6) comes after the interruption and is not the last. The
# plateau counter is at 2 when the run dies, so `ReduceLROnPlateau` cuts the
# rate on the first resumed epoch only if its state came back with it.
SCORES = [0.1, 0.5, 0.3, 0.2, 0.2, 0.2, 0.6, 0.3]
INTERRUPTED_AT = 4
INPUTS = {"config": {"lr": 0.05}, "encodings_digest": "abc"}


class _Interrupted(Exception):
    """Stands in for a kill arriving while an epoch runs."""


class _NoisyModel(Model):
    """Trains on shuffled batches with noise drawn from the global RNG."""

    default_selection_metrics = ("class_micro_f1",)

    def __init__(self, dies_at: int | None = None, **config: object) -> None:
        super().__init__(
            config=ModelConfig(
                **{
                    "model_class": "NERClassificationModel",
                    "num_epochs": len(SCORES),
                    "patience": len(SCORES),
                    "ramp_epochs": 0,
                    "lr": 0.05,
                    "lr_scheduler": "reduce_on_plateau",
                }
                | config
            ),
            device="cpu",
        )
        self.head = torch.nn.Linear(4, 1)
        self.dies_at = dies_at
        self.epochs: list[int] = []

    def run_epoch(self, data, epoch, update):
        if epoch == self.dies_at:
            raise _Interrupted
        self.epochs.append(epoch)
        batches = 0
        for batch in data:
            update.zero_grad()
            noisy = batch + torch.randn_like(batch)
            loss = (self.head(noisy) - batch.sum()).square().mean()
            update(loss)
            batches += 1
        return {"class": loss.detach().item()}, batches

    def evaluate_model(
        self, data, tau_cls=0.5, prefix="test", log_reports=True, step=None
    ):
        return {f"{prefix}/class_micro_f1": SCORES[step]}


def _loader() -> DataLoader:
    # `shuffle=True` seeds its sampler off torch's global generator.
    return DataLoader(
        torch.arange(24.0).reshape(6, 4) / 24, batch_size=2, shuffle=True
    )


def _fit(model: Model, resume_file: ResumeFile | None = None, resume=None):
    trainer = Trainer(model)
    best = trainer.fit(
        _loader(),
        _loader(),
        resume_file=resume_file,
        resume_from=resume,
    )
    return trainer, best


def test_a_resumed_run_ends_where_an_uninterrupted_one_does(tmp_path):
    """Parameters, best epoch and score all match bit for bit, whatever the
    restarted process seeded its RNG with or initialised its model to."""
    torch.manual_seed(0)
    uninterrupted, best = _fit(_NoisyModel())

    resume_file = ResumeFile(tmp_path / "run.resume.pt", INPUTS)
    torch.manual_seed(0)
    with pytest.raises(_Interrupted):
        _fit(_NoisyModel(dies_at=INTERRUPTED_AT), resume_file)

    torch.manual_seed(1234)
    restarted = _NoisyModel()
    resumed, resumed_best = _fit(restarted, resume_file, resume_file.read())

    assert restarted.epochs == list(range(INTERRUPTED_AT, len(SCORES)))
    assert resumed.best_epoch == uninterrupted.best_epoch == 6
    assert resumed.best_selection_score == uninterrupted.best_selection_score
    assert best is not None and resumed_best is not None
    assert best.keys() == resumed_best.keys()
    for key in best:
        assert torch.equal(best[key], resumed_best[key]), key
    for (name, ours), theirs in zip(
        restarted.state_dict().items(),
        uninterrupted.model.state_dict().values(),
        strict=True,
    ):
        assert torch.equal(ours, theirs), name
    assert (
        resumed.optimizer.param_groups[0]["lr"]
        == uninterrupted.optimizer.param_groups[0]["lr"]
        < 0.05
    )


def test_a_run_that_stopped_early_trains_no_further_on_resume(tmp_path):
    """The kill can land between the early stop and the checkpoint write; the
    resumed run must hand back the same best epoch, not train on past it."""
    resume_file = ResumeFile(tmp_path / "run.resume.pt", INPUTS)
    torch.manual_seed(0)
    first, best = _fit(_NoisyModel(patience=3), resume_file)
    assert first.model.epochs == [0, 1, 2, 3, 4, 5]

    restarted = _NoisyModel(patience=3)
    resumed, resumed_best = _fit(restarted, resume_file, resume_file.read())

    assert restarted.epochs == []
    assert resumed.best_epoch == first.best_epoch == 1
    assert best is not None and resumed_best is not None
    assert torch.equal(best["head.weight"], resumed_best["head.weight"])
    assert torch.equal(restarted.head.weight, first.model.head.weight)


@pytest.mark.parametrize(
    "changed",
    [
        {"config": {"lr": 0.1}, "encodings_digest": "abc"},
        {"config": {"lr": 0.05}, "encodings_digest": "def"},
    ],
    ids=["config", "store-digest"],
)
def test_a_resume_under_different_inputs_is_refused(tmp_path, changed):
    """Mixing epochs trained on one config or store with epochs on another
    would produce a model neither describes."""
    path = tmp_path / "run.resume.pt"
    torch.manual_seed(0)
    _fit(_NoisyModel(num_epochs=1), ResumeFile(path, INPUTS))

    assert ResumeFile(path, INPUTS).read()["epochs_run"] == 1
    with pytest.raises(ValueError, match="different inputs"):
        ResumeFile(path, changed).read()
