"""What `train` writes into the checkpoint.

The heads' weights are the whole product of a run, and which epoch's they are
is decided two files away, in `Trainer.fit`. This drives `train.main` with a
scripted validation schedule whose best epoch is not its last, and asserts at
the *file* — not at the model object — that the best epoch is what landed.
"""

import argparse
import contextlib

import pytest
import torch
from d3text import checkpoint
from d3text.cli import train
from d3text.data.data import EntityRelationDataset
from d3text.models.config import ModelConfig
from d3text.models.base import Model, Step
from d3text.training.trainer import Trainer
from d3text.vocabulary import Vocabulary
from torch.utils.data import DataLoader

VOCABULARY = Vocabulary.from_class_map(
    {"enzymes": {"enz7"}, "bacteria": {"bac42"}}
)

# Not a value any training step could reach, so a checkpoint holding it is
# unambiguously the post-`fit` model rather than the best epoch.
SCRIBBLE = -12345.0

# 3.0, then the best at 1.0, then two worse epochs the schedule sits through:
# `patience` is high enough that the run ends on epoch 3, not on its best.
VAL_LOSSES = [3.0, 1.0, 2.0, 2.5]
BEST_EPOCH = 1


class _ScriptedModel(Model):
    """A real `Model` that trains one synthetic batch an epoch and reads its
    validation losses off a script, so the schedule is deterministic."""

    def __init__(self) -> None:
        super().__init__(
            config=ModelConfig(
                base_model="prajjwal1/bert-mini",
                num_epochs=len(VAL_LOSSES),
                patience=len(VAL_LOSSES),
                ramp_epochs=0,
                lr=0.1,
            ),
            device="cpu",
        )
        self.head = torch.nn.Linear(4, 1)
        self.weights: dict[int, torch.Tensor] = {}

    def run_epoch(self, data, step, epoch, update):
        if step == Step.TRAINING:
            update.zero_grad()
            loss = self.head(torch.ones(1, 4)).sum().square()
            update(loss)
            self.weights[epoch] = self.head.weight.detach().clone()
            return {"class": loss.detach().item()}, 1
        return {"class": VAL_LOSSES[epoch]}, 1


class _ScribblingTrainer(Trainer):
    """A trainer that leaves the model holding something that is *not* the
    best epoch.

    It stands in for any future change that drops `fit`'s restore — the defect
    `02eff32` fixed on one of two exit paths, which lived unnoticed for the
    whole life of the code because the call site read the weights off the
    model and so could not tell the two apart.
    """

    def fit(self, *args, **kwargs):
        best_state = super().fit(*args, **kwargs)
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.fill_(SCRIBBLE)
        return best_state


@pytest.fixture
def trained(tmp_path, tiny_brenda, monkeypatch):
    """Run `train.main` over the scripted schedule, with everything but the
    epoch loop and the checkpoint write stubbed out."""
    model = _ScriptedModel()
    output = tmp_path / "model.pt"
    config = tmp_path / "config.toml"
    config.write_text("")

    dataset = EntityRelationDataset(
        data={split: tiny_brenda.present for split in ("train", "val", "test")},
        entity_index=VOCABULARY.entity_index,
        class_map=VOCABULARY.as_class_map(),
        class_matrix=torch.zeros(len(VOCABULARY), 2),
    )

    monkeypatch.setattr(train.runtime, "configure", lambda: None)
    monkeypatch.setattr(train.runtime, "compile_model", lambda _model: False)
    monkeypatch.setattr(
        train,
        "command_line_args",
        lambda: argparse.Namespace(
            config=str(config),
            output=str(output),
            prof=False,
            limit=None,
            log_checkpoint=False,
        ),
    )
    monkeypatch.setattr(train, "load_model_config", lambda _path: model.config)
    monkeypatch.setattr(train, "brenda_dataset", lambda **_kwargs: dataset)
    monkeypatch.setattr(
        train.data, "compute_frequencies", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        train.data,
        "get_batch_loader",
        lambda **_kwargs: DataLoader([0], batch_size=1),
    )
    monkeypatch.setattr(
        train.factory, "build_model", lambda *_args, **_kwargs: model
    )
    monkeypatch.setattr(train, "Trainer", _ScribblingTrainer)

    monkeypatch.setattr(
        train.tracking,
        "run",
        contextlib.contextmanager(lambda **_k: iter([None])),
    )
    monkeypatch.setattr(
        train.tracking, "log_metrics", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        train.tracking, "log_artifact", lambda *_args, **_kwargs: None
    )

    train.main()

    return model, checkpoint.load(output)


def test_the_checkpoint_holds_the_best_epoch_not_the_live_model(trained):
    """`train` writes what `fit` handed back. Reading the weights off the
    model instead was correct only because `fit` happens to load the snapshot
    back into it on the way out — a side effect the call site never named and
    could not see change."""
    model, saved = trained

    assert torch.equal(
        saved.state_dict["head.weight"], model.weights[BEST_EPOCH]
    )
    assert not torch.equal(
        model.head.weight.detach(), model.weights[BEST_EPOCH]
    )


def test_the_checkpoint_still_carries_the_datasets_vocabulary(trained):
    """The weights and the columns that interpret them are written together;
    changing where the weights come from must not drop the vocabulary."""
    _model, saved = trained

    assert saved.vocabulary == VOCABULARY
