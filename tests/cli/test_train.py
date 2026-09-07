"""What `train` writes into the checkpoint.

The heads' weights are the whole product of a run, and which epoch's they are
is decided two files away, in `Trainer.fit`. This drives `train.main` with a
scripted validation schedule whose best epoch is not its last, and asserts at
the *file* — not at the model object — that the best epoch is what landed.
"""

import argparse
import contextlib

import h5py
import pandas as pd
import pytest
import torch
from d3text import checkpoint, encodings_store, surface_forms, token_labels
from d3text.cli import train
from d3text.datasets import brenda
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

    def __init__(self, token_labels_store: str = "") -> None:
        super().__init__(
            config=ModelConfig(
                base_model="prajjwal1/bert-mini",
                num_epochs=len(VAL_LOSSES),
                patience=len(VAL_LOSSES),
                ramp_epochs=0,
                lr=0.1,
                token_labels_store=token_labels_store,
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


def run_train(
    tmp_path,
    tiny_brenda,
    monkeypatch,
    token_labels_store="",
    *,
    trainer=_ScribblingTrainer,
    compile_model=lambda _model: False,
    tag_calls=None,
):
    """Run `train.main` over the scripted schedule, with everything but the
    epoch loop and the checkpoint write stubbed out.

    `tag_calls`, when given, collects `("run", tags)` for the tags the run
    opened with and `("set_tags", tags)` for every retag after it, in order.
    """
    model = _ScriptedModel(token_labels_store)
    recorded = [] if tag_calls is None else tag_calls
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
    monkeypatch.setattr(train.runtime, "compile_model", compile_model)
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
    monkeypatch.setattr(train, "Trainer", trainer)

    def start_run(**kwargs):
        recorded.append(("run", dict(kwargs.get("tags") or {})))
        return iter([None])

    monkeypatch.setattr(
        train.tracking, "run", contextlib.contextmanager(start_run)
    )
    monkeypatch.setattr(
        train.tracking,
        "set_tags",
        lambda tags: recorded.append(("set_tags", dict(tags))),
    )
    monkeypatch.setattr(
        train.tracking, "log_metrics", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        train.tracking, "log_artifact", lambda *_args, **_kwargs: None
    )

    train.main()

    return model, checkpoint.load(output)


@pytest.fixture
def trained(tmp_path, tiny_brenda, monkeypatch):
    return run_train(tmp_path, tiny_brenda, monkeypatch)


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


def test_the_checkpoint_records_the_label_store_its_targets_came_from(
    tmp_path, tiny_brenda, monkeypatch
):
    """Which strings the store's dictionary named is what set the span
    targets, and nothing in the weights or the vocabulary says. Without it a
    checkpoint scored against a store rebuilt from another index is scored
    against a different count of gold spans, with both existing guards
    silent."""
    store = tmp_path / "labels.hdf5"
    stamp = token_labels.IndexStamp.from_index(
        surface_forms.build_index({"enz7": ["catalase"]}),
        sources=("split.csv",),
    )
    with h5py.File(store, "w-", libver="latest") as handle:
        token_labels.write_label_space(handle, stamp=stamp)

    _model, saved = run_train(
        tmp_path, tiny_brenda, monkeypatch, token_labels_store=str(store)
    )

    assert saved.token_labels_digest == stamp.digest


def test_a_run_that_reads_no_label_store_records_no_digest(trained):
    """The field is a provenance record, not a requirement: a config that
    names no store trains exactly the model it always did."""
    _model, saved = trained

    assert saved.token_labels_digest is None


class _EagerFallbackTrainer(Trainer):
    """A trainer whose epochs leave the model executing eagerly, the way
    `runtime._install_eager_fallback` does when the backend fails at a
    forward."""

    def fit(self, *args, **kwargs):
        self.model._compiled_call_impl = None
        return super().fit(*args, **kwargs)


def _compile_that_takes(model):
    """Stand in for `runtime.compile_model` on a Triton-capable machine:
    install a graph and report that it took."""
    model._compiled_call_impl = model._call_impl
    return True


def test_the_compiled_tag_reports_what_the_epochs_ran(
    tmp_path, tiny_brenda, monkeypatch
):
    """The backend does not run until the first batch, so the tag the run
    opens with is a prediction. A run that fell back to eager and kept
    `compiled=true` reads in MLflow exactly like one that stayed compiled,
    which misattributes any speed difference between the two."""
    recorded: list[tuple[str, dict[str, str]]] = []

    run_train(
        tmp_path,
        tiny_brenda,
        monkeypatch,
        trainer=_EagerFallbackTrainer,
        compile_model=_compile_that_takes,
        tag_calls=recorded,
    )

    opened = [tags for call, tags in recorded if call == "run"]
    after_fit = [tags for call, tags in recorded if call == "set_tags"]

    assert opened[0]["compiled"] == "true"
    assert after_fit == [{"compiled": "false"}]


def test_the_checkpoint_records_the_tokenization_its_inputs_came_from(
    tmp_path, tiny_brenda, tiny_hdf5, monkeypatch
):
    """Which ids the store holds is what the heads ever saw, and neither the
    weights nor the vocabulary nor the store's own model-and-window stamp says
    it. Without this, a checkpoint scored against a corpus re-tokenized under
    a newer tokenizer is scored on inputs it never trained on, with every
    existing guard silent."""
    with h5py.File(tiny_hdf5, "r+") as handle:
        digest = encodings_store.stamp_content_digest(handle)
    # Named relative to the data directory, as a config names it: the digest
    # has to be read from the file the dataset opens, and an absolute path
    # would pass whether or not the two were joined.
    monkeypatch.setattr(brenda, "DATA_DIR", tiny_hdf5.parent)
    monkeypatch.setitem(train.encodings, "prajjwal1/bert-mini", tiny_hdf5.name)

    _model, saved = run_train(tmp_path, tiny_brenda, monkeypatch)

    assert saved.encodings_digest == digest


def test_a_run_over_an_unstamped_store_records_no_encodings_digest(
    tmp_path, tiny_brenda, tiny_hdf5, monkeypatch
):
    """Every encodings file written before the digest existed is unstamped, so
    a run against one has to train and write its checkpoint as it always
    did."""
    monkeypatch.setitem(train.encodings, "prajjwal1/bert-mini", str(tiny_hdf5))

    _model, saved = run_train(tmp_path, tiny_brenda, monkeypatch)

    assert saved.encodings_digest is None


class _StopAfterDatasetBuild(Exception):
    """Raised as soon as the dataset is built, so this test never has to drive
    a model or a trainer through the rest of `main`."""


def _split_frame() -> pd.DataFrame:
    """One document, in the shape `brenda_references` hands a split over."""
    return pd.DataFrame(
        [
            {
                "pubmed_id": 10,
                "fulltext": "<p>body</p>",
                "relations": [],
                "entities": ["enz7"],
                "strains": [],
                "bacteria": [],
                "other_organisms": [],
                "enzymes": [7],
            }
        ]
    )


def test_training_builds_no_split_it_never_reads(monkeypatch):
    """`train` reads `train` and `val` and nothing else, so asking the corpus
    for `test` costs a pass over a 75 MB CSV that is then discarded. The
    assertion is on the dataset `main` actually receives, so a future reader
    reaching for `data["test"]` fails here rather than at the KeyError."""
    loaded = []
    built = {}

    def loader(split):
        def load(noise=0, limit=0):
            loaded.append(split)
            return _split_frame()

        return load

    for split in ("training", "validation", "test"):
        monkeypatch.setattr(
            brenda.brenda_references, f"{split}_data", loader(split)
        )

    def build(**kwargs):
        built["dataset"] = brenda.brenda_dataset(**kwargs)
        raise _StopAfterDatasetBuild

    config = ModelConfig(base_model="prajjwal1/bert-mini")
    monkeypatch.setattr(train.runtime, "configure", lambda: None)
    monkeypatch.setattr(
        train,
        "command_line_args",
        lambda: argparse.Namespace(
            config="unused.toml",
            output="unused.pt",
            prof=False,
            limit=None,
            log_checkpoint=False,
        ),
    )
    monkeypatch.setattr(train, "load_model_config", lambda _path: config)
    monkeypatch.setitem(train.encodings, config.base_model, "nowhere.hdf5")
    monkeypatch.setattr(train, "brenda_dataset", build)

    with pytest.raises(_StopAfterDatasetBuild):
        train.main()

    assert set(built["dataset"].data) == {"train", "val"}
    assert loaded == ["training", "validation"]
