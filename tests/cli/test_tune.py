"""What `tune` records about a trial: its config dump and its tags.

`pprint.pp` (the old call) hardcodes `sort_dicts=False`; `pprint.pformat`
(what it was replaced with) defaults `sort_dicts` to `True`. Left at that
default the dump prints alphabetically instead of in `ModelConfig` field
order, which is what a reader expects when comparing it against the TOML.
"""

import argparse
import contextlib
import types

import pytest
import torch

from d3text.cli import tune
from d3text.models.config import ModelConfig


class _StopAfterDump(Exception):
    """Raised from the first thing `main` does after the log line, so the
    test never has to drive a real dataset/model/trainer through it."""


@pytest.fixture
def stop_after_config_dump(monkeypatch):
    monkeypatch.setattr(
        tune,
        "command_line_args",
        lambda: argparse.Namespace(
            config="unused.toml", output="unused.csv", limit=None
        ),
    )
    monkeypatch.setattr(
        tune, "load_tuning_config", lambda path: [ModelConfig()]
    )
    monkeypatch.setitem(tune.encodings, ModelConfig().base_model, "unused.hdf5")

    calls = []

    def blow_up(**kwargs):
        calls.append(kwargs)
        raise _StopAfterDump

    monkeypatch.setattr(tune, "brenda_dataset", blow_up)
    return calls


def test_dump_key_order_matches_model_dump_field_order(
    stop_after_config_dump, capsys
):
    with pytest.raises(_StopAfterDump):
        tune.main()

    dump = ModelConfig().model_dump()
    field_order = list(dump.keys())

    printed = capsys.readouterr().out
    dump_start = printed.index("{")
    dump_end = printed.index("}\n", dump_start) + 1
    dump_text = printed[dump_start:dump_end]

    printed_order = [
        line.split(":", 1)[0].strip().strip("'")
        for line in dump_text.strip("{}").splitlines()
        if line.strip()
    ]
    assert printed_order == field_order


def test_a_trial_asks_for_no_split_it_never_reads(stop_after_config_dump):
    """Each trial rebuilds the dataset, so a split nobody reads is a pass over
    a 75 MB CSV per trial. `tune` loads batches from `train` and `val` only."""
    with pytest.raises(_StopAfterDump):
        tune.main()

    (call,) = stop_after_config_dump
    assert call["split_names"] == ("train", "val")


class _Model(torch.nn.Module):
    """The least a model has to be for `tune.main` to drive it: `is_compiled`
    reads a real `nn.Module` attribute, and nothing else here is exercised."""

    device = "cpu"


class _EagerFallbackTrainer:
    """Stands in for `Trainer`, leaving the model executing eagerly the way
    `runtime._install_eager_fallback` does when the backend fails at a
    forward."""

    def __init__(self, model):
        self.model = model
        self.best_val_loss = 1.0

    def fit(self, **_kwargs):
        self.model._compiled_call_impl = None


def _compile_that_takes(model):
    """Stand in for `runtime.compile_model` on a Triton-capable machine:
    install a graph and report that it took."""
    model._compiled_call_impl = model._call_impl
    return True


def stub_tune(monkeypatch, model, trainer, tag_calls):
    """Stub every part of a trial but the trainer, collecting `("run", tags)`
    for the tags the run opened with and `("set_tags", tags)` for every retag
    after it, in order."""
    config = ModelConfig()

    def start_run(**kwargs):
        # A generator rather than a one-item iterator: `contextmanager` throws
        # into it when the block raises, which a plain iterator cannot take.
        tag_calls.append(("run", dict(kwargs.get("tags") or {})))
        yield

    monkeypatch.setattr(tune.runtime, "configure", lambda: None)
    monkeypatch.setattr(
        tune,
        "command_line_args",
        lambda: argparse.Namespace(
            config="unused.toml", output="unused.csv", limit=None
        ),
    )
    monkeypatch.setattr(tune, "load_tuning_config", lambda _path: [config])
    monkeypatch.setitem(tune.encodings, config.base_model, "unused.hdf5")
    monkeypatch.setattr(
        tune,
        "brenda_dataset",
        lambda **_kwargs: types.SimpleNamespace(data={"train": [], "val": []}),
    )
    monkeypatch.setattr(
        tune.data, "compute_frequencies", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(tune.data, "get_batch_loader", lambda **_kwargs: None)
    monkeypatch.setattr(
        tune.factory, "build_model", lambda *_args, **_kwargs: model
    )
    monkeypatch.setattr(tune.factory, "dataset_metrics", lambda _dataset: {})
    monkeypatch.setattr(tune.factory, "model_metrics", lambda _model: {})
    monkeypatch.setattr(tune.runtime, "compile_model", _compile_that_takes)
    monkeypatch.setattr(tune, "Trainer", trainer)
    monkeypatch.setattr(tune.utils, "log_config", lambda *_a, **_k: None)
    monkeypatch.setattr(
        tune.tracking, "run", contextlib.contextmanager(start_run)
    )
    monkeypatch.setattr(tune.tracking, "log_metrics", lambda *_a, **_k: None)
    monkeypatch.setattr(
        tune.tracking,
        "set_tags",
        lambda tags: tag_calls.append(("set_tags", dict(tags))),
    )


def test_the_compiled_tag_reports_what_the_trial_ran(monkeypatch):
    """Every trial reuses one sweep's tag conventions, so a trial that fell
    back to eager and kept `compiled=true` is the one row in the sweep whose
    epoch times cannot be compared with its neighbours' — and nothing on the
    run says so."""
    recorded: list[tuple[str, dict[str, str]]] = []
    stub_tune(monkeypatch, _Model(), _EagerFallbackTrainer, recorded)

    tune.main()

    opened = [tags for call, tags in recorded if call == "run"]
    after_fit = [tags for call, tags in recorded if call == "set_tags"]

    assert opened[0]["compiled"] == "true"
    assert after_fit == [{"compiled": "false"}]


class _TrialDied(Exception):
    """Stands in for anything an epoch can die on — an OOM, a bad batch, a
    backend that fails past its own fallback."""


class _DyingTrainer(_EagerFallbackTrainer):
    """A trainer whose epochs fall back to eager and then raise, which is the
    order that leaves the opening tag both wrong and final."""

    def fit(self, **kwargs):
        super().fit(**kwargs)
        raise _TrialDied


def test_a_trial_whose_epochs_die_still_retags_what_they_ran(monkeypatch):
    """A sweep is read by filtering it, and a failed trial is exactly the row
    someone filters for when asking whether the compiler was implicated — so
    it must not be left holding the prediction `compile_model` made before the
    first batch."""
    recorded: list[tuple[str, dict[str, str]]] = []
    stub_tune(monkeypatch, _Model(), _DyingTrainer, recorded)

    with pytest.raises(_TrialDied):
        tune.main()

    assert [tags for call, tags in recorded if call == "set_tags"] == [
        {"compiled": "false"}
    ]
