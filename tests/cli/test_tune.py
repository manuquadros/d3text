"""What `tune` records about a trial: its config dump and its tags.

`pprint.pp` (the old call) hardcodes `sort_dicts=False`; `pprint.pformat`
(what it was replaced with) defaults `sort_dicts` to `True`. Left at that
default the dump prints alphabetically instead of in `ModelConfig` field
order, which is what a reader expects when comparing it against the TOML.
"""

import argparse
import contextlib
import math
import sys
import types
import weakref

import pytest
import torch

from d3text.cli import tune
from d3text.models.config import ModelConfig


@pytest.fixture(autouse=True)
def _clear_dataset_cache():
    """`_dataset_for` is a module-level `lru_cache`, so a dataset built by one
    test's mock would otherwise be handed to the next test sharing the same
    `(base_model, limit)` key."""
    tune._dataset_for.cache_clear()
    yield
    tune._dataset_for.cache_clear()


class _StopAfterDump(BaseException):
    """Raised from the first thing `main` does after the log line, so the
    test never has to drive a real dataset/model/trainer through it.

    A `BaseException`, not an `Exception`: setup now runs inside the trial's
    own `except Exception` (so a setup failure gets a FAILED run and a NaN
    row like any other), which would otherwise swallow this and log a row
    instead of letting it escape `main` for `pytest.raises` to see.
    """


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
        tune,
        "load_tuning_config",
        lambda path, **_kwargs: [
            ModelConfig(model_class="NERClassificationModel")
        ],
    )
    monkeypatch.setitem(
        tune.encodings,
        ModelConfig(model_class="NERClassificationModel").base_model,
        "unused.hdf5",
    )

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

    dump = ModelConfig(model_class="NERClassificationModel").model_dump()
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


def test_logged_configs_reads_prior_csv_rows(tmp_path):
    """CSV rows must become exclusions when a sweep resumes."""
    output = tmp_path / "results.csv"
    config = ModelConfig(
        model_class="NERClassificationModel",
        hidden_layers=[256, 128, 64],
        common_hidden_block=False,
    )
    tune.utils.log_config(str(output), config, selection_score=1.0)

    assert tune._logged_configs(str(output)) == [config]


class _Model(torch.nn.Module):
    """The least a model has to be for `tune.main` to drive it:
    `compile_trunk`/`trunk_is_compiled` track one mutable flag, standing in
    for `Model`'s real ones since `tune` calls the model directly rather
    than through `runtime.compile_model`, and nothing else here is
    exercised."""

    device = "cpu"

    def __init__(self) -> None:
        super().__init__()
        self._trunk_compiled = False

    def compile_trunk(self) -> bool:
        """Stand in for a Triton-capable machine: install a graph and
        report that it took."""
        self._trunk_compiled = True
        return True

    def trunk_is_compiled(self) -> bool:
        return self._trunk_compiled


class _EagerFallbackTrainer:
    """Stands in for `Trainer`, leaving the model executing eagerly the way
    `runtime._install_eager_fallback` does when the backend fails at a
    forward."""

    def __init__(self, model):
        self.model = model
        self.best_selection_score = 1.0

    def fit(self, **_kwargs):
        self.model._trunk_compiled = False


def stub_tune(
    monkeypatch, model, trainer, tag_calls, configs=None, brenda_dataset=None
):
    """Stub every part of a trial but the trainer, collecting `("run", tags)`
    for the tags the run opened with, `("set_tags", tags)` for every retag
    after it, and `("run_failed", tags)` when the run's block raised, in
    order. `configs` defaults to one trial; `brenda_dataset` defaults to a
    stub returning an empty dataset."""
    configs = configs or [ModelConfig(model_class="NERClassificationModel")]

    def start_run(**kwargs):
        # A generator rather than a one-item iterator: `contextmanager` throws
        # into it when the block raises, which a plain iterator cannot take.
        tags = dict(kwargs.get("tags") or {})
        tag_calls.append(("run", tags))
        try:
            yield
        except Exception:
            tag_calls.append(("run_failed", tags))
            raise

    monkeypatch.setattr(tune.runtime, "configure", lambda: None)
    monkeypatch.setattr(
        tune,
        "command_line_args",
        lambda: argparse.Namespace(
            config="unused.toml", output="unused.csv", limit=None
        ),
    )
    monkeypatch.setattr(
        tune, "load_tuning_config", lambda _path, **_kwargs: configs
    )
    for config in configs:
        monkeypatch.setitem(tune.encodings, config.base_model, "unused.hdf5")
    monkeypatch.setattr(
        tune,
        "brenda_dataset",
        brenda_dataset
        or (
            lambda **_kwargs: types.SimpleNamespace(
                data={"train": [], "val": []}
            )
        ),
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
    run says so. `compiled` is unknown when the run opens (it depends on
    `compile_trunk`, which runs after), so it is retagged, not an opening
    tag."""
    recorded: list[tuple[str, dict[str, str]]] = []
    stub_tune(monkeypatch, _Model(), _EagerFallbackTrainer, recorded)

    tune.main()

    opened = [tags for call, tags in recorded if call == "run"]
    retags = [tags for call, tags in recorded if call == "set_tags"]

    assert "compiled" not in opened[0]
    assert retags == [{"compiled": "true"}, {"compiled": "false"}]


def test_a_sweep_only_rebuilds_the_dataset_when_base_model_changes(
    monkeypatch,
):
    """Most swept fields (lr, dropout, batch_max_chunks, ...) don't change
    the dataset, so two trials sharing `base_model` must build it once; a
    trial that changes `base_model` must rebuild."""
    configs = [
        ModelConfig(model_class="NERClassificationModel", base_model="a"),
        ModelConfig(model_class="NERClassificationModel", base_model="a"),
        ModelConfig(model_class="NERClassificationModel", base_model="b"),
    ]
    calls: list[str] = []

    def counting_brenda_dataset(*, base_model, **_kwargs):
        calls.append(base_model)
        return types.SimpleNamespace(data={"train": [], "val": []})

    recorded: list[tuple[str, dict[str, str]]] = []
    stub_tune(
        monkeypatch,
        _Model(),
        _EagerFallbackTrainer,
        recorded,
        configs=configs,
        brenda_dataset=counting_brenda_dataset,
    )

    tune.main()

    assert calls == ["a", "b"]


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
    it must not be left holding the prediction `compile_trunk` made before the
    first batch."""
    recorded: list[tuple[str, dict[str, str]]] = []
    stub_tune(monkeypatch, _Model(), _DyingTrainer, recorded)

    with pytest.raises(SystemExit):
        tune.main()

    assert [tags for call, tags in recorded if call == "set_tags"] == [
        {"compiled": "true"},
        {"compiled": "false"},
    ]


def _recording_log_config(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Replace `utils.log_config` with a stub collecting each row's
    `selection_score`, in the order the rows were written."""
    rows: list[float] = []
    monkeypatch.setattr(
        tune.utils,
        "log_config",
        lambda _output, _config, **metrics: rows.append(
            metrics["selection_score"]
        ),
    )
    return rows


def test_a_failed_trial_is_recorded_and_the_sweep_goes_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A random grid can draw an invalid cell without the sweep being wrong,
    so one trial dying must not take the trials after it with it — and the
    dead one must stay visible: its run closed as failed, and a row in the
    results with no loss to read, so the sweep's coverage is legible."""
    configs = [
        ModelConfig(model_class="NERClassificationModel"),
        ModelConfig(model_class="NERClassificationModel"),
    ]
    fits: list[int] = []

    class _DiesOnceTrainer(_EagerFallbackTrainer):
        def fit(self, **kwargs: object) -> None:
            fits.append(len(fits))
            if len(fits) == 1:
                raise _TrialDied
            super().fit(**kwargs)

    recorded: list[tuple[str, dict[str, str]]] = []
    stub_tune(
        monkeypatch, _Model(), _DiesOnceTrainer, recorded, configs=configs
    )
    rows = _recording_log_config(monkeypatch)

    tune.main()

    assert fits == [0, 1]
    assert [call for call, _tags in recorded] == [
        "run",
        "set_tags",
        "set_tags",
        "run_failed",
        "run",
        "set_tags",
        "set_tags",
    ]
    assert math.isnan(rows[0])
    assert rows[1:] == [1.0]


def test_a_trial_that_dies_in_setup_still_gets_a_failed_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A config a model constructor rejects dies before `fit` ever starts,
    but every param and tag a run opens with comes from `config`/`args`
    alone, so it must still get the run a `fit` failure gets — FAILED, not
    silently missing — and the NaN row, with the sweep going on to the next
    trial."""
    configs = [
        ModelConfig(model_class="NERClassificationModel"),
        ModelConfig(model_class="NERClassificationModel"),
    ]
    builds: list[int] = []

    def dies_on_first_build(*_args, **_kwargs):
        builds.append(len(builds))
        if len(builds) == 1:
            raise _TrialDied
        return _Model()

    recorded: list[tuple[str, dict[str, str]]] = []
    stub_tune(
        monkeypatch,
        _Model(),
        _EagerFallbackTrainer,
        recorded,
        configs=configs,
    )
    monkeypatch.setattr(tune.factory, "build_model", dies_on_first_build)
    rows = _recording_log_config(monkeypatch)

    tune.main()

    assert builds == [0, 1]
    assert [call for call, _tags in recorded] == [
        "run",
        "run_failed",
        "run",
        "set_tags",
        "set_tags",
    ]
    assert math.isnan(rows[0])
    assert rows[1:] == [1.0]


def test_a_sweep_where_every_trial_failed_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sweep that produced no result must not end like one that did:
    silence read as a clean sweep is the trap."""
    configs = [
        ModelConfig(model_class="NERClassificationModel"),
        ModelConfig(model_class="NERClassificationModel"),
    ]
    recorded: list[tuple[str, dict[str, str]]] = []
    stub_tune(monkeypatch, _Model(), _DyingTrainer, recorded, configs=configs)
    rows = _recording_log_config(monkeypatch)

    with pytest.raises(SystemExit) as exc_info:
        tune.main()

    assert exc_info.value.code not in (None, 0)
    assert len(rows) == 2 and all(math.isnan(row) for row in rows)


def test_an_interrupt_still_ends_the_sweep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ctrl-C is the operator ending the sweep, not a trial failing: it must
    not be logged as one failure and followed by the next trial."""
    configs = [
        ModelConfig(model_class="NERClassificationModel"),
        ModelConfig(model_class="NERClassificationModel"),
    ]

    class _InterruptedTrainer(_EagerFallbackTrainer):
        def fit(self, **_kwargs: object) -> None:
            raise KeyboardInterrupt

    recorded: list[tuple[str, dict[str, str]]] = []
    stub_tune(
        monkeypatch, _Model(), _InterruptedTrainer, recorded, configs=configs
    )
    rows = _recording_log_config(monkeypatch)

    with pytest.raises(KeyboardInterrupt):
        tune.main()

    assert rows == []
    assert [call for call, _tags in recorded if call == "run"] == ["run"]


def test_a_negative_limit_is_refused_at_the_command_line(monkeypatch, capsys):
    """`--limit -1` would otherwise surface as a `ValueError` out of the
    corpus loader, far from the flag that caused it."""
    monkeypatch.setattr(
        sys, "argv", ["tuning", "config.toml", "out.csv", "--limit", "-1"]
    )

    with pytest.raises(SystemExit) as exc_info:
        tune.command_line_args()

    assert exc_info.value.code == 2
    assert "--limit" in capsys.readouterr().err


def test_a_trial_releases_its_model_before_the_next_one_builds(monkeypatch):
    """A sweep runs every trial in one process, so a trial still holding its
    model while the next allocates puts two in memory at once — which on a
    unified-memory device ends the sweep at the kernel OOM killer, with no
    message to assert on. The eager fallback leaves a cycle on the model, so
    the release only holds if the cycle collector runs.
    """
    configs = [
        ModelConfig(model_class="NERClassificationModel"),
        ModelConfig(model_class="NERClassificationModel"),
    ]
    live: list[weakref.ref[_Model]] = []
    earlier_trials_were_released: list[bool] = []

    def build_model(*_args, **_kwargs):
        earlier_trials_were_released.append(all(ref() is None for ref in live))
        model = _Model()
        live.append(weakref.ref(model))
        return model

    recorded: list[tuple[str, dict[str, str]]] = []
    stub_tune(
        monkeypatch,
        _Model(),
        _EagerFallbackTrainer,
        recorded,
        configs=configs,
    )
    # `stub_tune`'s own stub closes over one model, keeping it alive.
    monkeypatch.setattr(tune.factory, "build_model", build_model)

    tune.main()

    assert earlier_trials_were_released == [True, True]
    assert all(ref() is None for ref in live)
