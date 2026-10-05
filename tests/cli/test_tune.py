"""What `tune` records about a trial: its config dump and its tags.

`pprint.pp` (the old call) hardcodes `sort_dicts=False`; `pprint.pformat`
(what it was replaced with) defaults `sort_dicts` to `True`. Left at that
default the dump prints alphabetically instead of in `ModelConfig` field
order, which is what a reader expects when comparing it against the TOML.
"""

import argparse
import contextlib
import math
import pathlib
import sys
import types
import weakref

import pytest
import torch

from d3text import utils
from d3text.cli import tune
from d3text.models.base import Model
from d3text.models.config import ModelConfig, load_model_config
from d3text.token_labels import TokenizerStamp
from d3text.training.trainer import Trainer
from torch.utils.data import DataLoader


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
            config="unused.toml", output="unused-results.csv", limit=None
        ),
    )
    monkeypatch.setattr(
        tune,
        "load_tuning_config",
        lambda path, **_kwargs: [
            ModelConfig(model_class="NERClassificationModel")
        ],
    )
    monkeypatch.setattr(tune, "encodings_path", lambda _model: "unused.hdf5")

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


def _stamp(digest):
    return TokenizerStamp(
        base_model="prajjwal1/bert-mini",
        digest=digest,
        window_length=512,
        window_stride=20,
    )


def test_a_trial_hands_the_label_stores_tokenizer_stamp_to_the_dataset(
    stop_after_config_dump, monkeypatch
):
    """The dataset build checks the stamp against the encodings store's
    digest; a trial that passes None instead skips that check silently."""
    stamp = _stamp("d" * 64)
    monkeypatch.setattr(
        tune.token_labels, "store_tokenizer_stamp", lambda _path: stamp
    )

    with pytest.raises(_StopAfterDump):
        tune.main()

    (call,) = stop_after_config_dump
    assert call["tokenizer"] is stamp


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


def test_logged_configs_retries_a_failed_trial(tmp_path):
    """A `NaN` score marks a trial that raised, often for an environment fault
    shared by every trial; excluding it would leave a resume after the fix
    unable to reach the configurations that failed."""
    output = tmp_path / "results.csv"
    failed = ModelConfig(model_class="NERClassificationModel", lr=1e-3)
    scored = ModelConfig(model_class="NERClassificationModel", lr=1e-4)
    tune.utils.log_config(str(output), failed, selection_score=float("nan"))
    tune.utils.log_config(str(output), scored, selection_score=1.0)

    assert tune._logged_configs(str(output)) == [scored]


def test_logged_configs_preserves_a_string_field_that_looks_like_a_literal(
    tmp_path,
):
    """`base_model` is a plain `str` field: a value that also parses as a
    Python literal (a bare digit string, here) must still come back as that
    string, not whatever type `ast.literal_eval` decides it names."""
    output = tmp_path / "results.csv"
    config = ModelConfig(
        model_class="NERClassificationModel",
        token_supervision=False,
        base_model="42",
    )
    tune.utils.log_config(str(output), config, selection_score=1.0)

    assert tune._logged_configs(str(output)) == [config]


def test_a_results_file_from_before_token_supervision_still_resumes(
    tmp_path, machine_stores
):
    """Its rows carry the store path, not the flag; read through the field
    list alone the column is dropped, every row reads as unsupervised and
    `ETEBrendaModel` refuses it, so the resume dies instead of excluding."""
    store = tmp_path / "labels.hdf5"
    machine_stores(token_labels_store={"prajjwal1/bert-mini": store})
    output = tmp_path / "results.csv"
    output.write_text(
        "model_class,base_model,token_labels_store\n"
        f"ETEBrendaModel,prajjwal1/bert-mini,{store}\n"
    )

    (config,) = tune._logged_configs(str(output))

    assert config.token_supervision is True


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
            config="unused.toml", output="unused-results.csv", limit=None
        ),
    )
    monkeypatch.setattr(
        tune, "load_tuning_config", lambda _path, **_kwargs: configs
    )
    monkeypatch.setattr(tune, "encodings_path", lambda _model: "unused.hdf5")
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
        tune.factory,
        "build_model_for_dataset",
        lambda *_args, **_kwargs: model,
    )
    monkeypatch.setattr(
        tune.factory, "dataset_metrics", lambda _dataset, _schema: {}
    )
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
    assert retags == [
        {"compiled": "true"},
        {"compiled": "false", "recompile_limit_hit": "false"},
    ]


class _RecompileLimitTrainer(_EagerFallbackTrainer):
    """Trains with the compiled trunk left in place; the first trial's frame
    hits dynamo's recompile limit, which dynamo counts and nothing raises."""

    fits = 0

    def fit(self, **_kwargs):
        from torch._dynamo.utils import counters

        if _RecompileLimitTrainer.fits == 0:
            counters["unimplemented"]["Dynamo recompile limit exceeded"] += 1
        _RecompileLimitTrainer.fits += 1


def test_each_trial_is_tagged_with_whether_it_hit_the_recompile_limit(
    monkeypatch, dynamo_counters
):
    """The process-wide counter survives `torch._dynamo.reset()` between
    trials, so a trial is tagged from its own delta: a sticky flag would mark
    every trial after the first."""
    monkeypatch.setattr(_RecompileLimitTrainer, "fits", 0)
    recorded: list[tuple[str, dict[str, str]]] = []
    stub_tune(
        monkeypatch,
        _Model(),
        _RecompileLimitTrainer,
        recorded,
        configs=[
            ModelConfig(model_class="NERClassificationModel"),
            ModelConfig(model_class="NERClassificationModel"),
        ],
    )

    tune.main()

    hits = [
        tags["recompile_limit_hit"]
        for call, tags in recorded
        if call == "set_tags" and "recompile_limit_hit" in tags
    ]
    assert hits == ["true", "false"]


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


def test_a_rebuilt_label_store_rebuilds_the_dataset_under_one_base_model(
    monkeypatch,
):
    """The stamp is part of the dataset cache's key: a dataset checked
    against one label store's tokenizer must not be handed to a trial reading
    a store built under another."""
    configs = [
        ModelConfig(model_class="NERClassificationModel", base_model="a"),
        ModelConfig(model_class="NERClassificationModel", base_model="a"),
    ]
    stamps = iter([_stamp("a" * 64), _stamp("b" * 64)])
    monkeypatch.setattr(
        tune.token_labels, "store_tokenizer_stamp", lambda _path: next(stamps)
    )
    built: list[TokenizerStamp | None] = []

    def counting_brenda_dataset(*, tokenizer, **_kwargs):
        built.append(tokenizer)
        return types.SimpleNamespace(data={"train": [], "val": []})

    stub_tune(
        monkeypatch,
        _Model(),
        _EagerFallbackTrainer,
        [],
        configs=configs,
        brenda_dataset=counting_brenda_dataset,
    )

    tune.main()

    assert [stamp.digest[0] for stamp in built] == ["a", "b"]


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
        {"compiled": "false", "recompile_limit_hit": "false"},
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
    monkeypatch.setattr(
        tune.factory, "build_model_for_dataset", dies_on_first_build
    )
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
    monkeypatch.setattr(tune.factory, "build_model_for_dataset", build_model)

    tune.main()

    assert earlier_trials_were_released == [True, True]
    assert all(ref() is None for ref in live)


def _sweep_scoring(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    configs: list[ModelConfig],
    scores: list[float],
) -> str:
    """Stub a sweep over `configs` whose trials score `scores` in order,
    writing real rows to a results CSV under `tmp_path`, and return its
    path."""
    output = str(tmp_path / "results.csv")
    pending = iter(scores)

    class _ScoringTrainer(_EagerFallbackTrainer):
        def fit(self, **kwargs: object) -> None:
            super().fit(**kwargs)
            self.best_selection_score = next(pending)

    log_config = utils.log_config
    stub_tune(monkeypatch, _Model(), _ScoringTrainer, [], configs=configs)
    monkeypatch.setattr(
        tune,
        "command_line_args",
        lambda: argparse.Namespace(
            config=str(tmp_path / "sweep.toml"), output=output, limit=None
        ),
    )
    monkeypatch.setattr(tune.utils, "log_config", log_config)
    return output


def test_a_sweep_keeps_its_best_configuration_beside_the_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """The best trial so far is written as a config `train` can take, so a
    sweep interrupted at any point leaves its winner ready to use. A later,
    worse trial must not overwrite it."""
    configs = [
        ModelConfig(model_class="NERClassificationModel", lr=0.1),
        ModelConfig(model_class="NERClassificationModel", lr=0.2),
        ModelConfig(model_class="NERClassificationModel", lr=0.3),
    ]
    _sweep_scoring(monkeypatch, tmp_path, configs, [0.5, 0.9, 0.1])

    tune.main()

    assert load_model_config(str(tmp_path / "results.toml")) == configs[1]


def test_a_resumed_sweep_measures_against_earlier_sessions(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A resume starts a fresh process, so the best must come from every row
    in the results file, not only the trials this session ran."""
    earlier = ModelConfig(model_class="NERClassificationModel", lr=0.4)
    fresh = ModelConfig(model_class="NERClassificationModel", lr=0.2)
    output = _sweep_scoring(monkeypatch, tmp_path, [fresh], [0.9])
    utils.log_config(output, earlier, selection_score=2.0)

    tune.main()

    assert load_model_config(str(tmp_path / "results.toml")) == earlier


def test_a_results_file_named_after_the_sweep_config_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """`tuning sweep.toml sweep.csv` is a natural pairing, and its best
    config would overwrite the sweep configuration itself."""
    trials: list[int] = []
    _sweep_scoring(
        monkeypatch,
        tmp_path,
        [ModelConfig(model_class="NERClassificationModel")],
        [1.0],
    )
    monkeypatch.setattr(
        tune,
        "command_line_args",
        lambda: argparse.Namespace(
            config=str(tmp_path / "sweep.toml"),
            output=str(tmp_path / "sweep.csv"),
            limit=None,
        ),
    )
    monkeypatch.setattr(
        tune, "load_tuning_config", lambda *_a, **_k: trials.append(0) or []
    )

    with pytest.raises(SystemExit, match="sweep configuration"):
        tune.main()

    assert trials == []


def test_a_stale_results_header_is_refused_before_any_trial_trains(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """The columns a row carries are known from `ModelConfig` before
    training, so a results file under a different column set must not cost
    a trial that is then reported as a model failure."""
    config = ModelConfig(model_class="NERClassificationModel")
    output = _sweep_scoring(monkeypatch, tmp_path, [config], [1.0])
    columns = [*config.model_dump(), "selection_score"]
    (tmp_path / "results.csv").write_text(",".join(columns[1:]) + "\n")
    trained: list[str] = []
    monkeypatch.setattr(
        tune.factory,
        "build_model_for_dataset",
        lambda *_a, **_k: trained.append("built"),
    )

    with pytest.raises(SystemExit, match="header does not match"):
        tune.main()

    assert trained == []
    assert output.endswith("results.csv")


class _NoisyModel(Model):
    """A one-`Linear` `Model` the real `Trainer` can drive, drawing from
    torch's global RNG each epoch and scoring off its own weights, so a trial
    whose restored state or RNG differed would log a different row."""

    default_selection_metrics = ("class_micro_f1",)

    def __init__(
        self, config: ModelConfig, epochs: list[tuple[float, int]], dies_at
    ) -> None:
        super().__init__(config=config, device="cpu")
        self.head = torch.nn.Linear(4, 1)
        self.epochs = epochs
        self.dies_at = dies_at

    def compile_trunk(self) -> bool:
        return False

    def trunk_is_compiled(self) -> bool:
        return False

    def run_epoch(self, data, epoch, update):
        if (self.config.lr, epoch) == self.dies_at:
            raise KeyboardInterrupt
        self.epochs.append((self.config.lr, epoch))
        update.zero_grad()
        loss = (self.head(torch.randn(2, 4)) - 1).square().mean()
        update(loss)
        return {"class": loss.detach().item()}, 1

    def evaluate_model(
        self, data, tau_cls=0.5, prefix="test", log_reports=True, step=None
    ):
        return {f"{prefix}/class_micro_f1": self.head.weight.sum().item()}


_RESUMED_CONFIGS = [
    ModelConfig(
        model_class="NERClassificationModel",
        num_epochs=4,
        patience=4,
        ramp_epochs=0,
        lr=lr,
    )
    for lr in (0.1, 0.05)
]


def _real_sweep(monkeypatch, output, run_ids, dies_at=None):
    """Run `tune.main` over `_RESUMED_CONFIGS` with the real `Trainer`,
    writing real rows to `output`; return the `(lr, epoch)`s it trained.

    `run_ids` collects the `run_id` each tracking run was opened with, and
    hands out a fresh id to each run opened without one."""
    epochs: list[tuple[float, int]] = []
    log_config = utils.log_config
    stub_tune(monkeypatch, _Model(), Trainer, [], configs=_RESUMED_CONFIGS)
    current: list[str] = []

    @contextlib.contextmanager
    def start_run(**kwargs):
        run_ids.append(kwargs.get("run_id"))
        current.append(kwargs.get("run_id") or f"run-{len(run_ids)}")
        yield

    monkeypatch.setattr(tune.tracking, "run", start_run)
    monkeypatch.setattr(tune.tracking, "active_run_id", lambda: current[-1])
    monkeypatch.setattr(
        tune,
        "load_tuning_config",
        lambda _path, excluded=(), **_kwargs: [
            config for config in _RESUMED_CONFIGS if config not in excluded
        ],
    )
    monkeypatch.setattr(
        tune,
        "command_line_args",
        lambda: argparse.Namespace(
            config="unused.toml", output=str(output), limit=None
        ),
    )
    monkeypatch.setattr(
        tune.data,
        "get_batch_loader",
        lambda **_kwargs: DataLoader([0], batch_size=1),
    )
    monkeypatch.setattr(
        tune.factory,
        "build_model_for_dataset",
        lambda config, *_a, **_k: _NoisyModel(config, epochs, dies_at),
    )
    monkeypatch.setattr(tune.utils, "log_config", log_config)
    tune.main()
    return epochs


def test_an_interrupted_trial_resumes_from_its_last_finished_epoch(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A sweep killed mid-trial and restarted trains only the epochs the
    trial had left, logs them to the trial's own tracking run, and ends with
    the rows an uninterrupted sweep writes; the resume file is gone after."""
    whole = tmp_path / "whole" / "results.csv"
    whole.parent.mkdir()
    _real_sweep(monkeypatch, whole, [])

    split = tmp_path / "split" / "results.csv"
    split.parent.mkdir()
    run_ids: list[str | None] = []
    with pytest.raises(KeyboardInterrupt):
        _real_sweep(monkeypatch, split, run_ids, dies_at=(0.1, 2))
    resumed = _real_sweep(monkeypatch, split, run_ids)

    assert resumed == [(0.1, 2), (0.1, 3)] + [(0.05, e) for e in range(4)]
    assert run_ids == [None, "run-1", None]
    assert tune._scored_configs(str(split)) == tune._scored_configs(str(whole))
    assert list(split.parent.glob("*.pt")) == []


def test_a_trial_killed_after_its_row_is_not_scored_twice(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A kill between a trial's results row and its resume file's removal
    leaves a scored trial's resume file behind; the restart must drop it, not
    chain the trial again and append the row a second time."""
    output = tmp_path / "results.csv"
    unlink = pathlib.Path.unlink
    killed: list[bool] = []

    def die_once(self, *args, **kwargs):
        if self.name.endswith(".trial.resume.pt") and not killed:
            killed.append(True)
            raise KeyboardInterrupt
        unlink(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(pathlib.Path, "unlink", die_once)
        with pytest.raises(KeyboardInterrupt):
            _real_sweep(monkeypatch, output, [])
    assert tune._resume_path(str(output)).exists()
    assert len(tune._scored_configs(str(output))) == 1

    run_ids: list[str | None] = []
    resumed = _real_sweep(monkeypatch, output, run_ids)

    assert resumed == [(0.05, e) for e in range(4)]
    assert [c.lr for c, _ in tune._scored_configs(str(output))] == [0.1, 0.05]
    assert run_ids == [None]
    assert list(output.parent.glob("*.pt")) == []
