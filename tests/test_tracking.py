"""Tracking must be invisible when off and harmless when it breaks."""

import subprocess
import sys
import types
import warnings
from typing import Any

import pytest
from d3text import tracking
from d3text.models.models import Step, print_epoch_stats


@pytest.fixture(autouse=True)
def reset_tracking(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear the module's memoised state between tests.

    `_mlflow` and `_disabled` are process-globals by design (the import and the
    give-up decision must happen once per run, not once per call), so a test
    that disables tracking would otherwise disable it for the whole session.
    """
    monkeypatch.delenv(tracking.TRACKING_URI_VAR, raising=False)
    monkeypatch.delenv(tracking.EXPERIMENT_VAR, raising=False)
    monkeypatch.setattr(tracking, "_mlflow", None)
    monkeypatch.setattr(tracking, "_disabled", False)
    tracking.git_commit.cache_clear()


def fake_mlflow() -> types.ModuleType:
    """A stand-in recording every call `tracking` makes into it."""
    module = types.ModuleType("mlflow")
    calls: list[tuple[str, Any]] = []
    module.calls = calls

    def record(name: str):
        def call(*args: Any, **kwargs: Any) -> None:
            calls.append((name, (args, kwargs)))

        return call

    for name in (
        "set_experiment",
        "start_run",
        "end_run",
        "log_params",
        "log_metrics",
        "log_artifact",
    ):
        setattr(module, name, record(name))

    return module


def enable(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    module = fake_mlflow()
    monkeypatch.setenv(tracking.TRACKING_URI_VAR, "http://127.0.0.1:5000")
    monkeypatch.setattr(tracking, "_mlflow", module)
    return module


def test_disabled_without_tracking_uri() -> None:
    assert not tracking.enabled()


def test_mlflow_is_not_imported_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The no-op path must not pay for the import.

    `models.py` calls into this module twice an epoch; if `_module()` imported
    mlflow before checking the environment, every CPU test run and every
    notebook would carry the tracking client.
    """
    monkeypatch.delitem(sys.modules, "mlflow", raising=False)
    monkeypatch.setattr(
        tracking,
        "_disable",
        lambda reason: pytest.fail(f"tracking touched mlflow: {reason}"),
    )

    with tracking.run(name="x", params={"lr": 1.0}):
        tracking.log_metrics({"loss": 1.0}, step=0)
        tracking.log_artifact(__file__)

    assert "mlflow" not in sys.modules


def test_missing_mlflow_warns_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Asking for tracking and not getting it must not be silent."""
    monkeypatch.setenv(tracking.TRACKING_URI_VAR, "http://127.0.0.1:5000")
    # `None` in sys.modules is the documented way to make `import x` fail.
    monkeypatch.setitem(sys.modules, "mlflow", None)

    with pytest.warns(RuntimeWarning, match="mlflow is missing"):
        assert not tracking.enabled()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not tracking.enabled()


def test_run_forwards_params_metrics_and_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = enable(monkeypatch)

    with tracking.run(name="trial-000", params={"lr": 0.003}):
        tracking.log_metrics({"training/total": 2.5}, step=3)

    names = [name for name, _ in module.calls]
    assert names == [
        "set_experiment",
        "start_run",
        "log_params",
        "log_metrics",
        "end_run",
    ]

    by_name = dict(module.calls)
    assert by_name["set_experiment"][0] == (tracking.DEFAULT_EXPERIMENT,)
    assert by_name["start_run"][1]["run_name"] == "trial-000"
    assert by_name["log_params"][0] == ({"lr": 0.003},)
    assert by_name["log_metrics"] == (({"training/total": 2.5},), {"step": 3})
    assert by_name["end_run"][1] == {"status": "FINISHED"}


def test_run_marks_a_crash_as_failed_and_reraises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = enable(monkeypatch)

    with pytest.raises(ZeroDivisionError):
        with tracking.run(name="trial-000"):
            raise ZeroDivisionError

    assert dict(module.calls)["end_run"][1] == {"status": "FAILED"}


def test_a_broken_server_does_not_break_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The property the whole module exists for.

    A tracking server that dies mid-epoch must cost the run its metrics, not
    its remaining hours of training.
    """
    module = enable(monkeypatch)

    def explode(*args: Any, **kwargs: Any) -> None:
        raise ConnectionError("server went away")

    monkeypatch.setattr(module, "log_metrics", explode)

    with pytest.warns(RuntimeWarning, match="could not log metrics"):
        tracking.log_metrics({"training/total": 1.0}, step=0)

    # Disabled from here on, so the next epoch does not re-raise or re-warn.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tracking.log_metrics({"training/total": 1.0}, step=1)
        assert not tracking.enabled()


def test_experiment_name_is_overridable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = enable(monkeypatch)
    monkeypatch.setenv(tracking.EXPERIMENT_VAR, "sweep-2026-08")

    with tracking.run():
        pass

    assert dict(module.calls)["set_experiment"][0] == ("sweep-2026-08",)


def test_log_text_forwards_a_report_and_skips_an_empty_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = enable(monkeypatch)

    tracking.log_text("precision recall f1", "test/class_report.txt")
    tracking.log_text("", "test/relation_report.txt")

    assert [name for name, _ in module.calls] == ["log_text"]
    assert module.calls[0][1][0] == (
        "precision recall f1",
        "test/class_report.txt",
    )


def test_environment_tags_describe_the_machine() -> None:
    """These are read to explain a run that was slower, or numerically
    different, from the run beside it in the list."""
    tags = tracking.environment_tags()

    assert tags["host"]
    assert tags["torch"]
    # "cpu" or a device name; never absent, so the column is never blank.
    assert tags["accelerator"]


def test_environment_tags_survive_a_torch_free_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`tracking` is a leaf that imports torch lazily; a caller without it
    still gets the machine it ran on rather than an ImportError."""
    monkeypatch.setitem(sys.modules, "torch", None)

    tags = tracking.environment_tags()

    assert tags == {"host": tags["host"]}


def test_print_epoch_stats_returns_what_it_prints() -> None:
    """The metrics logged per epoch are the printed averages, not a re-derivation."""
    stats = print_epoch_stats(
        losses={"entity": 6.0, "class": 4.0},
        denominator=2,
        step=Step.TRAINING,
    )

    assert stats == {
        "training/entity": 3.0,
        "training/class": 2.0,
        "training/total": 5.0,
    }


def test_git_commit_reports_the_working_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hash comes from the checkout the *package* lives in, not the cwd."""
    recorded: list[tuple[str, ...]] = []

    def fake_git(*args: str) -> subprocess.CompletedProcess[str]:
        recorded.append(args)
        if args[0] == "rev-parse":
            return subprocess.CompletedProcess(args, 0, "a1b2c3d\n", "")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(tracking, "_git", fake_git)
    assert tracking.git_commit() == "a1b2c3d"
    assert recorded[0] == ("rev-parse", "--short", "HEAD")


def test_git_commit_marks_a_dirty_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run from an edited tree is not reproducible from its hash alone.

    The check must be `diff --quiet HEAD` — tracked files only. This repo
    keeps `CLAUDE.md`, `design/` and `ncbitax/` untracked and un-ignored on
    purpose, so a `status --porcelain` check would call every run dirty.
    """

    def fake_git(*args: str) -> subprocess.CompletedProcess[str]:
        if args[0] == "rev-parse":
            return subprocess.CompletedProcess(args, 0, "a1b2c3d\n", "")
        assert args == ("diff", "--quiet", "HEAD")
        return subprocess.CompletedProcess(args, 1, "", "")

    monkeypatch.setattr(tracking, "_git", fake_git)
    assert tracking.git_commit() == "a1b2c3d-dirty"


@pytest.mark.parametrize(
    "failure",
    [
        subprocess.CompletedProcess(("rev-parse",), 128, "", "not a git repo"),
        subprocess.CompletedProcess(("rev-parse",), 0, "\n", ""),
    ],
    ids=["no-repository", "empty-head"],
)
def test_git_commit_is_none_when_it_would_be_a_guess(
    monkeypatch: pytest.MonkeyPatch,
    failure: subprocess.CompletedProcess[str],
) -> None:
    monkeypatch.setattr(tracking, "_git", lambda *args: failure)
    assert tracking.git_commit() is None


def test_git_commit_survives_a_missing_git(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(*args: str) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("git")

    monkeypatch.setattr(tracking, "_git", explode)
    assert tracking.git_commit() is None


def test_provenance_reaches_the_run_name_and_tags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = enable(monkeypatch)
    monkeypatch.setattr(tracking, "git_commit", lambda: "a1b2c3d")

    with tracking.run(
        name=tracking.stamped("trial-000"),
        tags={
            "stage": "tuning",
            **tracking.provenance_tags(
                "ETEBrendaModel", "michiyasunaga/BioLinkBERT-base"
            ),
        },
    ):
        pass

    start = dict(module.calls)["start_run"][1]
    assert start["run_name"] == "trial-000@a1b2c3d"
    assert start["tags"] == {
        "stage": "tuning",
        "model": "ETEBrendaModel",
        "base_model": "michiyasunaga/BioLinkBERT-base",
        "git_commit": "a1b2c3d",
    }


def test_provenance_omits_an_unknowable_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-editable install has no repo; the run is still worth tracking."""
    monkeypatch.setattr(tracking, "git_commit", lambda: None)

    assert tracking.stamped("trial-000") == "trial-000"
    assert "git_commit" not in tracking.provenance_tags("M", "base")
