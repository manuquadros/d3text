"""Tracking must be invisible when off and harmless when it breaks."""

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
