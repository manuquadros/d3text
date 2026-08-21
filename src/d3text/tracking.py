"""Optional MLflow experiment tracking.

Every entry point here is a **no-op unless ``MLFLOW_TRACKING_URI`` is set**, so
importing this module — or calling it from the training loop — changes nothing
for tests, notebooks, or a run on a machine with no tracking server. Opting in
is one environment variable:

.. code-block:: bash

   export MLFLOW_TRACKING_URI=http://127.0.0.1:5000   # or file:./mlruns

The variable, rather than a config key, is what selects tracking because the
tracking server is a property of the *machine* the run happens on, exactly like
the torch flavour — the same ``config.toml`` has to work on the VM that has a
server and on the laptop that does not.

This module is a **leaf**: it imports nothing from ``d3text``, and ``mlflow``
itself only on first use. That is what lets ``models.py`` log without dragging
a tracking client into every import of the package.

Tracking never propagates a failure into the run. A server that is down, an
expired token, or a client too old for the API disables tracking for the rest
of the process with a single warning; a multi-hour training run must not die
because a metric could not be posted.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from types import ModuleType
from typing import Any

TRACKING_URI_VAR = "MLFLOW_TRACKING_URI"
EXPERIMENT_VAR = "MLFLOW_EXPERIMENT_NAME"
DEFAULT_EXPERIMENT = "d3text"

_mlflow: ModuleType | None = None
_disabled = False


def _disable(reason: str) -> None:
    global _disabled
    _disabled = True
    warnings.warn(
        f"MLflow tracking disabled for this process: {reason}",
        RuntimeWarning,
        stacklevel=3,
    )


def _module() -> Any | None:
    """Return the ``mlflow`` module, or ``None`` when tracking is off.

    ``None`` is the ordinary case (no ``MLFLOW_TRACKING_URI``), not an error;
    callers treat it as "skip".
    """
    global _mlflow

    if _disabled or not os.environ.get(TRACKING_URI_VAR):
        return None

    if _mlflow is None:
        try:
            import mlflow
        except ImportError as exc:
            # The variable is set, so tracking was asked for: say why it is
            # not happening instead of silently dropping every metric.
            _disable(f"{TRACKING_URI_VAR} is set but mlflow is missing ({exc})")
            return None
        _mlflow = mlflow

    return _mlflow


def enabled() -> bool:
    """Whether metrics logged from this process will reach a tracking server."""
    return _module() is not None


def log_params(params: Mapping[str, Any]) -> None:
    """Record hyperparameters on the active run.

    Values are whatever ``ModelConfig.model_dump()`` produces; MLflow stores
    every one as its string repr, so lists and enums need no conversion.
    """
    mlflow = _module()
    if mlflow is None or not params:
        return
    try:
        mlflow.log_params(dict(params))
    except Exception as exc:
        _disable(f"could not log parameters ({exc})")


def log_metrics(metrics: Mapping[str, float], step: int | None = None) -> None:
    """Record metrics on the active run, ``step`` being the epoch number."""
    mlflow = _module()
    if mlflow is None or not metrics:
        return
    try:
        mlflow.log_metrics(dict(metrics), step=step)
    except Exception as exc:
        _disable(f"could not log metrics ({exc})")


def log_artifact(path: str | os.PathLike[str]) -> None:
    """Upload a file — a checkpoint, a config, a results CSV — to the run."""
    mlflow = _module()
    if mlflow is None:
        return
    try:
        mlflow.log_artifact(str(path))
    except Exception as exc:
        _disable(f"could not log artifact {path!r} ({exc})")


@contextmanager
def run(
    name: str | None = None,
    params: Mapping[str, Any] | None = None,
    tags: Mapping[str, str] | None = None,
) -> Iterator[None]:
    """Scope a tracking run around a block, or do nothing if tracking is off.

    The run is closed as ``FAILED`` when the block raises, so a crashed
    training run is distinguishable in the UI from one that merely stopped
    early — and the exception is re-raised untouched either way.
    """
    mlflow = _module()
    if mlflow is None:
        yield
        return

    try:
        mlflow.set_experiment(
            os.environ.get(EXPERIMENT_VAR) or DEFAULT_EXPERIMENT
        )
        mlflow.start_run(run_name=name, tags=dict(tags) if tags else None)
    except Exception as exc:
        _disable(f"could not start a run ({exc})")
        yield
        return

    log_params(params or {})

    status = "FINISHED"
    try:
        yield
    except BaseException:
        status = "FAILED"
        raise
    finally:
        try:
            mlflow.end_run(status=status)
        except Exception as exc:
            _disable(f"could not close the run ({exc})")
