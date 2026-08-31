"""Optional MLflow experiment tracking.

Every entry point here is a **no-op unless ``MLFLOW_TRACKING_URI`` is set**, so
importing this module — or calling it from the training loop — changes nothing
for tests, notebooks, or a run on a machine with no tracking server. Opting in
is one environment variable:

.. code-block:: bash

   export MLFLOW_TRACKING_URI=http://127.0.0.1:5000   # must be http(s)

The variable, rather than a config key, is what selects tracking because the
tracking server is a property of the *machine* the run happens on, exactly like
the torch flavour — the same ``config.toml`` has to work on the VM that has a
server and on the laptop that does not. It has to name an ``http(s)://``
server: the dependency is ``mlflow-skinny``, which ships no local store
backend.

This module is a **leaf** but for ``d3text.metric_docs``, which is itself one
(no mlflow, no torch, no data layer); ``mlflow`` is imported only on first use. That is what lets ``models.py`` log without dragging
a tracking client into every import of the package.

Tracking never propagates a failure into the run. A server that is down, an
expired token, or a client too old for the API disables tracking for the rest
of the process with a single warning; a multi-hour training run must not die
because a metric could not be posted.
"""

from __future__ import annotations

import functools
import os
import pathlib
import platform
import subprocess
import warnings
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from types import ModuleType
from typing import Any

from d3text import metric_docs

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


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    # Asked of the *package's* directory, not the cwd: an editable install puts
    # this file in the checkout that produced the run, while the cwd is
    # wherever the operator happened to launch from.
    return subprocess.run(
        ("git", "-C", str(pathlib.Path(__file__).resolve().parent), *args),
        capture_output=True,
        text=True,
        timeout=10,
    )


@functools.cache
def git_commit() -> str | None:
    """The short hash the run was launched from, ``-dirty`` if it was edited.

    `None` when the answer would be a guess: no git, no repository (a
    non-editable install into site-packages), or a detached/empty HEAD.

    The dirty check is `git diff --quiet HEAD`, which compares **tracked**
    files only. `git status --porcelain` would be wrong here: `CLAUDE.md`,
    `design/` and `ncbitax/` live in the tree untracked and un-ignored on
    purpose, so it would report every run as dirty and the flag would stop
    meaning anything.
    """
    try:
        head = _git("rev-parse", "--short", "HEAD")
        if head.returncode != 0:
            return None
        commit = head.stdout.strip()
        if not commit:
            return None
        return (
            commit
            if _git("diff", "--quiet", "HEAD").returncode == 0
            else f"{commit}-dirty"
        )
    except (OSError, subprocess.SubprocessError):
        return None


def stamped(name: str) -> str:
    """`name` with the short commit appended, when one can be determined.

    The run name is the only column always visible in a run list, so the
    commit goes there as well as into the tags — scanning a sweep for "which
    of these ran before the pooling change" should not need a click per run.
    """
    commit = git_commit()
    return f"{name}@{commit}" if commit else name


def default_experiment_name() -> str:
    """The experiment to use when `MLFLOW_EXPERIMENT_NAME` is unset.

    `DEFAULT_EXPERIMENT` suffixed with the short commit, so runs from
    different code auto-namespace into different experiments rather than
    piling into one. Falls back to the bare `DEFAULT_EXPERIMENT` when no
    commit can be determined (a non-editable install, no repository), the
    same condition under which `stamped()` and `provenance_tags()` fall back.
    Setting `MLFLOW_EXPERIMENT_NAME` still overrides this outright, for a
    sweep that wants every trial in one place regardless of commit.
    """
    commit = git_commit()
    return f"{DEFAULT_EXPERIMENT}_{commit}" if commit else DEFAULT_EXPERIMENT


def provenance_tags(model: str, base_model: str) -> dict[str, str]:
    """What was trained, from which code — as tags rather than params.

    Both names are already in the params via `ModelConfig.model_dump()`, but
    a param is one click deep. These are the questions asked while *scanning*
    a run list, so they also go where they can be shown as columns and
    filtered on (`tags.model = "ETEBrendaModel"`).
    """
    tags = {"model": model, "base_model": base_model}
    commit = git_commit()
    if commit is not None:
        tags["git_commit"] = commit

    return tags


def environment_tags() -> dict[str, str]:
    """The machine and torch build the run happened on.

    A sweep is normally spread over the machines that were free — a P100 VM, an
    RTX Ada box, a laptop on CPU — and the accelerator is what explains a run
    that is three times slower, or that differs numerically, from the run
    beside it in the list. `torch.__version__` carries the flavour suffix
    (`+cu128`, `+rocm…`, bare for CPU), which is the same thing `TORCH_FLAVOUR`
    selected at lock time.

    `torch` is imported inside the function so this module stays a leaf: a
    caller that never asks for tags pays nothing, exactly as with mlflow.
    """
    tags = {"host": platform.node()}
    try:
        import torch
    except ImportError:
        return tags

    tags["torch"] = str(torch.__version__)
    if torch.cuda.is_available():
        # `get_device_name(0)` also answers for a ROCm build, which reports
        # itself through the CUDA API.
        tags["accelerator"] = torch.cuda.get_device_name(0)
        tags["accelerator_count"] = str(torch.cuda.device_count())
    else:
        tags["accelerator"] = "cpu"

    return tags


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


def log_text(text: str, artifact_file: str) -> None:
    """Store a block of text — a classification report — as a run artifact.

    A per-class table is not a metric: it has one row per label and is read
    whole, once, when a micro-average turns out to hide something. Writing it
    beside the metrics keeps the run self-contained, rather than in a terminal
    scrollback that outlives nothing.
    """
    mlflow = _module()
    if mlflow is None or not text:
        return
    try:
        mlflow.log_text(text, artifact_file)
    except Exception as exc:
        _disable(f"could not log text to {artifact_file!r} ({exc})")


def set_description(text: str) -> None:
    """Post `text` as the run's description, which MLflow renders as Markdown.

    It is written as the `mlflow.note.content` tag because that is the only
    free-text field the UI shows on the run page itself. The metric glossary
    goes here: MLflow charts a metric under its key and offers nowhere to
    record what the y-axis measures or in what unit, so the units have to
    travel beside the charts rather than inside them.
    """
    mlflow = _module()
    if mlflow is None or not text:
        return
    try:
        mlflow.set_tag("mlflow.note.content", text)
    except Exception as exc:
        _disable(f"could not set the run description ({exc})")


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
            os.environ.get(EXPERIMENT_VAR) or default_experiment_name()
        )
        mlflow.start_run(run_name=name, tags=dict(tags) if tags else None)
    except Exception as exc:
        _disable(f"could not start a run ({exc})")
        yield
        return

    stage = (tags or {}).get("stage")
    if stage is not None:
        set_description(metric_docs.glossary(stage))

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
