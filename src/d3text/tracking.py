"""Optional MLflow experiment tracking.

Every entry point here is a no-op unless `MLFLOW_TRACKING_URI` names an
`http(s)://` server. mlflow and torch are imported only on first use, and a
tracking failure disables tracking behind one warning, never ends the run.
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
from typing import TYPE_CHECKING, Any

from d3text import metric_docs
from d3text.constraints import NonNegative

if TYPE_CHECKING:
    import torch

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
    """The `mlflow` module, or `None` when tracking is off.

    :return: the module, or None, which is the ordinary case rather than an
        error.
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
    """Whether metrics logged from this process will reach a tracking server.

    :return: whether tracking is on.
    """
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
    """The short hash the run was launched from, `-dirty` if it was edited.

    The dirty check compares tracked files only: `git status --porcelain` would
    report every run as dirty, since several files live in the tree untracked
    and un-ignored on purpose.

    :return: the hash, or None when the answer would be a guess — no git, no
        repository, or an empty HEAD. A detached HEAD still yields a hash.
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


@functools.cache
def git_describe() -> str | None:
    """The nearest release tag, the commits since it, and the hash.

    The citable half of provenance: `v0.2.0-12-gabc1234`. Matched against
    `v[0-9]*` so a tag that is not a release cannot become the anchor.

    :return: the description, or None when there is no release tag to describe
        against, no git, or no repository.
    """
    try:
        out = _git("describe", "--tags", "--dirty", "--match", "v[0-9]*")
        if out.returncode != 0:
            return None
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def stamped(name: str) -> str:
    """`name` with the short commit appended, when one can be determined.

    The run name is the only column always visible in a run list, so the commit
    goes there as well as into the tags.

    :param name: the run name to stamp.
    :return: the stamped name.
    """
    commit = git_commit()
    return f"{name}@{commit}" if commit else name


def default_experiment_name() -> str:
    """The experiment to use when `MLFLOW_EXPERIMENT_NAME` is unset.

    Suffixed with the short commit, so runs from different code auto-namespace
    rather than piling into one experiment.

    :return: the experiment name.
    """
    commit = git_commit()
    return f"{DEFAULT_EXPERIMENT}_{commit}" if commit else DEFAULT_EXPERIMENT


def provenance_tags(model: str, base_model: str) -> dict[str, str]:
    """What was trained, from which code — as tags rather than params.

    Both names are already in the params, but a param is one click deep and
    these are the questions asked while *scanning* a run list.

    :param model: the model class's name.
    :param base_model: the frozen transformer's name.
    :return: the tags to set.
    """
    tags = {"model": model, "base_model": base_model}
    commit = git_commit()
    if commit is not None:
        tags["git_commit"] = commit
    described = git_describe()
    if described is not None:
        tags["git_describe"] = described

    return tags


def _machine_tags(base_model: str) -> dict[str, str]:
    """The `config.toml` settings a run's numbers or duration depend on.

    `config.toml` is untracked, so the run is the only record of it; paths go
    in as whether they were set. `models.config` is imported here because it
    imports torch, which this module must not at import time.

    :param base_model: the run's base model, since `embeddings_store` is
        keyed by it — a store configured for a different model is unset here.
    :return: the tags to set.
    """
    from d3text.models.config import machine_config

    settings = machine_config()
    return {
        "float32_matmul_precision": settings.float32_matmul_precision,
        "cudnn_allow_tf32": str(settings.cudnn_allow_tf32),
        "expandable_segments": str(settings.expandable_segments),
        "tokenizers_parallelism": str(settings.tokenizers_parallelism),
        "cpu_embeddings_cache_mb": str(settings.cpu_embeddings_cache_mb),
        "embeddings_store": str(base_model in settings.embeddings_store),
        "linking_corpora": str(settings.linking_corpora is not None),
    }


def _coverage_tags(served: NonNegative, asked: NonNegative) -> dict[str, str]:
    """What an embeddings store answered this run, written as it closes.

    A configured store may still not serve the run; coverage says how much
    it carried, which decides whether two runs are comparable. No lookups
    reads as coverage 0: every embedding was computed by the base model.

    :param served: documents this run read from a store.
    :param asked: documents this run looked up in one.
    :return: the tags to set.
    """
    return {
        "embeddings_store_lookups": str(asked),
        "embeddings_store_coverage": (
            f"{served / asked:.4f}" if asked else "0.0000"
        ),
    }


def environment_tags(base_model: str) -> dict[str, str]:
    """The machine, its `config.toml` settings, and the torch build.

    The accelerator is what explains a run three times slower than the one
    beside it. `torch` is imported inside the function so this module stays a
    leaf.

    :param base_model: the run's base model, forwarded to `_machine_tags`.
    :return: the tags to set.
    """
    tags = {"host": platform.node(), **_machine_tags(base_model)}
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

    :param params: whatever `ModelConfig.model_dump()` produces; MLflow stores
        every value as its string repr, so lists and enums need no conversion.
    """
    mlflow = _module()
    if mlflow is None or not params:
        return
    try:
        mlflow.log_params(dict(params))
    except Exception as exc:
        _disable(f"could not log parameters ({exc})")


def log_metrics(
    metrics: Mapping[str, float], step: NonNegative | None = None
) -> None:
    """Record metrics on the active run.

    :param metrics: the values to log.
    :param step: the epoch number.
    """
    mlflow = _module()
    if mlflow is None or not metrics:
        return
    try:
        mlflow.log_metrics(dict(metrics), step=step)
    except Exception as exc:
        _disable(f"could not log metrics ({exc})")


def log_artifact(path: str | os.PathLike[str]) -> None:
    """Upload a file — a checkpoint, a config, a results CSV — to the run.

    :param path: the file to upload.
    """
    mlflow = _module()
    if mlflow is None:
        return
    try:
        mlflow.log_artifact(str(path))
    except Exception as exc:
        _disable(f"could not log artifact {path!r} ({exc})")


def register_model(
    model: torch.nn.Module,
    sidecar: str | os.PathLike[str],
    name: str,
    tags: Mapping[str, str],
) -> None:
    """Log `model` as an MLflow PyTorch model and register a version of it.

    The weights travel once, in the pickled module; `sidecar` rides along as
    an extra file carrying what interprets them. The version number is
    MLflow's own counter, so which release it came from goes in `tags`.

    :param model: the trained model, holding the parameters to register.
    :param sidecar: a checkpoint written with an empty state dict.
    :param name: the registered model to add a version to, created if absent.
    :param tags: tags to set on the new model version.
    """
    mlflow = _module()
    if mlflow is None:
        return
    try:
        # Pickle, not the default `pt2`: `torch.export` needs a traceable
        # forward and an input example, and the models take a batch dict.
        info = mlflow.pytorch.log_model(
            model,
            name="model",
            extra_files=[str(sidecar)],
            serialization_format="pickle",
        )
        mlflow.register_model(info.model_uri, name, tags=dict(tags))
    except Exception as exc:
        _disable(f"could not register model {name!r} ({exc})")


def log_text(text: str, artifact_file: str) -> None:
    """Store a block of text — a classification report — as a run artifact.

    A per-class table is not a metric: it is read whole, once, when a
    micro-average turns out to hide something.

    :param text: the text to store.
    :param artifact_file: the name to store it under.
    """
    mlflow = _module()
    if mlflow is None or not text:
        return
    try:
        mlflow.log_text(text, artifact_file)
    except Exception as exc:
        _disable(f"could not log text to {artifact_file!r} ({exc})")


def set_tags(tags: Mapping[str, str]) -> None:
    """Set tags on the active run, overwriting whatever is already there.

    :param tags: the tags to set.
    """
    mlflow = _module()
    if mlflow is None or not tags:
        return
    try:
        mlflow.set_tags(dict(tags))
    except Exception as exc:
        _disable(f"could not set tags ({exc})")


def set_description(text: str) -> None:
    """Post `text` as the run's description, which MLflow renders as Markdown.

    Written as the `mlflow.note.content` tag, the only free-text field the UI
    shows on the run page itself, which is where the metric glossary has to go.

    :param text: the Markdown to post.
    """
    mlflow = _module()
    if mlflow is None or not text:
        return
    try:
        mlflow.set_tag("mlflow.note.content", text)
    except Exception as exc:
        _disable(f"could not set the run description ({exc})")


def active_run_id() -> str | None:
    """The id of the tracking run in progress, for reopening it later.

    :return: the id, or None when tracking is off or no run is active.
    """
    mlflow = _module()
    if mlflow is None:
        return None
    active = mlflow.active_run()
    return None if active is None else str(active.info.run_id)


@contextmanager
def run(
    name: str | None = None,
    params: Mapping[str, Any] | None = None,
    tags: Mapping[str, str] | None = None,
    run_id: str | None = None,
) -> Iterator[None]:
    """Scope a tracking run around a block, or do nothing if tracking is off.

    The run is closed as `FAILED` when the block raises, so a crashed run is
    distinguishable from one that merely stopped early; the exception is
    re-raised untouched either way.

    :param name: the run's name, passed to MLflow as given. Ignored when
        `run_id` is given.
    :param params: hyperparameters to record.
    :param tags: tags to set on the run.
    :param run_id: an existing run to reopen instead of starting a new one.
        It stays in its own experiment: MLflow refuses to reopen a run while
        a different experiment is active.
    """
    mlflow = _module()
    if mlflow is None:
        yield
        return

    try:
        if run_id is None:
            mlflow.set_experiment(
                os.environ.get(EXPERIMENT_VAR) or default_experiment_name()
            )
            mlflow.start_run(run_name=name, tags=dict(tags) if tags else None)
        else:
            mlflow.start_run(run_id=run_id, tags=dict(tags) if tags else None)
    except Exception as exc:
        _disable(f"could not start a run ({exc})")
        yield
        return

    stage = (tags or {}).get("stage")
    if stage is not None:
        set_description(metric_docs.glossary(stage))

    log_params(params or {})

    # Lazy: `embeddings_store` imports torch. The counters are process-wide
    # and never reset, so this run's share is the difference across its scope,
    # written before `end_run` since the store closes only at `atexit`.
    from d3text.embeddings_store import lookup_totals

    served_before, asked_before = lookup_totals()

    status = "FINISHED"
    try:
        yield
    except BaseException:
        status = "FAILED"
        raise
    finally:
        served, asked = lookup_totals()
        set_tags(_coverage_tags(served - served_before, asked - asked_before))
        try:
            mlflow.end_run(status=status)
        except Exception as exc:
            _disable(f"could not close the run ({exc})")
