"""Tracking must be invisible when off and harmless when it breaks."""

import os
import pathlib
import subprocess
import sys
import types
import warnings
from typing import Any

import lmdb
import pytest
import torch
from d3text import metric_docs, tracking
from d3text.embeddings_store import (
    EmbeddingsStore,
    StoreProvenance,
    tensor_to_bytes,
    write_provenance,
)

BASE_MODEL = "michiyasunaga/BioLinkBERT-base"


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
        "log_text",
        "set_tag",
        "set_tags",
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
        "set_tags",
        "end_run",
    ]

    by_name = dict(module.calls)
    assert by_name["set_experiment"][0] == (tracking.default_experiment_name(),)
    assert by_name["start_run"][1]["run_name"] == "trial-000"
    assert by_name["log_params"][0] == ({"lr": 0.003},)
    assert by_name["log_metrics"] == (({"training/total": 2.5},), {"step": 3})
    assert by_name["end_run"][1] == {"status": "FINISHED"}


def test_a_staged_run_carries_the_metric_glossary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MLflow has nowhere to put a metric's unit, so the run description
    carries the glossary — otherwise the y-axis of every chart is a guess."""
    module = enable(monkeypatch)

    with tracking.run(name="brenda-ete", tags={"stage": "train"}):
        pass

    by_name = dict(module.calls)
    key, note = by_name["set_tag"][0]
    assert key == "mlflow.note.content"
    assert note == metric_docs.glossary("train")
    assert "batches per second" in note


def test_a_run_with_no_stage_posts_no_description(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The glossary is per stage: a run that names none gets no table rather
    than the training one."""
    module = enable(monkeypatch)

    with tracking.run(name="brenda-ete", tags={"model": "ETEBrendaModel"}):
        pass

    assert "set_tag" not in [name for name, _ in module.calls]


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


def test_default_experiment_name_carries_the_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tracking, "git_commit", lambda: "a1b2c3d")
    assert tracking.default_experiment_name() == "d3text_a1b2c3d"


def test_default_experiment_name_falls_back_with_no_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tracking, "git_commit", lambda: None)
    assert tracking.default_experiment_name() == tracking.DEFAULT_EXPERIMENT


def test_run_uses_the_commit_experiment_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = enable(monkeypatch)
    monkeypatch.setattr(tracking, "git_commit", lambda: "a1b2c3d")

    with tracking.run():
        pass

    assert dict(module.calls)["set_experiment"][0] == ("d3text_a1b2c3d",)


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


def test_set_tags_forwards_a_tag_and_skips_an_empty_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = enable(monkeypatch)

    tracking.set_tags({"compiled": "false"})
    tracking.set_tags({})

    assert [name for name, _ in module.calls] == ["set_tags"]
    assert module.calls[0][1][0] == ({"compiled": "false"},)


def test_set_tags_is_silent_with_no_tracking_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`train` and `tune` retag the run after every `fit`, so this call is on
    the path of every untracked run as well."""
    monkeypatch.delitem(sys.modules, "mlflow", raising=False)
    monkeypatch.setattr(
        tracking,
        "_disable",
        lambda reason: pytest.fail(f"tracking touched mlflow: {reason}"),
    )

    tracking.set_tags({"compiled": "false"})

    assert "mlflow" not in sys.modules


def test_a_failed_set_tags_does_not_break_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The retag happens between the last epoch and the checkpoint write, so a
    server that died during training must cost the run its tag and not the
    weights it has yet to save."""
    module = enable(monkeypatch)

    def explode(*args: Any, **kwargs: Any) -> None:
        raise ConnectionError("server went away")

    monkeypatch.setattr(module, "set_tags", explode)

    with pytest.warns(RuntimeWarning, match="could not set tags"):
        tracking.set_tags({"compiled": "false"})

    assert not tracking.enabled()


def opened_store(
    path: pathlib.Path, documents: dict[int, torch.Tensor]
) -> EmbeddingsStore:
    """An LMDB stamped for `BASE_MODEL`, holding `documents`, open to read."""
    env = lmdb.open(str(path), map_size=8 * 1024**2)
    write_provenance(
        env, StoreProvenance(base_model=BASE_MODEL, max_length=512, stride=20)
    )
    with env.begin(write=True) as transaction:
        for pubmed_id, embedding in documents.items():
            transaction.put(str(pubmed_id).encode(), tensor_to_bytes(embedding))
    env.close()

    return EmbeddingsStore(path, BASE_MODEL)


def coverage_tags(module: types.ModuleType) -> dict[str, str]:
    """Every tag the run set through `set_tags`, merged."""
    merged: dict[str, str] = {}
    for name, (args, _) in module.calls:
        if name == "set_tags":
            merged.update(args[0])

    return merged


def test_a_run_records_the_share_the_embeddings_store_served_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A run off the store and a run that recomputed are not numerically
    comparable, so which one this was has to survive the terminal it was
    launched from: the counters are otherwise reported only to the log."""
    module = enable(monkeypatch)
    store = opened_store(tmp_path / "embeddings", {100: torch.rand(12, 8)})

    with tracking.run(name="trial-000"):
        store.get(100, expected_tokens=12)
        store.get(101, expected_tokens=12)

    assert coverage_tags(module)["embeddings_store_lookups"] == "2"
    assert coverage_tags(module)["embeddings_store_coverage"] == "0.5000"


def test_each_run_in_one_process_reports_only_its_own_lookups(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """`tune` opens a run per trial in one process and the store's counters
    never reset, so a run stamped with them raw would report every trial
    before it too — a number meaning something different in each row."""
    store = opened_store(tmp_path / "embeddings", {100: torch.rand(12, 8)})

    first = enable(monkeypatch)
    with tracking.run(name="trial-000"):
        store.get(100, expected_tokens=12)
        store.get(101, expected_tokens=12)

    second = enable(monkeypatch)
    with tracking.run(name="trial-001"):
        store.get(100, expected_tokens=12)

    assert coverage_tags(first)["embeddings_store_lookups"] == "2"
    assert coverage_tags(second)["embeddings_store_lookups"] == "1"
    assert coverage_tags(second)["embeddings_store_coverage"] == "1.0000"


def test_a_run_with_no_store_still_carries_its_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tags are what a run list is filtered on, so a run that recomputed
    everything has to say so rather than leave the column empty."""
    module = enable(monkeypatch)

    with tracking.run(name="trial-000"):
        pass

    assert coverage_tags(module) == {
        "embeddings_store_lookups": "0",
        "embeddings_store_coverage": "0.0000",
    }


@pytest.mark.parametrize(
    ("available", "expected"),
    [
        pytest.param(
            True,
            {"accelerator": "fake-gpu-0", "accelerator_count": "2"},
            id="cuda-available",
        ),
        pytest.param(
            False,
            {"accelerator": "cpu"},
            id="cuda-unavailable",
        ),
    ],
)
def test_environment_tags_describe_the_machine(
    monkeypatch: pytest.MonkeyPatch,
    available: bool,
    expected: dict[str, str],
) -> None:
    """Both `environment_tags` branches run stubbed, asserted by value."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda _index: "fake-gpu-0"
    )
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    tags = tracking.environment_tags("some/base-model")

    assert tags["host"]
    assert tags["torch"]
    for key, value in expected.items():
        assert tags[key] == value
    if not available:
        assert "accelerator_count" not in tags


@pytest.mark.gpu
def test_environment_tags_describe_a_real_accelerator() -> None:
    """Smoke check against an unstubbed install on a real CUDA device.

    Auto-skipped without one (see the `gpu` marker in conftest.py), so it
    never takes a CUDA context on a CPU-only run; it exists to catch a
    `torch.cuda` signature drift the stubbed test above cannot see.
    """
    tags = tracking.environment_tags("some/base-model")

    assert tags["host"]
    assert tags["torch"]
    assert tags["accelerator"] != "cpu"
    assert tags["accelerator_count"]


def test_environment_tags_survive_a_torch_free_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`tracking` is a leaf that imports torch lazily; a caller without it
    still gets the machine it ran on rather than an ImportError."""
    monkeypatch.setitem(sys.modules, "torch", None)

    tags = tracking.environment_tags("some/base-model")

    assert tags["host"]
    assert "torch" not in tags
    assert "accelerator" not in tags


def test_environment_tags_carry_the_untracked_machine_config() -> None:
    """`config.toml` is per-machine and never committed, so the run is the
    only record of the numerics it was launched under — and a path in it is
    recorded as whether it was set, not as this machine's directory layout."""
    tags = tracking.environment_tags("some/base-model")

    assert tags["float32_matmul_precision"]
    assert tags["cudnn_allow_tf32"] in {"True", "False"}
    assert tags["embeddings_store"] in {"True", "False"}
    assert tags["linking_corpora"] in {"True", "False"}


def test_git_commit_reports_the_working_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The `rev-parse --short HEAD` output becomes the reported hash."""
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


def _scratch_git_env() -> dict[str, str]:
    """A subprocess env with no inherited `GIT_*` and no real git config."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_AUTHOR_NAME"] = env["GIT_COMMITTER_NAME"] = "d3text tests"
    env["GIT_AUTHOR_EMAIL"] = env["GIT_COMMITTER_EMAIL"] = (
        "tests@example.invalid"
    )
    return env


@pytest.mark.parametrize(
    "detach", [False, True], ids=["branch-tip", "detached-head"]
)
def test_git_commit_anchors_on_the_package_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    detach: bool,
) -> None:
    """`_git`'s `-C` reads the package's repo, never an unrelated cwd."""
    repo = tmp_path / "repo"
    repo.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    env = _scratch_git_env()

    def run_git(*args: str) -> str:
        """Run git against the scratch repo and return trimmed stdout."""
        result = subprocess.run(
            (
                "git",
                "-C",
                str(repo),
                "-c",
                "commit.gpgsign=false",
                "-c",
                "init.defaultBranch=main",
                *args,
            ),
            capture_output=True,
            text=True,
            env=env,
            check=True,
            timeout=10,
        )
        return result.stdout.strip()

    run_git("init")
    (repo / "a.txt").write_text("one\n")
    run_git("add", "a.txt")
    run_git("commit", "-m", "first")
    older = run_git("rev-parse", "--short", "HEAD")

    (repo / "a.txt").write_text("two\n")
    run_git("add", "a.txt")
    run_git("commit", "-m", "second")
    tip = run_git("rev-parse", "--short", "HEAD")

    if detach:
        run_git("checkout", older)
    expected = older if detach else tip

    monkeypatch.setattr(tracking, "__file__", str(repo / "tracking.py"))
    monkeypatch.chdir(elsewhere)
    monkeypatch.delenv("GIT_DIR", raising=False)
    monkeypatch.delenv("GIT_WORK_TREE", raising=False)
    monkeypatch.delenv("GIT_INDEX_FILE", raising=False)

    assert tracking.git_commit() == expected


def test_provenance_reaches_the_run_name_and_tags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = enable(monkeypatch)
    monkeypatch.setattr(tracking, "git_commit", lambda: "a1b2c3d")
    monkeypatch.setattr(tracking, "git_describe", lambda: "v0.1.0-2-ga1b2c3d")

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
        "git_describe": "v0.1.0-2-ga1b2c3d",
    }


def test_provenance_omits_an_unknowable_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-editable install has no repo; the run is still worth tracking."""
    monkeypatch.setattr(tracking, "git_commit", lambda: None)

    monkeypatch.setattr(tracking, "git_describe", lambda: None)

    assert tracking.stamped("trial-000") == "trial-000"
    assert "git_commit" not in tracking.provenance_tags("M", "base")
    assert "git_describe" not in tracking.provenance_tags("M", "base")


def test_provenance_carries_the_release_the_code_descends_from(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tag is what a paper cites; the suffix is what makes it exact, so
    both halves ride on the run rather than the reader reconstructing one."""
    monkeypatch.setattr(tracking, "git_describe", lambda: "v0.1.0-2-ga1b2c3d")

    tags = tracking.provenance_tags("M", "base")

    assert tags["git_describe"] == "v0.1.0-2-ga1b2c3d"
