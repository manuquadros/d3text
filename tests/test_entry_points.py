"""Smoke test for the `[project.scripts]` console-script entry points.

The contract is stronger than "the module imports": the shim runs with the
venv's `bin/` as `sys.path[0]`, never the repo root, so the module has to be
part of the *installed distribution*. An earlier version resolved each entry
point with `importlib` inside pytest, where the repo root happened to be
importable, and stayed green while `pdm run train` died with `No module named
'scripts'`. These execute the installed console script in a subprocess.
"""

import os
import pathlib
import subprocess
import sys
import tomllib

import pytest

_PYPROJECT = pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"
_BIN_DIR = pathlib.Path(sys.executable).parent


def _entry_points() -> list[tuple[str, str]]:
    with _PYPROJECT.open("rb") as pyproject:
        config = tomllib.load(pyproject)
    return sorted(config["project"]["scripts"].items())


def _run_help(name: str, cwd: pathlib.Path) -> subprocess.CompletedProcess[str]:
    script = _BIN_DIR / name
    assert script.exists(), (
        f"{script} is missing: the project is not installed into this "
        f"environment, so its entry points cannot be exercised. Run `pdm install`."
    )
    return subprocess.run(
        [str(script), "--help"],
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=300,
    )


def test_pipeline_entry_points_are_declared() -> None:
    """Every documented `pdm run` command must be a declared entry point.

    Guards the zero-entries corner where the parametrized test below would
    collect nothing and pass vacuously.
    """
    assert dict(_entry_points()) == {
        "train": "d3text.cli.train:main",
        "tuning": "d3text.cli.tune:main",
        "evaluate": "d3text.cli.evaluate:main",
        "precompute-encodings": "d3text.cli.precompute_encodings:main",
        "precompute-embeddings": "d3text.cli.precompute_embeddings:main",
        "precompute-token-labels": ("d3text.cli.precompute_token_labels:main"),
    }


@pytest.mark.slow
@pytest.mark.parametrize("name, reference", _entry_points(), ids=lambda v: v)
def test_console_script_runs(
    name: str, reference: str, tmp_path: pathlib.Path
) -> None:
    """The installed console script starts and parses its arguments.

    `--help` is enough: the shim's import must succeed before argparse can
    print anything. Runs from `tmp_path`, outside the repo, which reproduces
    the console script's real import environment.
    """
    result = _run_help(name, tmp_path)

    assert result.returncode == 0, (
        f"`{name} --help` ({reference}) exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )


@pytest.mark.slow
def test_a_command_writes_nothing_to_its_working_directory(
    tmp_path: pathlib.Path,
) -> None:
    """Starting a d3text command must leave the cwd exactly as it found it.

    `lpsn_interface` used to attach a `RotatingFileHandler` to the *relative*
    path `lpsn.log` at module scope. The assertion is deliberately wider than
    that one filename: no import in the chain gets to write to the cwd.
    """
    result = _run_help("train", tmp_path)

    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == [], (
        "starting a d3text command littered its working directory: "
        f"{sorted(p.name for p in tmp_path.iterdir())}"
    )


@pytest.mark.slow
def test_a_command_starts_in_a_read_only_working_directory(
    tmp_path: pathlib.Path,
) -> None:
    """The other half of the same contract: no write means no crash.

    A read-only working directory used to kill every command outright with
    `PermissionError: './lpsn.log'` before argparse ever ran.
    """
    workdir = tmp_path / "read-only"
    workdir.mkdir()
    workdir.chmod(0o500)
    if os.access(workdir, os.W_OK):
        pytest.skip("cannot make a directory unwritable for this user")

    try:
        result = _run_help("train", workdir)
    finally:
        workdir.chmod(0o700)

    assert result.returncode == 0, (
        "`train --help` failed in a read-only working directory\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
