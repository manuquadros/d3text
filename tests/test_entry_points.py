"""Smoke test for the ``[project.scripts]`` console-script entry points.

An entry point is a ``module:attr`` string, and the installer turns it into a
``bin/`` shim that does ``from <module> import <attr>``. The contract is
therefore stronger than "the module imports": the module has to be *part of the
installed distribution*, because the shim runs with the venv's ``bin/`` as
``sys.path[0]`` — never the repo root, and never the caller's cwd.

That distinction is the whole point of this file. An earlier version of this
test resolved each entry point with ``importlib`` inside the pytest process,
where the repo root happened to be importable; it stayed green while
``pdm run train`` died with ``ModuleNotFoundError: No module named 'scripts'``,
because the entry modules lived in a top-level ``scripts/`` directory the wheel
does not ship. So these tests execute the **installed console script** in a
subprocess, which is the only thing that reproduces how a user invokes it.
"""

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


@pytest.fixture
def read_only_cwd(tmp_path: pathlib.Path) -> pathlib.Path:
    """A directory a command can be run from but cannot write to.

    Outside the repo, so the repo root is not on ``sys.path``, and read-only,
    so a command that writes to its working directory just to start up cannot
    pass. Left empty on purpose: a run that puts anything here has done
    something it was never asked to do.
    """
    cwd = tmp_path / "read_only"
    cwd.mkdir(mode=0o555)

    return cwd


def test_pipeline_entry_points_are_declared() -> None:
    """Every documented ``pdm run`` command must be a declared entry point.

    Guards the zero-entries corner where the parametrized test below would
    collect nothing and pass vacuously. The targets must live under
    ``d3text.cli``: anything outside the installed package is unreachable from a
    console script, however well it imports under pytest.
    """
    assert dict(_entry_points()) == {
        "train": "d3text.cli.train:main",
        "tuning": "d3text.cli.tune:main",
        "evaluate": "d3text.cli.evaluate:main",
        "precompute-encodings": "d3text.cli.precompute_encodings:main",
        "precompute-embeddings": "d3text.cli.precompute_embeddings:main",
    }


@pytest.mark.slow
@pytest.mark.parametrize("name, reference", _entry_points(), ids=lambda v: v)
def test_console_script_runs(
    name: str, reference: str, read_only_cwd: pathlib.Path
) -> None:
    """The installed console script starts and parses its arguments.

    ``--help`` is enough to prove the contract: the shim's
    ``from <module> import <attr>`` must succeed before argparse can print
    anything, so a missing module, a module absent from the wheel, and a missing
    ``main`` all surface here as a non-zero exit.

    Starting up must also touch nothing outside the process, which the
    read-only working directory pins from both sides: a command that writes
    there dies, and one that somehow can (running as root) leaves the directory
    non-empty. Every command used to fail this. Importing ``lpsn_interface``
    (transitively, via ``brenda_references``) attached a ``RotatingFileHandler``
    to the *relative* path ``lpsn.log`` at import time, so a d3text command
    dropped a log file wherever it was invoked from, and raised
    ``PermissionError`` where it could not.
    """
    script = _BIN_DIR / name
    assert script.exists(), (
        f"{script} is missing: the project is not installed into this "
        f"environment, so its entry points cannot be exercised. Run `uv sync`."
    )

    result = subprocess.run(
        [str(script), "--help"],
        capture_output=True,
        text=True,
        cwd=read_only_cwd,
        timeout=300,
    )

    assert result.returncode == 0, (
        f"`{name} --help` ({reference}) exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    assert list(read_only_cwd.iterdir()) == [], (
        f"`{name} --help` ({reference}) wrote to its working directory: "
        f"{[path.name for path in read_only_cwd.iterdir()]}"
    )
