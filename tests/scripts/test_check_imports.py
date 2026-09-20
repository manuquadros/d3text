"""``check_imports.py``: the gate that catches an unimportable package.

Exercised against throwaway projects rather than against this one, for two
reasons. Importing the real entry points costs half a minute, and
`test_entry_points.py` already starts every console script; and the failure
this gate exists for cannot be staged in the tree under test, because a
`d3text` that will not import takes this test module's own collection down
with it. A toy package with a real cycle reproduces the shape in a tenth of
a second.
"""

import os
import pathlib
import subprocess
import sys

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2] / "scripts/check_imports.py"
)

_PYPROJECT = """\
[project]
name = "toy"
version = "0"

[project.scripts]
toy = "toy_pkg.entry:main"
"""

_HEALTHY_ENTRY = """\
MARK = "ok"


def main() -> int:
    return 0
"""

# `entry` is partially initialized when `other` reaches back into it, so
# `MARK` is not bound yet: the ImportError production would raise.
_CYCLIC_ENTRY = """\
from toy_pkg.other import OTHER

MARK = OTHER


def main() -> int:
    return 0
"""

_OTHER = """\
from toy_pkg.entry import MARK

OTHER = MARK
"""


def _toy_project(
    root: pathlib.Path, entry: str, pyproject: str = _PYPROJECT
) -> pathlib.Path:
    """Write a src-layout project whose console script is `toy_pkg.entry`."""
    package = root / "src" / "toy_pkg"
    package.mkdir(parents=True)
    (root / "pyproject.toml").write_text(pyproject)
    (package / "__init__.py").write_text("")
    (package / "entry.py").write_text(entry)
    (package / "other.py").write_text(_OTHER)
    return root


def _run_gate(
    root: pathlib.Path, python_path: str | None = None
) -> subprocess.CompletedProcess[str]:
    """Run the gate against `root` with `PYTHONPATH` under the test's control.

    Inheriting this session's `PYTHONPATH` would let the tree being gated
    reach the checkout's sources, which is the confusion these tests are
    about.
    """
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    if python_path is not None:
        env["PYTHONPATH"] = python_path

    return subprocess.run(
        [sys.executable, str(_SCRIPT), str(root)],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )


def test_a_tree_whose_entry_points_import_passes(
    tmp_path: pathlib.Path,
) -> None:
    """Guards the inverse of the test below: a gate that always failed would
    pass it."""
    result = _run_gate(_toy_project(tmp_path, _HEALTHY_ENTRY))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASSED  toy_pkg.entry" in result.stdout


def test_an_import_cycle_fails_the_gate_by_name(tmp_path: pathlib.Path) -> None:
    """The gap this script closes: `ruff` and `mypy src/` both pass on a tree
    with a circular import, because neither ever performs one, and pytest
    reports it as a heap of collection errors with no named failure among
    them."""
    result = _run_gate(_toy_project(tmp_path, _CYCLIC_ENTRY))

    assert result.returncode == 1
    assert "FAILED  toy_pkg.entry" in result.stdout
    assert "circular import" in result.stderr
    assert "1 of 1 entry points cannot be imported" in result.stderr


def test_a_project_declaring_no_entry_points_cannot_run(
    tmp_path: pathlib.Path,
) -> None:
    """An empty selection is a coverage hole, not a pass: without this the
    gate would report success on a `pyproject.toml` it failed to read
    anything out of."""
    root = _toy_project(
        tmp_path, _HEALTHY_ENTRY, pyproject='[project]\nname = "toy"\n'
    )

    result = _run_gate(root)

    assert result.returncode == 1
    assert "COULD NOT RUN" in result.stderr


def test_the_gate_answers_for_the_tree_it_is_given(
    tmp_path: pathlib.Path,
) -> None:
    """The gate must import the sources under `root`, not a same-named package
    the environment already offers.

    This is how every other check in this repository is read in a `git
    worktree`: the venv's `.pth` file names the main checkout's `src`, and
    `site` appends it, so anything that does not put the tree under test
    first reports on the checkout instead — green, while the tree it was
    asked about is broken.
    """
    broken = _toy_project(tmp_path / "broken", _CYCLIC_ENTRY)
    healthy = _toy_project(tmp_path / "healthy", _HEALTHY_ENTRY)

    result = _run_gate(broken, python_path=str(healthy / "src"))

    assert result.returncode == 1, (
        "the gate imported `toy_pkg` from the environment rather than from "
        f"the root it was given:\n{result.stdout}{result.stderr}"
    )
