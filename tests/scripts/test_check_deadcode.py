"""``check_deadcode.py``: the gate that fails on code nothing uses.

Run against throwaway projects, so each outcome is staged exactly rather than
depending on what this repository happens to contain.
"""

import pathlib
import subprocess
import sys

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2] / "scripts/check_deadcode.py"
)

_PYPROJECT = """\
[tool.deadcode]
only = ["src/*"]
"""

_LIBRARY = """\
def helper() -> int:
    return 1
"""

_CALLER = """\
from toy.library import helper


def test_helper() -> None:
    assert helper() == 1
"""


def _toy_project(
    root: pathlib.Path, library: str = _LIBRARY, pyproject: str = _PYPROJECT
) -> pathlib.Path:
    """Write a project whose only caller of `toy.library` is a test."""
    (root / "src" / "toy").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "pyproject.toml").write_text(pyproject)
    (root / "src" / "toy" / "library.py").write_text(library)
    (root / "tests" / "test_library.py").write_text(_CALLER)
    return root


def _run_gate(root: pathlib.Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), str(root)],
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_code_used_only_by_tests_passes(tmp_path: pathlib.Path) -> None:
    """Tests count as callers but are not reported on; also guards the
    inverse of the test below, which a gate that always failed would pass."""
    result = _run_gate(_toy_project(tmp_path))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASSED" in result.stdout


def test_unused_code_fails_the_gate_by_name(tmp_path: pathlib.Path) -> None:
    """The `deadcode` command itself exits 0 on this tree, so the failure
    has to come from the wrapper."""
    library = _LIBRARY + "\n\ndef orphan() -> None:\n    pass\n"

    result = _run_gate(_toy_project(tmp_path, library=library))

    assert result.returncode == 1
    assert "orphan" in result.stderr


def test_nothing_to_report_on_cannot_run(tmp_path: pathlib.Path) -> None:
    """An `only` matching no file checks nothing; that is a coverage hole,
    not a pass."""
    root = _toy_project(
        tmp_path, pyproject='[tool.deadcode]\nonly = ["lib/*"]\n'
    )

    result = _run_gate(root)

    assert result.returncode == 1
    assert "COULD NOT RUN" in result.stderr
