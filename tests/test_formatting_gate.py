"""Every tracked file under `brenda_references/` stays `ruff format`-clean.

The lint gate this project documents names `src/`, `scripts/` and `tests/`,
and no CI job runs `ruff` at all, so the nested path dependency's sources were
formatted by nobody: two of its scripts drifted out of the formatter and were
noticed only when an unrelated whole-tree `ruff format` reflowed them. pytest
is the one gate that runs everywhere, which is why the check lives here.

Formatting only. Widening the gate to `ruff check` was measured and rejected:
that reports 26 errors across five files, and two of the rules it fires
(`E711`, `E712`) would rewrite tinydb query expressions such as
`where("id") == None`, where the suggested `is None` builds no query at all.
"""

import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _tracked_python_files() -> list[str]:
    """The files git has, so a local scratch file cannot turn the gate red."""
    listing = subprocess.run(
        ["git", "ls-files", "-z", "--", "brenda_references/*.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )

    return [name for name in listing.stdout.split("\0") if name]


def test_tracked_brenda_references_files_are_ruff_format_clean() -> None:
    """Invokes ruff through this interpreter, whose version
    `test_dev_tooling_pinned.py` holds at the pin, so the verdict here is the
    one the documented gate gives."""
    paths = _tracked_python_files()

    assert paths, (
        "no tracked Python file under brenda_references/, so this check "
        "would pass without formatting anything"
    )

    result = subprocess.run(
        [sys.executable, "-m", "ruff", "format", "--check", *paths],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        "run `ruff format brenda_references/`:\n"
        f"{result.stdout}{result.stderr}"
    )
