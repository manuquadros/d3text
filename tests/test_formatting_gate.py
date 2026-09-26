"""Every tracked file under `brenda_references/` stays `ruff format`-clean.

The documented lint gate skips `brenda_references/` and no CI job runs ruff;
pytest runs everywhere. Format only: `ruff check`'s E711/E712 would rewrite
tinydb's `where("id") == None` to `is None`, which builds no query.
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
