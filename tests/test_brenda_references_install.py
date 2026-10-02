"""`brenda_references` installs editable, from the dev group alone.

pdm installs a local path editable only from a dev group; declared in
`[project] dependencies` it is copied into site-packages as a snapshot, and
later edits under `brenda_references/src/` never run.
"""

import pathlib
import subprocess
import sys
import tomllib

import pytest

REPO_ROOT = pathlib.Path(__file__).parents[1]
LOCKFILES = sorted((REPO_ROOT / "locks").glob("*.lock"))


def test_pyproject_does_not_declare_it_a_runtime_dependency() -> None:
    """A `[project] dependencies` entry would ship it as a non-editable copy."""
    with (REPO_ROOT / "pyproject.toml").open("rb") as f:
        dependencies = tomllib.load(f)["project"]["dependencies"]

    assert not [d for d in dependencies if d.startswith("brenda-references")]


def test_there_are_lockfiles_to_check() -> None:
    assert LOCKFILES


@pytest.mark.parametrize("lockfile", LOCKFILES, ids=lambda p: p.name)
def test_every_lockfile_installs_it_editable_outside_default(
    lockfile: pathlib.Path,
) -> None:
    """The lock entry, not the venv, decides what a fresh install gets."""
    with lockfile.open("rb") as f:
        packages = tomllib.load(f)["package"]
    [entry] = [p for p in packages if p["name"] == "brenda-references"]

    assert entry.get("editable") is True
    assert "default" not in entry["groups"]


def test_it_imports_from_the_checkout_source() -> None:
    """A snapshot under an in-project `.venv` is inside the repo too."""
    import brenda_references

    source = (REPO_ROOT / "brenda_references" / "src").resolve()
    path = pathlib.Path(brenda_references.__file__).resolve()

    assert path.is_relative_to(source), path


def test_importing_infer_loads_no_brenda_references_module(
    tmp_path: pathlib.Path,
) -> None:
    """`infer` must run on an install without the dev group.

    A subprocess, because the suite itself has already imported it.
    """
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, d3text.cli.infer; "
            "print(sorted(m for m in sys.modules "
            "if m.split('.')[0] == 'brenda_references'))",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=300,
    )

    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip() == "[]"
