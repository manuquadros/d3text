"""The package config `brenda_references` ships, and the host that is not in it.

`config.py` reads `config.toml` at module import and the package's own
`__init__` reaches it, so a tree without that file cannot execute
`import brenda_references` at all — the failure is not deferred to first use.
The file was invisible to git for years because `.gitignore` hid the
machine-local root `config.toml` with an unanchored glob that matched every
`config.toml` in the tree.

Committing it means the database server it named is published with it, hence
the second half of this module: the host belongs to the environment, and only
the schema names belong to the package.
"""

import os
import pathlib
import shutil
import subprocess
import sys
import tomllib

import pytest

REPO_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_CONFIG = (
    REPO_ROOT / "brenda_references/src/brenda_references/config.toml"
)


def tracked_tree(destination: pathlib.Path, pathspec: str) -> pathlib.Path:
    """Copy the files git has under `pathspec` into `destination`.

    The index rather than `HEAD`, so a staged-but-uncommitted fix counts, and
    a copy rather than the working tree, so untracked files cannot stand in
    for the ones a fresh clone would be missing.
    """
    listing = subprocess.run(
        ["git", "ls-files", "-z", "--", pathspec],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    )

    for name in listing.stdout.decode().split("\0"):
        if not name:
            continue
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO_ROOT / name, target)

    return destination


def test_the_package_is_importable_from_tracked_files_alone(tmp_path):
    """A fresh clone has exactly the tracked files and nothing else.

    Run in a subprocess with the copied tree first on `PYTHONPATH`: the suite
    itself imports the package from the editable checkout, where the untracked
    original sits on disk and hides the omission.
    """
    tree = tracked_tree(tmp_path / "clone", "brenda_references/src")
    env = os.environ | {
        "PYTHONPATH": str(tree / "brenda_references/src"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import brenda_references as b; print(b.__file__)",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        "a clone of the tracked files cannot import brenda_references:\n"
        f"{result.stderr}"
    )
    assert result.stdout.strip().startswith(str(tree)), (
        "the probe imported the editable checkout, not the copied tree: "
        f"{result.stdout.strip()}"
    )


def test_the_shipped_config_names_no_database_host():
    """The host is a private server; the package is published to anyone who
    clones it."""
    with PACKAGE_CONFIG.open("rb") as f:
        shipped = tomllib.load(f)

    assert "host" not in shipped["database"]


def test_get_engine_takes_the_host_from_the_environment(monkeypatch):
    from brenda_references import db

    urls = []
    monkeypatch.setattr(db, "create_engine", urls.append)
    for name, value in (
        ("BRENDA_HOST", "db.example.org"),
        ("BRENDA_USER", "user"),
        ("BRENDA_PASSWORD", "secret"),
    ):
        monkeypatch.setenv(name, value)

    db.get_engine()

    assert urls[0].host == "db.example.org"
    assert urls[0].database == "brenda_conn"


def test_get_engine_says_which_variables_are_missing(monkeypatch):
    """The note is the only place the new variable's name appears at the
    point of failure."""
    from brenda_references import db

    monkeypatch.delenv("BRENDA_HOST", raising=False)
    monkeypatch.setenv("BRENDA_USER", "user")
    monkeypatch.setenv("BRENDA_PASSWORD", "secret")

    with pytest.raises(KeyError) as caught:
        db.get_engine()

    assert "BRENDA_HOST" in "".join(caught.value.__notes__)
