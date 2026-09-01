"""The package config `brenda_references` ships, and the host that is not.

`config.py` reads `config.toml` at import and the package's `__init__` reaches
it, so a tree without that file cannot execute `import brenda_references` at
all. It was invisible to git for years because `.gitignore` hid the
machine-local root `config.toml` with an unanchored glob. Committing it means
publishing whatever it names, hence the second half: the host belongs to the
environment, only the schema names to the package.
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

    The index rather than `HEAD`, so a staged fix counts, and a copy rather
    than the working tree, so untracked files cannot stand in for missing ones.
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
    itself imports from the editable checkout, where the untracked original
    hides the omission.
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


def test_no_tracked_file_names_a_host_in_the_private_zone():
    """Hosts under the internal zone are private; tracked files are published.

    The zone rather than the one cluster name, and a regex whose escaped
    pattern does not contain the text it matches, so this file cannot match
    itself. The public `dsmz.de` alone would not do — maintainer addresses and
    StrainInfo URLs carry it. `--cached` searches the index: a working tree
    that merely deleted the file would still ship it.
    """
    internal_zone = r"\.dmz\.dsmz\.de"

    found = subprocess.run(
        ["git", "grep", "--cached", "--name-only", "-E", "-e", internal_zone],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert found.returncode in (0, 1), found.stderr
    assert found.returncode == 1, (
        "a host in the private zone is tracked in: "
        f"{', '.join(found.stdout.split())}"
    )
