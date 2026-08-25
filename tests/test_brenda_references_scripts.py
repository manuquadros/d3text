"""`brenda_references.scripts` must import without the root's dev group.

Its ``__init__`` installs beartype's import hook, and beartype is declared by
no runtime dependency anywhere in the tree: it is an extra of the *root*
project's ``dev`` group, so every module under ``brenda_references/scripts/``
— including the nine wired up as console entry points — was importable purely
as a side effect of a developer install.

The probe runs in a subprocess because that is the only place the failure
reproduces. Resolving the package with ``importlib`` inside the pytest process
cannot: the suite runs from a dev install, where beartype is already on
``sys.path``. Blocking it with a meta-path finder rather than uninstalling it
raises the same ``ModuleNotFoundError`` a non-dev environment would, without
touching the environment the rest of the suite shares.
"""

import os
import pathlib
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).parents[1]
SCRIPTS_PARENT = REPO_ROOT / "brenda_references"

_BLOCK_BEARTYPE = """
import sys


class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name == "beartype" or name.startswith("beartype."):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None


sys.meta_path.insert(0, Blocker())
"""

_REPORT = """
import sys
import scripts

print(scripts.__file__)
print("beartype.claw" in sys.modules)
"""


def _import_scripts(source: str, cwd: pathlib.Path) -> tuple[int, str, str]:
    """Import the package in a subprocess whose only path entry is the repo's.

    ``cwd`` is a tmp_path so the repo root — which has a top-level ``scripts/``
    of its own — cannot be what gets imported.
    """
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=cwd,
        env=os.environ
        | {
            "PYTHONPATH": str(SCRIPTS_PARENT),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout, result.stderr


def test_the_scripts_package_imports_with_beartype_missing(tmp_path) -> None:
    code, out, err = _import_scripts(_BLOCK_BEARTYPE + _REPORT, tmp_path)

    assert code == 0, (
        "importing brenda_references' scripts needs beartype, which no "
        f"runtime dependency provides:\n{err}"
    )
    location, claw_imported = out.split()
    assert location == str(SCRIPTS_PARENT / "scripts/__init__.py"), (
        f"the probe imported some other `scripts` package: {location}"
    )
    assert claw_imported == "False"


def test_the_scripts_package_still_installs_the_claw_when_it_can(
    tmp_path,
) -> None:
    """Making beartype optional must not mean dropping the checking."""
    pytest.importorskip("beartype")

    code, out, err = _import_scripts(_REPORT, tmp_path)

    assert code == 0, err
    assert out.split()[1] == "True", (
        "beartype is installed but importing the package did not reach the "
        "claw, so these scripts are no longer runtime type-checked"
    )
