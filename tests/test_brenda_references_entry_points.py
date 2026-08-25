"""`brenda_references` must declare no console script it cannot ship.

An entry point is a ``module:attr`` string that the installer turns into a
``bin/`` shim doing ``from <module> import <attr>``. The shim runs with the
venv's ``bin/`` as ``sys.path[0]`` — never the repo root, never the caller's
cwd — so the module has to be part of the *installed distribution*. This
package is src-layout: its editable install puts one path on ``sys.path``,
``brenda_references/src``, which leaves the sibling ``scripts/`` directory
reachable from nothing.

That is the same failure the root project hit with its own top-level
``scripts/``, and why its CLI moved into ``src/d3text/cli/``;
``tests/test_entry_points.py`` guards that half. This file guards the
sub-package, and lives in the root ``tests/`` tree because the sub-package's
own suite does not run in CI.
"""

import os
import pathlib
import subprocess
import sys
import tomllib

import pytest

_PACKAGE = pathlib.Path(__file__).resolve().parent.parent / "brenda_references"
_SRC = _PACKAGE / "src"


def _entry_points() -> dict[str, str]:
    with (_PACKAGE / "pyproject.toml").open("rb") as pyproject:
        config = tomllib.load(pyproject)
    return dict(config["project"].get("scripts", {}))


def _shipped_module(reference: str) -> pathlib.Path | None:
    """The file under ``src/`` a ``module:attr`` reference names, if any."""
    module = reference.split(":", 1)[0]
    relative = pathlib.Path(*module.split("."))
    for candidate in (
        _SRC / relative.with_suffix(".py"),
        _SRC / relative / "__init__.py",
    ):
        if candidate.exists():
            return candidate
    return None


def test_every_entry_point_target_ships_with_the_distribution() -> None:
    """No shim may name a module the wheel leaves behind.

    Collected into a dict rather than parametrized: with no entry points
    declared, a parametrization has nothing to collect and reports a
    permanently empty parameter set, whereas this states the invariant
    whatever the table holds — and names every offender at once when the
    table grows one that ``src/`` does not contain.
    """
    unshippable = {
        name: reference
        for name, reference in _entry_points().items()
        if _shipped_module(reference) is None
    }

    assert not unshippable, (
        f"{sorted(unshippable)} are declared as console scripts but their "
        f"targets are outside {_SRC}, so the installed shim cannot import "
        f"them: {unshippable}. Move the module under src/brenda_references, "
        f"or drop the entry point and run the file directly."
    )


@pytest.mark.slow
def test_the_scripts_package_is_unreachable_from_an_installed_shim(
    tmp_path: pathlib.Path,
) -> None:
    """The packaging fact the check above rests on.

    Run outside the repo and with ``PYTHONPATH`` cleared, which is the import
    environment a ``bin/`` shim actually gets. If ``scripts`` ever becomes
    importable from there, the constraint above is stricter than it needs to
    be and this test is where that gets noticed.
    """
    result = subprocess.run(
        [sys.executable, "-c", "import scripts"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        timeout=300,
    )

    assert result.returncode != 0, (
        "`scripts` imported from outside the repo, so the installed "
        "distribution now ships it after all"
    )
    assert "No module named 'scripts'" in result.stderr, result.stderr
