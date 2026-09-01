"""The pinned dev tooling must be the tooling this environment runs.

Two things can be true of a pin, and only one of them is checked elsewhere.
That ``pyproject.toml`` and the four lockfiles agree is CI's job, which runs
``pdm lock --check``. That the *installed* environment agrees with the pin is
nobody's, and that is the gap through which ``ruff`` went undeclared for the
life of the repo: ``pdm run ruff`` fell through ``PATH`` to a binary two years
stale, ``importlib.metadata.version("ruff")`` raised rather than answering, and
every gate reported a verdict as though it came from the documented version.
The same staleness follows a pin bumped without a ``pdm install`` — the venv
keeps the old version while the diff says otherwise, which is the failure mode
``CLAUDE.md`` already documents for a non-editable ``brenda_references``.

Only exact (``==``) pins are asserted. A floor (``mypy>=1.11``) names no
version this environment has to be running, so there is nothing here to check;
if one gains an exact pin it is covered from that moment without touching this
file.
"""

import importlib.metadata
import pathlib
import tomllib

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

_PYPROJECT = pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"


def _exact_pins() -> list[tuple[str, str]]:
    """``(distribution, version)`` for each dev requirement pinned with ``==``.

    A requirement whose environment marker excludes this interpreter is
    dropped: a pin that does not apply here is not one this environment has to
    satisfy. So is a wildcard such as ``sqlmodel==0.*``, which uses the ``==``
    operator to express a range rather than a version.
    """
    with _PYPROJECT.open("rb") as pyproject:
        config = tomllib.load(pyproject)

    pins: list[tuple[str, str]] = []
    for spec in config["dependency-groups"]["dev"]:
        requirement = Requirement(spec)
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        pins.extend(
            (requirement.name, specifier.version)
            for specifier in requirement.specifier
            if specifier.operator == "=="
            and not specifier.version.endswith(".*")
        )
    return sorted(pins)


def _installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def test_ruff_carries_an_exact_pin() -> None:
    """Guards the vacuous pass, on the pin the lint gate's verdict needs.

    ruff's answer moves in both directions with its version, so an unpinned
    ruff makes ``ruff check`` and ``ruff format`` report differently per
    machine. If this ever fails because the pin was deliberately relaxed, the
    parametrized test below has quietly stopped covering the tool it was
    written for.
    """
    assert "ruff" in dict(_exact_pins())


@pytest.mark.parametrize("distribution, pinned", _exact_pins(), ids=lambda v: v)
def test_pinned_dev_tool_is_the_installed_one(
    distribution: str, pinned: str
) -> None:
    installed = _installed_version(distribution)

    assert installed is not None, (
        f"{distribution} is pinned to {pinned} but is not installed in this "
        f"environment, so whatever the gates run is not the pinned version. "
        f"Run `TMPDIR=~/.cache/pdm-tmp pdm install -L locks/<flavour>.lock "
        f"--frozen-lockfile`."
    )
    assert Version(installed) == Version(pinned), (
        f"{distribution} is pinned to {pinned} but {installed} is installed: "
        f"the environment is stale relative to pyproject.toml. Run "
        f"`TMPDIR=~/.cache/pdm-tmp pdm install -L locks/<flavour>.lock "
        f"--frozen-lockfile`."
    )
