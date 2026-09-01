"""The pinned dev tooling must be the tooling this environment runs.

CI checks that `pyproject.toml` and the lockfiles agree; that the *installed*
environment agrees with the pin is nobody's job, and that is the gap through
which `ruff` went undeclared for the life of the repo — `pdm run ruff` fell
through `PATH` to a binary two years stale while every gate reported a verdict
as though it came from the documented version. Only exact (`==`) pins are
asserted: a floor names no version this environment has to be running.
"""

import importlib.metadata
import pathlib
import tomllib

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

_PYPROJECT = pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"


def _exact_pins() -> list[tuple[str, str]]:
    """`(distribution, version)` for each dev requirement pinned with `==`.

    A requirement whose marker excludes this interpreter is dropped, as is a
    wildcard such as `sqlmodel==0.*`, which uses `==` to express a range.
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

    If this fails because the pin was deliberately relaxed, the parametrized
    test below has quietly stopped covering the tool it was written for.
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
