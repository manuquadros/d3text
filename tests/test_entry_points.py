"""Smoke test for the ``[project.scripts]`` console-script entry points.

Each entry point is a ``module:attr`` string that must import cleanly and expose
a callable. Nothing exercised this contract before, so entries could reference a
module with no such function, or a module that no longer exists, and only fail
when a user typed ``pdm run <name>``. This test resolves every declared entry
point the way the installer does.
"""

import importlib
import pathlib
import tomllib

import pytest

_PYPROJECT = pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"


def _entry_points() -> list[tuple[str, str]]:
    with _PYPROJECT.open("rb") as pyproject:
        config = tomllib.load(pyproject)
    return sorted(config["project"]["scripts"].items())


def test_documented_pipeline_entry_points_are_declared() -> None:
    """The two documented ``pdm run`` commands must exist as entry points.

    Guards the zero-entries corner where the parametrized test below would
    collect nothing and pass vacuously.
    """
    declared = dict(_entry_points())
    assert declared["train"] == "scripts.train:main"
    assert declared["tuning"] == "scripts.tune:main"


@pytest.mark.parametrize(
    "name, reference",
    _entry_points(),
    ids=lambda v: v if ":" not in v else None,
)
def test_project_script_resolves_to_callable(name: str, reference: str) -> None:
    module_name, _, attribute = reference.partition(":")
    module = importlib.import_module(module_name)
    target = getattr(module, attribute, None)
    assert callable(target), (
        f"{name} -> {reference} does not resolve to a callable"
    )
