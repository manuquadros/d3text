"""Every tracked script under ``scripts/`` must name modules that exist.

``scripts/`` is the one corner of the tree no other gate covers: ``mypy`` runs
against ``src/`` only, and nothing executes these files, so a script can name a
module that has not existed for years and stay green forever. That is not
hypothetical — ``get_annotation_targets.py`` did ``from config import
species_list`` at module scope, and since the ``open(species_list)`` beside it
was also module scope, ``--help`` died before ``main()`` was ever reached.
Same story for ``brenda_references/scripts/``: a prior cleanup dropped the
console-script entry points that used to wrap those modules but left the
modules themselves importing names that had moved, so both script trees are
covered here.

The check is **static**: the imports are read with ``ast`` and resolved with
``importlib.util.find_spec``. Executing the scripts is not an option — several
open files, build a dataset, or load the base model at import scope. Resolving a
*dotted* name does import the parent package (that is documented ``find_spec``
behaviour), which is why the resolution is guarded: a missing top-level package
surfaces as ``ModuleNotFoundError`` from the parent's own import machinery
rather than as a ``None`` spec.

Only module-scope imports are collected. An import inside a function is a
deliberately deferred one — ``dec03_full/vm/preflight.py`` defers ``torch`` and
half of ``d3text`` precisely so that it stays a leaf — and demanding those
resolve would assert the opposite of what the script is arranging.
"""

import ast
import importlib.util
import pathlib
import subprocess

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _tracked_scripts() -> list[str]:
    listing = subprocess.run(
        ["git", "ls-files", "--", "scripts/", "brenda_references/scripts/"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return sorted(
        path for path in listing.stdout.split() if path.endswith(".py")
    )


def _module_scope_imports(source: str) -> list[str]:
    """The dotted module names a file imports at module scope.

    ``import a.b`` is resolved as ``a.b``; ``from a.b import c`` as ``a.b``,
    since ``c`` may be a submodule or an attribute and only the module part is
    resolvable without importing it. Relative imports are skipped: none of
    these scripts is inside a package that would give them a meaning.
    """
    names: list[str] = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module is not None:
                names.append(node.module)
    return names


def test_some_scripts_are_checked() -> None:
    """Guards the vacuous pass if the listing ever breaks."""
    assert len(_tracked_scripts()) > 5


@pytest.mark.parametrize("script", _tracked_scripts(), ids=lambda v: v)
def test_script_imports_resolve(script: str) -> None:
    path = _ROOT / script
    unresolved = []
    for module in _module_scope_imports(path.read_text()):
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            unresolved.append(module)

    assert not unresolved, (
        f"{script} imports {unresolved} at module scope, which cannot be "
        f"resolved in this environment: the script cannot start. Fix the "
        f"import, or delete the script."
    )
