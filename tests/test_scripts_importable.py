"""Every tracked script under `scripts/` must name modules that exist.

`scripts/` is the one corner no other gate covers: mypy runs against `src/`
only and nothing executes these files, so a script can name a module that has
not existed for years and stay green. The module part is checked statically
with `ast` and `find_spec`, since several of these scripts open files, build a
dataset or load the base model at import scope. The imported *name* in `from
a.b import c` is checked too, falling back to importing `a.b` so a dynamic
re-export resolves the way it would at run time. Only module-scope imports are
collected: an import inside a function is a deliberately deferred one.
"""

import ast
import dataclasses
import importlib
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


@dataclasses.dataclass(frozen=True)
class _ImportRef:
    """One module-scope import, as ``from module import name`` (or a bare
    ``import module``/``from module import *``, where ``name`` is ``None``
    and only the module part is checked)."""

    module: str
    name: str | None


def _module_scope_imports(source: str) -> list[_ImportRef]:
    """The module-scope imports of a file, module part plus imported name.

    A star import names nothing to check. Relative imports are skipped: none of
    these scripts is inside a package that would give them a meaning.
    """
    refs: list[_ImportRef] = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.Import):
            refs.extend(_ImportRef(alias.name, None) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module is not None:
                for alias in node.names:
                    if alias.name == "*":
                        refs.append(_ImportRef(node.module, None))
                    else:
                        refs.append(_ImportRef(node.module, alias.name))
    return refs


def _module_resolves(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _name_resolves(module: str, name: str) -> bool:
    """Whether `name` in `from module import name` names something real.

    Tried as a submodule first, then by importing `module` and looking it up
    with `hasattr`, which honours a module-level `__getattr__` or an
    `__all__`-driven re-export.
    """
    try:
        if importlib.util.find_spec(f"{module}.{name}") is not None:
            return True
    except (ImportError, ValueError, AttributeError):
        pass

    try:
        imported = importlib.import_module(module)
    except Exception:
        return False
    return hasattr(imported, name)


def _unresolved_imports(source: str) -> list[str]:
    unresolved = []
    for ref in _module_scope_imports(source):
        if not _module_resolves(ref.module):
            unresolved.append(ref.module)
        elif ref.name is not None and not _name_resolves(ref.module, ref.name):
            unresolved.append(f"{ref.module}.{ref.name}")
    return unresolved


def test_some_scripts_are_checked() -> None:
    """Guards the vacuous pass if the listing ever breaks."""
    assert len(_tracked_scripts()) > 5


@pytest.mark.parametrize("script", _tracked_scripts(), ids=lambda v: v)
def test_script_imports_resolve(script: str) -> None:
    path = _ROOT / script
    unresolved = _unresolved_imports(path.read_text())

    assert not unresolved, (
        f"{script} imports {unresolved} at module scope, which cannot be "
        f"resolved in this environment: the script cannot start. Fix the "
        f"import, or delete the script."
    )


def test_nonexistent_name_from_existing_package_is_unresolved() -> None:
    """The gap this suite closes: a package that resolves is not the same
    as one of its members existing (``from brenda_references import
    brenda_types``, a submodule that no longer exists, used to pass because
    ``brenda_references`` itself still resolves)."""
    source = "from os import definitely_not_a_real_attribute_xyz\n"

    assert _unresolved_imports(source) == [
        "os.definitely_not_a_real_attribute_xyz"
    ]


def test_existing_name_from_existing_package_resolves() -> None:
    source = "from os import path\n"

    assert _unresolved_imports(source) == []


def test_dynamically_exported_name_resolves(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A name produced by a module-level ``__getattr__`` (PEP 562) is a
    legitimate re-export, not a broken import: it must not be flagged just
    because it is absent from the module's static namespace, and a name that
    the same ``__getattr__`` does not know must still be flagged."""
    package = tmp_path / "dynamic_pkg"
    package.mkdir()
    (package / "__init__.py").write_text(
        "def __getattr__(name):\n"
        "    if name == 'made_up':\n"
        "        return 42\n"
        "    raise AttributeError(name)\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    assert _unresolved_imports("from dynamic_pkg import made_up\n") == []
    assert _unresolved_imports("from dynamic_pkg import missing\n") == [
        "dynamic_pkg.missing"
    ]
