"""The package `__init__` must not make a leaf import pay for the model stack.

`d3text.models.__init__` used to re-export the three model classes eagerly, so
importing any submodule of the package ran `base` and everything behind it —
transformers, lmdb, sklearn, `d3text.utils` — whatever the importer actually
wanted. The classes now resolve through a module-level `__getattr__`.

Every check runs in a subprocess: the suite as a whole imports the model
stack, so an in-process `sys.modules` assertion would pass on a tree that had
regressed.
"""

import pathlib
import subprocess
import sys

_STACK = (
    "'d3text.models.base' in sys.modules, "
    "'d3text.utils' in sys.modules, "
    "'transformers' in sys.modules"
)


def _probe(source: str, cwd: pathlib.Path) -> subprocess.CompletedProcess[str]:
    """Run `source` in a fresh interpreter, outside the repo."""
    return subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=300,
    )


def test_a_leaf_import_does_not_load_the_model_stack(
    tmp_path: pathlib.Path,
) -> None:
    """`d3text.models.config` reads a TOML file and needs nothing else."""
    result = _probe(
        f"import sys, d3text.models.config; print(any(({_STACK})))", tmp_path
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("False"), (
        "importing d3text.models.config pulled in the model stack: "
        f"{result.stdout!r} {result.stderr}"
    )


def test_importing_the_runtime_does_not_load_the_utilities(
    tmp_path: pathlib.Path,
) -> None:
    """The payoff: `d3text.runtime` is no longer downstream of `d3text.utils`.

    Its only `d3text` imports are `logs` and `models.config`, but the eager
    re-exports made every one of them transitively import `utils` — which is
    what kept a device helper from moving out of `models.base` and into
    `runtime`, since `utils` could not then import it back.
    """
    result = _probe(
        f"import sys, d3text.runtime; print(any(({_STACK})))", tmp_path
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("False"), (
        "importing d3text.runtime pulled in the model stack: "
        f"{result.stdout!r} {result.stderr}"
    )


def test_the_model_classes_are_still_reachable_on_the_package(
    tmp_path: pathlib.Path,
) -> None:
    """`from d3text.models import ETEBrendaModel` must keep resolving, to the
    same object the owning submodule exports rather than a second copy — and
    `dir()` must keep listing the three, which it does not do for a name only
    a `__getattr__` knows about.
    """
    result = _probe(
        "import d3text.models as m, d3text.models.entity_linking as el, "
        "d3text.models.ete as ete, d3text.models.ner as ner; "
        "names = {'BrendaClassificationModel', 'ETEBrendaModel', "
        "'NERClassificationModel'}; "
        "print(m.BrendaClassificationModel is el.BrendaClassificationModel "
        "and m.ETEBrendaModel is ete.ETEBrendaModel "
        "and m.NERClassificationModel is ner.NERClassificationModel "
        "and names <= set(dir(m)))",
        tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith(
        "True"
    ), f"the lazy names did not resolve: {result.stdout!r} {result.stderr}"


def test_an_unknown_attribute_still_raises_attribute_error(
    tmp_path: pathlib.Path,
) -> None:
    """A `__getattr__` that answered anything else would break `hasattr` and
    every other "does this module have X" probe."""
    result = _probe(
        "import d3text.models; d3text.models.does_not_exist", tmp_path
    )

    assert result.returncode != 0
    assert "AttributeError" in result.stderr
    assert "does_not_exist" in result.stderr
