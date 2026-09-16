"""The package `__init__` must not make a non-BRENDA import pay for BRENDA.

`d3text.datasets.__init__` used to import `d3text.datasets.brenda` eagerly,
so importing any submodule of the package pulled in the whole BRENDA data
layer regardless of what was actually wanted. `BRENDA_SCHEMA` and
`brenda_dataset` now resolve lazily through a module-level `__getattr__`.
"""

import subprocess
import sys


def test_a_non_brenda_submodule_does_not_import_the_data_layer(tmp_path):
    """Checked in a subprocess: the suite as a whole imports `d3text.data`,
    so an in-process check would pass no matter what."""
    probe = (
        "import sys; import d3text.datasets.s800; "
        "prefixes = ('d3text.data.', 'brenda_references', "
        "'lpsn_interface', 'torch'); "
        "print(any(m == 'd3text.data' or m.startswith(prefixes) "
        "for m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=True,
    )

    assert result.stdout.strip().endswith("False"), (
        "d3text.datasets.s800 pulled in the BRENDA data layer: "
        f"{result.stdout!r} {result.stderr}"
    )


def test_brenda_dataset_and_schema_still_resolve_lazily(tmp_path):
    """The two names a caller still expects `from d3text.datasets import
    brenda_dataset` to find, resolved on first access rather than at
    package-import time."""
    probe = (
        "import sys; from d3text.datasets import BRENDA_SCHEMA, "
        "brenda_dataset; "
        "print(any(m == 'd3text.data' or m.startswith('d3text.data.') "
        "or m.startswith('brenda_references') for m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=True,
    )

    assert result.stdout.strip().endswith("True"), (
        "accessing BRENDA_SCHEMA/brenda_dataset did not load the BRENDA "
        f"data layer: {result.stdout!r} {result.stderr}"
    )


def test_an_unknown_attribute_still_raises_attribute_error(tmp_path):
    probe = "import d3text.datasets; " "d3text.datasets.does_not_exist"
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert result.returncode != 0
    assert "AttributeError" in result.stderr
    assert "does_not_exist" in result.stderr
