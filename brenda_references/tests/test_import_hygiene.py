"""Importing brenda_references must leave global pandas options alone."""

import subprocess
import sys


def test_import_leaves_copy_on_write_unchanged() -> None:
    """Importing brenda_references leaves copy_on_write as it found it.

    The option is process-global, so setting it on import makes pandas
    semantics depend on import order. Compared before/after, not against a
    fixed value: pandas seeds it from the PANDAS_COPY_ON_WRITE env var.
    """
    code = """\
import pandas
print(pandas.options.mode.copy_on_write)
import brenda_references
print(pandas.options.mode.copy_on_write)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    before, after = result.stdout.split()
    assert before == after
