"""mypy still sees a misspelt attribute on `ETEBrendaModel`.

The class composes its entity/class machinery and reaches through to it in
`__getattr__`. mypy resolves every name a class does not declare through
that method's return type, so declaring it `Any` would type-check a typo —
even one called with the wrong arity — as clean across the whole class.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

SNIPPET = """\
from d3text.models.ete import ETEBrendaModel


def check(model: ETEBrendaModel) -> None:
    threshold: float = model.entity_threshold
    index: dict[str, int] = model.entity_to_index
    bogus: int = model.entity_thresold
    model.entity_thresold(1, 2, 3)
"""
TYPO_LINES = {7, 8}


@pytest.mark.slow
def test_a_misspelt_attribute_is_a_mypy_error(tmp_path: Path) -> None:
    """The typo lines error and the real attributes beside them do not.

    `MYPYPATH` names this tree's `src/` so the snippet is checked against the
    module beside this test: mypy searches it before site-packages, where the
    venv's `.pth` may point at another checkout.
    """
    pytest.importorskip("mypy")
    snippet = tmp_path / "snippet.py"
    snippet.write_text(SNIPPET)
    env = dict(os.environ, MYPYPATH=str(REPO / "src"))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--config-file",
            str(REPO / "pyproject.toml"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--no-color-output",
            "--no-error-summary",
            str(snippet),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    errors = {
        int(m.group(1))
        for m in re.finditer(
            rf"^{re.escape(str(snippet))}:(\d+): error:",
            result.stdout,
            flags=re.MULTILINE,
        )
    }
    assert errors == TYPO_LINES, result.stdout + result.stderr
