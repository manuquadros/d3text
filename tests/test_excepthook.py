"""`d3text.excepthook`: the context an exception carries must reach the user.

`add_note` is where this package puts the line that names the file or the
setting a third-party exception could not name itself. stackprinter's hook
renders only the traceback, so on the console -- the one path those notes were
written for -- they used to vanish.
"""

import pathlib
import subprocess
import sys
import types

import pytest
from d3text import excepthook

REPO_ROOT = pathlib.Path(__file__).parent.parent

# The note has to survive a real uncaught exception under the hook the package
# installs at import, which only a fresh interpreter can show: pytest keeps its
# own excepthook, and the stdlib one prints notes regardless.
_UNCAUGHT_NOTE = """
import d3text

error = ValueError("invalid literal")
error.add_note("while reading /nowhere/config.toml")
raise error
"""


def _note_carrying_exception() -> BaseException:
    try:
        raise ValueError("invalid literal")
    except ValueError as error:
        error.add_note("while reading /nowhere/config.toml")
        return error


def _traceback_only(
    exc_type: type[BaseException],
    exc: BaseException,
    traceback: types.TracebackType | None,
) -> None:
    print("the traceback", file=sys.stderr)


def test_installed_hook_prints_the_note() -> None:
    probe = subprocess.run(
        [sys.executable, "-c", _UNCAUGHT_NOTE],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert probe.returncode == 1, probe.stderr
    assert "invalid literal" in probe.stderr
    assert "while reading /nowhere/config.toml" in probe.stderr


def test_notes_follow_the_trace(capsys: pytest.CaptureFixture[str]) -> None:
    error = _note_carrying_exception()

    excepthook.with_notes(_traceback_only)(
        type(error), error, error.__traceback__
    )

    printed = capsys.readouterr().err
    assert printed.index("the traceback") < printed.index("while reading")


def test_an_exception_without_notes_is_unchanged(
    capsys: pytest.CaptureFixture[str],
) -> None:
    error = ValueError("invalid literal")

    excepthook.with_notes(_traceback_only)(type(error), error, None)

    assert capsys.readouterr().err == "the traceback\n"
