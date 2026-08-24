"""`d3text.logs`: where the library's console output goes, and who decides.

Two guarantees. The library must not decide — importing `d3text` installs no
handler and touches no level, so an application that embeds it keeps its own
logging — and the handler an entry point *does* install must not smear a live
progress bar, which is the property the training loop had for free while it
was calling `tqdm.write` directly.
"""

import ast
import io
import json
import logging
import pathlib
import subprocess
import sys

import pytest
from d3text import logs
from tqdm import tqdm

REPO_ROOT = pathlib.Path(__file__).parent.parent

# Read the logging state, import the library, read it again. The sentinel is
# needed because `d3text/__init__` prints on a missing optional dependency.
_IMPORT_PROBE = """
import json, logging

def snapshot():
    package = logging.getLogger("d3text")
    root = logging.getLogger()
    return {
        "package_handlers": len(package.handlers),
        "package_level": package.level,
        "package_propagate": package.propagate,
        "root_handlers": len(root.handlers),
        "root_level": root.level,
    }

before = snapshot()
import d3text
import d3text.models.models
after = snapshot()
print("@@" + json.dumps({"before": before, "after": after}))
"""


@pytest.mark.slow
def test_importing_the_library_installs_no_logging() -> None:
    """Can only be checked in a fresh interpreter: pytest imported these long
    ago, and a handler installed at import time is invisible afterwards."""
    probe = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert probe.returncode == 0, probe.stderr

    state = json.loads(
        next(
            line[2:]
            for line in probe.stdout.splitlines()
            if line.startswith("@@")
        )
    )

    assert state["after"] == state["before"]
    assert state["after"]["package_handlers"] == 0
    assert state["after"]["package_propagate"] is True


def test_configure_installs_exactly_one_handler(
    restore_package_logger: logging.Logger,
) -> None:
    logger = logs.configure(logging.INFO)

    assert logger is restore_package_logger
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logs.TqdmLoggingHandler)


def test_configuring_twice_does_not_double_every_line(
    restore_package_logger: logging.Logger,
) -> None:
    """A second entry point call must replace the handler, not stack on it."""
    stream = io.StringIO()
    logs.configure(logging.INFO, stream=stream)
    logger = logs.configure(logging.INFO, stream=stream)

    logger.info("once")

    assert stream.getvalue() == "once\n"


def test_the_root_logger_is_left_alone(
    restore_package_logger: logging.Logger,
) -> None:
    root = logging.getLogger()
    before = (list(root.handlers), root.level)

    logger = logs.configure(logging.INFO, stream=io.StringIO())

    assert (list(root.handlers), root.level) == before
    assert logger.propagate is False


def test_a_live_bar_is_cleared_and_redrawn_around_a_log_line(
    restore_package_logger: logging.Logger,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The hazard the handler exists for.

    A `StreamHandler` writes into whatever terminal line the bar occupies, so
    the bar is left duplicated and half-overwritten. Going through
    `tqdm.write` erases every live bar, writes, and redraws them — which shows
    up as the bar's *own* stream being written to by a call that only logged.
    """
    logger = logs.configure(logging.INFO)

    with tqdm(total=3, mininterval=0) as bar:
        bar.update(1)
        capsys.readouterr()

        logger.info("epoch 1 done")

        captured = capsys.readouterr()

    assert captured.out == "epoch 1 done\n"
    assert captured.err, "the bar was neither cleared nor redrawn"


def test_info_is_verbatim_and_a_warning_names_its_level(
    restore_package_logger: logging.Logger,
) -> None:
    stream = io.StringIO()
    logger = logs.configure(logging.INFO, stream=stream)

    logger.info("Average training loss: 0.5")
    logger.warning("No samples found.")

    assert stream.getvalue() == (
        "Average training loss: 0.5\nWARNING: No samples found.\n"
    )


def test_the_level_silences_what_is_below_it(
    restore_package_logger: logging.Logger,
) -> None:
    stream = io.StringIO()
    logger = logs.configure(logging.WARNING, stream=stream)

    logger.info("Average training loss: 0.5")
    logger.warning("No samples found.")

    assert stream.getvalue() == "WARNING: No samples found.\n"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, logging.INFO),
        ("DEBUG", logging.DEBUG),
        ("warning", logging.WARNING),
        ("  ERROR  ", logging.ERROR),
        ("30", logging.WARNING),
        ("chatty", logging.INFO),
        ("", logging.INFO),
    ],
)
def test_level_from_env(value: str | None, expected: int) -> None:
    """A typo in a verbosity knob must not cost a multi-hour run."""
    env = {} if value is None else {logs.LEVEL_VARIABLE: value}

    assert logs.level_from_env(env) == expected


def test_configure_reads_the_environment_when_given_no_level(
    restore_package_logger: logging.Logger,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(logs.LEVEL_VARIABLE, "ERROR")

    assert logs.configure().level == logging.ERROR


#: `logs.py` is the one sanctioned exit — the handler *is* the `tqdm.write`.
#: `__init__.py` is exempt because its two missing-dependency notices fire
#: while the package is being imported, before any entry point could have
#: configured a handler for them to reach.
_EXEMPT = frozenset({"__init__.py", "logs.py"})

_MUST_NOT_PRINT = tuple(
    sorted(
        str(path.relative_to(REPO_ROOT))
        for path in (REPO_ROOT / "src" / "d3text").rglob("*.py")
        if path.name not in _EXEMPT
    )
)


def _writes_to_the_console(node: ast.AST) -> bool:
    """A bare `print(...)`, or the `tqdm.write(...)` it was traded for."""
    if not isinstance(node, ast.Call):
        return False

    if isinstance(node.func, ast.Name):
        return node.func.id == "print"

    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "write"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "tqdm"
    )


@pytest.mark.parametrize("module", _MUST_NOT_PRINT)
def test_no_module_writes_to_the_console_directly(module: str) -> None:
    """A `print` here is a decision about someone else's terminal.

    It cannot be levelled and cannot be redirected by the process that owns
    the output. `tqdm.write` fixes only the second half of that — the bar
    survives, the verbosity is still the library's to choose — which is why it
    counts as the same defect.
    """
    tree = ast.parse((REPO_ROOT / module).read_text())

    calls = [
        node.lineno for node in ast.walk(tree) if _writes_to_the_console(node)
    ]

    assert calls == [], f"{module} writes to the console at lines {calls}"
