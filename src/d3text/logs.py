"""Console logging for the entry points.

The library installs nothing on import: deciding where anyone else's records go
is the same first-writer-wins hazard `runtime.configure` exists for.
`configure` puts one handler on the `d3text` logger with `propagate = False`,
and it writes through `tqdm.write`, since a plain stream write smears the live
progress bar.
"""

import logging
import os
import sys
from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from tqdm import tqdm

PACKAGE_LOGGER = "d3text"

#: Selects the verbosity of a run rather than of a machine, so it is an
#: environment variable and not a `config.toml` key — and it has to be read
#: before `command_line_args()`, since `runtime.configure()` runs first.
LEVEL_VARIABLE = "D3TEXT_LOG_LEVEL"

DEFAULT_LEVEL = logging.INFO


@runtime_checkable
class WritableStream(Protocol):
    """What `tqdm.write` needs of its destination.

    Narrower than `typing.TextIO`, which `io.StringIO` does not satisfy — and a
    stream a test can read back is the only way to pin what the handler wrote.
    """

    def write(self, text: str, /) -> int: ...

    def flush(self) -> None: ...


class TqdmLoggingHandler(logging.Handler):
    """Write records with `tqdm.write`, which redraws the bars around them.

    `stream` is resolved at emit time rather than stored, so a handler
    installed before a stream is swapped still writes where stdout currently
    points.
    """

    def __init__(self, stream: WritableStream | None = None) -> None:
        super().__init__()
        self.stream = stream

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=self.stream or sys.stdout)
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


class LevelPrefixFormatter(logging.Formatter):
    """Name the level of anything more urgent than INFO, and nothing else.

    INFO is narration these commands printed verbatim before it moved behind
    `logging`, and a warning that looks exactly like narration is one nobody
    reads.
    """

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)

        if record.levelno > logging.INFO:
            return f"{record.levelname}: {message}"

        return message


def level_from_env(environ: Mapping[str, str] | None = None) -> int:
    """Resolve `D3TEXT_LOG_LEVEL` to a level, defaulting to INFO.

    :param environ: the environment to read; the process's own by default.
    :return: the level, falling back to the default on an unparseable value
        rather than raising.
    """
    env: Mapping[str, str] = os.environ if environ is None else environ
    requested = env.get(LEVEL_VARIABLE)

    if requested is None:
        return DEFAULT_LEVEL

    resolved = logging.getLevelName(requested.strip().upper())

    if isinstance(resolved, int):
        return resolved

    if requested.strip().isdigit():
        return int(requested.strip())

    return DEFAULT_LEVEL


def configure(
    level: int | None = None, *, stream: WritableStream | None = None
) -> logging.Logger:
    """Install the package's console handler. Call once, from an entry point.

    Replaces any handler a previous call left, so calling it twice does not
    double every line.

    :param level: the verbosity; `None` reads `D3TEXT_LOG_LEVEL`.
    :param stream: where to write; the process's stdout by default.
    :return: the configured `d3text` logger.
    """
    logger = logging.getLogger(PACKAGE_LOGGER)

    for installed in list(logger.handlers):
        logger.removeHandler(installed)
        installed.close()

    handler = TqdmLoggingHandler(stream)
    handler.setFormatter(LevelPrefixFormatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level_from_env() if level is None else level)

    # Nothing above `d3text` needs to have been configured for the package's
    # own output to appear, and nothing above it should receive a duplicate of
    # every record it emits.
    logger.propagate = False

    return logger
