"""Console logging for the entry points.

The library logs through `logging.getLogger(__name__)` and installs nothing on
the way in: importing `d3text` must not decide where anyone else's records go,
the same first-writer-wins hazard `runtime.configure` exists for. `configure`
is called from an entry point — `runtime.configure` does it for `train`, `tune`
and `evaluate`, the precompute commands call it themselves — and puts one
handler on the ``d3text`` logger with ``propagate = False``, so the root logger
and any configuration the importing application already has are left alone.

`d3text/__init__.py`'s two missing-dependency notices stay bare `print`s on
purpose: they fire while the package is being imported, before any entry point
could have configured a handler, so a logger would drop them.

The handler writes through `tqdm.write`. A plain stream write lands in
whatever terminal line a live progress bar occupies and smears it, which is why
the training loop wrote its epoch numbers with `tqdm.write` in the first place;
routing them through `logging` had to keep that property, not trade it for a
verbosity knob.
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

    Narrower than `typing.TextIO`, which is a protocol wide enough that
    `io.StringIO` does not satisfy it — and a stream a test can read back is
    the only way to pin what the handler wrote.
    """

    def write(self, text: str, /) -> int: ...

    def flush(self) -> None: ...


class TqdmLoggingHandler(logging.Handler):
    """Write records with `tqdm.write`, which redraws the live bars around them.

    `stream` is resolved at emit time rather than stored, so a handler
    installed before a stream is swapped — pytest's capture, a redirect — still
    writes where the process's stdout currently points.
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

    INFO is the narration these commands printed verbatim before it moved
    behind `logging`, so it has to keep printing verbatim; a warning that looks
    exactly like narration is a warning nobody reads.
    """

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)

        if record.levelno > logging.INFO:
            return f"{record.levelname}: {message}"

        return message


def level_from_env(environ: Mapping[str, str] | None = None) -> int:
    """Resolve ``D3TEXT_LOG_LEVEL`` to a level, defaulting to INFO.

    An unparseable value falls back to the default rather than raising: losing
    a multi-hour run to a typo in a verbosity knob would be a poor trade.
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

    Replaces any handler a previous call left, so calling it twice in one
    process does not double every line. ``level=None`` reads
    ``D3TEXT_LOG_LEVEL``.
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
