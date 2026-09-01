"""The uncaught-exception handler, and the notes stackprinter drops.

stackprinter renders only the traceback and the exception's own message, so a
note attached with `add_note` reaches pytest and a plain `python -c` and is
dropped from every console script — the one path it was written for.
"""

import sys
import types
from collections.abc import Callable

ExceptHook = Callable[
    [type[BaseException], BaseException, types.TracebackType | None], None
]


def with_notes(hook: ExceptHook) -> ExceptHook:
    """Wrap `hook` so an exception's `__notes__` follow its traceback.

    :param hook: the excepthook to extend.
    :return: the wrapped hook.
    """

    def excepthook(
        exc_type: type[BaseException],
        exc: BaseException,
        traceback: types.TracebackType | None,
    ) -> None:
        hook(exc_type, exc, traceback)
        for note in getattr(exc, "__notes__", ()):
            print(note, file=sys.stderr)

    return excepthook


def install(**kwargs: object) -> None:
    """Install stackprinter's excepthook, extended to print notes.

    :param kwargs: passed to `stackprinter.set_excepthook`; a missing
        stackprinter raises `ModuleNotFoundError` and leaves the hook alone.
    """
    import stackprinter

    previous = sys.excepthook
    stackprinter.set_excepthook(**kwargs)
    # Under IPython, `set_excepthook` patches IPython's own printer and leaves
    # `sys.excepthook` alone -- wrapping it then would decorate the stdlib hook,
    # which prints the notes itself, and every note would appear twice.
    if sys.excepthook is not previous:
        sys.excepthook = with_notes(sys.excepthook)
