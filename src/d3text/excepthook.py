"""The uncaught-exception handler, and the notes stackprinter drops.

``BaseException.add_note`` is the obvious way to attach context to an exception
raised by someone else's code -- pydantic's ``ValidationError``, say -- but
stackprinter renders only the traceback and the exception's own message. A note
attached anywhere in this package therefore reaches pytest and a plain
``python -c``, which use the stdlib hook, and is dropped from every console
script, which is the one path it was written for.
"""

import sys
import types
from collections.abc import Callable

ExceptHook = Callable[
    [type[BaseException], BaseException, types.TracebackType | None], None
]


def with_notes(hook: ExceptHook) -> ExceptHook:
    """Wrap ``hook`` so an exception's ``__notes__`` follow its traceback."""

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

    ``kwargs`` go to ``stackprinter.set_excepthook``; a missing stackprinter
    raises ``ModuleNotFoundError`` and leaves the hook alone.
    """
    import stackprinter

    previous = sys.excepthook
    stackprinter.set_excepthook(**kwargs)
    # Under IPython, `set_excepthook` patches IPython's own printer and leaves
    # `sys.excepthook` alone -- wrapping it then would decorate the stdlib hook,
    # which prints the notes itself, and every note would appear twice.
    if sys.excepthook is not previous:
        sys.excepthook = with_notes(sys.excepthook)
