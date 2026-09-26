#!/usr/bin/env python
"""Report code nothing uses, through deadcode, and fail when there is any.

Exit codes: 0 nothing unused · 1 unused code found, or nothing was checked.
Settings live in `[tool.deadcode]`; rationale for the gate, and for this
wrapper rather than the `deadcode` command: docs/how-to/run-the-checks.md
(Dead code).
"""

import argparse
import fnmatch
import os
import pathlib
import sys

from deadcode.actions.find_python_filenames import find_python_filenames
from deadcode.actions.parse_arguments import parse_arguments
from deadcode.cli import main as deadcode_main

REPO = pathlib.Path(__file__).resolve().parent.parent

# Every directory whose code counts as a caller. `only` in `[tool.deadcode]`
# narrows which of them are reported on.
PATHS = ["src", "scripts", "tests"]


def main() -> int:
    """Run deadcode over the project and turn its report into an exit code.

    :return: the process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "root",
        nargs="?",
        type=pathlib.Path,
        default=REPO,
        help="project root to check; defaults to this repository",
    )
    root = parser.parse_args().root.resolve()

    # deadcode reads `pyproject.toml` from the working directory only.
    os.chdir(root)
    command = [*PATHS, "--no-color"]
    args = parse_arguments(command)
    reported = [
        name
        for name in find_python_filenames(args=args)
        if not args.only
        or any(fnmatch.fnmatch(name, pattern) for pattern in args.only)
    ]
    if not reported:
        print(
            f"COULD NOT RUN: no Python file under {', '.join(PATHS)} in "
            f"{root} matches `only`, so this gate checked nothing.",
            file=sys.stderr,
        )
        return 1

    # The `deadcode` console script exits 0 whatever it finds; `main`
    # returns the report instead, which is the only failure signal it gives.
    report = deadcode_main(command)
    if report:
        print(report, file=sys.stderr)
        return 1

    print(f"PASSED  no unused code in {len(reported)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
