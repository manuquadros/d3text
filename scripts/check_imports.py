#!/usr/bin/env python
"""Import every module `[project.scripts]` names, one subprocess each.

The gate the static checks cannot be. ruff imports nothing and mypy resolves
circular imports on paper, so both report a clean tree while `import
d3text.runtime` raises `ImportError: cannot import name ... (most likely due
to a circular import)` and every command the project ships is dead. pytest
does notice, but not usefully: a package that will not import also breaks
collection of the test modules that would report it, and pytest abandons the
session on a collection error, so the entry-point test written to name this
failure never runs. What comes out is dozens of identical tracebacks and no
test result at all.

One subprocess per entry point, never one process for all of them: a module
already in `sys.modules` from an earlier import launders the very ordering
that produces the cycle, so a single process can report success on a tree
that fails in production.

Exit codes: 0 every entry point imported · 1 one or more did not, or there
was nothing to import.
"""

import argparse
import os
import pathlib
import subprocess
import sys
import tomllib

REPO = pathlib.Path(__file__).resolve().parent.parent


def entry_point_modules(root: pathlib.Path) -> list[str]:
    """The distinct modules the project's console scripts name.

    :param root: project root holding the `pyproject.toml` to read.
    :return: the module part of each `module:attr` reference, deduplicated.
    """
    with (root / "pyproject.toml").open("rb") as pyproject:
        scripts = tomllib.load(pyproject)["project"].get("scripts", {})
    return sorted(
        {reference.split(":", 1)[0] for reference in scripts.values()}
    )


def import_path(root: pathlib.Path) -> list[str]:
    """The source directories of `root` and of any path dependency beside it.

    Prepended to the subprocess's `PYTHONPATH` so the gate answers for the
    tree it was pointed at. Without it the answer comes from whatever the
    environment has installed — in a `git worktree` that is the main
    checkout, whose `.pth` file `site` appends after `PYTHONPATH`, so an
    unprefixed check passes while the tree under test is broken.

    :param root: project root to collect source directories from.
    :return: existing `src` directories, outermost first.
    """
    candidates = [root / "src", *sorted(root.glob("*/src"))]
    return [str(path) for path in candidates if path.is_dir()]


def import_in_a_fresh_interpreter(
    module: str, root: pathlib.Path
) -> subprocess.CompletedProcess[str]:
    """Import `module` in its own interpreter, echoing the file it resolved to.

    :param module: dotted module name to import.
    :param root: project root whose sources take precedence on the path.
    :return: the finished subprocess, its last stdout line the module's file.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        path for path in (*import_path(root), env.get("PYTHONPATH")) if path
    )
    return subprocess.run(
        [sys.executable, "-c", f"import {module}; print({module}.__file__)"],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )


def main() -> int:
    """Check every declared entry point and report each one by name.

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

    modules = entry_point_modules(root)
    if not modules:
        print(
            f"COULD NOT RUN: {root}/pyproject.toml declares no "
            f"[project.scripts], so this gate imported nothing.",
            file=sys.stderr,
        )
        return 1

    failed = []
    for module in modules:
        result = import_in_a_fresh_interpreter(module, root)
        if result.returncode == 0:
            # The resolved file, so a pass names the tree it vouches for.
            lines = result.stdout.splitlines()
            print(f"PASSED  {module}  {lines[-1] if lines else '?'}")
        else:
            failed.append(module)
            print(f"FAILED  {module}")
            print(result.stdout, result.stderr, sep="", end="", file=sys.stderr)

    if failed:
        print(
            f"\n{len(failed)} of {len(modules)} entry points cannot be "
            f"imported: {', '.join(failed)}",
            file=sys.stderr,
        )
        return 1

    print(f"\nAll {len(modules)} entry points import.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
