"""Argparse `type=` callbacks and defaults shared by more than one CLI."""

import argparse
import pathlib
from collections.abc import Sequence


def non_negative_limit(value: str) -> int:
    """Reject a negative `--limit` before it reaches the corpus loaders.

    `load_split` refuses one too, but a `ValueError` out of the data layer
    names neither the flag nor the command, so a typo like `-1` would be
    reported far from the argument that caused it.

    :param value: the raw command-line argument.
    :return: the parsed value.
    :raises argparse.ArgumentTypeError: if it parses to a negative integer.
    """
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(
            f"must be non-negative (0 or omitted means no limit); "
            f"got {parsed}."
        )
    return parsed


def readable_path(path: str) -> pathlib.Path:
    """Reject a path that is not a readable file, at parse time.

    :param path: the raw command-line argument.
    :return: the path.
    :raises argparse.ArgumentTypeError: if it names no file.
    """
    resolved = pathlib.Path(path)
    if not resolved.is_file():
        raise argparse.ArgumentTypeError(f"{path} is not a readable file")
    return resolved


def resolve_datasets(
    parser: argparse.ArgumentParser, given: Sequence[pathlib.Path | str]
) -> list[pathlib.Path]:
    """The corpus files to read: those named, or the configured ones.

    Defaulting rather than requiring the list is what keeps a store's
    contents and a training run's contents the same set. `load_split`
    appends a block of each noise pool to every split, so a list retyped
    per invocation silently omitted the pools, and every one of their
    documents was dropped from its batch or masked out of the tagger loss.

    :param parser: the parser to report a missing default through.
    :param given: the paths named on the command line, already checked by
        `readable_path`; empty to take the configured set.
    :return: the files to read.
    """
    if given:
        return [pathlib.Path(path) for path in given]

    # Imported here, not at module scope: `precompute-token-labels` takes
    # each document's gold set off the split frame's own columns and is
    # tested for importing none of the data layer, so only a run that
    # actually defaults its file list pays for that import.
    import brenda_references

    configured = list(brenda_references.corpus_files())
    missing = [path for path in configured if not path.is_file()]
    if missing:
        parser.error(
            "the configured corpus files are not all present: "
            + ", ".join(str(path) for path in missing)
            + " — fetch the data, point BRENDA_DATA_DIR at it, or name the "
            "files to read on the command line"
        )
    return configured
