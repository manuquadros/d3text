"""Argparse `type=` callbacks shared by more than one CLI entry point."""

import argparse


def non_negative_limit(value: str) -> int:
    """Reject a negative `--limit` before it silently empties the split.

    `load_split` truncates a `RangeIndex` at `limit - 1`, which keeps zero
    rows for any negative `limit`, so a typo like `-1` would otherwise size
    the entity vocabulary to nothing far from the flag that caused it.

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
