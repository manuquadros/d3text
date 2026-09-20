"""Argparse `type=` callbacks shared by more than one CLI entry point."""

import argparse


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
