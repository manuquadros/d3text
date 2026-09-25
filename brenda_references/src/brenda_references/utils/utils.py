"""Utility functions for brenda_references"""

import string

import nltk
from aiotinydb.middleware import AIOMiddlewareMixin
from rapidfuzz import fuzz
from tinydb.middlewares import CachingMiddleware as SyncCachingMiddleware


class CachingMiddleware(SyncCachingMiddleware, AIOMiddlewareMixin):
    """Adding async powers to CachingMiddleware."""


def ratio(a: str, b: str) -> float:
    """Compute the normalized Indel similarity of `a` and `b`.

    :returns: average of the similarity computation over two conditions:
              lower-casing the arguments vs not doing so.
    """
    return (
        fuzz.ratio(a, b, processor=lambda s: s.lower()) + fuzz.ratio(a, b)
    ) / 2


def fuzzy_find_all(
    text: str,
    pattern: str,
    threshold: int = 83,
    *,
    try_abbrev: bool = False,
) -> list[tuple[int, int]]:
    """Find all fuzzy matches of `pattern` in `text` with given `threshold`."""
    matches = []

    if not pattern.strip():
        return []

    if text:
        words = text.split()

        for i, group in enumerate(nltk.ngrams(words, len(pattern.split()))):
            test_str = " ".join(group).strip(string.punctuation)
            ratio_pass = ratio(test_str, pattern) >= threshold
            abbrev_ratio_pass = (
                ratio(test_str, abbreviate_bacteria(pattern)) >= threshold
                if try_abbrev
                else False
            )
            if ratio_pass or abbrev_ratio_pass:
                start = sum(len(w) + 1 for w in words[:i])
                end = start + len(test_str)
                matches.append((start, end))

    return matches


def abbreviate_bacteria(name: str) -> str:
    """Abbreviate the genus component of `name`."""
    if name:
        parts = name.split()
        parts[0] = parts[0][0] + "."

        return " ".join(parts)

    return name
