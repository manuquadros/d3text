"""Utility functions for brenda_references"""

import string
from collections.abc import Iterable

import nltk
import pandas as pd
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


def entities_in_dataset(
    data: pd.DataFrame,
    entity_columns: Iterable = (
        "bacteria",
        "strains",
        "enzymes",
        "other_organisms",
    ),
) -> dict[str, set[int]]:
    """Retrieve the set of entity ids for each class in `data`

    This function expects that the dataset contains one or more of the
    following columns: "bacteria", "strains", "enzymes", "other_organisms",
    where each column contains a collection (e.g., list) of entity IDs. It
    returns a dictionary where each key is the column name and its value is the
    set of all entity IDs (as ints) found in that column.
    """
    entities: dict[str, set[int]] = {}
    for col in entity_columns:
        entities[col] = set()
        if col in data.columns:
            # Iterate over the non-null values in the column
            for value in data[col].dropna():
                # If the column entry is a list (or set), extend the set;
                # otherwise add the single value.
                if isinstance(value, (list, set)):
                    entities[col].update(int(x) for x in value if x is not None)
                else:
                    entities[col].add(int(value))
    return entities


def jaccard_similarity(
    a: pd.DataFrame,
    b: pd.DataFrame,
    entity_columns: Iterable = (
        "bacteria",
        "strains",
        "enzymes",
        "other_organisms",
    ),
) -> dict[str, float]:
    """Compute the Jaccard similarity between `a` and `b` entity columns."""
    aents = entities_in_dataset(a, entity_columns=entity_columns)
    bents = entities_in_dataset(b, entity_columns=entity_columns)
    jaccard_indices = {}

    for col in entity_columns:
        intersection = aents[col] & bents[col]
        union = aents[col] | bents[col]
        if not union:
            msg = f"Union is empty for {col}."
            raise ValueError(msg)
        jaccard_indices[col] = len(intersection) / len(union)

    return jaccard_indices
