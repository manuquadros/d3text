#!/usr/bin/env python
"""Did the store get built with the guarded dictionary, and what came out?

Two checks that have to happen between building the token labels and training
on them, because **nothing in the store records which dictionary produced it**
(BUG-60). A store built before the designation guard and one built after are
byte-comparable only in the places they differ, so a run pointed at a stale
store trains on precisely the mislabelled targets DEC-04 exists to remove, and
reports nothing unusual while doing it.

The first check is on the *index*: the ordinary-English designations must be
gone and the near-threshold taxonomic names must not be. It rebuilds the index
rather than reading the store, so it fails before the two hours rather than
after.

The second is on the *store*: the realised share of each target. FEAT-05
recorded 1.83% positive / 3.12% ignore / 95.05% negative at word level under
the unguarded dictionary, and the guard should move `ignore` and `positive`
down without touching the shape. A distribution far from that is the signal
that something other than the guard changed.

Exits non-zero if the index check fails, so `run.sh` can gate on it.
"""

import argparse
import collections
import json
import pathlib
import sys

import h5py
import numpy
from d3text import corpus, logs, surface_forms, token_labels

# Ordinary English that BRENDA registers as strain designations, plus the two
# other-organism category nouns. Each fires on between 6% and 27% of the
# corpus, and every one of them is a mislabelled mention wherever it appears.
MUST_BE_ABSENT = (
    "sensitive",
    "original",
    "yielding",
    "hybrid",
    "aerobic",
    "shanghai",
    "california",
    "chinese",
    "animal",
    "unidentified",
)

# The legitimate names closest to the cutoff from below. They are what a
# raised threshold or a re-estimated frequency table would take first, and
# losing them would cost most of the bacterial channel — silently, since a
# missing surface form produces no error, only a mention that stops being
# found.
MUST_BE_PRESENT = (
    "escherichia",
    "pseudomonas",
    "bacillus",
    "streptomyces",
    "mycobacterium",
    "catalase",
    "trypsin",
    "lysozyme",
)

STREAM_BATCH = 1000


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="label-audit",
        description="Check the guard took, and report the label distribution.",
    )
    parser.add_argument("entity_tables", type=pathlib.Path)
    parser.add_argument("store", type=pathlib.Path)
    parser.add_argument("datasets", nargs="+", type=pathlib.Path)
    parser.add_argument(
        "--documents",
        type=int,
        default=400,
        help="Documents to sample for the distribution (default 400).",
    )
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def build_index(
    entity_tables: pathlib.Path, datasets: list[pathlib.Path]
) -> surface_forms.SurfaceFormIndex:
    """The index `precompute-token-labels` builds, by the same route."""
    tables = surface_forms.load_entity_tables(entity_tables)
    return surface_forms.build_index(
        surface_forms.brenda_surface_forms(
            tables,
            (
                names
                for dataset in datasets
                for names in corpus.other_organism_names(dataset, STREAM_BATCH)
            ),
        )
    )


def audit_index(index: surface_forms.SurfaceFormIndex) -> dict[str, list[str]]:
    """Which of the two watch lists are in the index when they should not be.

    Looked up through `index.lookup`, not by inspecting the tables, so this
    asks the question the labeller asks: not "is the key there" but "does
    this word reach an entity".
    """
    present = [
        word for word in MUST_BE_ABSENT if index.lookup([word]) != frozenset()
    ]
    absent = [
        word for word in MUST_BE_PRESENT if index.lookup([word]) == frozenset()
    ]
    return {"leaked": present, "lost": absent}


def distribution(
    store_path: pathlib.Path, sample: int
) -> dict[str, float | int]:
    """The realised share of each target over the store's first `sample` keys.

    Counted per token rather than per word, which is the geometry the loss
    sees; FEAT-05's recorded shares are word-level, so the two are close but
    not the same number.
    """
    counts: collections.Counter[str] = collections.Counter()
    documents = 0
    with h5py.File(store_path, "r") as store:
        for key in store:
            if documents >= sample:
                break
            labels = token_labels.load_token_labels(store, key)
            codes = labels.codes
            counts["ignore"] += int(
                numpy.count_nonzero(codes == token_labels.IGNORE_INDEX)
            )
            counts["negative"] += int(
                numpy.count_nonzero(codes == token_labels.OUTSIDE)
            )
            counts["positive"] += int(
                numpy.count_nonzero(
                    (codes != token_labels.IGNORE_INDEX)
                    & (codes != token_labels.OUTSIDE)
                )
            )
            counts["mentions"] += int(labels.spans.shape[0])
            documents += 1

    total = counts["ignore"] + counts["negative"] + counts["positive"]
    if total == 0:
        return {"documents": documents, "tokens": 0}
    return {
        "documents": documents,
        "tokens": total,
        "positive": round(counts["positive"] / total, 4),
        "ignore": round(counts["ignore"] / total, 4),
        "negative": round(counts["negative"] / total, 4),
        "mentions_per_document": round(counts["mentions"] / documents, 1),
    }


def main() -> int:
    logs.configure()
    args = read_args()

    index = build_index(args.entity_tables, args.datasets)
    verdict = audit_index(index)
    summary: dict[str, object] = {
        "surface_forms": len(index),
        "entities": len(index.entity_ids),
        **verdict,
    }

    print(f"index: {len(index)} forms over {len(index.entity_ids)} entities")
    for word in MUST_BE_ABSENT:
        reached = index.lookup([word])
        state = f"LEAKED -> {sorted(reached)[:3]}" if reached else "dropped"
        print(f"  {word:16s} {state}")
    for word in MUST_BE_PRESENT:
        reached = index.lookup([word])
        print(f"  {word:16s} {'kept' if reached else 'LOST'}")

    if args.store.exists():
        summary["distribution"] = distribution(args.store, args.documents)
        print(f"\nstore: {json.dumps(summary['distribution'], indent=2)}")

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(summary, indent=2))

    if verdict["leaked"] or verdict["lost"]:
        print(
            f"\nFAIL: {len(verdict['leaked'])} ordinary words still name an "
            f"entity, {len(verdict['lost'])} real names were dropped.",
            file=sys.stderr,
        )
        print(
            "The store was built with an unguarded dictionary, or "
            "COMMON_WORD_ZIPF moved. Regenerate it before training.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
