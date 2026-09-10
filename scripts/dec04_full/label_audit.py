#!/usr/bin/env python
"""Did the store get built with the guarded dictionary, and what came out?

**Nothing in the store records which dictionary produced it**, so a run pointed
at a stale one trains on precisely the mislabelled targets the guard exists to
remove and reports nothing unusual. The first check is on the *index* — the
ordinary-English designations gone, the near-threshold taxonomic names kept —
and rebuilds it rather than reading the store, so it fails before the two hours
rather than after. The second is on the *store*: the realised share of each
target, against the 1.83% positive / 3.12% ignore / 95.05% negative recorded at
word level under the unguarded dictionary.

Exits non-zero if the index check fails, so `run.sh` can gate on it.
"""

import argparse
import collections
import json
import pathlib
import sys

import h5py
import numpy
from d3text import logs, surface_forms, token_labels
from d3text.cli.precompute_token_labels import build_index

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


def audit_index(index: surface_forms.SurfaceFormIndex) -> dict[str, list[str]]:
    """Which of the two watch lists are in the index when they should not be.

    Looked up through `index.lookup` rather than by inspecting the tables, so
    this asks the labeller's question: not "is the key there" but "does this
    word reach an entity".
    """
    present = [
        word for word in MUST_BE_ABSENT if index.lookup([word]) != frozenset()
    ]
    absent = [
        word for word in MUST_BE_PRESENT if index.lookup([word]) == frozenset()
    ]
    return {"leaked": present, "lost": absent}


def check_store_index(
    store: h5py.File, stamp: token_labels.IndexStamp
) -> str | None:
    """Whether the store's recorded index agrees with the one this run built.

    :param store: an open label store.
    :param stamp: the stamp of the index this audit just built.
    :return: ``None`` if the store agrees; otherwise a message naming both
        digest prefixes, fit to print as the audit's failure reason.
    """
    try:
        token_labels.check_index(store, stamp)
    except ValueError as error:
        return (
            f"this audit's index is {stamp.digest[:12]}, and the store "
            f"disagrees: {error}"
        )
    except KeyError as error:
        return (
            f"this audit's index is {stamp.digest[:12]}, but the store "
            f"records no index to compare it to: {error}"
        )
    return None


def distribution(
    store_path: pathlib.Path, sample: int
) -> dict[str, float | int]:
    """The realised share of each target over the store's first `sample` keys.

    Counted per token, which is the geometry the loss sees; the recorded
    reference shares are word-level, so the two are close but not the same
    number.
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
    stamp = token_labels.IndexStamp.from_index(
        index,
        sources=[str(args.entity_tables), *(str(d) for d in args.datasets)],
    )
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

    index_mismatch: str | None = None
    if args.store.exists():
        summary["distribution"] = distribution(args.store, args.documents)
        print(f"\nstore: {json.dumps(summary['distribution'], indent=2)}")

        with h5py.File(args.store, "r") as store:
            index_mismatch = check_store_index(store, stamp)
        if index_mismatch is None:
            print(f"\nindex matches the store ({stamp.digest[:12]}).")
        else:
            summary["index_mismatch"] = index_mismatch
            print(f"\nFAIL: {index_mismatch}", file=sys.stderr)

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
    if verdict["leaked"] or verdict["lost"] or index_mismatch:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
