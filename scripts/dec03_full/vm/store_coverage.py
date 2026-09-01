"""Does the store hold every document the run will ask it for?

A store covering 95% of the corpus is not 95% as good: the missing 5% is
re-embedded on every epoch at the 2.1 doc/s the store exists to avoid, and
nothing says so — a miss is silent by design, since it is also what an
unconfigured store looks like.

The check is a floor rather than an equality because a few documents are
legitimately never stored, so `--min-coverage` is what separates those from a
store built against the wrong corpus or a build that stopped early and got
stamped as finished. The ids are compared as sets, which keeps the corpus's
repeated pubmed_ids from reading as missing documents.
"""

import argparse
import json
import pathlib
import sys

import lmdb
import polars as pl

from d3text.embeddings_store import _PROVENANCE_KEY

CORPUS = pathlib.Path(__file__).resolve().parents[3] / (
    "brenda_references/src/brenda_references/data"
)
SOURCES = (
    "training_data.csv",
    "validation_data.csv",
    "test_data.csv",
    "pmc_linguistics_articles.json",
)


def ids(path: pathlib.Path) -> set[str]:
    frame = pl.scan_csv(path) if path.suffix == ".csv" else pl.scan_ndjson(path)
    return {
        str(value)
        for value in frame.select("pubmed_id").collect()["pubmed_id"].to_list()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("store")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.0,
        help=(
            "fail unless every source is at least this fraction stored "
            "(0 reports without asserting)"
        ),
    )
    args = parser.parse_args()

    env = lmdb.open(args.store, readonly=True, lock=False, readahead=False)
    with env.begin() as transaction:
        stored = {
            key.decode()
            for key in transaction.cursor().iternext(keys=True, values=False)
            if key != _PROVENANCE_KEY
        }

    report: dict[str, object] = {"store": args.store, "keys": len(stored)}
    short: list[str] = []
    for name in SOURCES:
        wanted = ids(CORPUS / name)
        missing = wanted - stored
        held = len(wanted) - len(missing)
        coverage = held / len(wanted) if wanted else 1.0
        report[name] = {
            "documents": len(wanted),
            "stored": held,
            "missing": len(missing),
            "coverage": coverage,
            "missing_examples": sorted(missing)[:10],
        }
        if coverage < args.min_coverage:
            short.append(
                f"{name}: {held:,} of {len(wanted):,} stored "
                f"({coverage:.1%}, floor {args.min_coverage:.1%})"
            )

    report["min_coverage"] = args.min_coverage
    report["below_floor"] = short
    print(json.dumps(report, indent=2))
    pathlib.Path(args.out).write_text(json.dumps(report, indent=2))

    if short:
        print("\nSTORE COVERAGE TOO LOW:", file=sys.stderr)
        for line in short:
            print(f"  - {line}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
