#!/usr/bin/env python
"""Keep only the candidates a corpus screen certifies as enzyme-free.

The third step after `collect_microbiology_sample.py` (draws candidates) and
`screen_enzyme_negatives.py` (measures the yield): this writes out the
candidates themselves, filtered to the ones that survive::

    python scripts/build_enzyme_negative_pool.py \\
        brenda_references/src/brenda_references/data/documents.json \\
        candidates.json pool.json

Screens under the **literal** reading — every exact match disqualifies, not
only a descriptive name — per feat-12's own recipe: a survivor is what makes
the negative true, and strictness is the point. Kept lines are copied
verbatim, in the same line-delimited shape the input already carries, so the
output needs no conversion to be read as a corpus.
"""

import argparse
import json
import pathlib

from d3text import corpus, logs, negative_screen, surface_forms

STREAM_BATCH = 1000


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="build-enzyme-negative-pool",
        description=(
            "Filter a candidate pool to the documents with zero matches "
            "under the strict (literal) enzyme screen."
        ),
    )
    parser.add_argument("entity_tables", type=pathlib.Path)
    parser.add_argument("candidates", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument(
        "--type",
        default="enzymes",
        choices=sorted(surface_forms.BRENDA_PREFIXES),
        help="the entity type screened for (default enzymes)",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        type=pathlib.Path,
        default=[],
        help="splits whose inline other-organism names join the index",
    )
    return parser.parse_args()


def build_index(
    entity_tables: pathlib.Path, datasets: list[pathlib.Path]
) -> surface_forms.SurfaceFormIndex:
    """The index `precompute-token-labels` builds, by the same route."""
    return surface_forms.build_index(
        surface_forms.brenda_surface_forms(
            surface_forms.load_entity_tables(entity_tables),
            (
                names
                for dataset in datasets
                for names in corpus.other_organism_names(dataset, STREAM_BATCH)
            ),
        )
    )


def main() -> None:
    logs.configure()
    args = read_args()

    index = build_index(args.entity_tables, args.datasets)
    digest = surface_forms.index_digest(index)
    prefix = surface_forms.BRENDA_PREFIXES[args.type]
    print(
        f"index {digest[:12]}: {len(index)} forms over {len(index.entity_ids)} entities"
    )

    total = kept = 0
    with (
        args.candidates.open(encoding="utf8") as source,
        args.output.open("w", encoding="utf8") as sink,
    ):
        for line in source:
            total += 1
            record = json.loads(line)
            text = corpus.document_text(
                record.get("abstract"), record.get("body")
            )
            matches = negative_screen.matched_forms(text, index, prefix=prefix)
            if negative_screen.LITERAL.accepts(matches):
                sink.write(line if line.endswith("\n") else line + "\n")
                kept += 1

    print(f"{kept} of {total} candidates kept (zero matches, literal reading)")


if __name__ == "__main__":
    main()
