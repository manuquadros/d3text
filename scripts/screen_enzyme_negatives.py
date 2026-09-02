#!/usr/bin/env python
"""How many documents of a candidate pool name no enzyme, and why not.

The measurement that decides whether a hard-negative pool is worth sourcing.
It takes **several** corpora and screens them side by side, which is the whole
design: a yield on its own is uninterpretable, and it was two controls — a
pool that is enzyme-free by construction and one that is enzyme-positive by
construction — that showed the literal reading of the index was measuring
which documents avoid common acronyms rather than which name no enzyme::

    data=brenda_references/src/brenda_references/data
    python scripts/screen_enzyme_negatives.py $data/documents.json \\
        ~/Downloads/microbiology_sample.json \\
        $data/pmc_linguistics_articles.json $data/test_data.csv \\
        --limit 600

Every pool is read through `d3text.corpus`, so the line-delimited shape
`collect_microbiology_sample.py` writes and the split CSVs are both accepted
with no conversion. Pass the splits as `--datasets` to pool the inline
other-organism names into the index, which is what the labelling command does;
without them no `oth` form is in it, and a span that would have been read as an
organism can be read as an enzyme instead.
"""

import argparse
import dataclasses
import json
import pathlib

from d3text import corpus, logs, negative_screen, surface_forms

STREAM_BATCH = 1000

METADATA = ("journal", "year")
"""Columns the survivors are characterised by, where a pool carries them."""

SCREENS = (negative_screen.DESCRIPTIVE, negative_screen.LITERAL)
"""Both readings, always: the difference between them is the measurement."""


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="screen-enzyme-negatives",
        description=(
            "Report the zero-match rate of one or more corpora under the "
            "guarded surface-form index, and the matches behind it."
        ),
    )
    parser.add_argument(
        "entity_tables",
        type=pathlib.Path,
        help="TinyDB dump carrying the entity tables",
    )
    parser.add_argument(
        "pools",
        nargs="+",
        type=pathlib.Path,
        help="the corpora to screen; pass known negatives and positives too",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        type=pathlib.Path,
        default=[],
        help="splits whose inline other-organism names join the index",
    )
    parser.add_argument(
        "--type",
        default="enzymes",
        choices=sorted(surface_forms.BRENDA_PREFIXES),
        help="the entity type screened for (default enzymes)",
    )
    parser.add_argument(
        "--metadata",
        nargs="*",
        default=list(METADATA),
        help=f"columns to characterise the survivors by (default {METADATA})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="documents to screen per pool; equal sizes make the rows "
        "comparable",
    )
    parser.add_argument(
        "--fuzzy-disqualifies",
        action="store_true",
        help="also reject a near-miss, under both readings",
    )
    parser.add_argument("--out", type=pathlib.Path, default=None)
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


def report(survey: negative_screen.Survey) -> dict[str, object]:
    """One survey as the JSON the `--out` file carries."""
    return {
        "screen": survey.screen.label,
        "documents": survey.documents,
        "negatives": survey.negatives,
        "negative_rate": survey.negative_rate,
        "match_counts": dict(sorted(survey.match_counts.items())),
        "descriptive_forms": dict(survey.descriptive_forms.most_common(100)),
        "symbolic_forms": dict(survey.symbolic_forms.most_common(100)),
        "fuzzy_forms": dict(survey.fuzzy_forms.most_common(100)),
        "screened_values": {
            column: dict(counts)
            for column, counts in survey.screened_values.items()
        },
        "negative_values": {
            column: dict(counts)
            for column, counts in survey.negative_values.items()
        },
    }


def main() -> None:
    logs.configure()
    args = read_args()

    index = build_index(args.entity_tables, args.datasets)
    digest = surface_forms.index_digest(index)
    print(
        f"index {digest[:12]}: {len(index)} forms over "
        f"{len(index.entity_ids)} entities"
    )

    screens = [
        dataclasses.replace(screen, fuzzy_disqualifies=args.fuzzy_disqualifies)
        for screen in SCREENS
    ]
    surveyed = {}
    for pool in args.pools:
        print(f"\n=== {pool}")
        surveys = negative_screen.survey_corpus(
            pool,
            index,
            screens=screens,
            prefix=surface_forms.BRENDA_PREFIXES[args.type],
            metadata_columns=args.metadata,
            limit=args.limit,
            batch_size=STREAM_BATCH,
        )
        surveyed[pool.name] = surveys
        for survey in surveys:
            print(survey.summary())
            print()

    print(negative_screen.comparison(surveyed))

    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "index_digest": digest,
                    "type": args.type,
                    "limit": args.limit,
                    "pools": {
                        name: [report(survey) for survey in surveys]
                        for name, surveys in surveyed.items()
                    },
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
