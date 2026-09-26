#!/usr/bin/env python
"""Score the dictionary linker on S800's hand-assigned NCBI taxids.

Three reports, bacteria, other organisms, then both as a run of its own:
summing the per-type reports would count a taxon both types carry twice.
Usage is in docs/how-to/evaluate-linking.md, how to read the score in
docs/explanation/evaluation.md.
"""

import argparse
import pathlib

from d3text import corpus
from d3text.datasets.s800 import load_s800
from d3text.identifier_bridge import NCBI_TAXID, load_bridge
from d3text.linking import DictionaryLinker
from d3text.linking_corpora import organism_linking
from d3text.surface_forms import (
    brenda_surface_forms,
    build_index,
    load_entity_tables,
)

BACTERIA = "bacteria"
OTHER_ORGANISMS = "other_organisms"

BATCH_SIZE = 512


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="score_species_linking.py",
        description=(
            "Score the dictionary linker against S800's NCBI taxids, with "
            "the subset coverage the score is only readable beside."
        ),
    )
    parser.add_argument(
        "documents", help="TinyDB dump carrying the entity tables"
    )
    parser.add_argument(
        "bridge", help="table written by build_organism_taxid_bridge.py"
    )
    parser.add_argument("s800", help="Species-800 corpus root")
    parser.add_argument(
        "corpora",
        nargs="+",
        help="split CSVs carrying the inline other-organism names",
    )

    return parser.parse_args()


def main() -> None:
    args = read_args()

    index = build_index(
        brenda_surface_forms(
            load_entity_tables(args.documents),
            other_organisms=[
                column
                for path in args.corpora
                for column in corpus.other_organism_names(
                    pathlib.Path(path), BATCH_SIZE
                )
            ],
        )
    )
    linker = DictionaryLinker(index)
    bridge = load_bridge(args.bridge, expect=NCBI_TAXID)
    annotated = load_s800(args.s800)

    for entity_types in (
        [BACTERIA],
        [OTHER_ORGANISMS],
        [BACTERIA, OTHER_ORGANISMS],
    ):
        report = organism_linking(
            mentions=annotated.mentions,
            bridge=bridge,
            linker=linker,
            entity_types=entity_types,
        )

        print(report.summary())
        for key, value in sorted(report.metrics().items()):
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
