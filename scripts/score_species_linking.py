#!/usr/bin/env python
"""Score the dictionary linker on S800's hand-assigned NCBI taxids.

The one linking measurement in reach that is not circular. BRENDA supplies the
candidate entities and their surface forms; S800 supplies, per species span, a
taxid a human assigned against the NCBI taxonomy; and the bridge table
`build_organism_taxid_bridge.py` writes joins the two without either side
consulting the other's names at scoring time. Read every score with the
coverage it is printed beside — the judged subset is the mentions that pair
with exactly one BRENDA entity, which is the easy half.

Three reports: one per entity type, then both together. They are three runs
rather than a sum, because a taxon BRENDA curates as a bacterium *and* as an
other organism is gold for neither when the type is not given, and summing the
per-type reports would count it twice and count the corpus twice with it.

Offline, and needs no BRENDA SQL connection: the entity tables are read off
the tail of the TinyDB dump, and the other organisms' names off the splits'
inline column, which polars reads without parsing the rest of the row::

    python scripts/score_species_linking.py \\
        brenda_references/src/brenda_references/data/documents.json \\
        data/organism_taxids.tsv \\
        ~/Downloads/Species-800 \\
        brenda_references/src/brenda_references/data/training_data.csv \\
        brenda_references/src/brenda_references/data/validation_data.csv \\
        brenda_references/src/brenda_references/data/test_data.csv

The splits are arguments rather than an option because BRENDA's
other-organism IDs are named nowhere else: an index built without them holds
no `oth` form at all, and the linker would then answer NIL to every one of
those spans — a score, not a missing report.
"""

import argparse
import pathlib

from d3text import corpus
from d3text.datasets.s800 import load_s800
from d3text.identifier_bridge import NCBI_TAXID, load_bridge
from d3text.linking import DictionaryLinker
from d3text.linking_eval import score_linking
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
        report = score_linking(
            mentions=annotated.mentions,
            bridge=bridge,
            linker=linker,
            entity_types=entity_types,
            namespace=NCBI_TAXID,
        )

        print(report.summary())
        for key, value in sorted(report.metrics().items()):
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
