#!/usr/bin/env python
"""Score the dictionary linker on S800's hand-assigned NCBI taxids.

The one linking measurement in reach that is not circular. BRENDA supplies the
candidate entities and their surface forms; S800 supplies, per species span, a
taxid a human assigned against the NCBI taxonomy; and the bridge table
`build_bacteria_taxid_bridge.py` writes joins the two without either side
consulting the other's names at scoring time. Read the score with the coverage
it is printed beside — the judged subset is the mentions that pair with
exactly one BRENDA bacterium, which is the easy half.

Offline, and needs no BRENDA SQL connection: the entity tables are read off
the tail of the TinyDB dump::

    python scripts/score_species_linking.py \\
        brenda_references/src/brenda_references/data/documents.json \\
        data/bacteria_taxids.tsv \\
        ~/Downloads/Species-800
"""

import argparse

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
        "bridge", help="table written by build_bacteria_taxid_bridge.py"
    )
    parser.add_argument("s800", help="Species-800 corpus root")

    return parser.parse_args()


def main() -> None:
    args = read_args()

    index = build_index(
        brenda_surface_forms(load_entity_tables(args.documents))
    )
    linker = DictionaryLinker(index)
    bridge = load_bridge(args.bridge, expect=NCBI_TAXID)
    corpus = load_s800(args.s800)

    report = score_linking(
        mentions=corpus.mentions,
        bridge=bridge,
        linker=linker,
        entity_type=BACTERIA,
        namespace=NCBI_TAXID,
    )

    print(report.summary())
    for key, value in sorted(report.metrics().items()):
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
