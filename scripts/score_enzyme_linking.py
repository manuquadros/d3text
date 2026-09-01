#!/usr/bin/env python
"""Score the dictionary linker on enzymeNER's spans, grounded through Expasy.

The weakest of the three external evaluations, and the caveat is printed with
every number: enzymeNER marks spans without naming them, so the gold EC number
is itself a dictionary lookup. That the dictionary is Expasy's rather than
BRENDA's is what makes the score evidence at all, but a name the nomenclature
resolves wrongly is charged to the linker with nothing to distinguish it.

Offline, and needs no BRENDA SQL connection or split CSV — the enzyme table is
read off the tail of the TinyDB dump::

    python scripts/score_enzyme_linking.py \\
        brenda_references/src/brenda_references/data/documents.json \\
        data/enzyme_ec_numbers.tsv \\
        ~/Downloads/enzymeNER \\
        ~/Downloads/expasy-enzyme/enzyme.dat

The subset selection is the whole argument. Every EC number names exactly one
BRENDA enzyme, so the bridge excludes nothing and the judged population is
chosen by Expasy alone — never by which spans BRENDA's own index resolves
uniquely, which would make the linker's answer the gold.
"""

import argparse
import collections

from d3text.datasets.enzymener import article_of, load_enzymener
from d3text.datasets.expasy import load_nomenclature
from d3text.identifier_bridge import EC_NUMBER, load_bridge
from d3text.linking import DictionaryLinker
from d3text.linking_eval import score_linking
from d3text.surface_forms import (
    brenda_surface_forms,
    build_index,
    load_entity_tables,
)

ENZYMES = "enzymes"

CAVEAT = (
    "Read as silver, and out of domain. The gold EC number is a lookup in an "
    "outside nomenclature, not an identifier a human assigned to this span, "
    "so the resolver's own errors are charged to the linker; and enzymeNER is "
    "general biomedical text where this project's corpus is BRENDA's enzyme "
    "literature, so relative comparisons transfer and absolute values do not."
)


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="score_enzyme_linking.py",
        description=(
            "Score the dictionary linker against EC numbers the ENZYME "
            "nomenclature assigns enzymeNER's spans, with the subset "
            "coverage the score is only readable beside."
        ),
    )
    parser.add_argument(
        "documents", help="TinyDB dump carrying the entity tables"
    )
    parser.add_argument(
        "bridge", help="table written by build_enzyme_ec_bridge.py"
    )
    parser.add_argument("enzymener", help="enzymeNER TestSet directory")
    parser.add_argument("nomenclature", help="Expasy ENZYME enzyme.dat")

    return parser.parse_args()


def main() -> None:
    args = read_args()

    linker = DictionaryLinker(
        build_index(brenda_surface_forms(load_entity_tables(args.documents)))
    )
    bridge = load_bridge(args.bridge, expect=EC_NUMBER)
    corpus = load_enzymener(args.enzymener)
    nomenclature = load_nomenclature(args.nomenclature)

    articles = {article_of(document) for document in corpus.texts}
    print(
        f"{len(corpus.mentions)} annotations over {len(corpus.texts)} "
        f"sentences from {len(articles)} articles; {len(corpus.misplaced)} "
        f"rows dropped for addressing text other than their own surface form."
    )
    print(
        f"{len(nomenclature)} nomenclature names, "
        f"{nomenclature.unambiguous} of them naming exactly one EC number."
    )

    resolved = collections.Counter(
        min(len(nomenclature.ec_numbers(mention.surface)), 2)
        for mention in corpus.mentions
    )
    print(
        f"Of the annotated spans, {resolved[1]} resolve to exactly one EC "
        f"number, {resolved[2]} to several, and {resolved[0]} to none."
    )

    report = score_linking(
        mentions=nomenclature.assign(corpus.mentions),
        bridge=bridge,
        linker=linker,
        entity_types=[ENZYMES],
        namespace=EC_NUMBER,
    )
    print(report.summary())
    print(
        f"Of the {report.outside_bridge} spans outside the bridge, "
        f"{resolved[0]} are names the nomenclature does not hold and "
        f"{report.outside_bridge - resolved[0]} are EC numbers BRENDA "
        f"curates no enzyme for."
    )
    for key, value in sorted(report.metrics().items()):
        print(f"  {key}: {value:.4f}")
    print(CAVEAT)


if __name__ == "__main__":
    main()
