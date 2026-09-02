#!/usr/bin/env python
"""Score the dictionary linker on NLP4Pheno's strain spans, by deposit number.

Read what this measures before reading the number. The gold accession is
extracted from the span and joined against BRENDA's own `cultures` table, and
that table is also part of the surface-form index the linker queries — so
unlike the species and enzyme evaluations, the two sides are one resource read
two ways. What separates them is the reading: the gold joins on a *canonical*
accession, the index is keyed by a form's *words as written*, and the corpus
spells deposits both ways. So this scores the matcher's normalization, not
BRENDA's vocabulary, and the spans where the two spellings already agree are
spans the linker cannot get wrong. The report prints how many those are.

Offline, and needs no BRENDA SQL connection or split CSV — the strain table is
read off the tail of the TinyDB dump::

    export=~/Nextcloud/dev/datasets/nlp4pheno
    python scripts/score_strain_linking.py \\
        brenda_references/src/brenda_references/data/documents.json \\
        data/strain_numbers.tsv \\
        "$export/project-10-at-2025-08-21-21-08-cb43bf25.json"
"""

import argparse
import collections
from collections.abc import Iterable

from d3text.datasets.culture_numbers import assign, find
from d3text.datasets.nlp4pheno import STRAIN, load_nlp4pheno
from d3text.identifier_bridge import (
    STRAIN_NUMBER,
    ExternalMention,
    IdentifierBridge,
    load_bridge,
)
from d3text.linking import DictionaryLinker
from d3text.linking_eval import LinkingReport, score_linking
from d3text.schema import BRENDA_SCHEMA
from d3text.surface_forms import (
    SurfaceFormIndex,
    brenda_surface_forms,
    build_index,
    form_words,
    load_entity_tables,
)

STRAINS = "strains"

CAVEAT = (
    "Read as a measurement of the matcher, and out of domain. The gold "
    "accession joins the same BRENDA culture numbers the linker's index is "
    "built from, so this cannot show that BRENDA names the right strain — "
    "only whether the index's word-keyed forms recover the strain a "
    "canonical accession names. And NLP4Pheno is general microbiology text "
    "where this project's corpus is BRENDA's enzyme literature, so relative "
    "comparisons transfer and absolute values do not."
)


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="score_strain_linking.py",
        description=(
            "Score the dictionary linker against the culture-collection "
            "accessions NLP4Pheno's strain spans carry, with the subset "
            "coverage the score is only readable beside."
        ),
    )
    parser.add_argument(
        "documents", help="TinyDB dump carrying the entity tables"
    )
    parser.add_argument(
        "bridge", help="table written by build_strain_number_bridge.py"
    )
    parser.add_argument("nlp4pheno", help="Label Studio export of project 10")

    return parser.parse_args()


def held_verbatim(index: SurfaceFormIndex, surface: str, prefix: str) -> bool:
    """Whether the index holds an accession of `surface` as the text spells it.

    The circular share, made countable: where it is true the linker looks the
    gold's own key up and can only agree.
    """
    return any(
        any(
            entity_id.startswith(prefix)
            for entity_id in index.lookup(form_words(accession.written))
        )
        for accession in find(surface)
    )


def report_for(
    mentions: Iterable[ExternalMention],
    bridge: IdentifierBridge,
    linker: DictionaryLinker,
) -> LinkingReport:
    """The strain report over `mentions`, with their accessions assigned."""
    return score_linking(
        mentions=assign(mentions),
        bridge=bridge,
        linker=linker,
        entity_types=[STRAINS],
        namespace=STRAIN_NUMBER,
    )


def main() -> None:
    args = read_args()

    prefix = {
        entity_type.name: entity_type.prefix
        for entity_type in BRENDA_SCHEMA.entity_types
    }[STRAINS]
    index = build_index(
        brenda_surface_forms(load_entity_tables(args.documents))
    )
    linker = DictionaryLinker(index)
    bridge = load_bridge(args.bridge, expect=STRAIN_NUMBER)
    corpus = load_nlp4pheno(args.nlp4pheno)
    mentions = corpus.labelled(STRAIN)

    print(
        f"{len(mentions)} {STRAIN} span results over {len(corpus.texts)} "
        f"sentences, {len({m.surface for m in mentions})} distinct surface "
        f"forms; {corpus.relations} relation results not read."
    )

    # Keyed the way `score_linking` keys them, so the two annotations of one
    # span cannot make these counts and its populations disagree.
    spans = {
        (mention.document, mention.start, mention.end): mention
        for mention in mentions
    }
    carried = collections.Counter(
        min(len({found.canonical for found in find(mention.surface)}), 2)
        for mention in spans.values()
    )
    verbatim = sum(
        1
        for mention in spans.values()
        if held_verbatim(index, mention.surface, prefix)
    )
    print(
        f"Of the {len(spans)} distinct spans, {carried[1]} carry one "
        f"culture-collection accession, {carried[2]} carry several and "
        f"{carried[0]} carry none. Of the ones that carry any, {verbatim} "
        f"spell it the way the surface-form index already holds it, where "
        f"the linker is looking up the gold's own key."
    )

    report = report_for(spans.values(), bridge, linker)
    print(report.summary())
    print(
        f"Of the {report.outside_bridge} spans outside the bridge, "
        f"{carried[0]} carry no accession at all and "
        f"{report.outside_bridge - carried[0]} carry one BRENDA records no "
        f"deposit under."
    )
    for key, value in sorted(report.metrics().items()):
        print(f"  {key}: {value:.4f}")

    unheld = report_for(
        [
            mention
            for mention in spans.values()
            if not held_verbatim(index, mention.surface, prefix)
        ],
        bridge,
        linker,
    )
    print(
        "Dropping the spans whose accession the index already holds leaves "
        f"{unheld.judged} judged, at strict accuracy "
        f"{unheld.strict.accuracy:.3f}. That is a floor and not a fair "
        "score — the selection is on the linker's side and keeps exactly the "
        "spellings the dictionary is known to miss — but it is the part of "
        "the headline that is not the linker reading back its own key."
    )
    print(CAVEAT)


if __name__ == "__main__":
    main()
