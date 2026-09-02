"""The linking block an evaluation reports, over corpora BRENDA did not make.

Kept out of `d3text.linking_eval` because assembling it costs a surface-form
index over the whole BRENDA dump and the BRENDA data layer with it, neither of
which the scorer may need. A machine that has no corpora skips the block and
finishes the evaluation, the way an unset `MLFLOW_TRACKING_URI` skips
tracking. See the evaluation page of the documentation.
"""

import logging
import os
import pathlib
from collections.abc import Iterable
from dataclasses import dataclass

from brenda_references.brenda_references import DATA_DIR

from d3text import corpus, schema, surface_forms
from d3text.datasets import (
    culture_numbers,
    enzymener,
    expasy,
    nlp4pheno,
    s800,
)
from d3text.identifier_bridge import (
    EC_NUMBER,
    NCBI_TAXID,
    STRAIN_NUMBER,
    ExternalMention,
    IdentifierBridge,
    load_bridge,
)
from d3text.linking import DictionaryLinker, Linker
from d3text.linking_eval import LinkingReport, score_linking

logger = logging.getLogger(__name__)

S800 = "Species-800"
"""S800's directory, relative to the corpus root."""

ENZYMENER = "enzymeNER"
"""enzymeNER's directory, relative to the corpus root."""

NLP4PHENO = "nlp4pheno"
"""NLP4Pheno's directory, relative to the corpus root."""

NLP4PHENO_EXPORT = pathlib.Path(NLP4PHENO) / "export.json"
"""The Label Studio export, relative to the corpus root.

A name this project fixes, where the other two corpora are found by the
filename their publisher ships. Upstream releases several dated exports of the
annotation project and they do not annotate the same spans, so a glob would
score whichever one the directory listing returned first and would move the
gold set the day a second download landed beside it. A symlink is enough; the
export it resolves to is logged.
"""

NOMENCLATURE = pathlib.Path("expasy-enzyme") / expasy.NOMENCLATURE
"""The ENZYME nomenclature, relative to the corpus root."""

ORGANISM_BRIDGE = "organism_taxids.tsv"
"""The taxid table, relative to the repository's `data/`."""

ENZYME_BRIDGE = "enzyme_ec_numbers.tsv"
"""The EC table, relative to the repository's `data/`."""

STRAIN_BRIDGE = "strain_numbers.tsv"
"""The culture-number table, relative to the repository's `data/`."""

ORGANISM_TYPES = ("bacteria", "other_organisms")
"""The types S800's taxids may name. Strains have no bridge and no gold."""

ENZYME_TYPES = ("enzymes",)

STRAIN_TYPES = ("strains",)
"""The type a culture-collection accession grounds a span in."""

SPLITS = ("training", "validation", "test")
"""The corpus files pooled for the other-organism names, which live nowhere
else: an index built without them holds no `oth` form at all, so the linker
answers NIL to every one of those spans — a score, not a missing report."""

STREAM_BATCH = 1000

CAVEAT = (
    "A property of the surface-form index, not of the checkpoint: "
    "DictionaryLinker holds no learned parameters, so this block is the same "
    "for every model evaluated against the same index and moves only when the "
    "index does. The enzyme report is silver — its gold EC number is a lookup "
    "in an outside nomenclature rather than an identifier a human assigned to "
    "the span — and the corpora are general biomedical text where this "
    "project's is BRENDA's enzyme literature, so relative comparisons "
    "transfer and absolute values do not."
)

STRAIN_CAVEAT = (
    "Read as a measurement of the matcher, and out of domain. The gold "
    "accession joins the same BRENDA culture numbers the linker's index is "
    "built from, so this cannot show that BRENDA names the right strain — "
    "only whether the index's word-keyed forms recover the strain a "
    "canonical accession names. And NLP4Pheno is general microbiology text "
    "where this project's corpus is BRENDA's enzyme literature, so relative "
    "comparisons transfer and absolute values do not."
)

CAVEATS = {STRAIN_NUMBER: STRAIN_CAVEAT}
"""What `CAVEAT` does not cover, by the namespace it qualifies.

The shared caveat holds for a report whose gold was assigned outside this
project. The strain gold is not: its accession joins the same BRENDA table the
index is built from, so its own qualifier is printed beside its score rather
than left to the reader to know.
"""


@dataclass(frozen=True, slots=True)
class LinkingBlock:
    """The reports an evaluation logs, and the index they were taken against.

    Empty is a legitimate value and the one a machine without the corpora
    produces. `index_digest` is carried because the reports move with the
    index and with nothing else, so two evaluations are comparable exactly
    when it matches.
    """

    reports: tuple[LinkingReport, ...] = ()
    index_digest: str = ""

    def metrics(self) -> dict[str, float]:
        """Every report's metrics, in one dict.

        :return: the keys to log.
        :raises ValueError: if two reports key the same metric, which would
            silently leave one of them unlogged.
        """
        metrics: dict[str, float] = {}
        for report in self.reports:
            emitted = report.metrics()
            collision = sorted(metrics.keys() & emitted.keys())
            if collision:
                raise ValueError(
                    f"two linking reports emit {collision}: one would "
                    "overwrite the other, and the chart would name neither"
                )
            metrics.update(emitted)
        return metrics

    def summary(self) -> str:
        """The reports as prose, with what they are and are not evidence of.

        :return: one paragraph per report, each followed by any caveat the
            shared one does not cover, then the shared caveat; empty when no
            report was produced.
        """
        if not self.reports:
            return ""
        paragraphs: list[str] = []
        for report in self.reports:
            paragraphs.append(report.summary())
            caveat = CAVEATS.get(report.namespace)
            if caveat is not None:
                paragraphs.append(caveat)
        paragraphs.append(f"Surface-form index {self.index_digest}. {CAVEAT}")
        return "\n\n".join(paragraphs)


def brenda_index() -> surface_forms.SurfaceFormIndex:
    """The index the linker queries, over all four ID namespaces.

    :return: the surface forms BRENDA's entity tables and the splits' inline
        other-organism column define.
    """
    tables = surface_forms.load_entity_tables(_brenda_data("documents.json"))
    return surface_forms.build_index(
        surface_forms.brenda_surface_forms(
            tables,
            (
                names
                for split in SPLITS
                for names in corpus.other_organism_names(
                    _brenda_data(f"{split}_data.csv"), STREAM_BATCH
                )
            ),
        )
    )


def organism_report(root: pathlib.Path, linker: Linker) -> LinkingReport | None:
    """Score `linker` on S800's hand-assigned taxids.

    :param root: the corpus root holding S800.
    :param linker: the linker under test.
    :return: the report, or None where the corpus is not on disk.
    """
    directory = root / S800
    if not (directory / s800.ANNOTATIONS).is_file():
        return None
    return score_linking(
        mentions=s800.load_s800(directory).mentions,
        bridge=load_bridge(
            schema.DATA_DIR / ORGANISM_BRIDGE, expect=NCBI_TAXID
        ),
        linker=linker,
        entity_types=list(ORGANISM_TYPES),
        namespace=NCBI_TAXID,
    )


def enzyme_report(root: pathlib.Path, linker: Linker) -> LinkingReport | None:
    """Score `linker` on the EC numbers ENZYME assigns enzymeNER's spans.

    :param root: the corpus root holding enzymeNER and the nomenclature.
    :param linker: the linker under test.
    :return: the report, or None where either is not on disk — enzymeNER names
        no identifier itself, so without the nomenclature there is no gold and
        a report would read as total bridge failure rather than as absence.
    """
    directory = root / ENZYMENER
    nomenclature_path = root / NOMENCLATURE
    if not (directory / enzymener.ANNOTATIONS).is_file():
        return None
    if not nomenclature_path.is_file():
        logger.warning(
            "no ENZYME nomenclature at %s, so enzymeNER's spans carry no "
            "gold EC number and the enzyme linking report is skipped",
            nomenclature_path,
        )
        return None
    nomenclature = expasy.load_nomenclature(nomenclature_path)
    return score_linking(
        mentions=nomenclature.assign(
            enzymener.load_enzymener(directory).mentions
        ),
        bridge=load_bridge(schema.DATA_DIR / ENZYME_BRIDGE, expect=EC_NUMBER),
        linker=linker,
        entity_types=list(ENZYME_TYPES),
        namespace=EC_NUMBER,
    )


def strain_linking(
    mentions: Iterable[ExternalMention],
    bridge: IdentifierBridge,
    linker: Linker,
) -> LinkingReport:
    """Score `linker` on strain spans, each stamped with its own accessions.

    :param mentions: the corpus's strain spans, which carry no identifier of
        their own.
    :param bridge: the table pairing culture numbers with BRENDA strains.
    :param linker: the linker under test.
    :return: the report.
    """
    return score_linking(
        mentions=culture_numbers.assign(mentions),
        bridge=bridge,
        linker=linker,
        entity_types=list(STRAIN_TYPES),
        namespace=STRAIN_NUMBER,
    )


def strain_report(root: pathlib.Path, linker: Linker) -> LinkingReport | None:
    """Score `linker` on the deposit numbers NLP4Pheno's strain spans carry.

    :param root: the corpus root holding the NLP4Pheno export.
    :param linker: the linker under test.
    :return: the report, or None where the export is not on disk — the name
        it is looked up under is this project's rather than the publisher's,
        so a corpus present without it is warned about rather than skipped in
        silence.
    """
    export = _strain_export(root)
    if export is None:
        return None
    logger.info("scoring the NLP4Pheno export at %s", export.resolve())
    return strain_linking(
        nlp4pheno.load_nlp4pheno(export).labelled(nlp4pheno.STRAIN),
        load_bridge(schema.DATA_DIR / STRAIN_BRIDGE, expect=STRAIN_NUMBER),
        linker,
    )


def linking_block(root: str | os.PathLike[str] | None) -> LinkingBlock:
    """The linking reports for whichever corpora are under `root`.

    The index is built only once a corpus has been found, since building it
    reads the 1.1 GB entity dump and scans every split.

    :param root: the directory holding the corpora, or None on a machine that
        has none.
    :return: the block, empty where nothing could be scored.
    """
    if root is None:
        return LinkingBlock()
    directory = pathlib.Path(root).expanduser()
    if not directory.is_dir():
        logger.warning(
            "no directory at %s, so the linking block is skipped", directory
        )
        return LinkingBlock()

    scorers = [
        scorer
        for found, scorer in (
            ((directory / S800 / s800.ANNOTATIONS).is_file(), organism_report),
            (
                (directory / ENZYMENER / enzymener.ANNOTATIONS).is_file(),
                enzyme_report,
            ),
            (_strain_export(directory) is not None, strain_report),
        )
        if found
    ]
    if not scorers:
        logger.warning(
            "%s holds none of %s, so the linking block is skipped",
            directory,
            ", ".join((S800, ENZYMENER, str(NLP4PHENO_EXPORT))),
        )
        return LinkingBlock()

    index = brenda_index()
    linker = DictionaryLinker(index)
    return LinkingBlock(
        reports=_produced(scorer(directory, linker) for scorer in scorers),
        index_digest=surface_forms.index_digest(index),
    )


def _strain_export(root: pathlib.Path) -> pathlib.Path | None:
    """The NLP4Pheno export to score, or None where there is none.

    The fixed name is one the operator makes by copy or symlink, so a corpus
    directory holding nothing but the publisher's dated exports is the layout
    that ships — and skipping that in silence looks exactly like a machine
    that never downloaded NLP4Pheno at all.
    """
    export = root / NLP4PHENO_EXPORT
    if export.is_file():
        return export
    if (root / NLP4PHENO).is_dir():
        logger.warning(
            "no NLP4Pheno export at %s, so its strain spans carry no gold "
            "accession and the strain linking report is skipped; that name "
            "is this project's rather than the publisher's, and is made by "
            "copying or symlinking the dated export to be scored",
            export,
        )
    return None


def _produced(
    reports: Iterable[LinkingReport | None],
) -> tuple[LinkingReport, ...]:
    """The reports that were actually produced."""
    return tuple(report for report in reports if report is not None)


def _brenda_data(name: str) -> pathlib.Path:
    """A file of the BRENDA data directory, wherever the package installed it.

    A package resource rather than a path under the checkout: a git worktree
    has the tracked files and none of the 1.8 GB the splits weigh.
    """
    return pathlib.Path(str(DATA_DIR / name))


__all__ = [
    "CAVEAT",
    "CAVEATS",
    "STRAIN_CAVEAT",
    "LinkingBlock",
    "brenda_index",
    "enzyme_report",
    "linking_block",
    "organism_report",
    "strain_linking",
    "strain_report",
]
