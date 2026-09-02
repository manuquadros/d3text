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
from d3text.datasets import enzymener, expasy, s800
from d3text.identifier_bridge import EC_NUMBER, NCBI_TAXID, load_bridge
from d3text.linking import DictionaryLinker, Linker
from d3text.linking_eval import LinkingReport, score_linking

logger = logging.getLogger(__name__)

S800 = "Species-800"
"""S800's directory, relative to the corpus root."""

ENZYMENER = "enzymeNER"
"""enzymeNER's directory, relative to the corpus root."""

NOMENCLATURE = pathlib.Path("expasy-enzyme") / expasy.NOMENCLATURE
"""The ENZYME nomenclature, relative to the corpus root."""

ORGANISM_BRIDGE = "organism_taxids.tsv"
"""The taxid table, relative to the repository's `data/`."""

ENZYME_BRIDGE = "enzyme_ec_numbers.tsv"
"""The EC table, relative to the repository's `data/`."""

ORGANISM_TYPES = ("bacteria", "other_organisms")
"""The types S800's taxids may name. Strains have no bridge and no gold."""

ENZYME_TYPES = ("enzymes",)

SPLITS = ("training", "validation", "test")
"""The corpus files pooled for the other-organism names, which live nowhere
else: an index built without them holds no `oth` form at all, so the linker
answers NIL to every one of those spans — a score, not a missing report."""

STREAM_BATCH = 1000

CAVEAT = (
    "A property of the surface-form index, not of the checkpoint: "
    "DictionaryLinker holds no learned parameters, so this block is the same "
    "for every model evaluated against the same index and moves only when the "
    "index does. The enzyme half is silver — its gold EC number is a lookup "
    "in an outside nomenclature rather than an identifier a human assigned to "
    "the span — and both corpora are general biomedical text where this "
    "project's is BRENDA's enzyme literature, so relative comparisons "
    "transfer and absolute values do not."
)


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

        :return: one paragraph per report, then the caveat; empty when no
            report was produced.
        """
        if not self.reports:
            return ""
        paragraphs = [report.summary() for report in self.reports]
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
        for marker, scorer in (
            (directory / S800 / s800.ANNOTATIONS, organism_report),
            (directory / ENZYMENER / enzymener.ANNOTATIONS, enzyme_report),
        )
        if marker.is_file()
    ]
    if not scorers:
        logger.warning(
            "%s holds neither %s nor %s, so the linking block is skipped",
            directory,
            S800,
            ENZYMENER,
        )
        return LinkingBlock()

    index = brenda_index()
    linker = DictionaryLinker(index)
    return LinkingBlock(
        reports=_produced(scorer(directory, linker) for scorer in scorers),
        index_digest=surface_forms.index_digest(index),
    )


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
    "LinkingBlock",
    "brenda_index",
    "enzyme_report",
    "linking_block",
    "organism_report",
]
