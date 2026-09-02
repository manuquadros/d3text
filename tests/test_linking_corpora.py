"""The linking block `evaluate` logs, and what keeps two authorities apart.

An evaluation reports one linking score per outside authority — NCBI's taxids
for the organisms, ENZYME's numbers for the enzymes, the collections' deposit
numbers for the strains — and `score_linking` refuses to put two authorities in
one report. So they are separate reports whose metrics land in one MLflow run,
and the failure that mattered is silent: keyed alike, the second overwrites the
first and the run charts one number under a name that fits both. Hence the
namespace in every key, asserted here against the glossary that has to resolve
it.

The corpora themselves are downloads. Everything below either fabricates one
in `tmp_path` or asserts that absence skips the block, so nothing here reads
BRENDA's 1.1 GB dump.
"""

import json
import logging
import os
import pathlib

import pytest
from d3text import linking_corpora, metric_docs, surface_forms
from d3text.datasets import enzymener, expasy, nlp4pheno, s800
from d3text.identifier_bridge import (
    EC_NUMBER,
    NCBI_TAXID,
    STRAIN_NUMBER,
    BridgeRow,
    ExternalMention,
    IdentifierBridge,
)
from d3text.linking import DictionaryLinker
from d3text.linking_corpora import LinkingBlock
from d3text.linking_eval import LinkingReport, score_linking

COLI = "Escherichia coli"
ADH = "alcohol dehydrogenase"

S800_ANNOTATIONS = "\n".join(
    (
        "562\tspecies001:111\t10\t25\tEscherichia coli",
        "1423\tspecies001:111\t45\t61\tBacillus subtilis",
        "5833\tspecies002:222\t4\t24\tPlasmodium falciparum",
    )
)
S800_TEXTS = {
    "species001": "Growth of Escherichia coli was compared with Bacillus "
    "subtilis in vitro.",
    "species002": "The Plasmodium falciparum genome.",
}

ENZYMENER_SENTENCES = "PMC1\tS01\tAssays of alcohol dehydrogenase were run."
ENZYMENER_ANNOTATIONS = "PMC1\tS01\t10\t31\talcohol dehydrogenase"
ENZYME_DAT = "\n".join(("ID   1.1.1.1", "DE   alcohol dehydrogenase.", "//"))

DEPOSIT = "ATCC 6538"
AUREUS = f"Staphylococcus aureus {DEPOSIT}"
STRAIN_ENTITY = "str11445"
"""The one strain `data/strain_numbers.tsv` records `ATCC 6538` under.

The bridge the strain report reads is the tracked table, not a fixture, so a
regenerated table that paired the deposit with another strain or with several
would fail here — which is the report becoming unjudgeable, not a stale test.
"""

NLP4PHENO_TEXT = f"Growth of {AUREUS} was measured."
DATED_EXPORT = "project-10-at-2025-08-21-21-08-cb43bf25.json"
"""How upstream names an export, and it publishes more than one of them."""


def _mention(
    surface: str, external_id: str | None, document: str = "d1"
) -> ExternalMention:
    return ExternalMention(
        document=document,
        start=0,
        end=len(surface),
        surface=surface,
        external_id=external_id,
    )


def _organism_report() -> LinkingReport:
    return score_linking(
        mentions=[_mention(COLI, "562")],
        bridge=IdentifierBridge.from_rows(
            NCBI_TAXID, [BridgeRow("bac1", "562", "lpsn_id")]
        ),
        linker=DictionaryLinker(surface_forms.build_index({"bac1": [COLI]})),
        entity_types=list(linking_corpora.ORGANISM_TYPES),
        namespace=NCBI_TAXID,
    )


def _enzyme_report() -> LinkingReport:
    return score_linking(
        mentions=[_mention(ADH, "1.1.1.1")],
        bridge=IdentifierBridge.from_rows(
            EC_NUMBER, [BridgeRow("enz1", "1.1.1.1", "ec_class")]
        ),
        linker=DictionaryLinker(surface_forms.build_index({"enz1": [ADH]})),
        entity_types=list(linking_corpora.ENZYME_TYPES),
        namespace=EC_NUMBER,
    )


def _strain_report() -> LinkingReport:
    return linking_corpora.strain_linking(
        mentions=[_mention(AUREUS, None)],
        bridge=IdentifierBridge.from_rows(
            STRAIN_NUMBER,
            [BridgeRow(STRAIN_ENTITY, DEPOSIT, "culture_number")],
        ),
        linker=DictionaryLinker(
            surface_forms.build_index({STRAIN_ENTITY: [AUREUS]})
        ),
    )


@pytest.fixture
def no_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make building the surface-form index an error.

    It reads the 1.1 GB entity dump and scans every split, so a root with
    nothing to score has to be settled before it is touched — and a test that
    merely returned an empty block would pass either way.
    """

    def refuse() -> surface_forms.SurfaceFormIndex:
        raise AssertionError("the surface-form index was built for nothing")

    monkeypatch.setattr(linking_corpora, "brenda_index", refuse)


@pytest.fixture
def tiny_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand a three-entry index in for BRENDA's, which is a 1.1 GB read."""
    monkeypatch.setattr(
        linking_corpora,
        "brenda_index",
        lambda: surface_forms.build_index(
            {"bac1": [COLI], "enz1": [ADH], STRAIN_ENTITY: [AUREUS]}
        ),
    )


def _s800_corpus(root: pathlib.Path) -> pathlib.Path:
    directory = root / linking_corpora.S800
    (directory / s800.ABSTRACTS).mkdir(parents=True)
    (directory / s800.ANNOTATIONS).write_text(
        S800_ANNOTATIONS + "\n", encoding="utf8"
    )
    for document, text in S800_TEXTS.items():
        (directory / s800.ABSTRACTS / f"{document}.txt").write_text(
            text, encoding="utf8"
        )
    return root


def _enzymener_corpus(root: pathlib.Path, nomenclature: bool) -> pathlib.Path:
    directory = root / linking_corpora.ENZYMENER
    directory.mkdir(parents=True)
    (directory / enzymener.SENTENCES).write_text(
        "\ufeff" + ENZYMENER_SENTENCES + "\n", encoding="utf8"
    )
    (directory / enzymener.ANNOTATIONS).write_text(
        "\ufeff" + ENZYMENER_ANNOTATIONS + "\n", encoding="utf8"
    )
    if nomenclature:
        path = root / linking_corpora.NOMENCLATURE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(ENZYME_DAT + "\n", encoding=expasy.ENCODING)
    return root


def _nlp4pheno_corpus(
    root: pathlib.Path, export: str | os.PathLike[str] | None = None
) -> pathlib.Path:
    directory = root / linking_corpora.NLP4PHENO
    directory.mkdir(parents=True, exist_ok=True)
    start = NLP4PHENO_TEXT.index(AUREUS)
    task = {
        "id": 1,
        "data": {"text": NLP4PHENO_TEXT},
        "annotations": [
            {
                "result": [
                    {
                        "type": nlp4pheno.SPAN_RESULT,
                        "value": {
                            "start": start,
                            "end": start + len(AUREUS),
                            "text": AUREUS,
                            "labels": [nlp4pheno.STRAIN],
                        },
                    }
                ]
            }
        ],
    }
    path = directory / (
        linking_corpora.NLP4PHENO_EXPORT.name if export is None else export
    )
    path.write_text(json.dumps([task]), encoding="utf8")
    return root


# --------------------------------------------------------------------------- #
# Two authorities, one run                                                     #
# --------------------------------------------------------------------------- #
def test_both_authorities_survive_being_logged_together() -> None:
    """The collision this module exists to prevent.

    Keys carrying no namespace make the two reports the same twelve keys, so
    whichever is logged second is the only one the run holds — and the chart
    is labelled in a way that fits either.
    """
    metrics = LinkingBlock((_organism_report(), _enzyme_report())).metrics()

    assert metrics[f"test/linking_{NCBI_TAXID}_strict_accuracy"] == 1.0
    assert metrics[f"test/linking_{EC_NUMBER}_strict_accuracy"] == 1.0
    assert metrics[f"test/linking_{NCBI_TAXID}_judged"] == 1.0
    assert metrics[f"test/linking_{EC_NUMBER}_judged"] == 1.0


def test_every_key_the_block_emits_is_documented() -> None:
    """MLflow charts a key and records no unit anywhere else, so a key the
    glossary cannot resolve reaches the server saying nothing about itself."""
    metrics = LinkingBlock(
        (_organism_report(), _enzyme_report(), _strain_report())
    ).metrics()

    assert [
        name for name in metrics if metric_docs.describe(name) is None
    ] == []


def test_two_reports_keying_the_same_metric_are_refused() -> None:
    """The guard behind the namespace: a future report that collides with an
    existing one must fail rather than quietly replace it."""
    block = LinkingBlock((_organism_report(), _organism_report()))

    with pytest.raises(ValueError, match="overwrite"):
        block.metrics()


def test_the_summary_says_whose_property_the_number_is() -> None:
    """The block moves with the surface-form index and with nothing else, so
    read as a model score it invites a conclusion about a checkpoint that had
    no part in it."""
    summary = LinkingBlock(
        (_organism_report(),), index_digest="deadbeef"
    ).summary()

    assert "deadbeef" in summary
    assert "no learned parameters" in summary


def test_the_strain_score_carries_the_caveat_the_shared_one_does_not() -> None:
    """The shared caveat holds for gold an outside authority assigned, and the
    strain gold is a form of the dictionary under test: its accession joins the
    same BRENDA table the index is built from. Printed without that beside it,
    and beside the coverage it was taken on, the number reads as evidence
    about BRENDA's vocabulary rather than about the matcher."""
    summary = LinkingBlock((_strain_report(),)).summary()
    without_it = LinkingBlock((_organism_report(),)).summary()

    assert linking_corpora.STRAIN_CAVEAT in summary
    assert "of 1 annotated mentions" in summary
    assert linking_corpora.STRAIN_CAVEAT not in without_it


# --------------------------------------------------------------------------- #
# A machine without the corpora                                                #
# --------------------------------------------------------------------------- #
def test_an_unset_corpus_root_skips_the_block(no_index: None) -> None:
    """The corpora are downloads, so the common machine has none. Skipping
    costs an evaluation an optional measurement; failing costs it the run."""
    block = linking_corpora.linking_block(None)

    assert block.reports == ()
    assert block.metrics() == {}
    assert block.summary() == ""


def test_a_corpus_root_that_is_not_there_skips_the_block(
    tmp_path: pathlib.Path, no_index: None
) -> None:
    """A configured path that has moved is the same situation as none, and a
    stale `config.toml` must not be what ends an evaluation."""
    assert linking_corpora.linking_block(tmp_path / "gone").reports == ()


def test_a_corpus_root_holding_neither_corpus_skips_the_block(
    tmp_path: pathlib.Path, no_index: None
) -> None:
    assert linking_corpora.linking_block(tmp_path).reports == ()


def test_enzymener_without_the_nomenclature_is_skipped(
    tmp_path: pathlib.Path, tiny_index: None
) -> None:
    """enzymeNER assigns no identifiers, so the nomenclature *is* its gold.
    Scored without one every span falls outside the bridge, which reads as a
    bridge that resolves nothing rather than as a corpus half installed."""
    root = _enzymener_corpus(tmp_path, nomenclature=False)

    assert linking_corpora.linking_block(root).reports == ()


def test_a_dated_export_is_not_taken_as_the_corpus(
    tmp_path: pathlib.Path, no_index: None
) -> None:
    """Upstream publishes several dated exports and they do not annotate the
    same spans, so the export scored is the one named at a path this project
    fixes. Found by a glob instead, a second download landing beside the first
    would move the gold set with nothing anywhere saying so."""
    root = _nlp4pheno_corpus(tmp_path, export=DATED_EXPORT)

    assert linking_corpora.linking_block(root).reports == ()


def test_the_corpus_without_the_name_this_project_fixes_says_so(
    tmp_path: pathlib.Path, no_index: None, caplog: pytest.LogCaptureFixture
) -> None:
    """The fixed name is the operator's to make, so the corpus as published
    scores nothing — which is the layout on disk, not a corner case. Skipped
    mutely it is indistinguishable in the log from a machine that never
    downloaded NLP4Pheno, so the one thing the run can be read for is the
    thing that has to be said."""
    (tmp_path / linking_corpora.NLP4PHENO).mkdir()

    with caplog.at_level(logging.WARNING, logger=linking_corpora.__name__):
        assert linking_corpora.linking_block(tmp_path).reports == ()

    (missing,) = [
        record.getMessage()
        for record in caplog.records
        if "symlink" in record.getMessage()
    ]
    assert str(linking_corpora.NLP4PHENO_EXPORT) in missing


# --------------------------------------------------------------------------- #
# A machine with them                                                          #
# --------------------------------------------------------------------------- #
def test_each_corpus_present_is_scored_under_its_own_namespace(
    tmp_path: pathlib.Path, tiny_index: None
) -> None:
    """The layout `config.toml.example` documents, end to end: the directory
    names are what the block finds the corpora by."""
    root = _enzymener_corpus(_s800_corpus(tmp_path), nomenclature=True)

    block = linking_corpora.linking_block(root)

    assert [report.namespace for report in block.reports] == [
        NCBI_TAXID,
        EC_NUMBER,
    ]
    assert block.index_digest
    metrics = block.metrics()
    assert metrics[f"test/linking_{NCBI_TAXID}_annotated"] == 3.0
    assert metrics[f"test/linking_{EC_NUMBER}_annotated"] == 1.0
    assert [
        name for name in metrics if metric_docs.describe(name) is None
    ] == []


def test_the_strain_corpus_is_scored_beside_the_other_two(
    tmp_path: pathlib.Path, tiny_index: None
) -> None:
    """The strain gold reaches an evaluation run the way the other two do.
    Scored only by a script it is a number nobody has beside the rest, and the
    three reports have to key their metrics apart to share one run."""
    root = _nlp4pheno_corpus(
        _enzymener_corpus(_s800_corpus(tmp_path), nomenclature=True)
    )

    block = linking_corpora.linking_block(root)

    assert [report.namespace for report in block.reports] == [
        NCBI_TAXID,
        EC_NUMBER,
        STRAIN_NUMBER,
    ]
    metrics = block.metrics()
    assert metrics[f"test/linking_{STRAIN_NUMBER}_judged"] == 1.0
    assert metrics[f"test/linking_{STRAIN_NUMBER}_strict_accuracy"] == 1.0
    assert [
        name for name in metrics if metric_docs.describe(name) is None
    ] == []


def test_the_strain_corpus_alone_is_scored_alone(
    tmp_path: pathlib.Path, tiny_index: None
) -> None:
    """The export is a download like the other two, and the one a machine has
    is the one it scores."""
    block = linking_corpora.linking_block(_nlp4pheno_corpus(tmp_path))

    assert [report.namespace for report in block.reports] == [STRAIN_NUMBER]


def test_one_corpus_present_is_scored_alone(
    tmp_path: pathlib.Path, tiny_index: None
) -> None:
    """Half the corpora is a measurement, not a broken checkout."""
    block = linking_corpora.linking_block(_s800_corpus(tmp_path))

    assert [report.namespace for report in block.reports] == [NCBI_TAXID]
