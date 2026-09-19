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
in `tmp_path` or asserts that absence skips the block, so nothing here pays
the entity dump's 256 MB tail read or the ~1.7 GB resident index it builds.
"""

import hashlib
import json
import logging
import os
import pathlib
from typing import Any

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
from d3text.linking_eval import (
    LinkingReport,
    PredictedLinkingReport,
    TaggedSpan,
    score_linking,
    score_predicted_linking,
)

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
TRUNCATED_S800_ANNOTATIONS = (
    "562\tspecies001:111\t10\t25\tEscherichia coli\n1423\tspecies001:111\t45"
)
"""A row cut mid-field, holding three of the five columns a row needs."""

ENZYMENER_SENTENCES = "PMC1\tS01\tAssays of alcohol dehydrogenase were run."
ENZYMENER_ANNOTATIONS = "PMC1\tS01\t10\t31\talcohol dehydrogenase"
ENZYME_DAT = "\n".join(("ID   1.1.1.1", "DE   alcohol dehydrogenase.", "//"))
TRUNCATED_ENZYME_DAT = "\n".join(("ID   1.1.1.1", "DE   alcohol dehydro"))
"""A download cut short of the `//` a record ends at, so it holds no record."""

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

OTHER_LABEL = "TAXON"
"""One of the labels the export carries and this project does not read."""


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


# --------------------------------------------------------------------------- #
# The scripts' call-through wrappers                                           #
# --------------------------------------------------------------------------- #
def test_organism_linking_forwards_namespace_and_caller_s_types() -> None:
    """`organism_linking` is what `scripts/score_species_linking.py` calls
    instead of inlining `score_linking`. Unlike `enzyme_linking` and
    `strain_linking`, it takes `entity_types` from the caller rather than
    fixing it — the species script scores bacteria alone, other organisms
    alone, and both together. A revert to the inline call, or a wrapper that
    stopped forwarding either argument, would diverge from the equivalent
    direct `score_linking` call asserted here."""
    bridge = IdentifierBridge.from_rows(
        NCBI_TAXID, [BridgeRow("bac1", "562", "lpsn_id")]
    )
    linker = DictionaryLinker(surface_forms.build_index({"bac1": [COLI]}))
    mentions = [_mention(COLI, "562")]

    for entity_types in (
        ["bacteria"],
        ["other_organisms"],
        list(linking_corpora.ORGANISM_TYPES),
    ):
        assert linking_corpora.organism_linking(
            mentions=mentions,
            bridge=bridge,
            linker=linker,
            entity_types=entity_types,
        ) == score_linking(
            mentions=mentions,
            bridge=bridge,
            linker=linker,
            entity_types=entity_types,
            namespace=NCBI_TAXID,
        )


def test_enzyme_linking_fixes_the_ec_namespace_and_entity_type() -> None:
    """`enzyme_linking` is what `scripts/score_enzyme_linking.py` calls
    instead of inlining `score_linking`, mirroring `strain_linking`'s shape:
    both `namespace` and `entity_types` are fixed inside the wrapper. A
    revert to the inline call, or a wrapper fixing the wrong namespace or
    type, would diverge from the equivalent direct `score_linking` call
    asserted here."""
    bridge = IdentifierBridge.from_rows(
        EC_NUMBER, [BridgeRow("enz1", "1.1.1.1", "ec_class")]
    )
    linker = DictionaryLinker(surface_forms.build_index({"enz1": [ADH]}))
    mentions = [_mention(ADH, "1.1.1.1")]

    assert linking_corpora.enzyme_linking(
        mentions=mentions, bridge=bridge, linker=linker
    ) == score_linking(
        mentions=mentions,
        bridge=bridge,
        linker=linker,
        entity_types=list(linking_corpora.ENZYME_TYPES),
        namespace=EC_NUMBER,
    )


# --------------------------------------------------------------------------- #
# Detection and linking composed through the wrapper                          #
# --------------------------------------------------------------------------- #
def test_organism_linking_forwards_predicted_spans_to_the_new_scorer() -> None:
    """Passing `predicted` switches `organism_linking` onto
    `score_predicted_linking`, forwarding every argument the gold-offset path
    already took plus the tagger's own spans -- proven by matching the
    equivalent direct call, the same way the gold-offset path is proven
    above."""
    bridge = IdentifierBridge.from_rows(
        NCBI_TAXID, [BridgeRow("bac1", "562", "lpsn_id")]
    )
    linker = DictionaryLinker(surface_forms.build_index({"bac1": [COLI]}))
    mentions = [_mention(COLI, "562", document="d1")]
    predicted = [
        TaggedSpan(
            document="d1",
            start=0,
            end=len(COLI),
            surface=COLI,
            entity_type="bacteria",
        )
    ]

    result = linking_corpora.organism_linking(
        mentions=mentions,
        bridge=bridge,
        linker=linker,
        entity_types=["bacteria"],
        predicted=predicted,
    )

    assert result == score_predicted_linking(
        predicted=predicted,
        gold=mentions,
        bridge=bridge,
        linker=linker,
        entity_types=["bacteria"],
        namespace=NCBI_TAXID,
    )
    assert isinstance(result, PredictedLinkingReport)
    assert result.strict.correct == 1


def test_organism_linking_charges_a_missed_span_against_the_score() -> None:
    """The ticket's own case, proven through the wrapper rather than the raw
    scorer: a tagger that proposes nothing at all for a document still has to
    cost the gold mention a linking opportunity, not leave the denominator
    where a bare detection miss would -- and an empty `predicted` list must
    not be mistaken for "no predicted spans given" and fall back to the
    circular gold-offset path."""
    bridge = IdentifierBridge.from_rows(
        NCBI_TAXID, [BridgeRow("bac1", "562", "lpsn_id")]
    )
    linker = DictionaryLinker(surface_forms.build_index({"bac1": [COLI]}))
    mentions = [_mention(COLI, "562", document="d1")]

    report = linking_corpora.organism_linking(
        mentions=mentions,
        bridge=bridge,
        linker=linker,
        entity_types=["bacteria"],
        predicted=[],
    )

    assert isinstance(report, PredictedLinkingReport)
    assert report.judged == 1
    assert report.strict.missed_detection == 1
    assert report.strict.accuracy == 0.0
    assert (
        report.metrics()[f"test/predicted_linking_{NCBI_TAXID}_judged"] == 1.0
    )


def test_enzyme_linking_composes_detection_and_linking_too() -> None:
    """The same wiring, on the wrapper `strain_linking` mirrors: a detection
    miss still lands in `judged`, not dropped from it."""
    bridge = IdentifierBridge.from_rows(
        EC_NUMBER, [BridgeRow("enz1", "1.1.1.1", "ec_class")]
    )
    linker = DictionaryLinker(surface_forms.build_index({"enz1": [ADH]}))
    mentions = [_mention(ADH, "1.1.1.1", document="d1")]

    report = linking_corpora.enzyme_linking(
        mentions=mentions, bridge=bridge, linker=linker, predicted=[]
    )

    assert isinstance(report, PredictedLinkingReport)
    assert (report.judged, report.strict.missed_detection) == (1, 1)


def test_strain_linking_composes_detection_and_linking_too() -> None:
    """Same wiring again, on `strain_linking`, which additionally runs its
    mentions through `culture_numbers.assign` before scoring either way."""
    bridge = IdentifierBridge.from_rows(
        STRAIN_NUMBER, [BridgeRow(STRAIN_ENTITY, DEPOSIT, "culture_number")]
    )
    linker = DictionaryLinker(
        surface_forms.build_index({STRAIN_ENTITY: [AUREUS]})
    )
    mentions = [_mention(AUREUS, None, document="d1")]

    report = linking_corpora.strain_linking(
        mentions=mentions, bridge=bridge, linker=linker, predicted=[]
    )

    assert isinstance(report, PredictedLinkingReport)
    assert (report.judged, report.strict.missed_detection) == (1, 1)


@pytest.fixture
def no_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make building the surface-form index an error.

    Building it costs a 256 MB tail read of the entity dump and a scan of
    every split, landing at ~1.7 GB resident, so a root with nothing to
    score has to be settled before it is touched — and a test that merely
    returned an empty block would pass either way.
    """

    def refuse() -> surface_forms.SurfaceFormIndex:
        raise AssertionError("the surface-form index was built for nothing")

    monkeypatch.setattr(linking_corpora, "brenda_index", refuse)


@pytest.fixture
def tiny_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand a three-entry index in for BRENDA's ~1.7 GB resident build."""
    monkeypatch.setattr(
        linking_corpora,
        "brenda_index",
        lambda: surface_forms.build_index(
            {"bac1": [COLI], "enz1": [ADH], STRAIN_ENTITY: [AUREUS]}
        ),
    )


def _s800_corpus(
    root: pathlib.Path, annotations: str = S800_ANNOTATIONS + "\n"
) -> pathlib.Path:
    directory = root / linking_corpora.S800
    (directory / s800.ABSTRACTS).mkdir(parents=True)
    (directory / s800.ANNOTATIONS).write_text(annotations, encoding="utf8")
    for document, text in S800_TEXTS.items():
        (directory / s800.ABSTRACTS / f"{document}.txt").write_text(
            text, encoding="utf8"
        )
    return root


def _enzymener_corpus(
    root: pathlib.Path,
    nomenclature: bool,
    nomenclature_text: str = ENZYME_DAT + "\n",
    annotations: str = ENZYMENER_ANNOTATIONS + "\n",
    sentences: bool = True,
) -> pathlib.Path:
    directory = root / linking_corpora.ENZYMENER
    directory.mkdir(parents=True)
    if sentences:
        (directory / enzymener.SENTENCES).write_text(
            "\ufeff" + ENZYMENER_SENTENCES + "\n", encoding="utf8"
        )
    (directory / enzymener.ANNOTATIONS).write_text(
        "\ufeff" + annotations, encoding="utf8"
    )
    if nomenclature:
        path = root / linking_corpora.NOMENCLATURE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(nomenclature_text, encoding=expasy.ENCODING)
    return root


def _nlp4pheno_task(label: str = nlp4pheno.STRAIN) -> dict[str, Any]:
    """One exported task marking the strain designation under `label`."""
    start = NLP4PHENO_TEXT.index(AUREUS)
    return {
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
                            "labels": [label],
                        },
                    }
                ]
            }
        ],
    }


def _nlp4pheno_corpus(
    root: pathlib.Path,
    export: str | os.PathLike[str] | None = None,
    tasks: list[dict[str, Any]] | None = None,
) -> pathlib.Path:
    directory = root / linking_corpora.NLP4PHENO
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (
        linking_corpora.NLP4PHENO_EXPORT.name if export is None else export
    )
    path.write_text(
        json.dumps([_nlp4pheno_task()] if tasks is None else tasks),
        encoding="utf8",
    )
    return root


DUMP = "documents.json"
"""The entity dump's name as `brenda_references` ships it, spelled out here so
that a constant drifting from it fails rather than agreeing with itself."""

BRENDA_INPUTS = (
    DUMP,
    *(f"{split}_data.csv" for split in linking_corpora.SPLITS),
)


def _brenda_data(directory: pathlib.Path, absent: str | None) -> pathlib.Path:
    """Every BRENDA input but `absent`, each the smallest the reader accepts.

    Readable rather than merely present, so that code which never looks for
    `absent` first reads its way to it and raises on it, not on a stand-in.
    Also writes a `MANIFEST` naming every file this wrote, matching its
    content's digest, so `brenda_index`'s checksum check passes on this
    fixture the same way it would on an untouched download.
    """
    directory.mkdir()
    manifest_lines = []
    for name in BRENDA_INPUTS:
        if name != absent:
            content = "{}" if name == DUMP else "id\n"
            (directory / name).write_text(content, encoding="utf8")
            digest = hashlib.sha256(content.encode("utf8")).hexdigest()
            manifest_lines.append(f"{digest}  {name}")
    (directory / linking_corpora.MANIFEST).write_text(
        "\n".join(manifest_lines) + "\n", encoding="utf8"
    )
    return directory


def _truncated_nlp4pheno_corpus(root: pathlib.Path) -> pathlib.Path:
    directory = root / linking_corpora.NLP4PHENO
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / linking_corpora.NLP4PHENO_EXPORT.name
    whole = json.dumps([_nlp4pheno_task()])
    path.write_text(whole[: len(whole) // 2], encoding="utf8")
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


def test_every_report_in_the_block_carries_its_own_corpus_s_digest(
    tmp_path: pathlib.Path, tiny_index: None
) -> None:
    """`index_digest` says whether the dictionary side of two evaluations
    matches; this is the other side, so all three corpora need it under one
    mechanism — a digest missing from one report would leave that corpus
    looking pinned when it is not."""
    root = _nlp4pheno_corpus(
        _enzymener_corpus(_s800_corpus(tmp_path), nomenclature=True)
    )

    block = linking_corpora.linking_block(root)

    digests = {
        report.namespace: report.corpus_digest for report in block.reports
    }
    assert set(digests) == {NCBI_TAXID, EC_NUMBER, STRAIN_NUMBER}
    assert all(digests.values())
    assert (
        digests[NCBI_TAXID]
        == hashlib.sha256(
            (root / linking_corpora.S800 / s800.ANNOTATIONS).read_bytes()
        ).hexdigest()
    )
    assert (
        digests[EC_NUMBER]
        == hashlib.sha256(
            (
                root / linking_corpora.ENZYMENER / enzymener.ANNOTATIONS
            ).read_bytes()
        ).hexdigest()
    )
    assert (
        digests[STRAIN_NUMBER]
        == hashlib.sha256(
            (root / linking_corpora.NLP4PHENO_EXPORT).read_bytes()
        ).hexdigest()
    )
    assert len(set(digests.values())) == 3


def test_the_corpus_digest_moves_with_the_annotation_file_s_content(
    tmp_path: pathlib.Path,
) -> None:
    """The whole point of carrying it is telling two runs' gold apart, so it
    has to actually be a function of the file's bytes — not merely present —
    and it must not move when the file doesn't."""
    linker = DictionaryLinker(surface_forms.build_index({"bac1": [COLI]}))
    first = linking_corpora.organism_report(
        _s800_corpus(tmp_path / "a"), linker
    )
    same_content = linking_corpora.organism_report(
        _s800_corpus(tmp_path / "b"), linker
    )
    changed = linking_corpora.organism_report(
        _s800_corpus(tmp_path / "c", annotations=S800_ANNOTATIONS + "\n\n"),
        linker,
    )

    assert (
        first is not None and same_content is not None and changed is not None
    )
    assert first.corpus_digest != ""
    assert first.corpus_digest == same_content.corpus_digest
    assert first.corpus_digest != changed.corpus_digest


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


@pytest.mark.parametrize("absent", BRENDA_INPUTS)
def test_a_missing_brenda_input_skips_the_block(
    absent: str,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """With gold on disk the block goes on to build its index from BRENDA's
    own files, and an evaluation from a recorded vocabulary needs only the
    test split. A file missing there raised after every other metric had
    been logged, which exits a finished evaluation non-zero."""
    data = _brenda_data(tmp_path / "brenda", absent=absent)
    monkeypatch.setattr(linking_corpora, "DATA_DIR", data)

    with caplog.at_level(logging.WARNING, logger=linking_corpora.__name__):
        block = linking_corpora.linking_block(_s800_corpus(tmp_path / "gold"))

    assert block.reports == ()
    (warning,) = [
        record.getMessage()
        for record in caplog.records
        if "linking block is skipped" in record.getMessage()
    ]
    assert str(data / absent) in warning


def test_the_brenda_files_all_there_build_the_index(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard's other side: one that skipped on a file that is there would
    drop the block on every machine, with a warning nobody reads as a bug."""
    monkeypatch.setattr(
        linking_corpora, "DATA_DIR", _brenda_data(tmp_path / "brenda", None)
    )

    block = linking_corpora.linking_block(_s800_corpus(tmp_path / "gold"))

    assert [report.namespace for report in block.reports] == [NCBI_TAXID]


SPLIT = "training_data.csv"

UNREADABLE_BRENDA_INPUTS = (
    pytest.param(DUMP, b'{"enz', id="dump-cut-short"),
    pytest.param(DUMP, b'{"enzymes": {"1": "\xff"}}', id="dump-not-utf8"),
    pytest.param(SPLIT, b"", id="split-empty"),
    pytest.param(
        SPLIT, b"id,other_organisms\n1,\"{'oth1'", id="split-cut-mid-quote"
    ),
    pytest.param(
        SPLIT, b"id,other_organisms\n1,{'oth1\n", id="split-cut-mid-cell"
    ),
    pytest.param(SPLIT, b"id,other_organisms\n1,oops\n", id="split-no-literal"),
    pytest.param(
        SPLIT,
        b"id,other_organisms\n1,\"{'oth1': '\xff'}\"\n",
        id="split-not-utf8",
    ),
)


def _rewrite_digest(data: pathlib.Path, name: str, content: bytes) -> None:
    """Point `_brenda_data`'s manifest entry for `name` at `content`'s digest.

    Lets a test overwrite one input's bytes after fixture creation while
    still passing the checksum gate, so what follows still exercises the
    parse-level failure the test is named for rather than a digest mismatch
    `brenda_index` now catches first.
    """
    manifest = data / linking_corpora.MANIFEST
    kept = [
        line
        for line in manifest.read_text().splitlines()
        if not line.endswith(f"  {name}")
    ]
    kept.append(f"{hashlib.sha256(content).hexdigest()}  {name}")
    manifest.write_text("\n".join(kept) + "\n", encoding="utf8")


def _skip_warning(root: pathlib.Path, caplog: pytest.LogCaptureFixture) -> str:
    """Build the block over a gold corpus under `root`, assert nothing was
    scored, and return the one warning saying the block was skipped."""
    with caplog.at_level(logging.WARNING, logger=linking_corpora.__name__):
        block = linking_corpora.linking_block(_s800_corpus(root / "gold"))

    assert block.reports == ()
    (warning,) = [
        record.getMessage()
        for record in caplog.records
        if "linking block is skipped" in record.getMessage()
    ]
    return warning


@pytest.mark.parametrize(("name", "content"), UNREADABLE_BRENDA_INPUTS)
def test_an_unreadable_brenda_input_skips_the_block(
    name: str,
    content: bytes,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A file present but truncated or malformed raised out of the index
    build, after every other metric had been logged, and exited a finished
    evaluation non-zero exactly as a missing one used to."""
    data = _brenda_data(tmp_path / "brenda", absent=None)
    (data / name).write_bytes(content)
    _rewrite_digest(data, name, content)
    monkeypatch.setattr(linking_corpora, "DATA_DIR", data)

    assert str(data / name) in _skip_warning(tmp_path, caplog)


def test_a_dump_whose_tail_holds_no_entity_table_skips_the_block(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The real dump is read off its tail, where one cut short of its entity
    tables raises the reader's own `ValueError`, not a decode error."""
    data = _brenda_data(tmp_path / "brenda", absent=None)
    dump_content = b'{"documents": {}}'
    (data / DUMP).write_bytes(dump_content)
    _rewrite_digest(data, DUMP, dump_content)
    monkeypatch.setattr(linking_corpora, "DATA_DIR", data)
    monkeypatch.setattr(surface_forms, "_TAIL_SEARCH_BYTES", 8)

    assert str(data / DUMP) in _skip_warning(tmp_path, caplog)


@pytest.mark.skipif(
    os.geteuid() == 0, reason="root opens a file whatever its mode"
)
@pytest.mark.parametrize("name", (DUMP, SPLIT))
def test_a_brenda_input_that_cannot_be_opened_skips_the_block(
    name: str,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A file `is_file` accepts but `open` refuses fails with an `OSError`,
    which no parse error covers."""
    data = _brenda_data(tmp_path / "brenda", absent=None)
    (data / name).chmod(0)
    monkeypatch.setattr(linking_corpora, "DATA_DIR", data)

    assert str(data / name) in _skip_warning(tmp_path, caplog)


def test_a_split_truncated_at_a_row_boundary_skips_the_block(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A split cut exactly at a row boundary parses without raising anything
    — a real, well-formed CSV holding a fraction of the other-organism names
    rather than none — so nothing short of a digest check catches it.
    Undetected, the index would be built from that fraction and the linker
    would answer NIL to every name past the cut: a score, where the honest
    outcome is no report."""
    data = _brenda_data(tmp_path / "brenda", absent=None)
    whole = (
        "id,other_organisms\n"
        "1,\"{'oth1': 'Bacillus cereus'}\"\n"
        "2,\"{'oth2': 'Vibrio cholerae'}\"\n"
    )
    _rewrite_digest(data, SPLIT, whole.encode("utf8"))
    # The manifest now names the whole file's digest; the split on disk holds
    # only its first row, cut exactly at the newline between the two — no
    # exception anywhere in the read.
    (data / SPLIT).write_text(whole[: whole.index("2,")], encoding="utf8")
    monkeypatch.setattr(linking_corpora, "DATA_DIR", data)

    assert str(data / SPLIT) in _skip_warning(tmp_path, caplog)


def test_a_missing_manifest_skips_the_block(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Every BRENDA input present and readable is not, by itself, evidence
    they are whole — that is exactly what the manifest is for, so a machine
    holding the data without it cannot have that checked and is treated the
    same as a machine missing a file outright."""
    data = _brenda_data(tmp_path / "brenda", absent=None)
    (data / linking_corpora.MANIFEST).unlink()
    monkeypatch.setattr(linking_corpora, "DATA_DIR", data)

    assert str(data / linking_corpora.MANIFEST) in _skip_warning(
        tmp_path, caplog
    )


def test_a_manifest_missing_one_input_s_entry_skips_the_block(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A present but incomplete manifest — one written before a required
    input existed, or itself truncated — must not verify the files it does
    name and silently wave the rest through: a row-truncated split beside a
    manifest missing that split's entry has to be treated exactly like a
    wholly-missing manifest, not like a match."""
    data = _brenda_data(tmp_path / "brenda", absent=None)
    manifest = data / linking_corpora.MANIFEST
    kept = [
        line
        for line in manifest.read_text().splitlines()
        if not line.endswith(f"  {SPLIT}")
    ]
    manifest.write_text("\n".join(kept) + "\n", encoding="utf8")
    monkeypatch.setattr(linking_corpora, "DATA_DIR", data)

    assert str(data / SPLIT) in _skip_warning(tmp_path, caplog)


def test_a_manifest_line_with_no_separator_raises(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A line missing the two-space separator must fail loud, not key an
    empty name — the same malformed-line check `pull_data.py`'s
    `read_manifest` already makes."""
    data = _brenda_data(tmp_path / "brenda", absent=None)
    manifest = data / linking_corpora.MANIFEST
    manifest.write_text(
        manifest.read_text() + "not-a-valid-line\n", encoding="utf8"
    )
    monkeypatch.setattr(linking_corpora, "DATA_DIR", data)

    with pytest.raises(ValueError, match="not-a-valid-line"):
        linking_corpora._brenda_manifest()


def test_a_failure_building_the_index_is_not_reported_as_bad_data(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the reads are guarded. A `ValueError` out of the builder, over
    files that all read cleanly, is a bug, and skipping on it would bury the
    bug under a warning that blames the data."""
    monkeypatch.setattr(
        linking_corpora, "DATA_DIR", _brenda_data(tmp_path / "brenda", None)
    )

    def broken(*_args: object) -> dict[str, list[str]]:
        raise ValueError("a builder bug")

    monkeypatch.setattr(surface_forms, "brenda_surface_forms", broken)

    with pytest.raises(ValueError, match="a builder bug"):
        linking_corpora.linking_block(_s800_corpus(tmp_path / "gold"))


# --------------------------------------------------------------------------- #
# A corpus that is on disk and holds nothing                                   #
# --------------------------------------------------------------------------- #
# Every test below asserts through `no_index`, which fails if the surface-form
# index is built: an empty corpus settled as absence is settled before the
# entity dump's 256 MB tail read, and a block that merely came back empty
# afterwards would pass a bare `reports == ()`.
@pytest.mark.parametrize("annotations", ("", "\n\n"), ids=("empty", "blank"))
def test_an_s800_table_annotating_nothing_is_skipped(
    annotations: str, tmp_path: pathlib.Path, no_index: None
) -> None:
    """A download that wrote no row is an absent corpus, not a corpus of no
    mentions. Scored, it logs a strict accuracy of 0.0 over an empty
    denominator, and an accuracy is what a reader compares."""
    root = _s800_corpus(tmp_path, annotations=annotations)

    assert linking_corpora.linking_block(root).reports == ()


@pytest.mark.parametrize(
    "nomenclature_text",
    ("", TRUNCATED_ENZYME_DAT),
    ids=("empty", "truncated"),
)
def test_a_nomenclature_naming_no_enzyme_is_skipped(
    nomenclature_text: str, tmp_path: pathlib.Path, no_index: None
) -> None:
    """The flat file yields a record only at its terminator, so a download cut
    short of the first one parses to nothing while still being a file. Every
    span then falls outside the bridge — the total-bridge-failure reading the
    missing-file guard exists to prevent, reached with the file present."""
    root = _enzymener_corpus(
        tmp_path, nomenclature=True, nomenclature_text=nomenclature_text
    )

    assert linking_corpora.linking_block(root).reports == ()


def test_enzymener_annotating_nothing_is_skipped(
    tmp_path: pathlib.Path, no_index: None
) -> None:
    """The other half of the same corpus: the nomenclature is whole and the
    annotation table holds no span, so the gold exists and nothing carries
    it."""
    root = _enzymener_corpus(tmp_path, nomenclature=True, annotations="")

    assert linking_corpora.linking_block(root).reports == ()


@pytest.mark.parametrize(
    "tasks",
    ([], [_nlp4pheno_task(label=OTHER_LABEL)]),
    ids=("no_tasks", "no_strain_span"),
)
def test_an_export_marking_no_strain_span_is_skipped(
    tasks: list[dict[str, Any]], tmp_path: pathlib.Path, no_index: None
) -> None:
    """The export is named by hand from one of several the publisher ships,
    and only one of its labels is read here, so an export carrying none of
    that label is the shape a wrong or half-written file takes — and it is not
    a strain accuracy of zero."""
    root = _nlp4pheno_corpus(tmp_path, tasks=tasks)

    assert linking_corpora.linking_block(root).reports == ()


def test_an_empty_nomenclature_does_not_cost_the_other_corpora(
    tmp_path: pathlib.Path, tiny_index: None
) -> None:
    """Half a download is one corpus missing, not the block. S800 still
    scores, and the enzyme keys are absent rather than zero — which is what
    separates a corpus nobody has from a linker that resolved nothing."""
    root = _enzymener_corpus(
        _s800_corpus(tmp_path), nomenclature=True, nomenclature_text=""
    )

    block = linking_corpora.linking_block(root)

    assert [report.namespace for report in block.reports] == [NCBI_TAXID]
    assert f"test/linking_{EC_NUMBER}_strict_accuracy" not in block.metrics()


# --------------------------------------------------------------------------- #
# A corpus that is on disk and corrupt                                        #
# --------------------------------------------------------------------------- #
# Every test below pairs the corrupt corpus with a valid one under `tiny_index`
# so that a per-corpus catch (rather than one around the whole block) is what
# is actually pinned: the corrupt corpus's report is absent and the valid
# one's is not.
def test_a_truncated_s800_table_is_skipped_not_raised(
    tmp_path: pathlib.Path,
    tiny_index: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A row cut mid-field raises `ValueError` from `s800.parse_annotations`,
    past the presence check that only looks for the table itself."""
    root = _enzymener_corpus(
        _s800_corpus(tmp_path, annotations=TRUNCATED_S800_ANNOTATIONS),
        nomenclature=True,
    )

    with caplog.at_level(logging.WARNING, logger=linking_corpora.__name__):
        block = linking_corpora.linking_block(root)

    assert [report.namespace for report in block.reports] == [EC_NUMBER]
    assert f"test/linking_{NCBI_TAXID}_strict_accuracy" not in block.metrics()
    (warning,) = [
        record.getMessage()
        for record in caplog.records
        if "could not be read" in record.getMessage()
    ]
    assert str(linking_corpora.S800) in warning


def test_a_truncated_nlp4pheno_export_is_skipped_not_raised(
    tmp_path: pathlib.Path,
    tiny_index: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A download cut mid-array raises `json.JSONDecodeError`, a `ValueError`
    subclass, past the presence check that only looks for the fixed name."""
    root = _truncated_nlp4pheno_corpus(_s800_corpus(tmp_path))

    with caplog.at_level(logging.WARNING, logger=linking_corpora.__name__):
        block = linking_corpora.linking_block(root)

    assert [report.namespace for report in block.reports] == [NCBI_TAXID]
    assert (
        f"test/linking_{STRAIN_NUMBER}_strict_accuracy" not in block.metrics()
    )
    (warning,) = [
        record.getMessage()
        for record in caplog.records
        if "could not be read" in record.getMessage()
    ]
    assert linking_corpora.NLP4PHENO_EXPORT.name in warning


def test_enzymener_missing_its_sentence_table_is_skipped_not_raised(
    tmp_path: pathlib.Path,
    tiny_index: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`GoldSet.txt` never arriving raises `FileNotFoundError` from
    `enzymener.load_enzymener`, reached past the presence check that only
    looks for the annotation table beside it."""
    root = _enzymener_corpus(
        _s800_corpus(tmp_path), nomenclature=True, sentences=False
    )

    with caplog.at_level(logging.WARNING, logger=linking_corpora.__name__):
        block = linking_corpora.linking_block(root)

    assert [report.namespace for report in block.reports] == [NCBI_TAXID]
    assert f"test/linking_{EC_NUMBER}_strict_accuracy" not in block.metrics()
    (warning,) = [
        record.getMessage()
        for record in caplog.records
        if "could not be read" in record.getMessage()
    ]
    assert str(linking_corpora.ENZYMENER) in warning


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


def test_predicted_linking_block_scores_both_corpora_through_their_spans(
    tmp_path: pathlib.Path, tiny_index: None
) -> None:
    """The layout `linking_block` scores by gold offset, scored instead
    through a tagger's own spans -- keyed by the store prefixes
    `predicted_spans_from_store` reads groups under."""
    root = _enzymener_corpus(_s800_corpus(tmp_path), nomenclature=True)
    predicted = {
        "s800": [
            TaggedSpan(
                document="species001",
                start=10,
                end=25,
                surface=COLI,
                entity_type="bacteria",
            ),
            TaggedSpan(
                document="species001",
                start=45,
                end=61,
                surface="Bacillus subtilis",
                entity_type="bacteria",
            ),
            TaggedSpan(
                document="species002",
                start=4,
                end=24,
                surface="Plasmodium falciparum",
                entity_type="other_organisms",
            ),
        ],
        "enzymener": [
            TaggedSpan(
                document="PMC1:S01",
                start=10,
                end=31,
                surface=ADH,
                entity_type="enzymes",
            )
        ],
    }

    block = linking_corpora.predicted_linking_block(root, predicted)

    assert [report.namespace for report in block.reports] == [
        NCBI_TAXID,
        EC_NUMBER,
    ]
    assert all(
        isinstance(report, PredictedLinkingReport) for report in block.reports
    )
    metrics = block.metrics()
    assert metrics[f"test/predicted_linking_{NCBI_TAXID}_annotated"] == 3.0
    assert (
        metrics[f"test/predicted_linking_{NCBI_TAXID}_missed_detection"] == 0.0
    )
    assert metrics[f"test/predicted_linking_{EC_NUMBER}_annotated"] == 1.0
    assert [
        name for name in metrics if metric_docs.describe(name) is None
    ] == []


def test_predicted_linking_block_charges_an_undetected_document(
    tmp_path: pathlib.Path, tiny_index: None
) -> None:
    """A corpus whose tagger proposed nothing at all still scores through
    `organism_linking`'s `predicted=[]` branch, every gold mention charged
    as a missed detection rather than left out of the denominator."""
    block = linking_corpora.predicted_linking_block(
        _s800_corpus(tmp_path), {"s800": []}
    )

    (report,) = block.reports
    assert isinstance(report, PredictedLinkingReport)
    assert report.strict.missed_detection == report.judged


def test_predicted_linking_block_skips_a_corpus_with_no_predicted_spans(
    tmp_path: pathlib.Path, tiny_index: None
) -> None:
    """A corpus `predicted` carries no key for is not scored here at all --
    `linking_block` already covers it by gold offset, and scoring it again
    would key the same namespace's predicted-linking metrics twice in one
    run for no tagger that ever ran over it."""
    root = _enzymener_corpus(_s800_corpus(tmp_path), nomenclature=True)

    block = linking_corpora.predicted_linking_block(root, {"s800": []})

    assert [report.namespace for report in block.reports] == [NCBI_TAXID]


def test_predicted_linking_block_skips_where_linking_block_does(
    no_index: None,
) -> None:
    """An unset root costs nothing here either -- `linking_block`'s own
    absence tests already cover the rest of the shared skip logic."""
    assert linking_corpora.predicted_linking_block(None, {}).reports == ()
