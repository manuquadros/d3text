"""Scoring a linker against gold it had no part in choosing.

The failure these tests exist to catch is silent and flattering. Reconstruct a
span's gold entity from the surface-form index and the linker — which reads
that same index — agrees with it by construction: measured that way the score
is 1.000 over ten thousand spans and says nothing about linking. So the
fixtures here are built to make the two sources **disagree**, which is only
possible when they really are two sources. A gold quietly re-derived from the
dictionary turns every one of these red.

Everything runs on small in-memory indexes through the real `DictionaryLinker`
and the real `SurfaceFormIndex`, so nothing needs the BRENDA files, the NCBI
dump or the S800 corpus.
"""

import pytest
from d3text import metric_docs, surface_forms
from d3text.identifier_bridge import (
    NCBI_TAXID,
    BridgeRow,
    ExternalMention,
    IdentifierBridge,
)
from d3text.linking import DictionaryLinker
from d3text.linking_eval import LinkingReport, score_linking
from d3text.mention_metrics import LinkingScores

COLI = "Escherichia coli"
SUBTILIS = "Bacillus subtilis"


def _linker(forms: dict[str, list[str]]) -> DictionaryLinker:
    return DictionaryLinker(surface_forms.build_index(forms))


def _bridge(taxids: dict[str, str]) -> IdentifierBridge:
    return IdentifierBridge.from_rows(
        NCBI_TAXID,
        [
            BridgeRow(entity_id, taxid, "organism")
            for entity_id, taxid in taxids.items()
        ],
    )


def _mention(
    surface: str, taxid: str, start: int = 0, document: str = "species001"
) -> ExternalMention:
    return ExternalMention(
        document=document,
        start=start,
        end=start + len(surface),
        surface=surface,
        external_id=taxid,
    )


def _populations(report: LinkingReport) -> tuple[int, int, int, int]:
    return (
        report.annotated,
        report.judged,
        report.outside_bridge,
        report.ambiguous_gold,
    )


def _score(mentions, bridge, linker) -> LinkingReport:
    return score_linking(
        mentions=mentions,
        bridge=bridge,
        linker=linker,
        entity_type="bacteria",
        namespace=NCBI_TAXID,
    )


# --------------------------------------------------------------------------- #
# The gold is not the dictionary's own answer                                  #
# --------------------------------------------------------------------------- #
def test_gold_from_the_bridge_can_contradict_the_dictionary() -> None:
    """The anti-circularity property in one fixture.

    The dictionary says this surface form names `bac1`; the outside authority
    says the taxid annotated on the span belongs to `bac2`. Gold derived from
    the dictionary would return the linker's own answer and score 1.000; gold
    from the bridge scores this wrong, which is the only outcome that shows
    the two resources are distinct.
    """
    report = _score(
        [_mention(COLI, "562")],
        _bridge({"bac2": "562"}),
        _linker({"bac1": [COLI]}),
    )

    assert report.judged == 1
    assert (report.strict.correct, report.strict.wrong) == (0, 1)
    assert report.lenient.correct == 0


def test_the_judged_subset_does_not_move_with_the_dictionary() -> None:
    """The subtler half of the same rule: selection is on the gold side.

    Keeping the spans the dictionary resolves *uniquely* would keep exactly
    the spans whose answer is a singleton, and that answer would then be the
    gold — rigorous-looking and circular. Swapping the dictionary for one
    that answers nothing must therefore move every score and no population
    count.
    """
    mentions = [_mention(COLI, "562"), _mention(SUBTILIS, "1423", start=40)]
    bridge = _bridge({"bac1": "562", "bac2": "1423"})

    knows = _score(
        mentions, bridge, _linker({"bac1": [COLI], "bac2": [SUBTILIS]})
    )
    blank = _score(mentions, bridge, _linker({"bac9": ["Thermus aquaticus"]}))

    assert _populations(knows) == _populations(blank) == (2, 2, 0, 0)
    assert knows.strict.accuracy == 1.0
    assert blank.strict.accuracy == 0.0


def test_an_identifier_two_entities_share_is_not_judged() -> None:
    """Gold-side ambiguity, and the only unambiguity test allowed to select:
    BRENDA curating one taxon twice makes the right answer unanswerable, so
    the span leaves the judged subset instead of being scored against a
    coin flip."""
    report = _score(
        [_mention(COLI, "562")],
        _bridge({"bac1": "562", "bac2": "562"}),
        _linker({"bac1": [COLI]}),
    )

    assert (report.judged, report.ambiguous_gold) == (0, 1)
    assert report.annotated == 1


def test_a_mention_outside_the_bridge_is_counted_not_scored() -> None:
    """A species BRENDA does not curate, or one the bridge failed to resolve
    — indistinguishable from here. Scoring it as NIL would charge the linker
    for the bridge's misses, so it is counted and left out."""
    report = _score(
        [_mention("Plasmodium falciparum", "5833")],
        _bridge({"bac1": "562"}),
        _linker({"bac1": [COLI]}),
    )

    assert (report.judged, report.outside_bridge) == (0, 1)
    assert report.coverage == 0.0


# --------------------------------------------------------------------------- #
# Strict scoring, which the gold-side subset is what licenses                  #
# --------------------------------------------------------------------------- #
def test_strict_refuses_the_answer_the_lenient_rule_accepts() -> None:
    """The metric upgrade the subset buys: a linker returning the gold among
    several candidates has not disambiguated, and on a subset whose gold is
    known to be one entity that can be said."""
    report = _score(
        [_mention("AS-A", "562")],
        _bridge({"bac1": "562"}),
        _linker({"bac1": ["AS-A"], "bac2": ["AS-A"]}),
    )

    assert report.lenient.correct == 1
    assert report.strict.wrong == 1
    assert report.candidates == {2: 1}


def test_an_unresolved_span_is_a_missed_link_not_a_correct_nil() -> None:
    report = _score(
        [_mention(COLI, "562")],
        _bridge({"bac1": "562"}),
        _linker({"bac9": ["Thermus aquaticus"]}),
    )

    assert report.strict.nil_missed == 1
    assert report.strict.nil_correct == 0
    assert report.candidates == {0: 1}


# --------------------------------------------------------------------------- #
# The score is unreportable without its denominator                            #
# --------------------------------------------------------------------------- #
def test_every_accuracy_is_keyed_beside_its_coverage() -> None:
    """The bias runs towards the easy half, so an accuracy logged without the
    share it was taken over is not a claim anyone can check."""
    report = _score(
        [_mention(COLI, "562"), _mention("Plasmodium falciparum", "5833", 40)],
        _bridge({"bac1": "562"}),
        _linker({"bac1": [COLI]}),
    )
    metrics = report.metrics()

    assert [name for name in metrics if name.endswith("_accuracy")]
    assert metrics["test/linking_coverage"] == pytest.approx(0.5)
    assert metrics["test/linking_judged"] == 1.0
    assert metrics["test/linking_annotated"] == 2.0


def test_the_summary_states_the_coverage_beside_the_score() -> None:
    report = _score(
        [_mention(COLI, "562"), _mention("Plasmodium falciparum", "5833", 40)],
        _bridge({"bac1": "562"}),
        _linker({"bac1": [COLI]}),
    )
    summary = report.summary()

    assert "1.000" in summary
    assert "50.0%" in summary
    assert "2 annotated mentions" in summary


def test_populations_that_do_not_add_up_are_refused() -> None:
    """The coverage denominator is made of the counts beside it; a report
    where they disagree describes a subset nobody can locate."""
    with pytest.raises(ValueError, match="coverage denominator"):
        LinkingReport(
            namespace=NCBI_TAXID,
            entity_type="bacteria",
            documents=1,
            annotated=10,
            judged=1,
            outside_bridge=1,
            ambiguous_gold=1,
            strict=LinkingScores(correct=1),
            lenient=LinkingScores(correct=1),
            candidates={1: 1},
        )


def test_documented_metric_keys() -> None:
    """Every key this report emits has a glossary entry, because MLflow
    charts a key and records no unit anywhere else."""
    report = _score(
        [_mention(COLI, "562")],
        _bridge({"bac1": "562"}),
        _linker({"bac1": [COLI]}),
    )

    assert [
        name for name in report.metrics() if metric_docs.describe(name) is None
    ] == []


# --------------------------------------------------------------------------- #
# Refusals                                                                     #
# --------------------------------------------------------------------------- #
def test_a_bridge_of_other_identifiers_is_refused() -> None:
    bridge = IdentifierBridge.from_rows(
        "ec_number", [BridgeRow("enz1", "1.1.1.1", "nomenclature")]
    )

    with pytest.raises(ValueError, match="ec_number"):
        score_linking(
            mentions=[_mention(COLI, "562")],
            bridge=bridge,
            linker=_linker({"bac1": [COLI]}),
            entity_type="bacteria",
            namespace=NCBI_TAXID,
        )


def test_an_unknown_entity_type_is_refused() -> None:
    with pytest.raises(ValueError, match="not an entity type"):
        score_linking(
            mentions=[_mention(COLI, "562")],
            bridge=_bridge({"bac1": "562"}),
            linker=_linker({"bac1": [COLI]}),
            entity_type="archaea",
            namespace=NCBI_TAXID,
        )


def test_spans_are_scored_per_document() -> None:
    """Two documents sharing an offset are two mentions, not one: the scores
    key on the span, and pooling the documents would collapse them."""
    report = _score(
        [
            _mention(COLI, "562", document="species001"),
            _mention(SUBTILIS, "1423", document="species002"),
        ],
        _bridge({"bac1": "562", "bac2": "1423"}),
        _linker({"bac1": [COLI], "bac2": [SUBTILIS]}),
    )

    assert (report.documents, report.judged) == (2, 2)
    assert report.strict.correct == 2
