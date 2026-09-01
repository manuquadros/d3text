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
    EC_NUMBER,
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


def _score(mentions, bridge, linker, types=("bacteria",)) -> LinkingReport:
    return score_linking(
        mentions=mentions,
        bridge=bridge,
        linker=linker,
        entity_types=list(types),
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
            entity_types=("bacteria",),
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
        EC_NUMBER, [BridgeRow("enz1", "1.1.1.1", "ec_class")]
    )

    with pytest.raises(ValueError, match="ec_number"):
        score_linking(
            mentions=[_mention(COLI, "562")],
            bridge=bridge,
            linker=_linker({"bac1": [COLI]}),
            entity_types=["bacteria"],
            namespace=NCBI_TAXID,
        )


def test_an_unknown_entity_type_is_refused() -> None:
    with pytest.raises(ValueError, match="not an entity type"):
        score_linking(
            mentions=[_mention(COLI, "562")],
            bridge=_bridge({"bac1": "562"}),
            linker=_linker({"bac1": [COLI]}),
            entity_types=["archaea"],
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


# --------------------------------------------------------------------------- #
# Other organisms: the same taxonomy, the other half of the table              #
# --------------------------------------------------------------------------- #
CEREVISIAE = "Saccharomyces cerevisiae"
BOTH = ("bacteria", "other_organisms")


def test_other_organism_gold_can_contradict_the_dictionary() -> None:
    """The anti-circularity property, for the half of the table whose entity
    IDs come from BRENDA's own documents rather than a curated table. The
    dictionary says this yeast is `oth1`; the outside authority says the
    annotated taxid belongs to `oth2`. A gold re-derived from the dictionary
    would return the linker's own answer and score this correct."""
    report = _score(
        [_mention(CEREVISIAE, "4932")],
        _bridge({"oth2": "4932"}),
        _linker({"oth1": [CEREVISIAE]}),
        types=["other_organisms"],
    )

    assert report.judged == 1
    assert (report.strict.correct, report.strict.wrong) == (0, 1)
    assert report.lenient.correct == 0


def test_the_other_organism_subset_does_not_move_with_the_dictionary() -> None:
    """Selection stays on the gold side for this type too: swapping the
    dictionary for one that knows nothing must change every score and no
    population count."""
    mentions = [_mention(CEREVISIAE, "4932")]
    bridge = _bridge({"oth1": "4932"})

    knows = _score(
        mentions,
        bridge,
        _linker({"oth1": [CEREVISIAE]}),
        types=["other_organisms"],
    )
    blank = _score(
        mentions,
        bridge,
        _linker({"oth9": ["Zea mays"]}),
        types=["other_organisms"],
    )

    assert _populations(knows) == _populations(blank) == (1, 1, 0, 0)
    assert knows.strict.accuracy == 1.0
    assert blank.strict.accuracy == 0.0


def test_the_gold_entity_names_the_type_the_linker_is_asked_for() -> None:
    """One corpus, two types, one call: S800 says nothing about which BRENDA
    table a species belongs to, so the type has to come from the gold entity
    — asking every span as one type would score the other type's spans
    against candidates that could not contain their gold."""
    report = _score(
        [_mention(COLI, "562"), _mention(CEREVISIAE, "4932", start=40)],
        _bridge({"bac1": "562", "oth1": "4932"}),
        _linker({"bac1": [COLI], "oth1": [CEREVISIAE]}),
        types=BOTH,
    )

    assert _populations(report) == (2, 2, 0, 0)
    assert report.strict.correct == 2
    assert "bacteria + other_organisms" in report.summary()


def test_a_taxid_only_another_type_carries_is_outside_this_bridge() -> None:
    """Restricting the report to one type must not turn the other type's
    entities into wrong answers: the taxid pairs with nothing of the type
    asked for, which is the same population as a species BRENDA never
    curated."""
    report = _score(
        [_mention(CEREVISIAE, "4932")],
        _bridge({"bac1": "562", "oth1": "4932"}),
        _linker({"bac1": [COLI], "oth1": [CEREVISIAE]}),
    )

    assert _populations(report) == (1, 0, 1, 0)
    assert report.strict.total == 0


def test_a_taxon_curated_under_two_types_is_ambiguous_across_them() -> None:
    """The reason one table holds both halves. Judged per type the taxid names
    one entity each time; judged over both at once nothing says which table
    the mention belongs to, so it leaves the subset instead of being scored
    twice against two different golds."""
    bridge = _bridge({"bac1": "4932", "oth1": "4932"})
    mentions = [_mention(CEREVISIAE, "4932")]
    linker = _linker({"bac1": [CEREVISIAE], "oth1": [CEREVISIAE]})

    combined = _score(mentions, bridge, linker, types=BOTH)
    alone = _score(mentions, bridge, linker, types=["other_organisms"])

    assert _populations(combined) == (1, 0, 0, 1)
    assert _populations(alone) == (1, 1, 0, 0)
    assert alone.strict.correct == 1


def test_no_entity_type_is_refused() -> None:
    with pytest.raises(ValueError, match="no entity type"):
        _score(
            [_mention(COLI, "562")],
            _bridge({"bac1": "562"}),
            _linker({"bac1": [COLI]}),
            types=[],
        )


# --------------------------------------------------------------------------- #
# EC numbers: a namespace whose bridge side filters nothing                   #
# --------------------------------------------------------------------------- #
ADH = "alcohol dehydrogenase"


def _ec_bridge(numbers: dict[str, str]) -> IdentifierBridge:
    return IdentifierBridge.from_rows(
        EC_NUMBER,
        [
            BridgeRow(entity_id, ec_number, "ec_class")
            for entity_id, ec_number in numbers.items()
        ],
    )


def _enzyme(
    surface: str, ec_number: str | None, start: int = 0
) -> ExternalMention:
    return ExternalMention(
        document="PMC1:S01",
        start=start,
        end=start + len(surface),
        surface=surface,
        external_id=ec_number,
    )


def _enzyme_score(mentions, bridge, linker) -> LinkingReport:
    return score_linking(
        mentions=mentions,
        bridge=bridge,
        linker=linker,
        entity_types=["enzymes"],
        namespace=EC_NUMBER,
    )


def test_enzyme_gold_can_contradict_the_dictionary() -> None:
    """The anti-circularity property for the enzyme half, where it has to
    come from somewhere else entirely.

    Every EC number names exactly one BRENDA enzyme, so `sole_entity` excludes
    nothing and the judged subset is the outside nomenclature's alone. The
    dictionary says this name is `enz1`; the nomenclature's EC number belongs
    to `enz2`. A gold re-derived from the dictionary — or a subset chosen from
    the spans it resolves uniquely — would return the linker's own answer and
    score this correct.
    """
    report = _enzyme_score(
        [_enzyme(ADH, "1.1.1.1")],
        _ec_bridge({"enz2": "1.1.1.1"}),
        _linker({"enz1": [ADH]}),
    )

    assert report.judged == 1
    assert (report.strict.correct, report.strict.wrong) == (0, 1)
    assert report.lenient.correct == 0


def test_the_enzyme_subset_does_not_move_with_the_dictionary() -> None:
    """Selection stays on the gold side for this type too: swapping the
    dictionary for one that knows nothing must change every score and no
    population count."""
    mentions = [_enzyme(ADH, "1.1.1.1")]
    bridge = _ec_bridge({"enz1": "1.1.1.1"})

    knows = _enzyme_score(mentions, bridge, _linker({"enz1": [ADH]}))
    blank = _enzyme_score(mentions, bridge, _linker({"enz9": ["catalase"]}))

    assert _populations(knows) == _populations(blank) == (1, 1, 0, 0)
    assert knows.strict.accuracy == 1.0
    assert blank.strict.accuracy == 0.0


def test_a_span_the_authority_named_nothing_for_is_counted_not_scored() -> None:
    """enzymeNER marks spans without naming them, so a name the nomenclature
    does not hold has no gold at all. Dropping it would report the coverage
    over the names the nomenclature knows — a denominator nobody asked about
    — and scoring it as NIL would charge the linker for the resolver's
    misses."""
    report = _enzyme_score(
        [_enzyme(ADH, "1.1.1.1"), _enzyme("Taq polymerase", None, start=40)],
        _ec_bridge({"enz1": "1.1.1.1"}),
        _linker({"enz1": [ADH], "enz2": ["Taq polymerase"]}),
    )

    assert _populations(report) == (2, 1, 1, 0)
    assert report.coverage == pytest.approx(0.5)


def test_a_surface_naming_two_ec_numbers_is_not_judged() -> None:
    """`luciferase` is four reactions sharing a word. The nomenclature says so
    by giving the span more than one identifier, which is the same shape as a
    taxon BRENDA curates twice and is scored the same way — out of the judged
    subset rather than against a coin flip."""
    bridge = _ec_bridge({"enz1": "1.13.12.5", "enz2": "1.13.12.7"})
    report = _enzyme_score(
        [
            _enzyme("luciferase", "1.13.12.5"),
            _enzyme("luciferase", "1.13.12.7"),
        ],
        bridge,
        _linker({"enz1": ["luciferase"]}),
    )

    assert _populations(report) == (1, 0, 0, 1)
    assert bridge.sole_entity("1.13.12.5") == "enz1"
