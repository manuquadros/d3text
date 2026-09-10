"""The three mention scores, pinned on fixtures with known answers.

Detection and linking are the deliverable of the span pipeline, so their
arithmetic is asserted here value by value — a fixture with known spans, known
gold IDs and a known ignore set — independent of any model or checkpoint.
"""

import numpy
import pytest
from d3text import mention_metrics, token_labels
from d3text.mention_metrics import (
    DetectionAccumulator,
    DetectionScores,
    GoldMention,
    Novelty,
    NoveltyScores,
    PredictedMention,
)
from d3text.token_labels import BRENDA_LABELS, IGNORE_INDEX, Mention

# Codes under BRENDA_LABELS' declaration order.
STRAINS, BACTERIA, OTHER, ENZYMES = BRENDA_LABELS.codes


# --------------------------------------------------------------------------- #
# Detection                                                                    #
# --------------------------------------------------------------------------- #
GOLD = [
    GoldMention(start=0, end=10, type_code=ENZYMES),
    GoldMention(start=20, end=30, type_code=BACTERIA),
    GoldMention(start=40, end=50, type_code=STRAINS),
    # The ignore set: a mention of an entity BRENDA did not link here.
    GoldMention(start=60, end=70, type_code=ENZYMES, assertable=False),
]

PREDICTED = [
    PredictedMention(start=0, end=10, type_code=ENZYMES),  # TP
    PredictedMention(start=20, end=30, type_code=ENZYMES),  # wrong type: FP
    PredictedMention(start=62, end=68, type_code=BACTERIA),  # ignored
    PredictedMention(start=80, end=90, type_code=ENZYMES),  # FP
]


def test_detection_counts_on_the_fixture() -> None:
    scores = mention_metrics.detection_scores(PREDICTED, GOLD)

    assert scores.true_positives == 1
    assert scores.false_positives == 2
    assert scores.false_negatives == 2  # bacteria and strain mentions missed
    assert scores.ignored == 1


def test_detection_rates_on_the_fixture() -> None:
    scores = mention_metrics.detection_scores(PREDICTED, GOLD)

    assert scores.precision == pytest.approx(1 / 3)
    assert scores.recall == pytest.approx(1 / 3)
    assert scores.f1 == pytest.approx(1 / 3)


def test_a_boundary_miss_is_not_a_detection() -> None:
    """Span AND type must match; off by one character is a miss."""
    scores = mention_metrics.detection_scores(
        [PredictedMention(start=0, end=11, type_code=ENZYMES)],
        [GoldMention(start=0, end=10, type_code=ENZYMES)],
    )

    assert scores.true_positives == 0
    assert scores.false_positives == 1
    assert scores.false_negatives == 1


def test_a_hit_on_the_ignore_set_is_masked_not_judged() -> None:
    """The FEAT-05 mask applied at evaluation: a prediction overlapping an
    unassertable mention is neither TP nor FP, and the mention itself is
    never an FN — nothing asserts it."""
    scores = mention_metrics.detection_scores(
        [PredictedMention(start=60, end=70, type_code=ENZYMES)],
        [GoldMention(start=60, end=70, type_code=ENZYMES, assertable=False)],
    )

    assert scores == DetectionScores(
        true_positives=0, false_positives=0, false_negatives=0, ignored=1
    )


def test_empty_denominators_score_zero_not_nan() -> None:
    empty = mention_metrics.detection_scores([], [])

    assert empty.precision == 0.0
    assert empty.recall == 0.0
    assert empty.f1 == 0.0


def test_ignore_firing_rate_counts_regions_fired_on() -> None:
    regions, fired = mention_metrics.ignore_firing(PREDICTED, GOLD)

    assert (regions, fired) == (1, 1)

    regions, fired = mention_metrics.ignore_firing(
        [PredictedMention(start=0, end=10, type_code=ENZYMES)], GOLD
    )

    assert (regions, fired) == (1, 0)


# --------------------------------------------------------------------------- #
# Detection split by whether the training split named the entity               #
# --------------------------------------------------------------------------- #
NOVELTY_GOLD = [
    # seen: enz1 is in the training vocabulary below
    GoldMention(0, 10, ENZYMES, entity_ids=frozenset({"enz1"})),
    # unseen: named by no training document
    GoldMention(20, 30, BACTERIA, entity_ids=frozenset({"bac9"})),
    # unseen, and missed by the tagger
    GoldMention(40, 50, STRAINS, entity_ids=frozenset({"str9"})),
    # no BRENDA entity at all: neither seen nor unseen
    GoldMention(60, 70, ENZYMES, entity_ids=frozenset()),
    # the ignore set, of an entity the training split never named
    GoldMention(
        80,
        90,
        ENZYMES,
        entity_ids=frozenset({"enz9"}),
        assertable=False,
    ),
]

NOVELTY_PREDICTED = [
    PredictedMention(0, 10, ENZYMES),
    PredictedMention(20, 30, BACTERIA),
    # the strain mention's span, typed as bacteria: an FP, and no detection
    PredictedMention(40, 50, BACTERIA),
    PredictedMention(60, 70, ENZYMES),
    PredictedMention(80, 90, ENZYMES),
]

TRAINING_ENTITIES = frozenset({"enz1", "bac1", "str1"})


def test_a_seen_entity_scores_only_in_the_seen_bucket() -> None:
    split = mention_metrics.detection_by_novelty(
        [PredictedMention(0, 10, ENZYMES)],
        [GoldMention(0, 10, ENZYMES, entity_ids=frozenset({"enz1"}))],
        TRAINING_ENTITIES,
    )

    assert split[Novelty.SEEN] == NoveltyScores(detected=1, missed=0)
    assert split[Novelty.UNSEEN] == NoveltyScores()
    assert split[Novelty.UNLINKED] == NoveltyScores()


def test_an_unseen_entity_scores_only_in_the_unseen_bucket() -> None:
    """The whole point of the split: a mention the training split could not
    have taught is counted apart from one it could, whether it was found or
    missed."""
    split = mention_metrics.detection_by_novelty(
        [PredictedMention(0, 10, ENZYMES)],
        [
            GoldMention(0, 10, ENZYMES, entity_ids=frozenset({"enz9"})),
            GoldMention(40, 50, STRAINS, entity_ids=frozenset({"str9"})),
        ],
        TRAINING_ENTITIES,
    )

    assert split[Novelty.UNSEEN] == NoveltyScores(detected=1, missed=1)
    assert split[Novelty.UNSEEN].recall == pytest.approx(0.5)
    assert split[Novelty.SEEN] == NoveltyScores()
    assert split[Novelty.UNLINKED] == NoveltyScores()


def test_one_training_entity_of_several_makes_a_mention_seen() -> None:
    """An ambiguous form carries every entity it names, so the model had the
    string in training if *any* of them was — the same `any` rule the linking
    score's intersection uses."""
    split = mention_metrics.detection_by_novelty(
        [],
        [GoldMention(0, 10, ENZYMES, entity_ids=frozenset({"enz1", "enz9"}))],
        TRAINING_ENTITIES,
    )

    assert split[Novelty.SEEN] == NoveltyScores(detected=0, missed=1)
    assert split[Novelty.UNSEEN] == NoveltyScores()


def test_a_mention_with_no_entity_is_neither_seen_nor_unseen() -> None:
    """Novelty is a property of an entity, and an assertable mention with no
    BRENDA entity has none: `seen` would credit memorisation that did not
    happen and `unseen` would charge a bucket no vocabulary could ever fill."""
    split = mention_metrics.detection_by_novelty(
        [PredictedMention(60, 70, ENZYMES)],
        [GoldMention(60, 70, ENZYMES, entity_ids=frozenset())],
        TRAINING_ENTITIES,
    )

    assert split[Novelty.UNLINKED] == NoveltyScores(detected=1, missed=0)
    assert split[Novelty.SEEN] == NoveltyScores()
    assert split[Novelty.UNSEEN] == NoveltyScores()


def test_the_novelty_split_never_reads_the_ignore_set() -> None:
    """A hit on an ignored mention is masked for detection, so it must not
    reappear as a detection in a novelty bucket — that would score the
    population the abstain target exists to leave unscored."""
    split = mention_metrics.detection_by_novelty(
        NOVELTY_PREDICTED, NOVELTY_GOLD, TRAINING_ENTITIES
    )

    assert sum(scores.annotated for scores in split.values()) == 4
    assert split[Novelty.SEEN] == NoveltyScores(detected=1, missed=0)
    assert split[Novelty.UNSEEN] == NoveltyScores(detected=1, missed=1)
    assert split[Novelty.UNLINKED] == NoveltyScores(detected=1, missed=0)


def test_a_right_span_with_the_wrong_type_reaches_no_bucket() -> None:
    """The buckets key gold on `(start, end, type_code)` — the tuple
    `detection_scores` matches predictions against — so the fixture's strain
    mention, predicted as bacteria, stays missed and its span is credited to
    nobody. Correcting only the type is what moves it, which a split blind to
    type would score as found either way."""
    split = mention_metrics.detection_by_novelty(
        NOVELTY_PREDICTED, NOVELTY_GOLD, TRAINING_ENTITIES
    )
    retyped = mention_metrics.detection_by_novelty(
        [
            PredictedMention(40, 50, STRAINS)
            if (span.start, span.end) == (40, 50)
            else span
            for span in NOVELTY_PREDICTED
        ],
        NOVELTY_GOLD,
        TRAINING_ENTITIES,
    )

    assert split[Novelty.UNSEEN] == NoveltyScores(detected=1, missed=1)
    assert sum(bucket.detected for bucket in split.values()) == 3
    assert retyped[Novelty.UNSEEN] == NoveltyScores(detected=2, missed=0)


def test_the_buckets_partition_the_mentions_detection_scores_judges() -> None:
    """Same population, split rather than filtered: anything else makes the
    two recalls disagree with the aggregate they are meant to explain."""
    scores = mention_metrics.detection_scores(NOVELTY_PREDICTED, NOVELTY_GOLD)
    split = mention_metrics.detection_by_novelty(
        NOVELTY_PREDICTED, NOVELTY_GOLD, TRAINING_ENTITIES
    )

    assert sum(bucket.annotated for bucket in split.values()) == (
        scores.true_positives + scores.false_negatives
    )
    assert sum(bucket.detected for bucket in split.values()) == (
        scores.true_positives
    )


def test_novelty_scores_add() -> None:
    total = NoveltyScores(1, 2) + NoveltyScores(10, 20)

    assert total == NoveltyScores(11, 22)
    assert total.annotated == 33
    assert total.recall == pytest.approx(11 / 33)


def test_an_empty_bucket_scores_zero_not_nan() -> None:
    assert NoveltyScores().recall == 0.0


# --------------------------------------------------------------------------- #
# Linking, conditional on detection                                            #
# --------------------------------------------------------------------------- #
LINK_GOLD = [
    GoldMention(0, 10, ENZYMES, entity_ids=frozenset({"enz1", "enz2"})),
    GoldMention(20, 30, BACTERIA, entity_ids=frozenset({"bac1"})),
    GoldMention(40, 50, STRAINS, entity_ids=frozenset({"str1"})),
    # A mention with no BRENDA entity: NIL is the right answer for it.
    GoldMention(60, 70, ENZYMES, entity_ids=frozenset()),
]

LINK_PREDICTED = [
    # correct: intersects the gold IDs
    PredictedMention(0, 10, ENZYMES, entity_ids=frozenset({"enz2", "enz9"})),
    # wrong ID
    PredictedMention(20, 30, BACTERIA, entity_ids=frozenset({"bac7"})),
    # NIL where an ID existed
    PredictedMention(40, 50, STRAINS, entity_ids=frozenset()),
    # NIL where none exists: correct
    PredictedMention(60, 70, ENZYMES, entity_ids=frozenset()),
    # detection miss: never reaches the linker's ledger
    PredictedMention(80, 90, ENZYMES, entity_ids=frozenset({"enz1"})),
]


def test_linking_counts_on_the_fixture() -> None:
    scores = mention_metrics.linking_scores(LINK_PREDICTED, LINK_GOLD)

    assert scores.correct == 1
    assert scores.wrong == 1
    assert scores.nil_missed == 1
    assert scores.nil_correct == 1
    assert scores.total == 4
    assert scores.accuracy == pytest.approx(0.5)


def test_linking_ignores_spans_the_detector_missed() -> None:
    """A wrong ID on a span that matched no mention is a detection FP, not a
    linking error — scoring it twice would double-charge stage 1's mistake."""
    scores = mention_metrics.linking_scores(
        [PredictedMention(80, 90, ENZYMES, entity_ids=frozenset({"enz1"}))],
        LINK_GOLD,
    )

    assert scores.total == 0


def test_an_id_for_a_mention_without_one_is_wrong() -> None:
    scores = mention_metrics.linking_scores(
        [PredictedMention(60, 70, ENZYMES, entity_ids=frozenset({"enz1"}))],
        LINK_GOLD,
    )

    assert scores.wrong == 1 and scores.total == 1


def test_linking_never_reads_the_ignore_set() -> None:
    scores = mention_metrics.linking_scores(
        [PredictedMention(0, 10, ENZYMES, entity_ids=frozenset({"enz1"}))],
        [
            GoldMention(
                0,
                10,
                ENZYMES,
                entity_ids=frozenset({"enz1"}),
                assertable=False,
            )
        ],
    )

    assert scores.total == 0


def test_the_strict_rule_refuses_an_answer_that_only_intersects() -> None:
    """What a gold-side-unambiguous subset buys. Under the default rule a
    linker asserting every candidate it found scores as one that picked the
    curated entity out of them; under `STRICT` it does not, which is the
    difference between resolving and disambiguating."""
    predicted = [
        PredictedMention(
            20, 30, BACTERIA, entity_ids=frozenset({"bac1", "bac7"})
        )
    ]

    lenient = mention_metrics.linking_scores(predicted, LINK_GOLD)
    strict = mention_metrics.linking_scores(
        predicted, LINK_GOLD, mention_metrics.LinkingRule.STRICT
    )

    assert (lenient.correct, lenient.wrong) == (1, 0)
    assert (strict.correct, strict.wrong) == (0, 1)


def test_the_strict_rule_keeps_the_nil_bookkeeping() -> None:
    """The two rules differ in one predicate and share the NIL columns, which
    is why the rule is a parameter: a second function could drift on the half
    that is subtle."""
    scores = mention_metrics.linking_scores(
        LINK_PREDICTED, LINK_GOLD, mention_metrics.LinkingRule.STRICT
    )

    assert (scores.nil_correct, scores.nil_missed) == (1, 1)
    assert (scores.correct, scores.wrong) == (0, 2)


# --------------------------------------------------------------------------- #
# Gold mentions come from the training-label machinery                         #
# --------------------------------------------------------------------------- #
def test_gold_mentions_follow_the_label_rules() -> None:
    """Type, assertability and the scoreable IDs all read off the same
    computation the token targets use, so evaluation and training cannot
    disagree about which mentions are gold."""
    mentions = [
        # gold-linked enzyme, ambiguously shared with a non-gold one
        Mention(start=0, end=4, entity_ids=frozenset({"enz1", "enz7"})),
        # matches only a non-gold entity: the ignore set
        Mention(start=10, end=14, entity_ids=frozenset({"bac9"})),
        # gold candidates of two types: abstains
        Mention(start=20, end=30, entity_ids=frozenset({"bac1", "str1"})),
    ]

    gold = mention_metrics.gold_mentions(mentions, {"enz1", "bac1", "str1"})

    assert gold[0] == GoldMention(
        0, 4, ENZYMES, entity_ids=frozenset({"enz1"}), assertable=True
    )
    assert not gold[1].assertable
    assert gold[1].entity_ids == frozenset()
    assert not gold[2].assertable


# --------------------------------------------------------------------------- #
# Token-axis decoding                                                          #
# --------------------------------------------------------------------------- #
def test_spans_from_codes_decodes_maximal_runs() -> None:
    codes = numpy.array(
        [0, ENZYMES, ENZYMES, 0, BACTERIA, IGNORE_INDEX, IGNORE_INDEX, 0],
        dtype=numpy.int8,
    )

    assert mention_metrics.spans_from_codes(codes) == [
        (1, 3, ENZYMES),
        (4, 5, BACTERIA),
        (5, 7, IGNORE_INDEX),
    ]


def test_token_gold_mentions_split_mentions_from_the_ignore_set() -> None:
    codes = numpy.array(
        [ENZYMES, ENZYMES, 0, IGNORE_INDEX, 0], dtype=numpy.int8
    )

    gold = mention_metrics.token_gold_mentions(codes)

    assert gold[0] == GoldMention(0, 2, ENZYMES, assertable=True)
    assert gold[1].assertable is False
    assert (gold[1].start, gold[1].end) == (3, 4)


def test_adjacent_runs_of_different_types_stay_separate() -> None:
    codes = numpy.array([ENZYMES, ENZYMES, BACTERIA], dtype=numpy.int8)

    assert mention_metrics.spans_from_codes(codes) == [
        (0, 2, ENZYMES),
        (2, 3, BACTERIA),
    ]


def test_gold_mentions_with_entities_attaches_the_owning_id() -> None:
    codes = numpy.array(
        [ENZYMES, ENZYMES, 0, BACTERIA, IGNORE_INDEX], dtype=numpy.int8
    )

    gold = mention_metrics.token_gold_mentions_with_entities(
        codes, {"enz1": [0, 1], "bac1": [3]}
    )

    assert gold[0] == GoldMention(0, 2, ENZYMES, entity_ids=frozenset({"enz1"}))
    assert gold[1] == GoldMention(
        3, 4, BACTERIA, entity_ids=frozenset({"bac1"})
    )
    # The ignore run gets no type and no entity, same as `token_gold_mentions`.
    assert gold[2].assertable is False
    assert gold[2].entity_ids == frozenset()


def test_gold_mentions_with_entities_unions_an_ambiguous_span() -> None:
    """One span, two entities anchored to it: both must come back, not the
    last one written."""
    codes = numpy.array([ENZYMES, ENZYMES], dtype=numpy.int8)

    gold = mention_metrics.token_gold_mentions_with_entities(
        codes, {"enz1": [0, 1], "enz2": [0, 1]}
    )

    assert gold == [
        GoldMention(0, 2, ENZYMES, entity_ids=frozenset({"enz1", "enz2"}))
    ]


def test_gold_mentions_with_entities_drops_an_out_of_span_position() -> None:
    codes = numpy.array([ENZYMES, ENZYMES, 0], dtype=numpy.int8)

    gold = mention_metrics.token_gold_mentions_with_entities(
        codes, {"enz1": [0], "bac1": [2]}
    )

    assert gold == [GoldMention(0, 2, ENZYMES, entity_ids=frozenset({"enz1"}))]


# --------------------------------------------------------------------------- #
# The accumulator the model's evaluation feeds                                 #
# --------------------------------------------------------------------------- #
def test_accumulator_totals_and_keys_on_a_known_split() -> None:
    accumulator = DetectionAccumulator(BRENDA_LABELS)

    # Document 1: gold enzyme run [1, 3) predicted exactly; an ignore run
    # [5, 7) fired on; a spurious bacteria span.
    gold_1 = numpy.zeros(10, dtype=numpy.int64)
    gold_1[1:3] = ENZYMES
    gold_1[5:7] = IGNORE_INDEX
    predicted_1 = numpy.zeros(10, dtype=numpy.int64)
    predicted_1[1:3] = ENZYMES
    predicted_1[5:6] = BACTERIA
    predicted_1[8:9] = BACTERIA

    # Document 2: a gold strain mention the tagger missed entirely.
    gold_2 = numpy.zeros(6, dtype=numpy.int64)
    gold_2[2:4] = STRAINS
    predicted_2 = numpy.zeros(6, dtype=numpy.int64)

    accumulator.add_document(predicted_1, gold_1)
    accumulator.add_document(predicted_2, gold_2)
    accumulator.missing_documents += 1

    metrics = accumulator.metrics()

    assert metrics["test/detection_true_positives"] == 1.0
    assert metrics["test/detection_false_positives"] == 1.0
    assert metrics["test/detection_false_negatives"] == 1.0
    assert metrics["test/detection_ignored_predictions"] == 1.0
    assert metrics["test/detection_precision"] == pytest.approx(0.5)
    assert metrics["test/detection_recall"] == pytest.approx(0.5)
    assert metrics["test/detection_f1"] == pytest.approx(0.5)
    assert metrics["test/detection_ignore_regions"] == 1.0
    assert metrics["test/detection_ignore_firing_rate"] == pytest.approx(1.0)
    assert metrics["test/detection_documents"] == 2.0
    assert metrics["test/detection_documents_missing_labels"] == 1.0

    # Per-type: the enzyme channel is clean, the strain channel all misses,
    # and the spurious bacteria span still masks against the typeless ignore
    # set (one span on it, one clear of it).
    assert metrics["test/detection_enzymes_recall"] == pytest.approx(1.0)
    assert metrics["test/detection_enzymes_precision"] == pytest.approx(1.0)
    assert metrics["test/detection_strains_recall"] == pytest.approx(0.0)
    assert metrics["test/detection_bacteria_precision"] == pytest.approx(0.0)


def test_accumulator_omits_the_firing_rate_without_ignore_regions() -> None:
    accumulator = DetectionAccumulator(BRENDA_LABELS)
    accumulator.add_document(
        numpy.zeros(4, dtype=numpy.int64), numpy.zeros(4, dtype=numpy.int64)
    )

    assert "test/detection_ignore_firing_rate" not in accumulator.metrics()


# The keys `metrics()` emitted before the novelty split existed. Written out
# rather than derived, so a key appearing or vanishing is a diff here.
LEGACY_METRIC_KEYS = {
    "test/detection_precision",
    "test/detection_recall",
    "test/detection_f1",
    "test/detection_true_positives",
    "test/detection_false_positives",
    "test/detection_false_negatives",
    "test/detection_ignored_predictions",
    "test/detection_ignore_regions",
    "test/detection_ignore_firing_rate",
    "test/detection_documents",
    "test/detection_documents_missing_labels",
    *(
        f"test/detection_{name}_{score}"
        for name in BRENDA_LABELS.types
        for score in ("precision", "recall", "f1")
    ),
}


def scored_document() -> tuple[numpy.ndarray, numpy.ndarray]:
    """One document with a hit, a miss, an ignore region and a spurious span,
    as `(predicted, gold)` code arrays."""
    gold = numpy.zeros(12, dtype=numpy.int64)
    gold[1:3] = ENZYMES
    gold[5:7] = IGNORE_INDEX
    gold[9:11] = STRAINS
    predicted = numpy.zeros(12, dtype=numpy.int64)
    predicted[1:3] = ENZYMES
    predicted[5:6] = BACTERIA
    predicted[8:9] = BACTERIA
    return predicted, gold


def test_metrics_gain_no_keys_without_a_training_vocabulary() -> None:
    """The novelty split is opt-in: an accumulator built the way the model
    builds one must key its metrics exactly as it did before the split
    existed, since those keys are charted across old and new runs alike."""
    accumulator = DetectionAccumulator(BRENDA_LABELS)
    accumulator.add_document(*scored_document())

    assert set(accumulator.metrics()) == LEGACY_METRIC_KEYS


def test_the_accumulator_splits_a_document_by_novelty() -> None:
    accumulator = DetectionAccumulator(
        BRENDA_LABELS, training_entity_ids=TRAINING_ENTITIES
    )
    accumulator.add_mentions(NOVELTY_PREDICTED, NOVELTY_GOLD)
    accumulator.add_mentions(NOVELTY_PREDICTED, NOVELTY_GOLD)

    metrics = accumulator.metrics()

    assert metrics["test/detection_novelty_seen_annotated"] == 2.0
    assert metrics["test/detection_novelty_seen_detected"] == 2.0
    assert metrics["test/detection_novelty_seen_recall"] == pytest.approx(1.0)
    assert metrics["test/detection_novelty_unseen_annotated"] == 4.0
    assert metrics["test/detection_novelty_unseen_recall"] == pytest.approx(0.5)
    assert metrics["test/detection_novelty_unlinked_annotated"] == 2.0
    # The aggregate the split explains is untouched by the split, and the
    # wrong-type span is charged there as a false positive rather than
    # reaching a bucket.
    assert metrics["test/detection_recall"] == pytest.approx(0.75)
    assert metrics["test/detection_precision"] == pytest.approx(0.75)


def test_an_empty_novelty_bucket_reports_its_count_and_no_recall() -> None:
    """0/0 is not a measurement, and an absent recall cannot be mistaken for a
    tagger that found none of the novel entities — but the count still says
    the split held none of them."""
    accumulator = DetectionAccumulator(
        BRENDA_LABELS, training_entity_ids=TRAINING_ENTITIES
    )
    accumulator.add_mentions(
        [PredictedMention(0, 10, ENZYMES)],
        [GoldMention(0, 10, ENZYMES, entity_ids=frozenset({"enz1"}))],
    )

    metrics = accumulator.metrics()

    assert metrics["test/detection_novelty_unseen_annotated"] == 0.0
    assert "test/detection_novelty_unseen_recall" not in metrics
    assert metrics["test/detection_novelty_seen_recall"] == pytest.approx(1.0)


def test_the_token_axis_refuses_a_training_vocabulary() -> None:
    """Codes carry no entity IDs, so a novelty split taken from them would put
    every mention in `unlinked` and read as a measurement of one."""
    accumulator = DetectionAccumulator(
        BRENDA_LABELS, training_entity_ids=TRAINING_ENTITIES
    )

    with pytest.raises(ValueError, match="no entity IDs"):
        accumulator.add_document(*scored_document())


def test_scores_add() -> None:
    total = DetectionScores(1, 2, 3, 4) + DetectionScores(10, 20, 30, 40)

    assert total == DetectionScores(11, 22, 33, 44)


def test_label_space_codes_are_the_declared_four() -> None:
    """The fixture names above index BRENDA's declaration order; if that
    order moves, this file must be re-read, not silently re-passed."""
    assert token_labels.BRENDA_LABELS.types == (
        "strains",
        "bacteria",
        "other_organisms",
        "enzymes",
    )
