"""The relation decision rule, and fitting its thresholds on held-out data.

Candidate pairs are mostly `none`, so the head's argmax favours it and rows
whose best typed label it still ranks highly come back null. A threshold per
typed label decides those rows instead; it has to be chosen by the same
scoring `evaluate_model` reports, or the threshold optimises a different
number from the one it is judged on.
"""

import numpy as np
import pytest
import torch
from d3text.models.ete import (
    ETEBrendaModel,
    decide_relations,
    fit_relation_thresholds,
    match_gold_rows,
    use_relation_thresholds,
)
from d3text.schema import EntityType, RelationType, Schema

HAS_ENZYME, HAS_SPECIES, NONE = 0, 1, 2
RELATIONS = ["HasEnzyme", "HasSpecies", "none"]

SCHEMA = Schema(
    entity_types=(
        EntityType(name="bacteria", prefix="bac"),
        EntityType(name="strains", prefix="str"),
        EntityType(name="enzymes", prefix="enz"),
    ),
    relation_types=(
        RelationType(
            name="HasEnzyme", subject_types=("bacteria",), object_type="enzymes"
        ),
        RelationType(
            name="HasSpecies",
            subject_types=("strains",),
            object_type="bacteria",
        ),
        RelationType(name="none", is_none=True),
    ),
)


def _probs(*rows: tuple[float, float, float]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.float32)


def test_without_thresholds_a_row_takes_its_argmax():
    """An uncalibrated checkpoint must decide exactly as before thresholds
    existed, or every old evaluation changes under it."""
    probs = _probs((0.3, 0.1, 0.6), (0.7, 0.2, 0.1))

    assert decide_relations(probs, NONE) == [NONE, HAS_ENZYME]


def test_a_typed_label_past_its_threshold_beats_a_more_probable_none():
    probs = _probs((0.3, 0.1, 0.6), (0.2, 0.1, 0.7))

    assert decide_relations(probs, NONE, [0.25, 0.9, 0.0]) == [
        HAS_ENZYME,
        NONE,
    ]


def test_each_typed_label_is_held_to_its_own_threshold():
    """The best typed label is the candidate; only its threshold applies, so
    a low threshold on another label cannot rescue it."""
    probs = _probs((0.4, 0.35, 0.25))

    assert decide_relations(probs, NONE, [0.5, 0.1, 0.0]) == [NONE]


def test_a_gold_relation_is_scored_against_a_covering_row_agreeing_with_it():
    """Many-to-many: one row disagreeing with the gold must not count as the
    miss when another covering row got it right."""
    gold = [(HAS_ENZYME, [0, 1]), (HAS_SPECIES, [2])]
    row_pred = {0: NONE, 1: HAS_ENZYME, 2: HAS_ENZYME}

    assert match_gold_rows(gold, row_pred) == (
        [HAS_ENZYME, HAS_SPECIES],
        [HAS_ENZYME, HAS_ENZYME],
    )


def test_fitting_lowers_a_threshold_the_null_class_was_winning():
    """Two gold HasEnzyme rows the argmax calls null at p=0.3, and one null
    row at p=0.1: a HasEnzyme threshold between them recovers both without
    admitting the null row, which the argmax cannot."""
    probs = np.array(
        [(0.3, 0.0, 0.7), (0.3, 0.0, 0.7), (0.1, 0.0, 0.9)], dtype=np.float32
    )
    gold = [(HAS_ENZYME, [0]), (HAS_ENZYME, [1])]

    calibration = fit_relation_thresholds(
        probs,
        gold,
        uncovered=[2],
        missed=[],
        relations=RELATIONS,
        none_index=NONE,
    )

    assert calibration.f1_argmax == 0.0
    assert calibration.f1_thresholds == 1.0
    assert calibration.thresholds is not None
    assert 0.1 < calibration.thresholds["HasEnzyme"] <= 0.3
    assert set(calibration.thresholds) == {"HasEnzyme", "HasSpecies"}


def test_missed_gold_counts_against_every_threshold_alike():
    """Gold no row covers is a miss whatever the thresholds; leaving it out
    would report an F1 the evaluation never reproduces."""
    probs = np.array([(0.9, 0.0, 0.1)], dtype=np.float32)

    calibration = fit_relation_thresholds(
        probs,
        [(HAS_ENZYME, [0])],
        uncovered=[],
        missed=[HAS_ENZYME],
        relations=RELATIONS,
        none_index=NONE,
    )

    assert calibration.f1_argmax == pytest.approx(2 / 3)
    assert calibration.f1_thresholds == pytest.approx(2 / 3)


def test_no_thresholds_are_kept_when_none_beats_the_argmax():
    """Calibration must never make the selection metric worse than argmax on
    the split it was fitted to."""
    probs = np.array([(0.9, 0.0, 0.1), (0.1, 0.0, 0.9)], dtype=np.float32)

    calibration = fit_relation_thresholds(
        probs,
        [(HAS_ENZYME, [0])],
        uncovered=[1],
        missed=[],
        relations=RELATIONS,
        none_index=NONE,
    )

    assert calibration.f1_argmax == 1.0
    assert calibration.thresholds is None


def test_calibration_metrics_carry_both_scores_and_each_kept_threshold():
    calibration = fit_relation_thresholds(
        np.array([(0.3, 0.0, 0.7)], dtype=np.float32),
        [(HAS_ENZYME, [0])],
        uncovered=[],
        missed=[],
        relations=RELATIONS,
        none_index=NONE,
    )

    metrics = calibration.metrics("validation")

    assert metrics["validation/relation_micro_f1_typed_argmax"] == 0.0
    assert metrics["validation/relation_micro_f1_typed_calibrated"] == 1.0
    assert "validation/relation_threshold/HasEnzyme" in metrics


def _ete(stub):
    return stub(
        ETEBrendaModel,
        schema=SCHEMA,
        relations=SCHEMA.relation_names,
        relations_none_index=SCHEMA.none_relation_index,
    )


def test_restored_thresholds_decide_the_models_relations(stub):
    model = _ete(stub)

    rule = use_relation_thresholds(model, {"HasEnzyme": 0.2, "HasSpecies": 0.5})

    assert rule == "calibrated"
    logits = torch.log(_probs((0.3, 0.1, 0.6)))
    assert model._relation_labels(logits) == [HAS_ENZYME]


def test_a_checkpoint_without_thresholds_leaves_the_argmax(stub):
    model = _ete(stub)

    assert use_relation_thresholds(model, None) == "argmax"
    assert model._relation_labels(torch.log(_probs((0.3, 0.1, 0.6)))) == [NONE]


def test_thresholds_naming_other_relations_are_refused(stub):
    """A partial or foreign mapping would leave a typed label with no
    threshold, which `_relation_labels` cannot decide."""
    with pytest.raises(ValueError, match="typed relations"):
        use_relation_thresholds(_ete(stub), {"HasEnzyme": 0.2})


def test_thresholds_on_a_model_without_a_relation_head_are_refused():
    with pytest.raises(ValueError, match="no relation head"):
        use_relation_thresholds(object(), {"HasEnzyme": 0.2})
