import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from d3text import kingdoms
from d3text.datasets.brenda import entity_ids_by_class, filter_relations
from d3text.models.base import shared_class_columns, shared_class_metrics
from d3text.models.config import ModelConfig
from d3text.schema import BRENDA_SCHEMA, KINGDOM_SCHEMA


@pytest.fixture
def placed(monkeypatch):
    table = {1: ("animalia", "ani"), 2: ("archaea", "arc")}
    monkeypatch.setattr(kingdoms, "_kingdom_by_id", lambda: table)
    return table


def _frame():
    return pd.DataFrame(
        {
            "other_organisms": [[1, 3], [2]],
            "relations": [
                [{("enz9", "oth1"): "a", ("enz9", "oth3"): "b"}],
                [{("enz9", "oth2"): "c"}],
            ],
        }
    )


def test_split_frame_moves_each_organism_and_its_relations(placed):
    """An organism and every relation argument naming it must move together,
    or `filter_relations` drops the relation as untyped."""
    frame = _frame()
    out = kingdoms.split_frame(frame)

    assert out["animalia"].tolist() == [[1], []]
    assert out["archaea"].tolist() == [[], [2]]
    assert out["other_organisms"].tolist() == [[3], []]
    assert out["relations"].tolist() == [
        [{("enz9", "ani1"): "a", ("enz9", "oth3"): "b"}],
        [{("enz9", "arc2"): "c"}],
    ]
    assert frame["other_organisms"].tolist() == [[1, 3], [2]]


def test_split_ids_are_typed_and_admitted_under_kingdom_schema(placed):
    frame = _frame().assign(
        strains=[[], []], bacteria=[[], []], enzymes=[[9], [9]]
    )
    out = kingdoms.split_frame(frame)

    ids = entity_ids_by_class(KINGDOM_SCHEMA, out)
    assert ids["animalia"] == {"ani1"} and ids["archaea"] == {"arc2"}
    label = np.array([1.0, 0.0, 0.0])
    kept = filter_relations([{("enz9", "ani1"): label}], KINGDOM_SCHEMA)
    assert kept == [{("enz9", "ani1"): label}]


def test_folding_ors_kingdoms_into_other_organisms():
    """A document whose only organism is in a kingdom column must count as an
    other_organisms positive, or the two arms are scored on different golds."""
    names = KINGDOM_SCHEMA.class_names
    row = np.zeros((1, len(names)), dtype=int)
    row[0, names.index("fungi")] = 1
    row[0, names.index("enzymes")] = 1

    folded = shared_class_columns(row, names)

    assert folded is not None
    assert dict(zip(BRENDA_SCHEMA.class_names, folded[0], strict=True)) == {
        "strains": 0,
        "bacteria": 0,
        "other_organisms": 1,
        "enzymes": 1,
    }


def test_shared_metrics_equal_plain_scores_for_a_plain_head():
    true = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
    pred = np.array([[1, 0, 0, 0], [0, 1, 0, 1]])

    metrics = shared_class_metrics(
        true, pred, "test", BRENDA_SCHEMA.class_names
    )

    assert metrics["test/class_shared_micro_f1"] == pytest.approx(6 / 7)
    assert metrics["test/class_f1/other_organisms"] == 0.0
    assert metrics["test/class_f1/enzymes"] == 1.0


def test_shared_metrics_skip_a_schema_without_the_four_classes():
    assert (
        shared_class_metrics(np.ones((1, 1)), np.ones((1, 1)), "t", ["x"]) == {}
    )


def test_kingdom_split_refuses_a_model_reading_the_label_store():
    with pytest.raises(ValidationError, match="kingdom_split"):
        ModelConfig(model_class="BrendaClassificationModel", kingdom_split=True)
    ModelConfig(model_class="NERClassificationModel", kingdom_split=True)
