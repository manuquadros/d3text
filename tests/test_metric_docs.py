"""Every tracked metric has to say what its y-axis measures.

The glossary is the only record of a metric's unit — MLflow charts a key and
nothing else — so what these tests protect is its agreement with the keys the
code actually emits.
"""

import ast
from pathlib import Path

import d3text.models
import numpy as np
import pytest
from d3text import metric_docs
from d3text.mention_metrics import (
    DetectionAccumulator,
    GoldMention,
    PredictedMention,
)
from d3text.models.base import (
    Step,
    epoch_rate_metrics,
    print_epoch_stats,
    relation_metrics,
    support_metrics,
)
from d3text.token_labels import BRENDA_LABELS


def evaluation_metric_names() -> set[str]:
    """The `test/*` keys an evaluation pass logs, from the helpers that key
    them — rather than a list here that a rename would leave behind."""
    true = np.array([0, 1, 2, 2])
    pred = np.array([0, 2, 2, 1])
    names = set(
        relation_metrics(true, pred, labels=np.array([0, 1, 2]), none_index=2)
    )
    names |= set(
        support_metrics(
            {
                "entity": (np.zeros((2, 3)), np.ones((2, 3))),
                "class": (np.zeros((2, 3)), np.ones((2, 3))),
            }
        )
    )

    return names


@pytest.mark.parametrize("metric", sorted(evaluation_metric_names()))
def test_evaluation_metrics_are_documented(metric: str) -> None:
    assert metric_docs.describe(metric) is not None


def literal_evaluation_metric_names() -> set[str]:
    """Every `test/*` string a model module uses as a dictionary key.

    `evaluate_model` mints keys of its own, beside the ones the helpers above
    return, and nothing drives it here — so these are read from the source.
    Only dictionary keys count: the artifact paths handed to `log_text` share
    the prefix and are not metrics.
    """
    names: set[str] = set()
    for module in Path(d3text.models.__file__).parent.glob("*.py"):
        for node in ast.walk(ast.parse(module.read_text(), str(module))):
            if isinstance(node, ast.Subscript):
                keys = [node.slice]
            elif isinstance(node, ast.Dict):
                keys = list(node.keys)
            else:
                continue
            names |= {
                key.value
                for key in keys
                if isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                and key.value.startswith("test/")
            }

    return names


@pytest.mark.parametrize("metric", sorted(literal_evaluation_metric_names()))
def test_literal_evaluation_metrics_are_documented(metric: str) -> None:
    """The keys `evaluate_model` writes itself go out through
    `tracking.log_metrics` like any other, so an undocumented one reaches
    MLflow silently — `describe` returning `None` is not a logging error."""
    assert metric_docs.describe(metric) is not None


def test_the_literal_keys_were_actually_read() -> None:
    """A collector that finds nothing passes every parametrized case by
    running none of them; a key every model module writes and the one
    `evaluate_model` mints alone are what prove it read the source."""
    names = literal_evaluation_metric_names()

    assert "test/class_micro_f1" in names
    assert "test/entity_macro_f1_support10" in names


def detection_metric_names() -> set[str]:
    """Every key a detection pass logs, from the accumulator that keys them
    rather than a list here that a rename would leave behind.

    The training vocabulary and the non-assertable mention are what reach the
    two conditional families: without them the novelty split and the ignore
    firing rate are omitted, and the drift check would cover less than an
    evaluation run emits.
    """
    code = BRENDA_LABELS.codes[0]
    accumulator = DetectionAccumulator(
        BRENDA_LABELS, training_entity_ids=frozenset({"enz1"})
    )
    accumulator.add_mentions(
        [PredictedMention(0, 4, code), PredictedMention(16, 19, code)],
        [
            GoldMention(0, 4, code, frozenset({"enz1"})),
            GoldMention(6, 9, code, frozenset({"enz9"})),
            GoldMention(11, 14, code, frozenset()),
            GoldMention(16, 19, code, frozenset(), assertable=False),
        ],
    )

    return set(accumulator.metrics())


def per_type_metric_names() -> dict[str, set[str]]:
    """The emitted per-entity-type keys, grouped by the type each names."""
    grouped: dict[str, set[str]] = {name: set() for name in BRENDA_LABELS.types}
    for metric in detection_metric_names():
        named = metric.removeprefix("test/detection_").rpartition("_")[0]
        if named in grouped:
            grouped[named].add(metric)

    return grouped


@pytest.mark.parametrize("metric", sorted(detection_metric_names()))
def test_detection_metrics_are_documented(metric: str) -> None:
    """`evaluate_model` logs whatever `metrics()` returns straight onto the
    run, and nothing else calls it, so this is the only thing standing between
    a new detection key and an MLflow chart with no stated unit."""
    assert metric_docs.describe(metric) is not None


def test_the_per_type_scores_are_documented_as_per_type() -> None:
    """Not merely resolved: several entries match with `\\w+`, so a key
    answered by another family would carry that family's unit, and a score
    the per-type pattern omits falls through to no entry at all."""
    grouped = per_type_metric_names()
    entries = {
        metric_docs.describe(metric)
        for metrics in grouped.values()
        for metric in metrics
    }

    assert all(grouped.values())
    assert len(entries) == 1
    entry = entries.pop()
    assert entry is not None
    assert "<type>" in entry.display


@pytest.mark.parametrize(
    "metric",
    sorted(name for name in detection_metric_names() if "novelty" in name),
)
def test_the_novelty_split_is_documented_as_itself(metric: str) -> None:
    """Not merely resolved: the per-type entry's `\\w+` matches these keys
    too, so an entry ordered after it would document a bucket as an entity
    type — a wrong unit, which is worse than a missing one."""
    entry = metric_docs.describe(metric)

    assert entry is not None
    assert "novelty" in entry.display


@pytest.mark.parametrize("step", [Step.TRAINING, Step.VALIDATION])
def test_epoch_metrics_are_documented(step: Step) -> None:
    """`Step.TESTING` is left out because no pass logs epoch stats under it:
    an evaluation keys its numbers `test/`, not `testing/`."""
    metrics = {
        **print_epoch_stats(
            losses={"entity": 1.0, "class": 1.0, "relation": 1.0, "token": 1.0},
            denominator=1,
            step=step,
        ),
        **epoch_rate_metrics(batches=4, seconds=2.0, step=step),
    }

    assert [
        name for name in metrics if metric_docs.describe(name) is None
    ] == []


def test_an_undocumented_metric_is_reported_as_such() -> None:
    """`describe` returning `None` is the signal the drift tests key on; it
    must not fall back to a nearby entry."""
    assert metric_docs.describe("training/loss") is None
    assert metric_docs.describe("training/mystery_rate") is None


def test_the_glossary_documents_every_metric_of_its_stage() -> None:
    for stage, entries in metric_docs.STAGES.items():
        table = metric_docs.glossary(stage)
        assert table.startswith("### Metrics")
        for entry in entries:
            assert entry.display in table
            assert entry.axis in table


def test_an_unknown_stage_gets_no_glossary() -> None:
    """A stage tag this module has no table for yields nothing, rather than a
    table describing metrics the run never logs."""
    assert metric_docs.glossary("profiling") == ""
