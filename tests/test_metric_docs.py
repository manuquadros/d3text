"""Every tracked metric has to say what its y-axis measures.

The glossary is the only record of a metric's unit — MLflow charts a key and
nothing else — so what these tests protect is its agreement with the keys the
code actually emits.
"""

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


def novelty_metric_names() -> set[str]:
    """The novelty keys a detection pass logs, from the accumulator that keys
    them rather than a list here that a rename would leave behind."""
    accumulator = DetectionAccumulator(
        BRENDA_LABELS, training_entity_ids=frozenset({"enz1"})
    )
    accumulator.add_mentions(
        [PredictedMention(0, 4, BRENDA_LABELS.codes[0])],
        [
            GoldMention(0, 4, BRENDA_LABELS.codes[0], frozenset({"enz1"})),
            GoldMention(6, 9, BRENDA_LABELS.codes[0], frozenset({"enz9"})),
            GoldMention(11, 14, BRENDA_LABELS.codes[0], frozenset()),
        ],
    )

    return {name for name in accumulator.metrics() if "novelty" in name}


@pytest.mark.parametrize("metric", sorted(novelty_metric_names()))
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
