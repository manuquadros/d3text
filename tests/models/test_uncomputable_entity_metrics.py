"""What an evaluation logs when sklearn cannot score the entity head.

Both entity-scoring models catch the `ValueError` so a degenerate split does
not end the pass. These pin that the key is logged anyway: MLflow cannot tell
an absent key from a run that never emitted it. A document with no gold
entity is the case sklearn does score, as a perfect 1.0 LRAP, and so must be
kept from it. The models are stubbed down to their entity logits and targets.
"""

import math
from typing import Any

import pytest
import torch
from d3text import tracking
from d3text.models.entity_linking import BrendaClassificationModel
from d3text.models.ete import ETEBrendaModel
from d3text.models.model_types import BatchLogits, GroundTruth
from torch.utils.data import DataLoader

DOCUMENTS = 2

# entity logits, the columns `drop_unk` keeps, entity targets
SCENARIOS = {
    # The head holds only `UNK`, so no column is left to score.
    "no-entity-columns": (
        torch.zeros(DOCUMENTS, 1),
        torch.empty(0, dtype=torch.long),
        torch.zeros(DOCUMENTS, 0),
    ),
    # A diverged head: thresholding still works, ranking does not.
    "non-finite-scores": (
        torch.full((DOCUMENTS, 3), float("nan")),
        torch.tensor([0, 1]),
        torch.eye(DOCUMENTS),
    ),
}


class Brenda(BrendaClassificationModel):
    def get_batch_logits(
        self, batch: Any, gold_relations: Any = None
    ) -> BatchLogits:
        return BatchLogits(self.stub_entity_logits, torch.zeros(len(batch), 3))

    def ground_truth(self, batch: Any) -> GroundTruth:
        return GroundTruth(self.stub_entity_targets, torch.ones(len(batch), 2))


class ETE(ETEBrendaModel):
    # `evaluate_model` folds unscored gold in as `none` predictions, which
    # needs the column the bypassed `__init__` would have set.
    relations_none_index = 2

    def get_batch_logits(
        self, batch: Any, gold_relations: Any = None
    ) -> BatchLogits:
        return BatchLogits(
            self.stub_entity_logits, torch.zeros(len(batch), 3), None
        )

    def ground_truth(self, batch: Any) -> GroundTruth:
        return GroundTruth(
            self.stub_entity_targets, torch.ones(len(batch), 2), []
        )


def evaluate(stub, monkeypatch, model_class, logits, columns, targets):
    """Run `evaluate_model` on the stub; return what it returned and logged."""
    logged: list[dict[str, float]] = []
    monkeypatch.delenv(tracking.TRACKING_URI_VAR, raising=False)
    monkeypatch.setattr(
        tracking,
        "log_metrics",
        lambda metrics, step=None: logged.append(dict(metrics)),
    )
    model = stub(
        model_class,
        _modules={},
        _parameters={},
        _buffers={},
        training=False,
        _detection_accumulator=lambda: None,
        classes=["a", "b", "OOS"],
        class_columns=torch.tensor([0, 1]),
        entity_columns=columns,
        stub_entity_logits=logits,
        stub_entity_targets=targets,
    )
    documents = len(targets)

    returned = model.evaluate_model(
        DataLoader([{}] * documents, batch_size=documents, collate_fn=list)
    )

    assert len(logged) == 1
    return returned, logged[0]


@pytest.mark.parametrize("model_class", [Brenda, ETE], ids=["brenda", "ete"])
def test_entity_lrap_is_averaged_over_documents_with_a_gold_entity(
    stub, monkeypatch, model_class
) -> None:
    """One document ranks its gold entity last of three; the three beside it
    have none. Averaged over all four, sklearn's 1.0 for each of those lifts
    1/3 to 0.833 — an amount set by the vocabulary, not by the head."""
    targets = torch.zeros(4, 3)
    targets[0, 2] = 1.0
    logits = torch.tensor([[2.0, 1.0, -1.0, 0.0]]).repeat(4, 1)

    returned, logged = evaluate(
        stub, monkeypatch, model_class, logits, torch.tensor([0, 1, 2]), targets
    )

    for metrics in (returned, logged):
        assert metrics["test/entity_lrap"] == pytest.approx(1 / 3)
        assert metrics["test/entity_lrap_documents"] == 1.0


@pytest.mark.parametrize("scenario", SCENARIOS.values(), ids=SCENARIOS)
@pytest.mark.parametrize(
    ("model_class", "ranking_keys"),
    [
        (Brenda, ("test/entity_lrap", "test/entity_micro_ap")),
        (ETE, ("test/entity_lrap",)),
    ],
    ids=["brenda", "ete"],
)
def test_an_uncomputable_entity_metric_is_still_logged(
    stub, monkeypatch, model_class, ranking_keys, scenario
) -> None:
    """Micro-F1 logs 0.0, the value `zero_division=0` gives no positives and
    no predictions; the ranking metrics log NaN, since they measured nothing
    and either end of their scale would claim a result. Checked on the dict
    returned and on the one handed to tracking, which must be the same."""
    returned, logged = evaluate(stub, monkeypatch, model_class, *scenario)

    for metrics in (returned, logged):
        assert metrics.get("test/entity_micro_f1") == 0.0
        for key in ranking_keys:
            assert key in metrics
            assert math.isnan(metrics[key])
