"""What an evaluation logs when sklearn cannot score the entity head.

Both entity-scoring models catch the `ValueError` so a degenerate split does
not end the pass. These pin that the key is logged anyway: MLflow cannot tell
an absent key from a run that never emitted it. The models are stubbed down
to their entity logits and targets.
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
    logits, columns, targets = scenario
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

    returned = model.evaluate_model(
        DataLoader([{}] * DOCUMENTS, batch_size=DOCUMENTS, collate_fn=list)
    )

    assert len(logged) == 1
    for metrics in (returned, logged[0]):
        assert metrics.get("test/entity_micro_f1") == 0.0
        for key in ranking_keys:
            assert key in metrics
            assert math.isnan(metrics[key])
