"""What an evaluation logs when sklearn cannot score the class head.

A diverged head scores NaN, which `average_precision_score` refuses outright.
`evaluate_model` hands its dict to tracking in a single call at the end, so a
raise there cost the whole pass — including the entity scores already
measured — rather than one number. These pin that the key arrives as NaN
instead, on the dict returned and on the one logged, which must be the same.
The models are stubbed down to their logits and targets.
"""

import math
from typing import Any

import pytest
import torch
from d3text import tracking
from d3text.models.entity_linking import BrendaClassificationModel
from d3text.models.model_types import BatchLogits, GroundTruth
from d3text.models.ner import NERClassificationModel
from torch import Tensor
from torch.utils.data import DataLoader

DOCUMENTS = 2

# Finite, so the entity head is scored normally and the class head is the only
# thing the pass could have died on. Both trailing columns are the `UNK`/`OOS`
# the two `drop_*` helpers remove.
ENTITY_LOGITS = torch.tensor([[2.0, -1.0, 0.0], [-1.0, 2.0, 0.0]])
CLASS_LOGITS = torch.full((DOCUMENTS, 3), float("nan"))
TARGETS = torch.eye(DOCUMENTS)


class Brenda(BrendaClassificationModel):
    def get_batch_logits(
        self, batch: Any, gold_relations: Any = None
    ) -> BatchLogits:
        return BatchLogits(ENTITY_LOGITS, CLASS_LOGITS)

    def ground_truth(self, batch: Any) -> GroundTruth:
        return GroundTruth(TARGETS, TARGETS)


class NER(NERClassificationModel):
    def get_batch_logits(self, batch: Any) -> Tensor:
        return CLASS_LOGITS

    def ground_truth(self, batch: Any) -> Tensor:
        return TARGETS


def evaluate(stub, monkeypatch, model_class):
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
        entity_columns=torch.tensor([0, 1]),
    )

    returned = model.evaluate_model(
        DataLoader([{}] * DOCUMENTS, batch_size=DOCUMENTS, collate_fn=list)
    )

    assert len(logged) == 1
    return returned, logged[0]


@pytest.mark.parametrize("model_class", [Brenda, NER], ids=["brenda", "ner"])
def test_an_uncomputable_class_ap_is_still_logged(
    stub, monkeypatch, model_class
) -> None:
    """NaN, since nothing was ranked and either end of the scale would claim a
    result. The two metrics beside it are what say the pass ran to the end:
    micro-F1 scores the thresholded predictions, which NaN makes empty rather
    than unshaped, and the coverage count is keyed before any of it."""
    returned, logged = evaluate(stub, monkeypatch, model_class)

    for metrics in (returned, logged):
        assert "test/class_micro_ap" in metrics
        assert math.isnan(metrics["test/class_micro_ap"])
        assert metrics["test/class_micro_f1"] == 0.0
        assert metrics["dataset/test_documents_scored"] == float(DOCUMENTS)


def test_the_entity_scores_survive_a_nan_class_head(stub, monkeypatch) -> None:
    """The entity block runs first, so its numbers were the ones a raise in
    the class block threw away. They are real values here, not the NaN the
    entity head logs when its own scores cannot be ranked."""
    returned, logged = evaluate(stub, monkeypatch, Brenda)

    for metrics in (returned, logged):
        assert metrics["test/entity_micro_ap"] == pytest.approx(1.0)
        assert metrics["test/entity_lrap"] == pytest.approx(1.0)
