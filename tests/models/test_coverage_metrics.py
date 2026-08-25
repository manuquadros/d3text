"""What an evaluation reports about the documents it actually scored.

`dataset/test_documents` is logged at run setup from the split frame; the
scores are computed over whatever the encodings file backs. These pin the
counts that state the difference, over a split holding one pmid the HDF5 does
not, drawn the way `evaluate` draws it (`batch_size=1`, so the missing row is
an empty batch).

The models are stubbed rather than built: `evaluate_model`'s own arithmetic is
what is under test, so `get_batch_logits` and `ground_truth` return fixed
tensors and no base model is constructed.
"""

from typing import Any

import pytest
import torch
from d3text.models.models import (
    BrendaClassificationModel,
    ETEBrendaModel,
    NERClassificationModel,
)
from d3text.data.data import get_batch_loader


class StubbedBrenda(BrendaClassificationModel):
    def get_batch_logits(self, batch: Any, gold_relations: Any = None) -> Any:
        return torch.zeros(len(batch), 3), torch.zeros(len(batch), 3)

    def ground_truth(self, batch: Any) -> Any:
        return torch.ones(len(batch), 2), torch.ones(len(batch), 2)


class StubbedNER(NERClassificationModel):
    def get_batch_logits(self, batch: Any) -> Any:
        return torch.zeros(len(batch), 2)

    def ground_truth(self, batch: Any) -> Any:
        return torch.ones(len(batch), 2)


class StubbedETE(ETEBrendaModel):
    def get_batch_logits(self, batch: Any, gold_relations: Any = None) -> Any:
        return torch.zeros(len(batch), 3), torch.zeros(len(batch), 3), None

    def ground_truth(self, batch: Any) -> Any:
        return torch.ones(len(batch), 2), torch.ones(len(batch), 2), []


@pytest.fixture(params=[StubbedBrenda, StubbedNER, StubbedETE])
def evaluator(request, stub):
    """A model class whose only live method is `evaluate_model`."""
    return stub(
        request.param,
        _modules={},
        _parameters={},
        _buffers={},
        training=False,
        classes=["a", "b", "OOS"],
        class_columns=torch.tensor([0, 1]),
        entity_columns=torch.tensor([0, 1]),
    )


def loader_over(dataset: Any) -> Any:
    """`evaluate`'s loader: one document per batch, in frame order."""
    return get_batch_loader(
        dataset,
        batch_size=1,
        sampler=torch.utils.data.SequentialSampler(range(len(dataset))),
    )


def test_scores_fewer_documents_than_the_split_planned(
    evaluator, tiny_brenda
) -> None:
    """The reported case: four rows in the frame, three in the HDF5."""
    metrics = evaluator.evaluate_model(loader_over(tiny_brenda.full))

    assert metrics["dataset/test_documents_scored"] == 3.0
    assert metrics["dataset/test_documents_missing"] == 1.0


def test_reports_no_shortfall_when_every_document_arrived(
    evaluator, tiny_brenda
) -> None:
    """A healthy split states the zero rather than omitting the key: an
    absent metric cannot be told from a run that logged none."""
    metrics = evaluator.evaluate_model(loader_over(tiny_brenda.present))

    assert metrics["dataset/test_documents_scored"] == 3.0
    assert metrics["dataset/test_documents_missing"] == 0.0
