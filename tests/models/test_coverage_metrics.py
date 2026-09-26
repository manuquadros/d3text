"""What an evaluation reports about the documents it actually scored.

`dataset/test_documents` is logged at setup from the split frame; the scores
are computed over whatever the encodings file backs. These pin the counts that
state the difference, over a split holding one pmid the HDF5 does not and drawn
the way `evaluate` draws it. The models are stubbed: `evaluate_model`'s own
arithmetic is what is under test.
"""

from typing import Any

import pytest
import torch
from d3text.models.base import Model
from d3text.models.entity_linking import BrendaClassificationModel
from d3text.models.ete import ETEBrendaModel
from d3text.models.config import ModelConfig
from d3text.models.model_types import BatchLogits, GroundTruth
from d3text.models.ner import NERClassificationModel
from d3text.data.data import get_batch_loader


class StubbedBrenda(BrendaClassificationModel):
    def get_batch_logits(
        self, batch: Any, gold_relations: Any = None
    ) -> BatchLogits:
        return BatchLogits(torch.zeros(len(batch), 3))

    def ground_truth(self, batch: Any) -> GroundTruth:
        return GroundTruth(torch.ones(len(batch), 2))


class StubbedNER(NERClassificationModel):
    def get_batch_logits(self, batch: Any) -> Any:
        return torch.zeros(len(batch), 2)

    def ground_truth(self, batch: Any) -> Any:
        return torch.ones(len(batch), 2)


class StubbedETE(ETEBrendaModel):
    # `evaluate_model` folds unscored gold in as `none` predictions, which
    # needs the column the real `__init__` — bypassed here — would have set.
    relations_none_index = 2

    def get_batch_logits(
        self, batch: Any, gold_relations: Any = None
    ) -> BatchLogits:
        return BatchLogits(torch.zeros(len(batch), 3), None)

    def ground_truth(self, batch: Any) -> GroundTruth:
        return GroundTruth(torch.ones(len(batch), 2), [])


@pytest.fixture(params=[StubbedBrenda, StubbedNER, StubbedETE])
def evaluator(request, stub):
    """A model class whose only live method is `evaluate_model`."""
    return stub(
        request.param,
        _modules={},
        _parameters={},
        _buffers={},
        training=False,
        _detection_accumulator=lambda: None,
        classes=["a", "b", "OOS"],
        class_columns=torch.tensor([0, 1]),
        # `evaluate_model` now wraps its loop in
        # `prefetch_layer_boundary_reads`, which reads
        # `config.unfrozen_top_layers`; the class name here doesn't matter
        # to that check (0 either way), only that `config` exists.
        config=ModelConfig(model_class="NERClassificationModel"),
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


def test_evaluate_model_routes_its_batch_loop_through_the_prefetch_wrapper(
    evaluator, tiny_brenda, monkeypatch
) -> None:
    """Every `evaluate_model` override must hand `batch_progress`'s iterator
    to `prefetch_layer_boundary_reads`, not iterate it directly -- that
    wrapper is what lets a configured layer-boundary store's reads for the
    next batch overlap this batch's replay (see
    `tests/models/test_layer_boundary_prefetch.py`). `evaluator` is
    parametrized over NER, entity-linking and end-to-end, so this covers
    all three overrides; dropping the wrapper from any one of them turns
    this red for that parameter alone. Spies on the class method rather
    than replacing it, delegating to the real implementation so the rest of
    the pass stays correct and only the wiring is pinned."""
    calls: list[object] = []
    real = Model.prefetch_layer_boundary_reads

    def spy(self, batches):
        calls.append(self)
        yield from real(self, batches)

    monkeypatch.setattr(Model, "prefetch_layer_boundary_reads", spy)

    evaluator.evaluate_model(loader_over(tiny_brenda.full))

    assert calls == [evaluator]
