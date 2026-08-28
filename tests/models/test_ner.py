"""Pure unit tests for `d3text.models.ner.NERClassificationModel`.

Every test here runs on CPU with tiny synthetic tensors and no data, network,
or GPU. Methods are exercised through the `stub` fixture (see
`tests/conftest.py`), which supplies only the attributes each method reads.
"""

import torch
from torch.utils.data import DataLoader

from d3text.models.base import Step
from d3text.models.config import ModelConfig
from d3text.models.ner import NERClassificationModel
from d3text.schema import EntityType, Schema
from d3text.training.update import BatchUpdate

SCHEMA = Schema(entity_types=(EntityType(name="enzymes", prefix="enz"),))


# --------------------------------------------------------------------------- #
# NERClassificationModel.ground_truth (batch dimension)                        #
# --------------------------------------------------------------------------- #
def test_ner_ground_truth_keeps_a_batch_dimension_across_documents(stub):
    m = stub(NERClassificationModel, device="cpu")
    batch = [
        {"classes": torch.tensor([1.0, 0.0])},
        {"classes": torch.tensor([0.0, 1.0])},
    ]
    class_targets = m.ground_truth(batch)

    # `torch.concat` would flatten these into a 1-D vector of length B*C; the
    # class head and loss expect one row per document instead.
    assert tuple(class_targets.shape) == (2, 2)
    assert class_targets.tolist() == [[1.0, 0.0], [0.0, 1.0]]


# --------------------------------------------------------------------------- #
# run_epoch, via the shared `Model.compute_losses` contract                    #
# --------------------------------------------------------------------------- #
def _loader() -> DataLoader:
    """One batch of one placeholder document.

    Its content is never read — `compute_batch_losses` is monkeypatched
    below — but it still has to satisfy `compute_losses`'s
    ``Sequence[BatchItem]`` signature, which a bare collated tensor would
    not.
    """
    return DataLoader([[{}]], batch_size=1, collate_fn=lambda items: items[0])


def test_run_epoch_applies_the_single_ner_loss_through_the_shared_update(
    patch_base_model, monkeypatch
):
    """The regression this guards: `NERClassificationModel` used to carry its
    own `run_epoch`. Now `Model.run_epoch` is the only implementation, and it
    drives NER exactly as it drives the other two model classes — one call
    to `compute_losses`, whose one key here is applied through the same
    `BatchUpdate` (backward, clip, step) the entity-linking and end-to-end
    models share.
    """
    model = NERClassificationModel(
        schema=SCHEMA,
        config=ModelConfig(base_model="prajjwal1/bert-mini", hidden_layers=[8]),
        device="cpu",
    )
    anchor = next(p for p in model.parameters() if p.requires_grad)
    monkeypatch.setattr(
        model, "compute_batch_losses", lambda batch: anchor.sum() * 0.0 + 3.0
    )

    class RecordingUpdate(BatchUpdate):
        """Spies on the losses `run_epoch` hands to `update`, then still
        performs the real backward/clip/step so the shared plumbing is
        exercised end to end."""

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.calls: list[tuple[float, ...]] = []

        def __call__(self, *losses: torch.Tensor) -> None:
            self.calls.append(tuple(loss.item() for loss in losses))
            super().__call__(*losses)

    update = RecordingUpdate(
        model, torch.optim.SGD(model.parameters(), lr=0.5), "cpu"
    )

    losses, denominator = model.run_epoch(
        data=_loader(), step=Step.TRAINING, epoch=0, update=update
    )

    assert losses == {"class": 3.0}
    assert denominator == 1
    # NER has one objective, so `update` must see exactly it and nothing else.
    assert update.calls == [(3.0,)]
    # A populated dict is only returned once a training step actually ran the
    # clip; NER's own former `run_epoch` reached this same code path.
    assert update.grad_norm_metrics() != {}
