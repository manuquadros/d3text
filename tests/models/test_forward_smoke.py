"""End-to-end shape-contract smoke tests for ETEBrendaModel.

These drive the document-level ``forward`` (hidden block -> classifier ->
relation extraction) plus the entity and relation losses on synthetic pooled
embeddings. They construct a *real* ETEBrendaModel but inject a tiny random
BERT for the frozen base model, so there is no network download.

Parametrized over the ``device`` fixture: the CPU variant runs everywhere; the
CUDA variant carries the ``gpu`` marker and is auto-skipped when no CUDA device
is present (see ``tests/conftest.py``), so on a GPU machine these also exercise
device placement of the heads, buffers, and pooled tensors.

Marked ``slow`` (deterministic). The tiny random BERT is injected by
monkeypatching ``load_base_model`` (the ``patch_base_model`` fixture) to keep
this test offline and network-free; loading the real ``prajjwal1/bert-mini``
(whose legacy config.json lacks a ``model_type`` key) is covered by the
``integration`` test ``test_load_base_model_handles_legacy_config``.
"""

import pytest
import torch
from d3text.models.config import ModelConfig
from d3text.models.model_types import IndexedRelation
from d3text.models.models import ETEBrendaModel

pytestmark = pytest.mark.slow


@pytest.fixture
def tiny_ete(patch_base_model, device, tiny_schema):
    """A real ETEBrendaModel backed by a tiny random BERT (see the
    ``patch_base_model`` fixture), placed on ``device``."""
    model = ETEBrendaModel(
        schema=tiny_schema,
        class_matrix=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        entity_index={"enz1": 0, "bac1": 1},
        config=ModelConfig(
            base_model="prajjwal1/bert-mini", hidden_layers=[8], ramp_epochs=0
        ),
        device=device,
    )
    model.to(device)  # mirrors scripts/train.py:85
    model.eval()
    return model


def _forward_inputs(device):
    batch, tokens, hidden = 2, 10, 256
    embeddings = torch.randn(batch, tokens, hidden, device=device)
    mask = torch.ones(batch, tokens, dtype=torch.bool, device=device)
    entities_in_batch = (
        torch.tensor([0], dtype=torch.int16, device=device),
        torch.tensor([1], dtype=torch.int16, device=device),
    )
    # Gold labels are consumed via int(tr.label), so they stay on CPU, as they
    # do when built from data.
    gold = [
        IndexedRelation(
            docix=0, subject="enz1", object="bac1", label=torch.tensor(0)
        )
    ]
    return embeddings, mask, entities_in_batch, gold


def test_forward_pools_document_logits(tiny_ete):
    embeddings, mask, entities_in_batch, gold = _forward_inputs(tiny_ete.device)
    with torch.no_grad():
        entity_logits, class_logits, rel = tiny_ete(
            embeddings, mask, entities_in_batch, gold_relations=gold
        )
    # logits are pooled to one row per document, full width (incl UNK / OOS)
    assert tuple(entity_logits.shape) == (2, tiny_ete.num_of_entities)
    assert tuple(class_logits.shape) == (2, tiny_ete.num_of_classes)
    assert torch.isfinite(entity_logits).all()
    assert torch.isfinite(class_logits).all()


def test_forward_emits_relation_candidates(tiny_ete):
    embeddings, mask, entities_in_batch, gold = _forward_inputs(tiny_ete.device)
    with torch.no_grad():
        *_, rel = tiny_ete(
            embeddings, mask, entities_in_batch, gold_relations=gold
        )
    assert rel is not None  # two distinct gold entities -> a candidate pair
    meta, rel_logits = rel
    assert rel_logits.shape[1] == tiny_ete.num_relations
    assert set(meta) == {"sequence", "arg_pred_i", "arg_pred_j"}


def test_forward_losses_are_finite_scalars(tiny_ete):
    device = tiny_ete.device
    embeddings, mask, entities_in_batch, gold = _forward_inputs(device)
    with torch.no_grad():
        entity_logits, class_logits, rel = tiny_ete(
            embeddings, mask, entities_in_batch, gold_relations=gold
        )
        entity_true = torch.tensor(
            [[1, 0], [0, 1]], dtype=torch.float32, device=device
        )
        class_true = torch.tensor(
            [[1, 0], [0, 1]], dtype=torch.float32, device=device
        )
        entity_loss, class_loss = tiny_ete.compute_entity_loss(
            (entity_logits, class_logits), (entity_true, class_true)
        )
        meta, rel_logits = rel
        relation_loss = tiny_ete.compute_relation_loss(gold, meta, rel_logits)

    for loss in (entity_loss, class_loss, relation_loss):
        assert loss.ndim == 0 and torch.isfinite(loss)
