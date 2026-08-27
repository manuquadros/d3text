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
def tiny_ete(patch_base_model, device):
    """A real ETEBrendaModel backed by a tiny random BERT (see the
    ``patch_base_model`` fixture), placed on ``device``."""
    model = ETEBrendaModel(
        classes={"enzymes": {"enz1"}, "bacteria": {"bac1"}},
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
    # Gold labels are consumed via int(tr.label), so they stay on CPU, as they
    # do when built from data.
    gold = [
        IndexedRelation(
            docix=0, subject="enz1", object="bac1", label=torch.tensor(0)
        )
    ]
    return embeddings, mask, gold


def test_forward_pools_document_logits(tiny_ete):
    embeddings, mask, gold = _forward_inputs(tiny_ete.device)
    with torch.no_grad():
        entity_logits, class_logits, rel = tiny_ete(
            embeddings, mask, gold_relations=gold
        )
    # logits are pooled to one row per document, full width (incl UNK / OOS)
    assert tuple(entity_logits.shape) == (2, tiny_ete.num_of_entities)
    assert tuple(class_logits.shape) == (2, tiny_ete.num_of_classes)
    assert torch.isfinite(entity_logits).all()
    assert torch.isfinite(class_logits).all()


def test_forward_emits_relation_candidates(tiny_ete):
    embeddings, mask, gold = _forward_inputs(tiny_ete.device)
    with torch.no_grad():
        *_, rel = tiny_ete(embeddings, mask, gold_relations=gold)
    assert rel is not None  # two distinct gold entities -> a candidate pair
    meta, rel_logits = rel
    assert rel_logits.shape[1] == tiny_ete.num_relations
    assert set(meta) == {"sequence", "arg_pred_i", "arg_pred_j"}


def test_forward_losses_are_finite_scalars(tiny_ete):
    device = tiny_ete.device
    embeddings, mask, gold = _forward_inputs(device)
    with torch.no_grad():
        entity_logits, class_logits, rel = tiny_ete(
            embeddings, mask, gold_relations=gold
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


# --------------------------------------------------------------------------- #
# OOM-01: the hard-entity-mask block must not be recorded by autograd          #
# --------------------------------------------------------------------------- #
@pytest.fixture
def asymmetric_ete(patch_base_model, device):
    """An ETEBrendaModel whose entity and class heads have *different* widths.

    ``tiny_ete`` has 2 entities + UNK and 2 classes + OOS, so both heads emit
    ``[B, T, 3]`` and a saved tensor cannot be attributed to one of them. Three
    entities makes the entity head 4 wide and the counting unambiguous.
    """
    model = ETEBrendaModel(
        classes={"enzymes": {"enz1", "enz2"}, "bacteria": {"bac1"}},
        class_matrix=torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        entity_index={"enz1": 0, "enz2": 1, "bac1": 2},
        config=ModelConfig(
            base_model="prajjwal1/bert-mini", hidden_layers=[8], ramp_epochs=0
        ),
        device=device,
    )
    model.to(device)
    model.train()
    return model


def _saved_shapes(model, embeddings, mask, gold):
    """Shapes of every tensor autograd packs into the graph during forward."""
    shapes: list[tuple[int, ...]] = []

    def pack(tensor):
        shapes.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        entity_logits, class_logits, rel = model(
            embeddings, mask, gold_relations=gold
        )
    return shapes, entity_logits, class_logits, rel


def test_forward_saves_the_entity_logits_once(asymmetric_ete):
    """The entropy/argmax block runs under ``no_grad``.

    Its four intermediates are each a full ``[document, token, entity]``
    tensor; recorded by autograd they are the largest thing in the step, held
    for a backward that never reads them. Only ``entity_logits`` itself, which
    the pooling genuinely needs, may be saved at that width: 6 saves before the
    fix, 1 after.
    """
    model = asymmetric_ete
    assert model.num_of_entities != model.num_of_classes  # no shape collision

    batch, tokens = 2, 10
    embeddings = torch.randn(
        batch, tokens, 256, device=model.device, requires_grad=True
    )
    mask = torch.ones(batch, tokens, dtype=torch.bool, device=model.device)
    gold = [
        IndexedRelation(
            docix=0, subject="enz1", object="bac1", label=torch.tensor(0)
        )
    ]

    shapes, *_ = _saved_shapes(model, embeddings, mask, gold)

    entity_width = (batch, tokens, model.num_of_entities)
    assert shapes.count(entity_width) == 1


def test_forward_still_backpropagates_into_both_heads(asymmetric_ete):
    """The guard against over-widening the ``no_grad`` block.

    Wrapping one statement too many would silently sever the real gradient
    path — the pooled logits would lose their ``grad_fn`` and the heads would
    stop training, with no error anywhere.
    """
    model = asymmetric_ete
    batch, tokens = 2, 10
    embeddings = torch.randn(
        batch, tokens, 256, device=model.device, requires_grad=True
    )
    mask = torch.ones(batch, tokens, dtype=torch.bool, device=model.device)
    gold = [
        IndexedRelation(
            docix=0, subject="enz1", object="bac1", label=torch.tensor(0)
        )
    ]

    entity_logits, class_logits, rel = model(
        embeddings, mask, gold_relations=gold
    )
    assert entity_logits.requires_grad and class_logits.requires_grad

    assert rel is not None  # the gold pair always yields a candidate
    _, relation_logits = rel
    assert relation_logits.requires_grad

    (
        entity_logits.sum() + class_logits.sum() + relation_logits.sum()
    ).backward()

    # the shared hidden block, the entity head and the relation head all learn
    assert model.hidden_layers[0][0].weight.grad is not None
    assert model.classifier.entity_classifier[-1].weight.grad is not None
    assert model.relation_classifier.bilinear.grad is not None
    assert embeddings.grad is not None
