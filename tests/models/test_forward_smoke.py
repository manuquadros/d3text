"""End-to-end shape-contract smoke tests for `ETEBrendaModel`.

A real model with a tiny random BERT injected for the frozen base, so there is
no download. Parametrized over the `device` fixture, so on a GPU machine these
also exercise device placement of the heads, buffers and pooled tensors.
"""

import pytest
import torch
from d3text.models.config import ModelConfig
from d3text.models.model_types import IndexedRelation
from d3text.models.ete import ETEBrendaModel
from d3text.schema import EntityType, RelationType, Schema

pytestmark = pytest.mark.slow

SCHEMA = Schema(
    entity_types=(
        EntityType(name="enzymes", prefix="enz"),
        EntityType(name="bacteria", prefix="bac"),
    ),
    relation_types=(
        RelationType(
            name="HasEnzyme", subject_types=("bacteria",), object_type="enzymes"
        ),
        RelationType(name="none", is_none=True),
    ),
)


@pytest.fixture
def tiny_ete(patch_base_model, device):
    """A real ETEBrendaModel backed by a tiny random BERT (see the
    ``patch_base_model`` fixture), placed on ``device``."""
    model = ETEBrendaModel(
        schema=SCHEMA,
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


def test_forward_document_logits_ignore_padding(tiny_ete):
    """Pooled logits are a function of each document's real tokens alone.

    The normaliser used to be the *padded* length, so a document's score
    depended on how long its batch companions were.
    """
    device = tiny_ete.device
    embeddings, mask, _ = _forward_inputs(device)
    pad = 30
    padded_embeddings = torch.cat(
        [
            embeddings,
            torch.zeros(2, pad, embeddings.shape[2], device=device),
        ],
        dim=1,
    )
    padded_mask = torch.cat(
        [mask, torch.zeros(2, pad, dtype=torch.bool, device=device)], dim=1
    )

    with torch.no_grad():
        entity_logits, class_logits, _ = tiny_ete(embeddings, mask)
        padded_entity, padded_class, _ = tiny_ete(
            padded_embeddings, padded_mask
        )

    # the pre-fix shift is log(40/10) ≈ 1.39, far outside this tolerance
    assert torch.allclose(entity_logits, padded_entity, atol=2e-2)
    assert torch.allclose(class_logits, padded_class, atol=2e-2)


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
    """An `ETEBrendaModel` whose entity and class heads differ in width.

    `tiny_ete` makes both heads 3 wide, so a saved tensor cannot be attributed
    to one of them.
    """
    model = ETEBrendaModel(
        schema=SCHEMA,
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
    """The entropy/argmax block runs under `no_grad`.

    Its four intermediates are each a full `[document, token, entity]` tensor,
    held for a backward that never reads them: 6 saves before the fix, 1 after.
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
    """The guard against over-widening the `no_grad` block.

    One statement too many silently severs the gradient path — the pooled
    logits lose their `grad_fn` and the heads stop training, with no error
    anywhere.
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
