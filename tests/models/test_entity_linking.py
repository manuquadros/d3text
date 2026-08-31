"""Pure unit tests for `d3text.models.entity_linking.BrendaClassificationModel`
— entity/class loss, the consistency penalty between the two heads, and
`ground_truth`'s batch handling.

Every test here runs on CPU with tiny synthetic tensors and no data, network,
or GPU. Methods are exercised through the `stub` fixture (see
`tests/conftest.py`), which supplies only the attributes each method reads.
"""

import pytest
import torch

from d3text.models.base import label_columns
from d3text.models.config import ModelConfig
from d3text.models.entity_linking import BrendaClassificationModel
from d3text.schema import EntityType, Schema

SCHEMA = Schema(
    entity_types=(
        EntityType(name="enzymes", prefix="enz"),
        EntityType(name="bacteria", prefix="bac"),
    )
)


# --------------------------------------------------------------------------- #
# UNK / OOS column handling (drop_unk, drop_oos, compute_entity_loss)          #
# --------------------------------------------------------------------------- #
def _loss_stub(
    stub, entities=("e0", "e1", "e2", "UNK"), classes=("c0", "c1", "OOS")
):
    """A stub carrying the sentinel columns the losses look up by name. The
    defaults put UNK/OOS last, as the BRENDA models do; pass them elsewhere to
    prove nothing depends on that position."""
    unk_index, entity_columns = label_columns(list(entities), "UNK")
    oos_index, class_columns = label_columns(list(classes), "OOS")
    return stub(
        BrendaClassificationModel,
        entities=list(entities),
        classes=list(classes),
        unk_index=unk_index,
        oos_index=oos_index,
        entity_columns=entity_columns,
        class_columns=class_columns,
        entity_pos_weight=torch.ones(len(entities) - 1),
        class_pos_weight=torch.ones(len(classes) - 1),
        consistency_weight=0.0,
        config=ModelConfig(),
        device="cpu",
    )


def test_compute_entity_loss_finite_with_correct_widths(stub):
    m = _loss_stub(stub)
    predictions = (
        torch.randn(2, 4),
        torch.randn(2, 3),
    )  # include UNK / OOS tail
    targets = (torch.zeros(2, 3), torch.zeros(2, 2))  # tail dropped
    entity_loss, class_loss = m.compute_entity_loss(predictions, targets)
    assert torch.isfinite(entity_loss) and entity_loss.ndim == 0
    assert torch.isfinite(class_loss) and class_loss.ndim == 0


def test_compute_entity_loss_slice_is_load_bearing(stub):
    """A full-width entity target must not line up with the narrowed logits."""
    m = _loss_stub(stub)
    predictions = (torch.randn(2, 4), torch.randn(2, 3))
    full_width_targets = (torch.zeros(2, 4), torch.zeros(2, 2))
    with pytest.raises((ValueError, RuntimeError)):
        m.compute_entity_loss(predictions, full_width_targets)


def test_drop_unk_and_drop_oos_remove_the_named_column_not_the_last(stub):
    m = _loss_stub(
        stub, entities=("UNK", "e0", "e1", "e2"), classes=("OOS", "c0", "c1")
    )
    assert m.drop_unk(torch.tensor([[9.0, 1.0, 2.0, 3.0]])).tolist() == [
        [1.0, 2.0, 3.0]
    ]
    assert m.drop_oos(torch.tensor([[9.0, 1.0, 2.0]])).tolist() == [[1.0, 2.0]]
    assert m.known_entities == ["e0", "e1", "e2"]
    assert m.known_classes == ["c0", "c1"]


def test_entity_loss_ignores_the_unk_column_wherever_it_sits(stub):
    """UNK is scored but never supervised, so its logit must not reach the loss
    — and it is located by name, so moving it off the tail changes nothing.
    """
    supervised = torch.tensor([[1.0, -2.0, 0.5]])
    class_logits = torch.tensor([[0.3, -0.7, 4.0]])  # OOS logit last
    targets = (torch.tensor([[1.0, 0.0, 1.0]]), torch.tensor([[1.0, 0.0]]))

    tail = _loss_stub(stub)  # UNK last, as BRENDA builds it
    tail_loss, _ = tail.compute_entity_loss(
        (torch.cat([supervised, torch.tensor([[99.0]])], dim=-1), class_logits),
        targets,
    )

    head = _loss_stub(stub, entities=("UNK", "e0", "e1", "e2"))
    head_loss, _ = head.compute_entity_loss(
        (
            torch.cat([torch.tensor([[-99.0]]), supervised], dim=-1),
            class_logits,
        ),
        targets,
    )

    assert head_loss.item() == pytest.approx(tail_loss.item())


# --------------------------------------------------------------------------- #
# BrendaClassificationModel._consistency_loss                                 #
# --------------------------------------------------------------------------- #
def _consistency_stub(stub, weight):
    unk_index, entity_columns = label_columns(["e0", "e1", "UNK"], "UNK")
    oos_index, class_columns = label_columns(["c0", "c1", "OOS"], "OOS")
    return stub(
        BrendaClassificationModel,
        consistency_weight=weight,
        device="cpu",
        unk_index=unk_index,
        oos_index=oos_index,
        entity_columns=entity_columns,
        class_columns=class_columns,
        # identity map: entity i belongs to class i (E-1 == C-1 == 2)
        class_matrix=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    )


def test_consistency_loss_zero_when_heads_agree(stub):
    m = _consistency_stub(stub, weight=1.0)
    entity_logits = torch.tensor([[10.0, -10.0, -10.0]])  # entity 0 present
    class_logits = torch.tensor(
        [[10.0, -10.0, -10.0]]
    )  # class 0 present -> agree
    penalty = m._consistency_loss(entity_logits, class_logits)
    assert penalty.item() == pytest.approx(0.0, abs=1e-4)


def test_consistency_loss_penalises_disagreement(stub):
    m = _consistency_stub(stub, weight=1.0)
    entity_logits = torch.tensor([[10.0, -10.0, -10.0]])  # entity 0 present
    agree = m._consistency_loss(
        entity_logits, torch.tensor([[10.0, -10.0, -10.0]])
    )
    disagree = m._consistency_loss(
        entity_logits, torch.tensor([[-10.0, 10.0, -10.0]])
    )
    assert disagree.item() > agree.item()


def test_consistency_loss_disabled_returns_exact_zero(stub):
    m = _consistency_stub(stub, weight=0.0)
    penalty = m._consistency_loss(
        torch.tensor([[10.0, -10.0, -10.0]]),
        torch.tensor([[10.0, -10.0, -10.0]]),
    )
    assert penalty.item() == 0.0


# --------------------------------------------------------------------------- #
# BrendaClassificationModel.ground_truth (batch dimension)                     #
# --------------------------------------------------------------------------- #
def test_ground_truth_keeps_a_batch_dimension_across_documents(stub):
    m = stub(BrendaClassificationModel, device="cpu")
    batch = [
        {
            "entities": torch.tensor([1.0, 0.0, 0.0]),
            "classes": torch.tensor([1.0, 0.0]),
        },
        {
            "entities": torch.tensor([0.0, 1.0, 0.0]),
            "classes": torch.tensor([0.0, 1.0]),
        },
    ]
    ground_truth = m.ground_truth(batch)
    entity_targets, class_targets = ground_truth.entities, ground_truth.classes

    # `torch.concat` would flatten these into 1-D vectors of length B*E / B*C;
    # the heads and loss expect one row per document instead.
    assert tuple(entity_targets.shape) == (2, 3)
    assert tuple(class_targets.shape) == (2, 2)
    assert entity_targets.tolist() == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    assert class_targets.tolist() == [[1.0, 0.0], [0.0, 1.0]]


# --------------------------------------------------------------------------- #
# BrendaClassificationModel carries no relation head — the composed          #
# ETEBrendaModel adds one without widening this model's return type.         #
# --------------------------------------------------------------------------- #
def test_ground_truth_and_forward_report_no_relations(stub, patch_base_model):
    """The two-head model returns the same typed container the three-head
    one does, with `relations` left `None` — not a narrower tuple. Before
    composition replaced inheritance, `ETEBrendaModel.ground_truth` and
    `.forward` had to widen this model's return arity to add their relation
    slot, which is exactly what tripped mypy's `[override]` check."""
    m = stub(BrendaClassificationModel, device="cpu")
    batch = [
        {"entities": torch.tensor([1.0, 0.0]), "classes": torch.tensor([1.0])}
    ]

    ground_truth = m.ground_truth(batch)
    assert ground_truth.relations is None

    model = BrendaClassificationModel(
        schema=SCHEMA,
        class_matrix=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        entity_index={"enz1": 0, "bac1": 1},
        config=ModelConfig(base_model="prajjwal1/bert-mini", hidden_layers=[8]),
        device="cpu",
    )
    embeddings = torch.randn(1, 4, 256)
    mask = torch.ones(1, 4, dtype=torch.bool)
    with torch.no_grad():
        logits = model(embeddings, mask)
    assert logits.relations is None


# --------------------------------------------------------------------------- #
# entity-column alignment (BrendaClassificationModel construction)            #
# --------------------------------------------------------------------------- #
def test_entities_stay_aligned_with_entity_index_when_classes_overlap(
    patch_base_model,
):
    """`entities[i]` must name the entity that entity logit column `i` scores.

    An entity belonging to two classes is one entity and one column. Deriving
    the list by flattening the per-class entity sets counts it twice, widening
    the entity head past the target width.
    """
    model = BrendaClassificationModel(
        schema=SCHEMA,
        class_matrix=torch.tensor(
            [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]  # shared is in both classes
        ),
        entity_index={"enz1": 0, "shared": 1, "bac1": 2},
        config=ModelConfig(base_model="prajjwal1/bert-mini", hidden_layers=[8]),
        device="cpu",
    )

    assert model.entities == ["enz1", "shared", "bac1", "UNK"]
    assert model.num_of_entities == 4  # 3 entities + UNK, not 4 + UNK
    entity_logits, _ = model.classifier(
        torch.randn(2, model.hidden_block_output_size)
    )
    assert entity_logits.shape[1] == 4
