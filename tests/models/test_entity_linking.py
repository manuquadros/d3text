"""Pure unit tests for `d3text.models.entity_linking.BrendaClassificationModel`
— the class loss, `ground_truth`'s batch handling, and the document-level
class-negative abstention.

Every test here runs on CPU with tiny synthetic tensors and no data, network,
or GPU. Methods are exercised through the `stub` fixture (see
`tests/conftest.py`), which supplies only the attributes each method reads.
"""

import h5py
import pytest
import torch

from d3text import token_labels
from d3text.models.base import label_columns
from d3text.models.config import ModelConfig
from d3text.models.entity_linking import BrendaClassificationModel
from d3text.schema import EntityType, Schema
from d3text.utils import WINDOW_LENGTH, WINDOW_STRIDE

CLASSES_WITH_OOS = ["enzymes", "bacteria", "strains", "OOS"]

SCHEMA = Schema(
    entity_types=(
        EntityType(name="enzymes", prefix="enz"),
        EntityType(name="bacteria", prefix="bac"),
    )
)


# --------------------------------------------------------------------------- #
# gradient checkpointing is opt-in                                             #
# --------------------------------------------------------------------------- #
def _build_brenda(patch_base_model, *, gradient_checkpointing: bool):
    return BrendaClassificationModel(
        schema=SCHEMA,
        config=ModelConfig(
            model_class="BrendaClassificationModel",
            base_model="prajjwal1/bert-mini",
            hidden_layers=[8],
            gradient_checkpointing=gradient_checkpointing,
        ),
        device="cpu",
    )


def test_gradient_checkpointing_off_by_default_leaves_hidden_unwrapped(
    patch_base_model,
):
    """`ModelConfig.gradient_checkpointing` defaults to False, so `self.hidden`
    must stay the plain per-layer stack `build_layers` set, not the
    `torch.utils.checkpoint`-wrapping closure `enable_gradient_checkpointing`
    swaps in."""
    model = _build_brenda(patch_base_model, gradient_checkpointing=False)
    assert model.hidden.__name__ == "hidden_forward"


def test_gradient_checkpointing_true_wraps_hidden(patch_base_model):
    model = _build_brenda(patch_base_model, gradient_checkpointing=True)
    assert model.hidden.__name__ == "hidden_with_checkpoint"


# --------------------------------------------------------------------------- #
# OOS column handling (drop_oos, compute_class_loss)                           #
# --------------------------------------------------------------------------- #
def _loss_stub(stub, classes=("c0", "c1", "OOS")):
    """A stub carrying the sentinel column the loss looks up by name. The
    default puts OOS last, as the BRENDA models do; pass it elsewhere to prove
    nothing depends on that position."""
    oos_index, class_columns = label_columns(list(classes), "OOS")
    return stub(
        BrendaClassificationModel,
        classes=list(classes),
        oos_index=oos_index,
        class_columns=class_columns,
        class_pos_weight=torch.ones(len(classes) - 1),
        config=ModelConfig(model_class="BrendaClassificationModel"),
        device="cpu",
    )


def test_compute_class_loss_finite_with_correct_widths(stub):
    m = _loss_stub(stub)

    loss = m.compute_class_loss(torch.randn(2, 3), torch.zeros(2, 2))

    assert torch.isfinite(loss) and loss.ndim == 0


def test_compute_class_loss_slice_is_load_bearing(stub):
    """A full-width class target must not line up with the narrowed logits."""
    m = _loss_stub(stub)

    with pytest.raises((ValueError, RuntimeError)):
        m.compute_class_loss(torch.randn(2, 3), torch.zeros(2, 3))


def test_drop_oos_removes_the_named_column_not_the_last(stub):
    m = _loss_stub(stub, classes=("OOS", "c0", "c1"))

    assert m.drop_oos(torch.tensor([[9.0, 1.0, 2.0]])).tolist() == [[1.0, 2.0]]
    assert m.known_classes == ["c0", "c1"]


def test_the_class_loss_ignores_the_oos_column_wherever_it_sits(stub):
    """OOS is scored but never supervised, so its logit must not reach the
    loss — and it is located by name, so moving it off the tail changes
    nothing."""
    supervised = torch.tensor([[1.0, -2.0]])
    targets = torch.tensor([[1.0, 0.0]])

    tail = _loss_stub(stub)  # OOS last, as BRENDA builds it
    tail_loss = tail.compute_class_loss(
        torch.cat([supervised, torch.tensor([[99.0]])], dim=-1), targets
    )

    head = _loss_stub(stub, classes=("OOS", "c0", "c1"))
    head_loss = head.compute_class_loss(
        torch.cat([torch.tensor([[-99.0]]), supervised], dim=-1), targets
    )

    assert head_loss.item() == pytest.approx(tail_loss.item())


def test_the_model_has_no_entity_loss_or_consistency_penalty():
    """`_consistency_loss` penalised an entity prediction the class head did
    not agree with, through the entity-to-class matrix; with one head left
    there is nothing for it to be consistent with. `compute_entity_loss` went
    with the BCE it scored."""
    for gone in (
        "_consistency_loss",
        "compute_entity_loss",
        "entity_loss_fn",
        "drop_unk",
        "known_entities",
        "register_entity_columns",
    ):
        assert not hasattr(BrendaClassificationModel, gone), gone


# --------------------------------------------------------------------------- #
# BrendaClassificationModel.ground_truth (batch dimension)                     #
# --------------------------------------------------------------------------- #
def test_ground_truth_keeps_a_batch_dimension_across_documents(stub):
    m = stub(BrendaClassificationModel, device="cpu")
    batch = [
        {"classes": torch.tensor([1.0, 0.0])},
        {"classes": torch.tensor([0.0, 1.0])},
    ]

    class_targets = m.ground_truth(batch).classes

    # `torch.concat` would flatten these into a 1-D vector of length B*C;
    # the head and loss expect one row per document instead.
    assert tuple(class_targets.shape) == (2, 2)
    assert class_targets.tolist() == [[1.0, 0.0], [0.0, 1.0]]


# --------------------------------------------------------------------------- #
# BrendaClassificationModel carries no relation head — the composed          #
# ETEBrendaModel adds one without widening this model's return type.         #
# --------------------------------------------------------------------------- #
def test_ground_truth_and_forward_report_no_relations(stub, patch_base_model):
    """The relation-free model returns the same typed container the
    end-to-end one does, with `relations` left `None` — not a narrower tuple.
    Before composition replaced inheritance, `ETEBrendaModel.ground_truth`
    and `.forward` had to widen this model's return arity to add their
    relation slot, which is exactly what tripped mypy's `[override]` check."""
    m = stub(BrendaClassificationModel, device="cpu")
    batch = [{"classes": torch.tensor([1.0])}]

    ground_truth = m.ground_truth(batch)
    assert ground_truth.relations is None

    model = BrendaClassificationModel(
        schema=SCHEMA,
        config=ModelConfig(
            model_class="BrendaClassificationModel",
            base_model="prajjwal1/bert-mini",
            hidden_layers=[8],
        ),
        device="cpu",
    )
    embeddings = torch.randn(1, 4, 256)
    mask = torch.ones(1, 4, dtype=torch.bool)
    with torch.no_grad():
        logits = model(embeddings, mask)
    assert logits.relations is None


# --------------------------------------------------------------------------- #
# class-column geometry (BrendaClassificationModel construction)               #
# --------------------------------------------------------------------------- #
def test_the_class_head_is_as_wide_as_the_schema_plus_oos(patch_base_model):
    """`classes[i]` must name the class that class logit column `i` scores,
    and the head's only extra column is the unsupervised `OOS` one."""
    model = BrendaClassificationModel(
        schema=SCHEMA,
        config=ModelConfig(
            model_class="BrendaClassificationModel",
            base_model="prajjwal1/bert-mini",
            hidden_layers=[8],
        ),
        device="cpu",
    )

    assert model.classes == ["enzymes", "bacteria", "OOS"]
    assert model.num_of_classes == 3
    class_logits = model.classifier(
        torch.randn(2, model.hidden_block_output_size)
    )
    assert class_logits.shape[1] == 3


# --------------------------------------------------------------------------- #
# BrendaClassificationModel.class_negative_abstain_mask                       #
# --------------------------------------------------------------------------- #
class _FakeReader:
    """Stands in for `TokenLabelReader`: returns fixed type codes per id,
    ignoring `min_chars` — the cutoff logic belongs to the reader, not to
    `class_negative_abstain_mask`, which this test does not re-exercise."""

    def __init__(self, mentioned_by_id: dict[int, frozenset[int]]):
        self._mentioned_by_id = mentioned_by_id

    def mentioned_types(self, pubmed_id, min_chars=0):
        return self._mentioned_by_id.get(pubmed_id, frozenset())


def _abstain_stub(stub, reader, classes=CLASSES_WITH_OOS):
    return stub(
        BrendaClassificationModel,
        classes=list(classes),
        _token_labels=reader,
        config=ModelConfig(
            model_class="BrendaClassificationModel",
            class_negative_abstention=True,
            token_supervision=True,
        ),
    )


def _abstain_batch():
    return [
        {"id": torch.tensor(100)},
        {"id": torch.tensor(200)},
        {"id": torch.tensor(300)},
    ]


def test_class_negative_abstain_mask_matches_gold_negatives_only(stub):
    """A dictionary match abstains a (document, class) cell only when that
    cell is a gold negative; positives and unmentioned classes stay False."""
    reader = _FakeReader(
        {100: frozenset({1, 3}), 200: frozenset({2})}
    )  # 300: nothing mentioned
    m = _abstain_stub(stub, reader)

    class_true = torch.tensor(
        [
            [0.0, 1.0, 0.0],  # bacteria gold-positive
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    mask = m.class_negative_abstain_mask(_abstain_batch(), class_true)

    assert mask is not None
    assert mask.dtype == torch.bool
    assert mask.tolist() == [
        [True, False, True],  # enzymes, strains mentioned; bacteria excluded
        [False, True, False],  # bacteria mentioned
        [False, False, False],  # nothing mentioned
    ]


def test_class_negative_abstain_mask_lands_on_class_true_device(stub):
    """The mask is transferred back to wherever the caller's targets live,
    whatever device that batch actually ran on."""
    reader = _FakeReader({100: frozenset({1})})
    m = _abstain_stub(stub, reader)

    class_true = torch.zeros(3, 3)
    mask = m.class_negative_abstain_mask(_abstain_batch(), class_true)

    assert mask is not None
    assert mask.device == class_true.device


def test_class_negative_abstain_mask_never_writes_elementwise_off_cpu(
    stub, monkeypatch
):
    """The mask must be built as a CPU tensor and moved once, not written one
    element at a time on `class_true`'s own device — that pattern is what
    launches a device kernel per matched type. Simulated with the `meta`
    device, which needs no GPU: the old `zeros_like(class_true, dtype=bool)`
    construction inherits `meta`, so any element write it makes is caught
    below; the fixed code builds on `cpu` and only ever writes there.
    """
    reader = _FakeReader({100: frozenset({1}), 200: frozenset({2})})
    m = _abstain_stub(stub, reader)

    original_setitem = torch.Tensor.__setitem__

    def guarded_setitem(self, index, value):
        if self.device.type != "cpu":
            msg = (
                f"element-wise write to a {self.device.type!r} tensor "
                "(expected every write to land on a CPU staging tensor)"
            )
            raise AssertionError(msg)
        return original_setitem(self, index, value)

    monkeypatch.setattr(torch.Tensor, "__setitem__", guarded_setitem)

    class_true = torch.zeros(3, 3, device="meta")
    mask = m.class_negative_abstain_mask(_abstain_batch(), class_true)

    assert mask is not None
    assert mask.device.type == "meta"


# --------------------------------------------------------------------------- #
# construction wires the store's tokenizer check                              #
# --------------------------------------------------------------------------- #
def test_construction_refuses_a_store_stamped_for_another_base_model(
    patch_base_model, machine_stores, tmp_path
):
    """`__init__` builds its `TokenLabelReader` with `config.base_model`, so a
    store tokenized under one base model must already refuse a config naming
    another one at construction time, before any batch is ever run -- a gap
    a reader built with no base model to check would let straight through."""
    path = tmp_path / "labels.hdf5"
    with h5py.File(path, "w") as store:
        token_labels.write_label_space(
            store,
            token_labels.BRENDA_LABELS,
            stamp=token_labels.IndexStamp(digest="test-index"),
            tokenizer=token_labels.TokenizerStamp(
                base_model="model-a",
                digest="digest-a",
                window_length=WINDOW_LENGTH,
                window_stride=WINDOW_STRIDE,
            ),
        )

    machine_stores(token_labels_store={"model-b": path})
    with pytest.raises(ValueError, match="model-a"):
        BrendaClassificationModel(
            schema=SCHEMA,
            config=ModelConfig(
                model_class="BrendaClassificationModel",
                base_model="model-b",
                hidden_layers=[8],
                token_supervision=True,
            ),
            device="cpu",
        )
