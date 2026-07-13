"""Pure unit tests for models.py.

Every test here runs on CPU with tiny synthetic tensors and no data, network,
or GPU. Methods are exercised through the `stub` fixture (see conftest.py),
which supplies only the attributes each method reads.

Where a known bug is documented, the test asserts the *intended* behaviour and
is marked ``xfail`` so the suite drives the fix instead of freezing the buggy
output.
"""

import math
import types

import pytest
import torch
from pydantic import ValidationError

from d3text.models.config import ModelConfig
from d3text.models.model_types import IndexedRelation
from d3text.models.models import (
    BiaffineRelationClassifier,
    BrendaClassificationModel,
    ClassificationHead,
    ETEBrendaModel,
    Model,
    balanced_class_weights,
    focal_cross_entropy,
    get_batch_entities,
    initialize_classifier_bias,
    label_columns,
    ordered_entities,
)


# --------------------------------------------------------------------------- #
# Model._pool_logits (entity_logits_pooling knob)                              #
# --------------------------------------------------------------------------- #
def _pool_stub(stub, pooling):
    return stub(Model, entity_logits_pooling=pooling)


def test_pool_logits_defaults_to_logsumexp(stub):
    m = _pool_stub(stub, "logsumexp")
    logits = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert torch.allclose(
        m._pool_logits(logits, dim=0), torch.logsumexp(logits, dim=0)
    )


def test_logsumexp_pooling_is_length_biased(stub):
    """Smooth-max: uniform per-token logits gain +log(T), so pooling is *not*
    length-invariant (intended for sparse-mention detection)."""
    m = _pool_stub(stub, "logsumexp")
    short = m._pool_logits(torch.full((3, 2), 1.0), dim=0)
    long = m._pool_logits(torch.full((6, 2), 1.0), dim=0)
    expected_gap = torch.full_like(short, math.log(6) - math.log(3))
    assert torch.allclose(long - short, expected_gap)


@pytest.mark.parametrize("pooling", ["logmeanexp", "max", "mean"])
def test_length_invariant_pooling_options(stub, pooling):
    """logmeanexp / max / mean pool identical per-token logits to the same value
    regardless of document length."""
    m = _pool_stub(stub, pooling)
    short = m._pool_logits(torch.full((3, 2), 1.0), dim=0)
    long = m._pool_logits(torch.full((6, 2), 1.0), dim=0)
    assert torch.allclose(short, long)


def test_pool_logits_rejects_unknown_pooling(stub):
    m = _pool_stub(stub, "bogus")
    with pytest.raises(ValueError):
        m._pool_logits(torch.zeros(2, 2), dim=0)


# --------------------------------------------------------------------------- #
# get_batch_entities                                                           #
# --------------------------------------------------------------------------- #
def test_get_batch_entities_extracts_indices_on_cpu():
    batch = [{"entities": torch.tensor([[0, 1, 0, 1]], dtype=torch.uint8)}]
    (entities,) = get_batch_entities(batch, device="cpu")
    assert entities.tolist() == [1, 3]
    assert entities.dtype == torch.int16


# --------------------------------------------------------------------------- #
# Model.batch_input_tensors                                                    #
# --------------------------------------------------------------------------- #
def test_batch_input_tensors_concatenates_chunks_into_2d(stub):
    """Per-document ``[n_chunks, token]`` sequences must concat along dim 0 into
    a single ``[sum(n_chunks), token]`` tensor per key.

    ``get_token_embeddings`` slices the base-model output back into
    per-document chunks via ``doc_id.shape[-1]``, so this contract must be 2-D;
    the old ``chain.from_iterable`` collapsed it to 1-D.
    """
    m = stub(Model)
    token = 4
    doc0 = torch.arange(2 * token).reshape(2, token)  # 2 chunks
    doc1 = torch.arange(3 * token).reshape(3, token)  # 3 chunks
    batch = [
        {
            "sequence": {
                "input_ids": doc0,
                "attention_mask": torch.ones_like(doc0),
            }
        },
        {
            "sequence": {
                "input_ids": doc1,
                "attention_mask": torch.ones_like(doc1),
            }
        },
    ]

    out = m.batch_input_tensors(batch)

    assert out["input_ids"].shape == (5, token)
    assert out["attention_mask"].shape == (5, token)
    assert torch.equal(out["input_ids"], torch.cat([doc0, doc1], dim=0))


def test_get_token_embeddings_unpacks_rows_back_to_each_document(
    stub, monkeypatch
):
    """The other half of the pack/unpack contract: after ``batch_input_tensors``
    packs all chunks into one ``[sum(n_chunks), token]`` tensor and the base
    model runs over it, ``get_token_embeddings`` must slice the output rows back
    to the *right* document via ``doc_id.shape[-1]`` — doc 0 gets rows [0, 1],
    doc 1 gets rows [2, 3, 4], with no cross-contamination.
    """
    token, hidden = 4, 6

    def fake_base_model(input_ids, attention_mask):
        # Behave like a real transformer: it requires a 2-D [n_seq, seq_len]
        # input (this unpacking raises if batch_input_tensors regresses to 1-D)
        # and emits one [seq_len, hidden] row per sequence, marked by its global
        # position so routing back to documents is traceable.
        n_seq, seq_len = input_ids.shape
        lhs = torch.zeros(n_seq, seq_len, hidden)
        for r in range(n_seq):
            lhs[r] = float(r)
        return types.SimpleNamespace(last_hidden_state=lhs)

    received: list[list[float]] = []

    def spy_aggregate(outs, masks):
        # Record which global rows this document received; return one row per
        # chunk so pad_sequence recovers the per-document length.
        received.append(outs[:, 0, 0].tolist())
        return outs[:, 0, :]

    monkeypatch.setattr(
        "d3text.models.models.aggregate_embeddings", spy_aggregate
    )

    m = stub(
        Model,
        device="cpu",
        amp_dtype=torch.bfloat16,
        base_model=fake_base_model,
    )

    def item(pmid, n_chunks):
        return {
            "id": torch.tensor(pmid),
            "doc_id": torch.zeros(n_chunks, dtype=torch.uint8),
            "sequence": {
                "input_ids": torch.zeros(n_chunks, token, dtype=torch.long),
                "attention_mask": torch.ones(n_chunks, token, dtype=torch.long),
            },
        }

    batch = [item(100, 2), item(200, 3)]

    embeddings, masks = m.get_token_embeddings(batch)

    # Reconstruction: each document received exactly its own contiguous rows.
    assert received == [[0.0, 1.0], [2.0, 3.0, 4.0]]
    # Padded to the longest document (3 chunks); mask reflects per-doc length.
    assert tuple(embeddings.shape) == (2, 3, hidden)
    assert masks.tolist() == [[True, True, False], [True, True, True]]


# --------------------------------------------------------------------------- #
# Model.get_loss_weights                                                       #
# --------------------------------------------------------------------------- #
def test_get_loss_weights_without_ramp(stub):
    m = stub(Model, ramp_epochs=0)
    assert m.get_loss_weights(0) == (1.0, 1.0)
    assert m.get_loss_weights(50) == (1.0, 1.0)


def test_get_loss_weights_ramps_relation_weight_monotonically(stub):
    m = stub(Model, ramp_epochs=4)
    weights = [m.get_loss_weights(e) for e in range(6)]
    w_ent = [w[0] for w in weights]
    w_rel = [w[1] for w in weights]
    assert w_ent == [1.0] * 6  # entity weight is held at 1.0
    assert w_rel == sorted(w_rel)  # non-decreasing
    assert w_rel[0] == pytest.approx(0.1)  # starts at w0
    assert w_rel[-1] == pytest.approx(1.0)  # saturates at 1.0


# --------------------------------------------------------------------------- #
# Model.early_stop                                                             #
# --------------------------------------------------------------------------- #
def _early_stopper(stub, patience):
    return stub(
        Model,
        best_val_loss=float("inf"),
        stop_counter=0,
        config=types.SimpleNamespace(patience=patience),
    )


def test_early_stop_never_triggers_on_improvement(stub):
    m = _early_stopper(stub, patience=2)
    stops = [
        m.early_stop(v, save_checkpoint=False) for v in (5.0, 4.0, 3.0, 2.0)
    ]
    assert stops == [False, False, False, False]
    assert m.stop_counter == 0
    assert m.best_val_loss == 2.0


def test_early_stop_triggers_after_patience_exceeded(stub):
    m = _early_stopper(stub, patience=2)
    stops = [
        m.early_stop(v, save_checkpoint=False) for v in (1.0, 2.0, 3.0, 4.0)
    ]
    # improvement, then patience(2) tolerated increases, then stop
    assert stops == [False, False, False, True]
    assert m.best_val_loss == 1.0  # best preserved


# --------------------------------------------------------------------------- #
# initialize_classifier_bias                                                   #
# --------------------------------------------------------------------------- #
def test_initialize_classifier_bias_sets_logits_and_unk_tail():
    linear = torch.nn.Linear(4, 3)
    initialize_classifier_bias(
        linear, torch.tensor([0.5, 0.1])
    )  # unk_prior=0.1
    bias = linear.bias.detach()
    assert bias[0].item() == pytest.approx(0.0, abs=1e-5)  # logit(0.5)
    logit_01 = math.log(0.1) - math.log1p(-0.1)
    assert bias[1].item() == pytest.approx(logit_01, abs=1e-4)
    assert bias[2].item() == pytest.approx(logit_01, abs=1e-4)  # UNK tail slot


def test_initialize_classifier_bias_rejects_wrong_length():
    with pytest.raises(ValueError):
        # 3 freqs but out_features-1 == 2
        initialize_classifier_bias(
            torch.nn.Linear(4, 3), torch.tensor([0.5, 0.1, 0.2])
        )


def test_initialize_classifier_bias_seeds_the_sentinel_by_index():
    """The frequencies fill the supervised columns *around* the sentinel, which
    is seeded from the prior — so moving the sentinel off the tail moves both.
    """
    linear = torch.nn.Linear(4, 3)
    initialize_classifier_bias(
        linear, torch.tensor([0.5, 0.1]), sentinel_index=0
    )
    bias = linear.bias.detach()
    logit_01 = math.log(0.1) - math.log1p(-0.1)
    assert bias[0].item() == pytest.approx(logit_01, abs=1e-4)  # sentinel prior
    assert bias[1].item() == pytest.approx(0.0, abs=1e-5)  # logit(0.5)
    assert bias[2].item() == pytest.approx(logit_01, abs=1e-4)  # logit(0.1)


def test_initialize_classifier_bias_without_sentinel_fills_every_column():
    linear = torch.nn.Linear(4, 2)
    initialize_classifier_bias(
        linear, torch.tensor([0.5, 0.5]), sentinel_index=None
    )
    assert linear.bias.detach().tolist() == pytest.approx([0.0, 0.0], abs=1e-5)


# --------------------------------------------------------------------------- #
# ClassificationHead                                                           #
# --------------------------------------------------------------------------- #
def test_classification_head_returns_entity_and_class_logits():
    head = ClassificationHead(input_size=8, n_entities=5, n_classes=3)
    entity_logits, class_logits = head(torch.randn(2, 8))
    assert tuple(entity_logits.shape) == (2, 5)
    assert tuple(class_logits.shape) == (2, 3)


def test_classification_head_rejects_bad_entity_freqs():
    with pytest.raises(ValueError):
        # entity_freqs length must be n_entities - 1 == 4
        ClassificationHead(
            input_size=8, n_entities=5, n_classes=3, entity_freqs=torch.rand(3)
        )


# --------------------------------------------------------------------------- #
# BiaffineRelationClassifier.forward                                           #
# --------------------------------------------------------------------------- #
def test_biaffine_forward_shape_and_gradient():
    model = BiaffineRelationClassifier(hidden_size=8, num_relations=3)
    out = model(torch.randn(4, 8), torch.randn(4, 8))
    assert tuple(out.shape) == (4, 3)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert model.bilinear.grad is not None


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
# ETEBrendaModel.align_relation_predictions                                    #
# --------------------------------------------------------------------------- #
def _align_stub(stub):
    return stub(
        ETEBrendaModel,
        entity_logits_pooling="logsumexp",
        entity_to_index={"A": 0, "B": 1},
        relations_none_index=2,
    )


def _rel_meta():
    # two candidate rows for the same (doc=0, subj=0, obj=1) triple
    return {
        "sequence": torch.tensor([0, 0]),
        "arg_pred_i": torch.tensor([0, 0]),
        "arg_pred_j": torch.tensor([1, 1]),
    }


def test_align_pools_duplicate_rows_and_uses_gold_label(stub):
    m = _align_stub(stub)
    rel_logits = torch.randn(2, 3)
    gold = [
        IndexedRelation(docix=0, subject="A", object="B", label=torch.tensor(0))
    ]
    meta, pooled_logits, targets = m.align_relation_predictions(
        gold, _rel_meta(), rel_logits
    )
    assert pooled_logits.shape[0] == 1  # two rows pooled into one
    assert pooled_logits.shape[1] == 3  # relation width preserved
    assert targets.tolist() == [0]  # gold "HasEnzyme"
    assert meta["arg_pred_i"].tolist() == [0]
    assert meta["arg_pred_j"].tolist() == [1]


def test_align_defaults_to_none_when_gold_entity_not_indexed(stub):
    m = _align_stub(stub)
    # subject "Z" is absent from entity_to_index -> gold is dropped
    gold = [
        IndexedRelation(docix=0, subject="Z", object="B", label=torch.tensor(0))
    ]
    _, _, targets = m.align_relation_predictions(
        gold, _rel_meta(), torch.randn(2, 3)
    )
    assert targets.tolist() == [2]  # relations_none_index


def test_align_returns_none_for_empty_logits(stub):
    m = _align_stub(stub)
    assert m.align_relation_predictions([], _rel_meta(), None) is None


# --------------------------------------------------------------------------- #
# Relation-loss class weighting                                                #
# --------------------------------------------------------------------------- #
def test_balanced_class_weights_are_inverse_frequency():
    weights = balanced_class_weights(
        torch.tensor([2, 2, 2, 0]),
        num_classes=3,  # three `none`, one positive
    )
    assert torch.allclose(weights, torch.tensor([4 / 3, 4 / 3, 4 / 9]))
    assert weights[0] > weights[2]  # the rare class outweighs `none`


def test_balanced_class_weights_stay_finite_when_a_class_is_absent():
    weights = balanced_class_weights(torch.tensor([0, 0]), num_classes=3)
    assert torch.isfinite(weights).all()


def test_focal_cross_entropy_with_zero_gamma_is_plain_cross_entropy():
    preds, targets = torch.randn(6, 3), torch.randint(0, 3, (6,))
    assert torch.isclose(
        focal_cross_entropy(preds, targets, gamma=0.0),
        torch.nn.functional.cross_entropy(preds, targets),
    )


def test_focal_cross_entropy_suppresses_easy_pairs_far_more_than_hard_ones():
    targets = torch.tensor([2])
    easy = torch.tensor([[-6.0, -6.0, 6.0]])  # p_t ~= 1: already learned
    hard = torch.tensor([[0.0, 0.0, 0.0]])  # p_t == 1/3: uninformed

    def suppression(preds):
        focal = focal_cross_entropy(preds, targets, gamma=2.0)
        return (
            focal / torch.nn.functional.cross_entropy(preds, targets)
        ).item()

    assert suppression(easy) < 1e-6
    assert suppression(hard) > 0.4


def _relation_loss_stub(stub, weighting):
    return stub(
        ETEBrendaModel,
        device="cpu",
        entity_logits_pooling="logsumexp",
        entity_to_index={"A": 0, "B": 1},
        relations_none_index=2,
        num_relations=3,
        relation_label_smoothing=0.0,
        relation_loss_weighting=weighting,
        relation_focal_gamma=2.0,
    )


def _imbalanced_pairs(n_none):
    """One mispredicted positive plus `n_none` confidently-correct `none` pairs.

    Mimics what the entropy hard mask actually proposes: a flood of easy
    negatives around the sparse gold relations. Every triple is distinct, so
    alignment pools them 1:1 and the loss sees exactly these rows.
    """
    gold = [
        IndexedRelation(docix=0, subject="A", object="B", label=torch.tensor(0))
    ]
    meta = {
        "sequence": torch.zeros(n_none + 1, dtype=torch.long),
        "arg_pred_i": torch.tensor([0] + [k + 2 for k in range(n_none)]),
        "arg_pred_j": torch.tensor([1] + [k + 3 for k in range(n_none)]),
    }
    logits = torch.tensor(
        [[-6.0, 0.0, 6.0]]  # gold "HasEnzyme", confidently called `none`
        + [[-6.0, -6.0, 6.0]] * n_none  # `none`, confidently correct
    )
    return gold, meta, logits


def test_unweighted_relation_loss_is_diluted_by_none_pairs(stub):
    """The smell itself: the same mistake on the same gold relation costs the
    model ~8x less once the mask floods the batch with easy negatives."""
    m = _relation_loss_stub(stub, "unweighted")
    few = m.compute_relation_loss(*_imbalanced_pairs(3))
    many = m.compute_relation_loss(*_imbalanced_pairs(30))
    assert many < few / 5


@pytest.mark.parametrize("weighting", ("balanced", "focal"))
def test_weighting_keeps_the_positive_from_being_diluted(stub, weighting):
    m = _relation_loss_stub(stub, weighting)
    few = m.compute_relation_loss(*_imbalanced_pairs(3))
    many = m.compute_relation_loss(*_imbalanced_pairs(30))
    assert torch.isclose(few, many, rtol=0.02)


def test_relation_loss_weighting_defaults_to_unweighted():
    assert ModelConfig().relation_loss_weighting == "unweighted"


def test_relation_loss_weighting_rejects_an_unknown_scheme():
    with pytest.raises(ValidationError):
        ModelConfig(relation_loss_weighting="bogus")


# --------------------------------------------------------------------------- #
# ETEBrendaModel._compute_relations_vectorized                                 #
# --------------------------------------------------------------------------- #
def _relations_stub(stub):
    return stub(
        ETEBrendaModel,
        device="cpu",
        relation_classifier=BiaffineRelationClassifier(
            hidden_size=8, num_relations=3
        ),
    )


def test_compute_relations_one_pair_for_two_distinct_entities(stub):
    m = _relations_stub(stub)
    positions = torch.tensor(
        [[0, 0], [0, 1]], dtype=torch.int64
    )  # doc 0, tokens 0/1
    reprs = torch.randn(2, 8)
    max_indices = torch.tensor(
        [[5, 7]], dtype=torch.int64
    )  # token 0->5, token 1->7
    meta, logits = m._compute_relations_vectorized(
        positions, reprs, max_indices
    )
    assert tuple(logits.shape) == (1, 3)
    assert meta["arg_pred_i"].tolist() == [5]
    assert meta["arg_pred_j"].tolist() == [7]


def test_compute_relations_none_for_single_entity(stub):
    m = _relations_stub(stub)
    positions = torch.tensor([[0, 0], [0, 1]], dtype=torch.int64)
    reprs = torch.randn(2, 8)
    max_indices = torch.tensor(
        [[5, 5]], dtype=torch.int64
    )  # both tokens -> entity 5
    assert (
        m._compute_relations_vectorized(positions, reprs, max_indices) is None
    )


# --------------------------------------------------------------------------- #
# ETEBrendaModel.ground_truth (relation loop)                                  #
# --------------------------------------------------------------------------- #
def test_ground_truth_builds_indexed_relation_from_argmax(stub):
    m = stub(ETEBrendaModel, device="cpu")
    batch = [
        {
            "entities": torch.tensor([[1, 0]]),
            "classes": torch.tensor([[1, 0]]),
            "relations": [{("A", "B"): torch.tensor([0, 1, 0])}],  # argmax == 1
        }
    ]
    _, _, relations = m.ground_truth(batch)
    assert len(relations) == 1
    rel = relations[0]
    assert (rel.docix, rel.subject, rel.object) == (0, "A", "B")
    assert int(rel.label) == 1


def test_ground_truth_yields_no_relations_for_empty_dict(stub):
    m = stub(ETEBrendaModel, device="cpu")
    batch = [
        {
            "entities": torch.tensor([[1, 0]]),
            "classes": torch.tensor([[1, 0]]),
            "relations": [{}],
        }
    ]
    _, _, relations = m.ground_truth(batch)
    assert relations == []


# --------------------------------------------------------------------------- #
# ordered_entities / entity-column alignment                                   #
# --------------------------------------------------------------------------- #
def test_label_columns_locates_the_sentinel_and_lists_the_rest():
    index, columns = label_columns(["e0", "UNK", "e1"], "UNK")
    assert index == 1
    assert columns.tolist() == [0, 2]
    assert columns.dtype == torch.int64


def test_label_columns_rejects_a_missing_sentinel():
    with pytest.raises(ValueError):
        label_columns(["c0", "c1"], "OOS")


def test_ordered_entities_follows_the_index_not_insertion_order():
    assert ordered_entities({"b": 1, "c": 2, "a": 0}) == ["a", "b", "c"]


@pytest.mark.parametrize(
    "entity_index",
    [
        {"a": 0, "b": 2},  # gap: no entity owns column 1
        {"a": 1, "b": 2},  # does not start at 0
        {"a": 0, "b": 0},  # two entities claiming one column
    ],
    ids=["gap", "offset", "duplicate"],
)
def test_ordered_entities_rejects_non_contiguous_index(entity_index):
    with pytest.raises(ValueError, match="contiguous"):
        ordered_entities(entity_index)


def test_entities_stay_aligned_with_entity_index_when_classes_overlap(
    patch_base_model,
):
    """`entities[i]` must name the entity that entity logit column `i` scores.

    An entity belonging to two classes is one entity and one column. Deriving
    the list by flattening the per-class entity sets counts it twice, widening
    the entity head past the target width.
    """
    model = BrendaClassificationModel(
        classes={
            "enzymes": {"enz1", "shared"},
            "bacteria": {"bac1", "shared"},
        },
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
