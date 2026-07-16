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
from torch.utils.data import DataLoader

from d3text.models.config import ModelConfig
from d3text.models.model_types import IndexedRelation
from d3text.models.models import (
    BiaffineRelationClassifier,
    BrendaModel,
    ClassificationHead,
    Logits,
    Model,
    NERClassificationModel,
    RelationExtractor,
    RelationPairs,
    Targets,
    balanced_class_weights,
    focal_cross_entropy,
    initialize_classifier_bias,
    label_columns,
    ordered_entities,
)


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


def test_batch_input_tensors_concatenates_chunks_into_2d(stub):
    """Per-document ``[n_chunks, token]`` sequences must concat along dim 0 into
    a single ``[sum(n_chunks), token]`` tensor per key.

    ``get_token_embeddings`` slices the base-model output back into
    per-document chunks via ``doc_id.shape[-1]``, so this contract must be 2-D —
    and the documents' chunk counts differ, which is exactly what a concat along
    dim 0 allows and a stack does not.
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
        "d3text.models.base.aggregate_embeddings", spy_aggregate
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


def _caching_stub(stub, calls, hidden, *, training):
    """A `Model` whose base model records how many sequences it was handed."""

    def fake_base_model(input_ids, attention_mask):
        n_seq, seq_len = input_ids.shape
        calls.append(n_seq)
        return types.SimpleNamespace(
            last_hidden_state=torch.ones(n_seq, seq_len, hidden)
        )

    return stub(
        Model,
        device="cpu",
        amp_dtype=torch.bfloat16,
        base_model=fake_base_model,
        training=training,
    )


def _cache_item(pmid, n_chunks, token=4):
    return {
        "id": torch.tensor(pmid),
        "doc_id": torch.zeros(n_chunks, dtype=torch.uint8),
        "sequence": {
            "input_ids": torch.zeros(n_chunks, token, dtype=torch.long),
            "attention_mask": torch.ones(n_chunks, token, dtype=torch.long),
        },
    }


def test_get_token_embeddings_serves_cached_documents_without_re_embedding(
    stub, monkeypatch, embeddings_cache
):
    """With the CPU cache on, a document embedded once never reaches the frozen
    base model again — that is the whole point of the cache.

    Pinned explicitly because whether this branch runs at all is decided by
    `cpu_embeddings_cache_size` in the machine-local, untracked `config.toml`:
    left to the environment, a developer with a cache configured and CI with
    none run different code here, and neither covers it on purpose.
    """
    hidden = 6
    calls: list[int] = []
    monkeypatch.setattr(
        "d3text.models.base.aggregate_embeddings",
        lambda outs, masks: outs[:, 0, :],
    )
    model = _caching_stub(stub, calls, hidden, training=True)
    batch = [_cache_item(100, 2), _cache_item(200, 3)]

    first, _ = model.get_token_embeddings(batch)

    assert calls == [5], (
        "both documents' 2 + 3 chunks go in one base-model pass"
    )
    assert embeddings_cache.get(100) is not None
    assert embeddings_cache.get(200) is not None

    second, _ = model.get_token_embeddings(batch)

    assert calls == [5], "a cached document must not be embedded a second time"
    assert torch.equal(second, first)


def test_get_token_embeddings_does_not_cache_outside_training(
    stub, monkeypatch, embeddings_cache
):
    """Evaluation must not populate the cache: it is sized for the training
    set, and an eval pass would evict the documents training is reusing."""
    calls: list[int] = []
    monkeypatch.setattr(
        "d3text.models.base.aggregate_embeddings",
        lambda outs, masks: outs[:, 0, :],
    )
    model = _caching_stub(stub, calls, 6, training=False)

    model.get_token_embeddings([_cache_item(100, 2)])

    assert embeddings_cache.get(100) is None


def test_relation_loss_weight_without_ramp(stub):
    m = stub(RelationExtractor, ramp_epochs=0)
    assert m.loss_weight(0) == 1.0
    assert m.loss_weight(50) == 1.0


def test_relation_loss_weight_ramps_monotonically(stub):
    m = stub(RelationExtractor, ramp_epochs=4)
    weights = [m.loss_weight(e) for e in range(6)]
    assert weights == sorted(weights)  # non-decreasing
    assert weights[0] == pytest.approx(0.1)  # starts at w0
    assert weights[-1] == pytest.approx(1.0)  # saturates at 1.0


def test_the_ramp_schedule_belongs_to_the_relation_head(stub):
    """The ramp is the relation objective's own, and lives on the component that
    owns that objective. A model has no schedule of its own to reach for and
    land on a head that should have been training at full weight all along —
    and a model with no relation extractor has no ramp at all.
    """
    assert hasattr(stub(RelationExtractor, ramp_epochs=4), "loss_weight")
    assert not hasattr(stub(Model, ramp_epochs=4), "loss_weight")
    assert not hasattr(stub(BrendaModel, ramp_epochs=4), "loss_weight")


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


def test_biaffine_forward_shape_and_gradient():
    model = BiaffineRelationClassifier(hidden_size=8, num_relations=3)
    out = model(torch.randn(4, 8), torch.randn(4, 8))
    assert tuple(out.shape) == (4, 3)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert model.bilinear.grad is not None


def _loss_stub(
    stub, entities=("e0", "e1", "e2", "UNK"), classes=("c0", "c1", "OOS")
):
    """A stub carrying the sentinel columns the losses look up by name. The
    defaults put UNK/OOS last, as the BRENDA models do; pass them elsewhere to
    prove nothing depends on that position."""
    unk_index, entity_columns = label_columns(list(entities), "UNK")
    oos_index, class_columns = label_columns(list(classes), "OOS")
    return stub(
        BrendaModel,
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


def _consistency_stub(stub, weight):
    unk_index, entity_columns = label_columns(["e0", "e1", "UNK"], "UNK")
    oos_index, class_columns = label_columns(["c0", "c1", "OOS"], "OOS")
    return stub(
        BrendaModel,
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


def _align_stub(stub):
    return stub(
        RelationExtractor,
        pooling="logsumexp",
        entity_to_index={"A": 0, "B": 1},
        none_index=2,
    )


def _duplicate_pairs(logits=None):
    """Two candidate rows for the same (doc 0, subj 0, obj 1) triple."""
    return RelationPairs(
        meta={
            "sequence": torch.tensor([0, 0]),
            "arg_pred_i": torch.tensor([0, 0]),
            "arg_pred_j": torch.tensor([1, 1]),
        },
        logits=torch.randn(2, 3) if logits is None else logits,
    )


def test_align_pools_duplicate_rows_and_uses_gold_label(stub):
    m = _align_stub(stub)
    gold = [
        IndexedRelation(docix=0, subject="A", object="B", label=torch.tensor(0))
    ]
    meta, pooled_logits, targets = m.align(gold, _duplicate_pairs())
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
    _, _, targets = m.align(gold, _duplicate_pairs())
    assert targets.tolist() == [2]  # the `none` column


def test_align_returns_none_without_pairs(stub):
    m = _align_stub(stub)
    assert m.align([], None) is None
    assert m.align([], _duplicate_pairs(logits=torch.zeros(0, 3))) is None


# The aligner scores only the pairs the entity head proposed, so gold it never
# proposed leaves no row. Unless the metrics add it back, it is not a false
# negative -- it is absent, and relation F1 is computed over a denominator the
# model chose for itself.
def _missed_stub(stub):
    return stub(
        RelationExtractor,
        pooling="logsumexp",
        entity_to_index={"A": 0, "B": 1, "C": 2},
        labels=("HasEnzyme", "HasSpecies", "none"),
        none_index=2,
    )


HAS_ENZYME, HAS_SPECIES, NONE = 0, 1, 2


def _gold(subject, object, label, docix=0):
    return IndexedRelation(
        docix=docix, subject=subject, object=object, label=torch.tensor(label)
    )


def test_gold_with_a_scored_row_is_not_missed(stub):
    m = _missed_stub(stub)
    scored = {
        "sequence": torch.tensor([0]),
        "arg_pred_i": torch.tensor([0]),
        "arg_pred_j": torch.tensor([1]),
    }
    assert m.unscored_gold([_gold("A", "B", HAS_ENZYME)], scored) == (
        [],
        [],
    )


def test_gold_never_proposed_is_missed_even_when_other_pairs_were(stub):
    m = _missed_stub(stub)
    # (A, B) was proposed; (A, C) was not -- one scored row is not licence to
    # forget the other gold relation.
    scored = {
        "sequence": torch.tensor([0]),
        "arg_pred_i": torch.tensor([0]),
        "arg_pred_j": torch.tensor([1]),
    }
    not_proposed, out_of_vocabulary = m.unscored_gold(
        [_gold("A", "B", HAS_ENZYME), _gold("A", "C", HAS_SPECIES)], scored
    )
    assert not_proposed == [HAS_SPECIES]
    assert out_of_vocabulary == []


def test_gold_in_another_document_is_missed(stub):
    m = _missed_stub(stub)
    # Same (subject, object), different document: the row scored for doc 0 says
    # nothing about doc 1.
    scored = {
        "sequence": torch.tensor([0]),
        "arg_pred_i": torch.tensor([0]),
        "arg_pred_j": torch.tensor([1]),
    }
    not_proposed, _ = m.unscored_gold(
        [_gold("A", "B", HAS_ENZYME, docix=1)], scored
    )
    assert not_proposed == [HAS_ENZYME]


def test_gold_with_unindexed_entity_is_reported_out_of_vocabulary(stub):
    m = _missed_stub(stub)
    # "Z" is absent from entity_to_index, so no relation head could ever
    # predict this pair: a real miss, but not one the relation head can fix.
    not_proposed, out_of_vocabulary = m.unscored_gold(
        [_gold("Z", "B", HAS_ENZYME)], None
    )
    assert not_proposed == []
    assert out_of_vocabulary == [HAS_ENZYME]


def test_every_gold_is_missed_when_nothing_was_scored(stub):
    m = _missed_stub(stub)
    not_proposed, out_of_vocabulary = m.unscored_gold(
        [_gold("A", "B", HAS_ENZYME), _gold("A", "C", HAS_SPECIES)], None
    )
    assert not_proposed == [HAS_ENZYME, HAS_SPECIES]
    assert out_of_vocabulary == []


def _true_x_pred_stub(stub, pairs, gold):
    """A `BrendaModel` holding a stubbed relation extractor, whose only real
    behaviour is the relation bookkeeping the metrics depend on."""
    m = stub(BrendaModel, relations=_missed_stub(stub))
    entity_logits = torch.zeros(1, 4)
    class_logits = torch.zeros(1, 3)
    object.__setattr__(
        m,
        "get_batch_logits",
        lambda batch: Logits(entity_logits, class_logits, pairs),
    )
    object.__setattr__(
        m,
        "ground_truth",
        lambda batch: Targets(torch.zeros(1, 3), torch.zeros(1, 2), gold),
    )
    return m


def _candidate_pair_favouring_has_enzyme():
    """One candidate row for (doc 0, A, B), predicted HasEnzyme."""
    return RelationPairs(
        meta={
            "sequence": torch.tensor([0]),
            "arg_pred_i": torch.tensor([0]),
            "arg_pred_j": torch.tensor([1]),
        },
        logits=torch.tensor([[10.0, 0.0, 0.0]]),
    )


def test_true_x_pred_counts_unproposed_gold_as_a_false_negative(stub):
    gold = [_gold("A", "B", HAS_ENZYME), _gold("A", "C", HAS_SPECIES)]
    m = _true_x_pred_stub(stub, _candidate_pair_favouring_has_enzyme(), gold)

    relations = m.compute_batch_true_x_pred([{}])["relations"]

    # The proposed pair is scored on its logits; the unproposed one counts as
    # `none`, rather than disappearing because it has no row.
    assert relations["true"].tolist() == [HAS_ENZYME, HAS_SPECIES]
    assert relations["pred"].tolist() == [HAS_ENZYME, NONE]


def test_true_x_pred_counts_out_of_vocabulary_gold_as_a_false_negative(stub):
    gold = [_gold("A", "B", HAS_ENZYME), _gold("Z", "B", HAS_SPECIES)]
    m = _true_x_pred_stub(stub, _candidate_pair_favouring_has_enzyme(), gold)

    relations = m.compute_batch_true_x_pred([{}])["relations"]

    assert relations["true"].tolist() == [HAS_ENZYME, HAS_SPECIES]
    assert relations["pred"].tolist() == [HAS_ENZYME, NONE]


def test_true_x_pred_counts_all_gold_when_no_pairs_were_proposed(stub):
    gold = [_gold("A", "B", HAS_ENZYME), _gold("A", "C", HAS_SPECIES)]
    m = _true_x_pred_stub(stub, None, gold)

    relations = m.compute_batch_true_x_pred([{}])["relations"]

    assert relations["true"].tolist() == [HAS_ENZYME, HAS_SPECIES]
    assert relations["pred"].tolist() == [NONE, NONE]


def _evaluate_stub(stub, pairs, gold):
    """A model whose only real behaviour is the relation bookkeeping.

    Entities are ``A B C UNK`` and classes ``enzyme species OOS``, so `drop_unk`
    and `drop_oos` narrow the logits to the width the targets carry.
    """
    m = _true_x_pred_stub(stub, pairs, gold)
    object.__setattr__(m, "eval", lambda: None)
    object.__setattr__(m, "classes", ["enzyme", "species", "OOS"])
    object.__setattr__(m, "entity_columns", torch.tensor([0, 1, 2]))
    object.__setattr__(m, "class_columns", torch.tensor([0, 1]))
    object.__setattr__(
        m,
        "ground_truth",
        lambda batch: Targets(
            torch.tensor([[1.0, 1.0, 0.0]]),
            torch.tensor([[1.0, 1.0]]),
            gold,
        ),
    )
    return m


def _single_batch_loader():
    """One batch of one (empty) document.

    `evaluate_model` is typed to a real `DataLoader` and beartype enforces it,
    so the batch has to arrive through one; the stubbed `get_batch_logits` and
    `ground_truth` ignore its contents.
    """
    return DataLoader([{}], batch_size=1, collate_fn=list)


def _report_row(output, label):
    row = next(
        line
        for line in output.splitlines()
        if line.strip().startswith(f"{label} ")
    )
    _, precision, recall, f1, support = row.split()
    # sklearn formats support as an int or a float depending on the report.
    return float(precision), float(recall), float(f1), int(float(support))


def test_evaluate_scores_unproposed_gold_against_the_model(stub, capsys):
    # The head proposes (A, B) and labels it HasEnzyme correctly; the gold
    # HasSpecies pair (A, C) it never proposed at all.
    gold = [_gold("A", "B", HAS_ENZYME), _gold("A", "C", HAS_SPECIES)]
    m = _evaluate_stub(stub, _candidate_pair_favouring_has_enzyme(), gold)

    m.evaluate_model(_single_batch_loader())

    out = capsys.readouterr().out
    assert "gold: 2" in out
    assert "missed, never proposed: 1" in out

    # The missed relation must reach the report as a false negative: without it
    # the model scores a perfect HasEnzyme and HasSpecies is simply not there.
    _, recall, _, support = _report_row(out, "HasSpecies")
    assert support == 1
    assert recall == 0.0


def test_evaluate_reports_gold_when_no_pairs_were_proposed(stub, capsys):
    gold = [_gold("A", "B", HAS_ENZYME)]
    m = _evaluate_stub(stub, None, gold)

    m.evaluate_model(_single_batch_loader())

    out = capsys.readouterr().out
    assert "missed, never proposed: 1" in out
    # A split on which the head proposes nothing scores zero, rather than
    # silently reporting no relations at all.
    _, recall, _, support = _report_row(out, "HasEnzyme")
    assert support == 1
    assert recall == 0.0


def test_evaluate_separates_out_of_vocabulary_gold_from_unproposed_gold(
    stub, capsys
):
    gold = [_gold("Z", "B", HAS_ENZYME), _gold("A", "C", HAS_SPECIES)]
    m = _evaluate_stub(stub, _candidate_pair_favouring_has_enzyme(), gold)

    m.evaluate_model(_single_batch_loader())

    out = capsys.readouterr().out
    assert "missed, never proposed: 1" in out
    assert "missed, entity out of vocabulary: 1" in out


def test_ner_evaluate_drops_the_named_oos_column_not_the_last(stub, capsys):
    """The NER report must locate OOS by name, as `BrendaModel.evaluate_model`
    does. Narrowing the logits to the target width by *truncation* instead would
    keep OOS's column here and drop c1's, while still labelling the rows with
    `known_classes` — so every class would be scored against its neighbour's
    logits, silently, since the widths still line up.
    """
    m = stub(
        NERClassificationModel,
        classes=["OOS", "c0", "c1"],
        class_columns=torch.tensor([1, 2]),
        eval=lambda: None,
        # OOS fires on both documents; c0 holds only on the second, c1 only on
        # the first, so truncation reads c0 and c1 off the wrong columns.
        get_batch_logits=lambda batch: torch.tensor(
            [[10.0, -10.0, 10.0], [10.0, 10.0, -10.0]]
        ),
        ground_truth=lambda batch: torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
    )

    m.evaluate_model(_single_batch_loader())

    out = capsys.readouterr().out
    assert "micro-F1: 1.0" in out
    for label in ("c0", "c1"):
        assert _report_row(out, label) == (1.0, 1.0, 1.0, 1)


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
        RelationExtractor,
        pooling="logsumexp",
        entity_to_index={"A": 0, "B": 1},
        none_index=2,
        num_relations=3,
        label_smoothing=0.0,
        loss_weighting=weighting,
        focal_gamma=2.0,
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
    pairs = RelationPairs(
        meta={
            "sequence": torch.zeros(n_none + 1, dtype=torch.long),
            "arg_pred_i": torch.tensor([0] + [k + 2 for k in range(n_none)]),
            "arg_pred_j": torch.tensor([1] + [k + 3 for k in range(n_none)]),
        },
        logits=torch.tensor(
            [[-6.0, 0.0, 6.0]]  # gold "HasEnzyme", confidently called `none`
            + [[-6.0, -6.0, 6.0]] * n_none  # `none`, confidently correct
        ),
    )
    return gold, pairs


def test_unweighted_relation_loss_is_diluted_by_none_pairs(stub):
    """The smell itself: the same mistake on the same gold relation costs the
    model ~8x less once the mask floods the batch with easy negatives."""
    m = _relation_loss_stub(stub, "unweighted")
    few = m.loss(*_imbalanced_pairs(3))
    many = m.loss(*_imbalanced_pairs(30))
    assert many < few / 5


@pytest.mark.parametrize("weighting", ("balanced", "focal"))
def test_weighting_keeps_the_positive_from_being_diluted(stub, weighting):
    m = _relation_loss_stub(stub, weighting)
    few = m.loss(*_imbalanced_pairs(3))
    many = m.loss(*_imbalanced_pairs(30))
    assert torch.isclose(few, many, rtol=0.02)


def test_relation_loss_weighting_defaults_to_unweighted():
    assert ModelConfig().relation_loss_weighting == "unweighted"


def test_relation_loss_weighting_rejects_an_unknown_scheme():
    with pytest.raises(ValidationError):
        ModelConfig(relation_loss_weighting="bogus")


def _relations_stub(stub):
    return stub(
        RelationExtractor,
        classifier=BiaffineRelationClassifier(hidden_size=8, num_relations=3),
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
    meta, logits = m._pairs_from_positions(positions, reprs, max_indices)
    assert tuple(logits.shape) == (1, 3)
    assert meta["arg_pred_i"].tolist() == [5]
    assert meta["arg_pred_j"].tolist() == [7]


def test_compute_relations_none_for_single_entity(stub):
    """Two token positions predicting the *same* entity are one argument, not
    two, so they propose no pair."""
    m = _relations_stub(stub)
    positions = torch.tensor([[0, 0], [0, 1]], dtype=torch.int64)
    reprs = torch.randn(2, 8)
    max_indices = torch.tensor(
        [[5, 5]], dtype=torch.int64
    )  # both tokens -> entity 5
    assert m._pairs_from_positions(positions, reprs, max_indices) is None


def _ground_truth_stub(stub, relations):
    """A model that does or does not extract relations. `ground_truth` reads the
    corpus' relations only when there is a head to supervise with them."""
    return stub(
        BrendaModel,
        device="cpu",
        relations=_missed_stub(stub) if relations else None,
    )


def test_ground_truth_builds_indexed_relation_from_argmax(stub):
    m = _ground_truth_stub(stub, relations=True)
    batch = [
        {
            "entities": torch.tensor([1, 0]),
            "classes": torch.tensor([1, 0]),
            "relations": [{("A", "B"): torch.tensor([0, 1, 0])}],  # argmax == 1
        }
    ]
    truth = m.ground_truth(batch)
    assert len(truth.relations) == 1
    rel = truth.relations[0]
    assert (rel.docix, rel.subject, rel.object) == (0, "A", "B")
    assert int(rel.label) == 1


def test_ground_truth_yields_no_relations_for_empty_dict(stub):
    m = _ground_truth_stub(stub, relations=True)
    batch = [
        {
            "entities": torch.tensor([1, 0]),
            "classes": torch.tensor([1, 0]),
            "relations": [{}],
        }
    ]
    assert m.ground_truth(batch).relations == []


def test_ground_truth_ignores_relations_without_a_relation_head(stub):
    """A model with no relation extractor has nothing to supervise with the
    corpus' relations, so it does not read them — and its `Targets` still has the
    field, empty, rather than one arity for this model and another for that one.
    """
    m = _ground_truth_stub(stub, relations=False)
    batch = [
        {
            "entities": torch.tensor([1, 0]),
            "classes": torch.tensor([1, 0]),
            "relations": [{("A", "B"): torch.tensor([0, 1, 0])}],
        }
    ]
    assert m.ground_truth(batch).relations == []


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
    patch_base_model, tiny_schema
):
    """`entities[i]` must name the entity that entity logit column `i` scores.

    An entity belonging to two classes is one entity and one column. Deriving
    the list by flattening the per-class entity sets counts it twice, widening
    the entity head past the target width.
    """
    model = BrendaModel(
        schema=tiny_schema,
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
