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

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from torch.utils.data import default_collate

from cacheout import Cache
from d3text.models.config import ModelConfig
from d3text.models.model_types import IndexedRelation
from d3text.utils import aggregate_embeddings
from d3text.models.models import (
    GRAD_CLIP_NORM,
    BiaffineRelationClassifier,
    BrendaClassificationModel,
    ClassificationHead,
    ETEBrendaModel,
    Model,
    Step,
    balanced_class_weights,
    document_token_count,
    embeddings_store,
    epoch_rate_metrics,
    relation_metrics,
    support_metrics,
    focal_cross_entropy,
    get_batch_entities,
    has_bf16_hardware,
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


def test_batch_input_tensors_survives_the_dataloader_collate(stub):
    """The same contract, on the shape a real run actually produces.

    `get_batch_loader` hands the `DataLoader` a `BatchSampler` as its *sampler*,
    so each drawn "index" is a list and the fetched value is already a list of
    per-document dicts. `default_collate` then batches that one-element list and
    stamps a leading 1 onto every field, so `batch_input_tensors` sees
    ``[1, n_chunks, token]``, never the bare ``[n_chunks, token]`` the test
    above builds. Concatenating that on dim 0 stacks documents on the chunk axis
    and raises the moment two of them differ in chunk count — which is every
    real batch. Collating here rather than hand-writing the leading 1 is the
    point: the fixture cannot drift away from what the loader does.
    """
    m = stub(Model)
    token = 4
    doc0 = torch.arange(2 * token).reshape(2, token)  # 2 chunks
    doc1 = torch.arange(3 * token).reshape(3, token)  # 3 chunks
    batch = default_collate(
        [
            [
                {
                    "sequence": {
                        "input_ids": doc,
                        "attention_mask": torch.ones_like(doc),
                    }
                }
                for doc in (doc0, doc1)
            ]
        ]
    )

    assert batch[0]["sequence"]["input_ids"].shape == (1, 2, token)

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


@pytest.mark.parametrize("training", [True, False])
def test_get_token_embeddings_caches_in_both_train_and_eval(
    stub, monkeypatch, training
):
    """A freshly computed document is cached whichever split it came from.

    The write used to be gated on ``self.training``, which read as a policy
    reserving the budget for training documents. It is not one: the cache is a
    single module-global budget and a cached document skips exactly one frozen
    base-model forward per epoch regardless of split, so the gate only kept
    validation permanently cold. With any ``maxsize`` the training pass does
    not exhaust — every ``--limit``ed run — it was the sole condition rejecting
    the write.
    """
    hidden = 6

    def fake_base_model(input_ids, attention_mask):
        n_seq, seq_len = input_ids.shape
        return types.SimpleNamespace(
            last_hidden_state=torch.zeros(n_seq, seq_len, hidden)
        )

    cache = Cache(maxsize=8)
    monkeypatch.setattr(
        "d3text.models.models.cpu_embeddings_cache", cache, raising=False
    )
    monkeypatch.setattr(
        "d3text.models.models.aggregate_embeddings",
        lambda outs, masks: outs[:, 0, :],
    )

    m = stub(
        Model,
        device="cpu",
        amp_dtype=torch.bfloat16,
        base_model=fake_base_model,
        training=training,
    )
    batch = [
        {
            "id": torch.tensor(777),
            "doc_id": torch.zeros(2, dtype=torch.uint8),
            "sequence": {
                "input_ids": torch.zeros(2, 4, dtype=torch.long),
                "attention_mask": torch.ones(2, 4, dtype=torch.long),
            },
        }
    ]

    m.get_token_embeddings(batch)

    assert cache.get(777) is not None

    # The second pass is served from the cache: the base model is not re-run.
    def exploding_base_model(input_ids, attention_mask):
        raise AssertionError("cache miss on a document already cached")

    object.__setattr__(m, "base_model", exploding_base_model)
    m.get_token_embeddings(batch)


def test_get_token_embeddings_does_not_write_to_a_full_cache(stub, monkeypatch):
    """The frozen-once policy is untouched: a full cache rejects new writes
    rather than evicting, which is what keeps the hit rate stable under a
    shuffled sampler."""
    hidden = 6

    def fake_base_model(input_ids, attention_mask):
        n_seq, seq_len = input_ids.shape
        return types.SimpleNamespace(
            last_hidden_state=torch.zeros(n_seq, seq_len, hidden)
        )

    cache = Cache(maxsize=1)
    cache.set(1, torch.zeros(1, hidden))
    monkeypatch.setattr(
        "d3text.models.models.cpu_embeddings_cache", cache, raising=False
    )
    monkeypatch.setattr(
        "d3text.models.models.aggregate_embeddings",
        lambda outs, masks: outs[:, 0, :],
    )

    m = stub(
        Model,
        device="cpu",
        amp_dtype=torch.bfloat16,
        base_model=fake_base_model,
        training=True,
    )
    m.get_token_embeddings(
        [
            {
                "id": torch.tensor(2),
                "doc_id": torch.zeros(1, dtype=torch.uint8),
                "sequence": {
                    "input_ids": torch.zeros(1, 4, dtype=torch.long),
                    "attention_mask": torch.ones(1, 4, dtype=torch.long),
                },
            }
        ]
    )

    assert cache.get(2) is None
    assert cache.get(1) is not None


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
# Epoch telemetry: loss weights, gradient norms, rates                         #
# --------------------------------------------------------------------------- #
def test_epoch_loss_weights_are_empty_for_a_model_that_does_not_ramp(stub):
    """`Model.run_epoch` applies no weight, so nothing should be logged as if
    it had."""
    assert stub(Model, ramp_epochs=4).epoch_loss_weights(0) == {}


def test_epoch_loss_weights_name_the_objective_each_weight_scales(stub):
    """`get_loss_weights` returns a bare pair whose second element is the class
    weight in one subclass and the relation ramp in the other; the keys are
    what make a logged weight readable beside the loss it scaled."""
    epoch = 2
    _, second = stub(Model, ramp_epochs=4).get_loss_weights(epoch)

    parent = stub(BrendaClassificationModel, ramp_epochs=4)
    assert parent.epoch_loss_weights(epoch) == {
        "entity": 1.0,
        "class": second,
    }

    ete = stub(ETEBrendaModel, ramp_epochs=4)
    # ETE's `run_epoch` scales the class loss by the *entity* weight.
    assert ete.epoch_loss_weights(epoch) == {
        "entity": 1.0,
        "class": 1.0,
        "relation": second,
    }


def _grad_norm_recorder(stub):
    model = stub(Model)
    model._reset_grad_norms()
    return model


def test_grad_norm_metrics_are_absent_without_an_optimizer_step(stub):
    """A validation pass never calls `_update`, so it must not report a
    gradient statistic for an epoch that computed no gradients."""
    assert _grad_norm_recorder(stub)._grad_norm_metrics() == {}


def test_grad_norm_metrics_average_the_preclip_norms(stub):
    model = _grad_norm_recorder(stub)
    for norm in (2.0, 0.5):
        model._record_grad_norm(torch.tensor(norm))

    metrics = model._grad_norm_metrics()
    assert metrics["training/grad_norm"] == pytest.approx(1.25)
    # Exactly one of the two exceeded the clip threshold.
    assert metrics["training/grad_clip_rate"] == pytest.approx(0.5)


def test_grad_clip_rate_saturates_when_every_step_clips(stub):
    """A rate pinned at 1.0 is the signal that the clip, not the learning
    rate, is setting the step size."""
    model = _grad_norm_recorder(stub)
    for _ in range(3):
        model._record_grad_norm(torch.tensor(GRAD_CLIP_NORM * 10))

    assert model._grad_norm_metrics()["training/grad_clip_rate"] == 1.0


def test_resetting_grad_norms_drops_the_previous_epoch(stub):
    model = _grad_norm_recorder(stub)
    model._record_grad_norm(torch.tensor(4.0))
    model._reset_grad_norms()

    assert model._grad_norm_metrics() == {}


def test_epoch_rate_metrics_are_keyed_by_step():
    metrics = epoch_rate_metrics(batches=10, seconds=2.0, step=Step.VALIDATION)
    assert metrics == {
        "validation/seconds": 2.0,
        "validation/batches_per_second": 5.0,
    }


def test_epoch_rate_metrics_omit_an_undefined_rate():
    """A zero-duration epoch still has a duration worth logging; the rate it
    implies is a division by zero."""
    metrics = epoch_rate_metrics(batches=3, seconds=0.0, step=Step.TRAINING)
    assert metrics == {"training/seconds": 0.0}


# --------------------------------------------------------------------------- #
# Evaluation metrics                                                           #
# --------------------------------------------------------------------------- #
def test_support_metrics_separate_predicting_nothing_from_predicting_wrong():
    """Both score micro-F1 0; only the predicted-positive count tells them
    apart, which is the whole reason the counts are logged."""
    gold = np.array([[1, 0], [0, 1]])
    silent = np.zeros_like(gold)
    wrong = np.array([[0, 1], [1, 0]])

    assert support_metrics({"class": (gold, silent)}) == {
        "test/class_gold_positives": 2.0,
        "test/class_predicted_positives": 0.0,
        "test/class_labels_predicted": 0.0,
    }
    assert (
        support_metrics({"class": (gold, wrong)})[
            "test/class_predicted_positives"
        ]
        == 2.0
    )


def test_support_metrics_count_columns_not_positives():
    """A head collapsed onto one frequent label predicts plenty of positives
    over a single column."""
    gold = np.array([[1, 0], [0, 1]])
    collapsed = np.array([[1, 0], [1, 0]])

    metrics = support_metrics({"entity": (gold, collapsed)})

    assert metrics["test/entity_predicted_positives"] == 2.0
    assert metrics["test/entity_labels_predicted"] == 1.0


def test_relation_metrics_exclude_none_from_the_typed_scores():
    """`none` is the majority class and the one nobody asked about; a macro-F1
    including it reports mostly how well the model says nothing."""
    labels = np.arange(3)
    none_index = 2
    # Every typed pair wrong, every `none` right.
    true = np.array([0, 1, 2, 2, 2, 2])
    pred = np.array([1, 0, 2, 2, 2, 2])

    metrics = relation_metrics(
        true=true, pred=pred, labels=labels, none_index=none_index
    )

    assert metrics["test/relation_macro_f1_typed"] == 0.0
    assert metrics["test/relation_accuracy"] == pytest.approx(4 / 6)
    assert metrics["test/relation_none_share"] == pytest.approx(4 / 6)
    assert metrics["test/relation_candidate_pairs"] == 6.0


def test_relation_metrics_report_an_empty_candidate_set():
    """The hard mask can propose no pairs at all; the count is the finding, and
    an accuracy over zero pairs is not."""
    metrics = relation_metrics(
        true=np.array([], dtype=int),
        pred=np.array([], dtype=int),
        labels=np.arange(3),
        none_index=2,
    )

    assert metrics["test/relation_candidate_pairs"] == 0.0
    assert "test/relation_accuracy" not in metrics
    assert "test/relation_none_share" not in metrics


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
        m.early_stop(v, epoch=e, save_checkpoint=False)
        for e, v in enumerate((5.0, 4.0, 3.0, 2.0))
    ]
    assert stops == [False, False, False, False]
    assert m.stop_counter == 0
    assert m.best_val_loss == 2.0


def test_early_stop_triggers_after_patience_exceeded(stub):
    m = _early_stopper(stub, patience=2)
    stops = [
        m.early_stop(v, epoch=e, save_checkpoint=False)
        for e, v in enumerate((1.0, 2.0, 3.0, 4.0))
    ]
    # improvement, then patience(2) tolerated increases, then stop
    assert stops == [False, False, False, True]
    assert m.best_val_loss == 1.0  # best preserved


def test_early_stop_records_the_epoch_that_produced_the_best_loss(stub):
    """`best_epoch` was initialised to -1 and never assigned, so a run that
    peaked at epoch 0 and then degraded reported having peaked at epoch -1."""
    m = _early_stopper(stub, patience=2)
    m.best_epoch = -1

    for epoch, val_loss in enumerate((1.0, 2.0, 3.0)):
        m.early_stop(val_loss, epoch=epoch, save_checkpoint=False)

    assert m.best_val_loss == 1.0
    assert m.best_epoch == 0


class _CheckpointableModel(Model):
    """The smallest real `Model`: `Model.__init__` loads no base model, so this
    has a genuine `state_dict` without a network download."""

    def __init__(self, device):
        super().__init__(config=ModelConfig(), device=device)
        self.head = torch.nn.Linear(4, 3)


def test_early_stop_snapshots_the_best_state_on_cpu(device):
    """The best-epoch snapshot must not sit on the GPU.

    `deepcopy(state_dict())` preserved each tensor's device, so on CUDA the
    snapshot was a second resident copy of the whole model — the frozen base
    model included — pinned for the rest of the run. The CPU variant passes
    either way and is here as the semantics guard; the CUDA variant is the red.
    """
    model = _CheckpointableModel(device).to(device)
    model.best_val_loss = float("inf")
    model.stop_counter = 0
    model.config = types.SimpleNamespace(patience=2)

    model.early_stop(1.0, epoch=0, save_checkpoint=True)

    assert model.best_model_state  # parameters and the _neg_inf buffer
    assert all(
        tensor.device.type == "cpu"
        for tensor in model.best_model_state.values()
    )
    # the live model has not moved
    assert model.head.weight.device.type == device


def test_early_stop_snapshot_does_not_alias_the_live_parameters(device):
    """`.to("cpu")` returns *self* for a tensor already there, so a CPU run
    would otherwise snapshot references that follow training."""
    model = _CheckpointableModel(device).to(device)
    model.best_val_loss = float("inf")
    model.stop_counter = 0
    model.config = types.SimpleNamespace(patience=2)

    model.early_stop(1.0, epoch=0, save_checkpoint=True)
    snapshot = model.best_model_state["head.weight"].clone()

    with torch.no_grad():
        model.head.weight.add_(1.0)

    assert torch.equal(model.best_model_state["head.weight"], snapshot)


def test_early_stop_snapshot_still_reloads_strictly(device):
    """The convergence path in `train_model`: the snapshot goes back in whole,
    and `load_state_dict` returns each tensor to the parameter's own device."""
    model = _CheckpointableModel(device).to(device)
    model.best_val_loss = float("inf")
    model.stop_counter = 0
    model.config = types.SimpleNamespace(patience=2)

    model.early_stop(1.0, epoch=0, save_checkpoint=True)
    # to CPU explicitly: this is a both-ways guard on the reload, so it must
    # not red merely because the snapshot's own device changed.
    best = model.best_model_state["head.weight"].detach().cpu().clone()

    with torch.no_grad():
        model.head.weight.add_(1.0)
    model.load_state_dict(model.best_model_state, strict=True)

    assert model.head.weight.device.type == device
    assert torch.equal(model.head.weight.detach().cpu(), best)


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


def test_biaffine_hidden_size_sets_the_bilinear_width():
    """The internal projection width is injectable, not a hardcoded 32: the
    bilinear parameter is (num_relations, width, width)."""
    model = BiaffineRelationClassifier(
        hidden_size=8, num_relations=3, biaff_hidden_size=16
    )
    assert tuple(model.bilinear.shape) == (3, 16, 16)


def test_config_knobs_reach_the_ete_model(patch_base_model):
    """entity_entropy_threshold and biaffine_hidden_size are ModelConfig fields
    that must reach the entropy-mask cutoff and the relation classifier's
    projection width, rather than the former hardcoded 0.8 / 32."""
    model = ETEBrendaModel(
        classes={"enzymes": {"enz1"}, "bacteria": {"bac1"}},
        class_matrix=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        entity_index={"enz1": 0, "bac1": 1},
        config=ModelConfig(
            base_model="prajjwal1/bert-mini",
            hidden_layers=[8],
            entity_entropy_threshold=0.5,
            biaffine_hidden_size=16,
        ),
        device="cpu",
    )
    assert model.entity_threshold == 0.5
    assert tuple(model.relation_classifier.bilinear.shape) == (3, 16, 16)


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


def test_ground_truth_reads_every_relations_dict_of_a_document(stub):
    m = stub(ETEBrendaModel, device="cpu")
    batch = [
        {
            "entities": torch.tensor([[1, 0]]),
            "classes": torch.tensor([[1, 0]]),
            "relations": [
                {("A", "B"): torch.tensor([0, 1, 0])},
                {("C", "D"): torch.tensor([1, 0, 0])},
            ],
        }
    ]
    _, _, relations = m.ground_truth(batch)
    assert {(r.subject, r.object, int(r.label)) for r in relations} == {
        ("A", "B", 1),
        ("C", "D", 0),
    }


def test_ground_truth_yields_no_relations_for_an_empty_relations_list(stub):
    m = stub(ETEBrendaModel, device="cpu")
    batch = [
        {
            "entities": torch.tensor([[1, 0]]),
            "classes": torch.tensor([[1, 0]]),
            "relations": [],
        }
    ]
    _, _, relations = m.ground_truth(batch)
    assert relations == []


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


# --------------------------------------------------------------------------- #
# The precomputed-embeddings store                                             #
# --------------------------------------------------------------------------- #
def _store_item(pmid, n_chunks, token=6):
    return {
        "id": torch.tensor(pmid),
        "doc_id": torch.zeros(n_chunks, dtype=torch.uint8),
        "sequence": {
            "input_ids": torch.zeros(n_chunks, token, dtype=torch.long),
            "attention_mask": torch.ones(n_chunks, token, dtype=torch.long),
        },
    }


@pytest.mark.parametrize("n_chunks", [1, 2, 5])
def test_document_token_count_is_what_the_aggregation_produces(n_chunks):
    """The row count guarding the store must equal the real thing for every
    chunk count, including one — the overlap arithmetic has a separate branch
    for the first sequence and for the tail, and a document of one chunk takes
    both."""
    token, hidden = 6, 3
    masks = torch.ones(n_chunks, token, dtype=torch.long)
    aggregated = aggregate_embeddings(
        torch.rand(n_chunks, token, hidden), masks
    )

    assert (
        document_token_count(_store_item(1, n_chunks, token))
        == (aggregated.shape[0])
    )


def test_a_stored_document_never_reaches_the_base_model(stub, monkeypatch):
    """The whole point of the store: the frozen base model is a pure function
    of the input ids, so a document it has already been run over must not be
    run over again."""

    def base_model_that_must_not_run(input_ids, attention_mask):
        raise AssertionError("the base model ran for a stored document")

    tokens = document_token_count(_store_item(100, 2))
    stored = torch.rand(tokens, 4)

    class FakeStore:
        def get(self, pubmed_id, expected_tokens):
            assert expected_tokens == tokens
            return stored.to(torch.bfloat16)

    monkeypatch.setattr(
        "d3text.models.models.embeddings_store", lambda: FakeStore()
    )
    monkeypatch.setattr("d3text.models.models.cpu_embeddings_cache", None)

    m = stub(
        Model,
        device="cpu",
        amp_dtype=torch.float16,
        base_model=base_model_that_must_not_run,
    )

    embeddings, masks = m.get_token_embeddings([_store_item(100, 2)])

    assert tuple(embeddings.shape) == (1, tokens, 4)
    # cast to the live path's dtype, not left as the store's bf16: on a card
    # without bf16 the two differ and the heads see one of them.
    assert embeddings.dtype == torch.float16
    assert masks.all()


def test_a_document_the_store_refuses_falls_back_to_the_base_model(
    stub, monkeypatch
):
    """A miss and a row-count mismatch are the same event here — the store
    returns None and the document is embedded live, which is what a run with no
    store configured does for every document."""
    hidden = 4
    ran = []

    def fake_base_model(input_ids, attention_mask):
        ran.append(input_ids.shape[0])
        n_seq, seq_len = input_ids.shape
        return types.SimpleNamespace(
            last_hidden_state=torch.zeros(n_seq, seq_len, hidden)
        )

    class RefusingStore:
        def get(self, pubmed_id, expected_tokens):
            return None

    monkeypatch.setattr(
        "d3text.models.models.embeddings_store", lambda: RefusingStore()
    )
    monkeypatch.setattr("d3text.models.models.cpu_embeddings_cache", None)

    m = stub(
        Model,
        device="cpu",
        amp_dtype=torch.bfloat16,
        base_model=fake_base_model,
    )

    m.get_token_embeddings([_store_item(100, 2)])

    assert ran == [2]


def test_the_cpu_cache_is_consulted_before_the_store(stub, monkeypatch):
    """Cheapest source first. A document in RAM must not cost an LMDB read and
    a blosc2 decompress."""
    cache = Cache(maxsize=4)
    cached = torch.rand(7, 4)
    cache.set(100, cached)

    class StoreThatMustNotBeRead:
        def get(self, pubmed_id, expected_tokens):
            raise AssertionError("the store was read for a cached document")

    monkeypatch.setattr("d3text.models.models.cpu_embeddings_cache", cache)
    monkeypatch.setattr(
        "d3text.models.models.embeddings_store",
        lambda: StoreThatMustNotBeRead(),
    )

    m = stub(Model, device="cpu", amp_dtype=torch.bfloat16, base_model=None)

    embeddings, _ = m.get_token_embeddings([_store_item(100, 2)])

    assert torch.equal(embeddings[0], cached)


def test_no_store_is_configured_by_default(monkeypatch):
    """The store is opt-in: absent the config key, `get_token_embeddings` is
    the function it always was."""
    monkeypatch.setattr(
        "d3text.models.models.mconfig",
        types.SimpleNamespace(embeddings_store=None),
    )
    embeddings_store.cache_clear()

    assert embeddings_store() is None

    embeddings_store.cache_clear()


# --------------------------------------------------------------------------- #
# has_bf16_hardware                                                            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "capability,expected",
    [((6, 0), False), ((7, 5), False), ((8, 0), True), ((9, 0), True)],
)
def test_bf16_is_claimed_only_where_there_are_bf16_units(
    monkeypatch, capability, expected
):
    """`torch.cuda.is_bf16_supported()` says yes on a Pascal card, because it
    counts emulation. Emulated bf16 measured 27% slower than fp16 and took
    close to three times the peak memory, so the question the dtype pick has to
    ask is about silicon: bf16 units arrive with Ampere."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)

    assert has_bf16_hardware() is expected


def test_bf16_is_not_claimed_without_a_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert has_bf16_hardware() is False
