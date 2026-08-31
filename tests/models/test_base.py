"""Pure unit tests for `d3text.models.base` — the shared `Model` base class
and the module-level helpers (pooling, telemetry, metrics, embeddings-store
plumbing) every model class builds on.

Every test here runs on CPU with tiny synthetic tensors and no data, network,
or GPU. Methods are exercised through the `stub` fixture (see
`tests/conftest.py`), which supplies only the attributes each method reads.
"""

import logging
import math
import types

import lmdb
import numpy as np
import pytest
import torch
from torch.utils.data import default_collate

from cacheout import Cache
from d3text.embeddings_store import (
    StoreProvenance,
    tensor_to_bytes,
    write_provenance,
)
from d3text.models.base import (
    Model,
    Step,
    balanced_class_weights,
    document_token_count,
    embeddings_store,
    epoch_rate_metrics,
    focal_cross_entropy,
    has_bf16_hardware,
    label_columns,
    ordered_entities,
    relation_metrics,
    select_amp_dtype,
    support_metrics,
)
from d3text.models.config import ModelConfig
from d3text.models.entity_linking import BrendaClassificationModel
from d3text.models.ete import ETEBrendaModel
from d3text.models.ner import NERClassificationModel
from d3text.training.update import BatchUpdate
from d3text.utils import aggregate_embeddings


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
        "d3text.models.base.aggregate_embeddings", spy_aggregate
    )

    m = stub(
        Model,
        device="cpu",
        amp_dtype=torch.bfloat16,
        base_model=fake_base_model,
        config=ModelConfig(),
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
        "d3text.models.base.cpu_embeddings_cache", cache, raising=False
    )
    monkeypatch.setattr(
        "d3text.models.base.aggregate_embeddings",
        lambda outs, masks: outs[:, 0, :],
    )

    m = stub(
        Model,
        device="cpu",
        amp_dtype=torch.bfloat16,
        base_model=fake_base_model,
        training=training,
        config=ModelConfig(),
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
        "d3text.models.base.cpu_embeddings_cache", cache, raising=False
    )
    monkeypatch.setattr(
        "d3text.models.base.aggregate_embeddings",
        lambda outs, masks: outs[:, 0, :],
    )

    m = stub(
        Model,
        device="cpu",
        amp_dtype=torch.bfloat16,
        base_model=fake_base_model,
        training=True,
        config=ModelConfig(),
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
# Unification: one `run_epoch`, one per-subclass `compute_losses`             #
# --------------------------------------------------------------------------- #
def test_every_model_class_shares_the_one_run_epoch():
    """The three model classes used to each carry their own `run_epoch`,
    differing only in which losses they accumulated. Only `compute_losses`
    may still differ between them; a subclass silently reintroducing its own
    `run_epoch` would pass every other test in the suite while breaking this
    identity."""
    assert NERClassificationModel.run_epoch is Model.run_epoch
    assert BrendaClassificationModel.run_epoch is Model.run_epoch
    assert ETEBrendaModel.run_epoch is Model.run_epoch


# --------------------------------------------------------------------------- #
# run_epoch's grad boundary: validation must not build an autograd graph      #
# --------------------------------------------------------------------------- #
def _loader_of_one_batch(batch):
    """A real `DataLoader` yielding exactly `batch`, unchanged.

    `run_epoch` is beartype-checked against `DataLoader`, so a hand-rolled
    stand-in is rejected at the boundary; `batch_size=None` disables
    collation, so the one-element "dataset" is handed back as-is.
    """
    return torch.utils.data.DataLoader([batch], batch_size=None)


class _NoOpUpdate(BatchUpdate):
    """A `BatchUpdate` that skips the real optimizer setup; `run_epoch` only
    calls this on the training step, and the fake loss here has no
    parameters worth stepping."""

    def __init__(self):  # no super().__init__: no optimizer to build
        pass

    def zero_grad(self):
        pass

    def __call__(self, *losses):
        pass


@pytest.mark.parametrize(
    "step,expect_requires_grad",
    [(Step.TRAINING, True), (Step.VALIDATION, False)],
)
def test_run_epoch_grad_tracking_follows_the_step(
    stub, step, expect_requires_grad
):
    """A tensor `compute_losses` builds from a tensor that requires grad
    keeps its graph on the training step and loses it on validation —
    `model.eval()` alone does not stop autograd from recording, only
    `run_epoch`'s grad context does."""
    captured: dict[str, torch.Tensor] = {}

    def fake_compute_losses(batch, step, epoch):
        weight = torch.nn.Parameter(torch.tensor(3.0))
        loss = (weight * 2).sum()
        captured["loss"] = loss
        return {"class": loss}

    obj = stub(Model, compute_losses=fake_compute_losses)
    obj.run_epoch(
        data=_loader_of_one_batch([object()]),
        step=step,
        epoch=0,
        update=_NoOpUpdate(),
    )

    assert captured["loss"].requires_grad is expect_requires_grad
    assert (captured["loss"].grad_fn is not None) is expect_requires_grad


# --------------------------------------------------------------------------- #
# Epoch telemetry: loss weights and rates                                      #
# --------------------------------------------------------------------------- #
def test_epoch_loss_weights_are_empty_for_a_model_that_does_not_ramp(stub):
    """`Model.run_epoch` applies no weight, so nothing should be logged as if
    it had."""
    assert stub(Model, ramp_epochs=4).epoch_loss_weights(0) == {}


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
# Relation-loss class weighting: the standalone functions                      #
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


# --------------------------------------------------------------------------- #
# ordered_entities / label_columns                                             #
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
        "d3text.models.base.embeddings_store", lambda _base_model: FakeStore()
    )
    monkeypatch.setattr("d3text.models.base.cpu_embeddings_cache", None)

    m = stub(
        Model,
        device="cpu",
        amp_dtype=torch.float16,
        base_model=base_model_that_must_not_run,
        config=ModelConfig(),
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
        "d3text.models.base.embeddings_store",
        lambda _base_model: RefusingStore(),
    )
    monkeypatch.setattr("d3text.models.base.cpu_embeddings_cache", None)

    m = stub(
        Model,
        device="cpu",
        amp_dtype=torch.bfloat16,
        base_model=fake_base_model,
        config=ModelConfig(),
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

    monkeypatch.setattr("d3text.models.base.cpu_embeddings_cache", cache)
    monkeypatch.setattr(
        "d3text.models.base.embeddings_store",
        lambda _base_model: StoreThatMustNotBeRead(),
    )

    m = stub(
        Model,
        device="cpu",
        amp_dtype=torch.bfloat16,
        base_model=None,
        config=ModelConfig(),
    )

    embeddings, _ = m.get_token_embeddings([_store_item(100, 2)])

    assert torch.equal(embeddings[0], cached)


def test_no_store_is_configured_by_default(monkeypatch):
    """The store is opt-in: absent the config key, `get_token_embeddings` is
    the function it always was."""
    monkeypatch.setattr(
        "d3text.models.base.mconfig",
        types.SimpleNamespace(embeddings_store=None),
    )
    embeddings_store.cache_clear()

    assert embeddings_store("michiyasunaga/BioLinkBERT-base") is None

    embeddings_store.cache_clear()


def _configured_store(tmp_path, monkeypatch, base_model):
    """A one-document store on disk, named by the machine config."""
    path = tmp_path / "store"
    with lmdb.open(str(path), map_size=2**20) as env:
        write_provenance(env, StoreProvenance(base_model, 512, 20))
        with env.begin(write=True) as transaction:
            transaction.put(b"100", tensor_to_bytes(torch.rand(4, 8)))

    monkeypatch.setattr(
        "d3text.models.base.mconfig",
        types.SimpleNamespace(embeddings_store=str(path)),
    )
    embeddings_store.cache_clear()


def test_a_store_written_by_another_model_disables_itself(
    tmp_path, monkeypatch, caplog
):
    """The run must lose the store, not the representation space it trains in.

    A store built with one 768-dim encoder and read under another answers
    every `get` with a matrix of exactly the right shape, so the heads see one
    model's activations for the documents it holds and the run's own for the
    documents it misses. Nothing raises and nothing is logged; the loss is
    merely worse than it should be.
    """
    _configured_store(tmp_path, monkeypatch, "prajjwal1/bert-mini")
    try:
        with caplog.at_level(logging.WARNING, logger="d3text.models.base"):
            store = embeddings_store("michiyasunaga/BioLinkBERT-base")
    finally:
        embeddings_store.cache_clear()

    assert store is None
    assert "prajjwal1/bert-mini" in caplog.text


def test_the_store_the_run_wrote_is_still_opened(tmp_path, monkeypatch):
    """The check must cost nothing to the run that is entitled to its store: a
    reader that refused everything would look exactly like one that is never
    hit."""
    _configured_store(tmp_path, monkeypatch, "michiyasunaga/BioLinkBERT-base")
    store = None
    try:
        store = embeddings_store("michiyasunaga/BioLinkBERT-base")

        assert store is not None
        assert store.get(100, expected_tokens=4) is not None
    finally:
        if store is not None:
            store.close()
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


# --------------------------------------------------------------------------- #
# select_amp_dtype                                                             #
# --------------------------------------------------------------------------- #
def _set_rocm(monkeypatch, is_rocm):
    monkeypatch.setattr(
        torch.version, "hip", "6.0" if is_rocm else None, raising=False
    )


def test_cuda_bf16_capable_card_gets_bf16(monkeypatch):
    _set_rocm(monkeypatch, False)
    monkeypatch.setattr("d3text.models.base.has_bf16_hardware", lambda: True)

    assert select_amp_dtype() is torch.bfloat16


def test_cuda_non_bf16_card_gets_fp16(monkeypatch):
    _set_rocm(monkeypatch, False)
    monkeypatch.setattr("d3text.models.base.has_bf16_hardware", lambda: False)

    assert select_amp_dtype() is torch.float16


@pytest.mark.parametrize(
    "device_name", ["AMD Instinct MI250X", "AMD Instinct MI300X"]
)
def test_rocm_allowlisted_card_gets_bf16_even_if_capability_would_say_no(
    monkeypatch, device_name
):
    """The device name, not `has_bf16_hardware`, must decide under ROCm: a
    gfx-derived compute capability could answer True for a card with no bf16
    units. `has_bf16_hardware` is never reached on the ROCm branch (Python's
    `and` short-circuits before it in the pre-fix code too), so this guard
    is a regression check against ever wiring it back in under ROCm, not a
    reproduction of the original bug's own mechanism."""
    _set_rocm(monkeypatch, True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda index: device_name
    )
    monkeypatch.setattr(
        "d3text.models.base.has_bf16_hardware",
        lambda: (_ for _ in ()).throw(
            AssertionError("has_bf16_hardware asked under ROCm")
        ),
    )

    assert select_amp_dtype() is torch.bfloat16


def test_rocm_non_allowlisted_card_gets_fp16(monkeypatch):
    _set_rocm(monkeypatch, True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda index: "AMD Instinct MI100"
    )

    assert select_amp_dtype() is torch.float16
