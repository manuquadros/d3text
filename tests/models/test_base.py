"""Pure unit tests for `d3text.models.base`.

The shared `Model` base class and the module-level helpers — pooling,
telemetry, metrics, embeddings-store plumbing. Everything runs on CPU with tiny
synthetic tensors through the `stub` fixture, which supplies only the
attributes each method reads.
"""

import logging
import math
import sys
import types
import weakref

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
from d3text.models import base as base_module
from d3text.models.base import (
    Model,
    Step,
    balanced_class_weights,
    cpu_cache_key,
    document_token_count,
    embeddings_store,
    epoch_rate_metrics,
    focal_cross_entropy,
    label_columns,
    relation_metrics,
    support_metrics,
)
from d3text.models.config import MachineConfig, ModelConfig
from d3text.models.entity_linking import BrendaClassificationModel
from d3text.models.ete import ETEBrendaModel
from d3text.models.ner import NERClassificationModel
from d3text.training.update import BatchUpdate
from d3text.utils import aggregate_embeddings


# --------------------------------------------------------------------------- #
# Shared fixtures for the embedding path                                       #
# --------------------------------------------------------------------------- #
def _batch_item(pmid, n_chunks, token=6, mask=None):
    """One collated document of `n_chunks` all-zero sequences.

    :param mask: attention mask to use in place of the default all-ones one,
        shaped `(n_chunks, token)`.
    """
    if mask is None:
        mask = torch.ones(n_chunks, token, dtype=torch.long)
    return {
        "id": torch.tensor(pmid),
        "doc_id": torch.zeros(n_chunks, dtype=torch.uint8),
        "sequence": {
            "input_ids": torch.zeros(n_chunks, token, dtype=torch.long),
            "attention_mask": mask,
        },
    }


def _fake_base_model(hidden, fill=0.0, device="cpu"):
    """A forward emitting a constant `[n_seq, seq_len, hidden]` on `device`."""

    def forward(input_ids, attention_mask):
        n_seq, seq_len = input_ids.shape
        return types.SimpleNamespace(
            last_hidden_state=torch.full(
                (n_seq, seq_len, hidden), fill, device=device
            )
        )

    return forward


def _embedding_model(stub, base_model, **attrs):
    """A `Model` stub for `get_token_embeddings`, on the CPU by default."""
    attrs = {
        "device": "cpu",
        "amp_dtype": torch.bfloat16,
        "config": ModelConfig(model_class="NERClassificationModel"),
        **attrs,
    }
    return stub(Model, base_model=base_model, **attrs)


def _one_row_per_chunk(monkeypatch):
    """Aggregate to the first row of each chunk, so a document's embedding
    has exactly `n_chunks` rows."""
    monkeypatch.setattr(
        "d3text.models.base.aggregate_embeddings",
        lambda outs, masks: outs[:, 0, :],
    )


class _MaxSizeCache(Cache):
    """`cacheout.Cache` plus the `would_admit` the real cache exposes.

    These tests admit by entry count, exactly what `full()` already
    decides, so `would_admit` here is just `full()` inverted — no byte
    accounting to duplicate. `from_store` is accepted and dropped for the
    same reason: what the real cache does with an entry's provenance is
    budget policy, pinned against the real cache in its own module.
    """

    def would_admit(self, key, cost, from_store=False):
        return not self.full()

    def set(self, key, value, ttl=None, from_store=False):
        super().set(key, value, ttl)


def _cpu_cache(monkeypatch, maxsize):
    """Install a fresh module-level CPU cache holding `maxsize` documents."""
    cache = _MaxSizeCache(maxsize=maxsize)
    monkeypatch.setattr("d3text.models.base.cpu_embeddings_cache", cache)
    return cache


# --------------------------------------------------------------------------- #
# Model._pool_logits (entity_logits_pooling knob)                              #
# --------------------------------------------------------------------------- #
def _pool_stub(stub, pooling):
    return stub(Model, entity_logits_pooling=pooling)


def _general_branch(tokens):
    """`[token, logits]`, `dim=0` — the shape/dim no model call site uses."""
    return tokens, 0


def _production_branch(tokens):
    """`[1, token, logits]`, `dim=1` — the shape/dim every model call site
    uses, routed through `pool_token_dim`."""
    return tokens.unsqueeze(0), 1


pool_branches = pytest.mark.parametrize(
    "to_branch",
    [_general_branch, _production_branch],
    ids=["general", "production"],
)


@pool_branches
def test_pool_logits_logsumexp_matches_torch(stub, to_branch):
    m = _pool_stub(stub, "logsumexp")
    logits, dim = to_branch(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert torch.allclose(
        m._pool_logits(logits, dim=dim), torch.logsumexp(logits, dim=dim)
    )


@pool_branches
def test_logsumexp_pooling_is_length_biased(stub, to_branch):
    """Smooth-max: uniform per-token logits gain +log(T), so pooling is *not*
    length-invariant (intended for sparse-mention detection)."""
    m = _pool_stub(stub, "logsumexp")
    short, dim = to_branch(torch.full((3, 2), 1.0))
    long, _ = to_branch(torch.full((6, 2), 1.0))
    short = m._pool_logits(short, dim=dim)
    long = m._pool_logits(long, dim=dim)
    expected_gap = torch.full_like(short, math.log(6) - math.log(3))
    assert torch.allclose(long - short, expected_gap)


@pool_branches
@pytest.mark.parametrize("pooling", ["logmeanexp", "max", "mean"])
def test_length_invariant_pooling_options(stub, to_branch, pooling):
    """logmeanexp / max / mean pool identical per-token logits to the same value
    regardless of document length."""
    m = _pool_stub(stub, pooling)
    short, dim = to_branch(torch.full((3, 2), 1.0))
    long, _ = to_branch(torch.full((6, 2), 1.0))
    short = m._pool_logits(short, dim=dim)
    long = m._pool_logits(long, dim=dim)
    assert torch.allclose(short, long)


@pool_branches
def test_pool_logits_rejects_unknown_pooling(stub, to_branch):
    m = _pool_stub(stub, "bogus")
    logits, dim = to_branch(torch.zeros(2, 2))
    with pytest.raises(ValueError):
        m._pool_logits(logits, dim=dim)


# --------------------------------------------------------------------------- #
# Model.batch_input_tensors                                                    #
# --------------------------------------------------------------------------- #
def test_batch_input_tensors_concatenates_chunks_into_2d(stub):
    """Per-document chunks concat along dim 0 into one 2-D tensor per key.

    `get_token_embeddings` slices the base-model output back per document via
    `doc_id.shape[-1]`, so the contract must be 2-D; the old
    `chain.from_iterable` collapsed it to 1-D.
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

    `default_collate` stamps a leading 1 onto every field, so the method sees
    `[1, n_chunks, token]`; concatenating that on dim 0 stacks documents on the
    chunk axis and raises as soon as two differ in chunk count. Collating here
    rather than hand-writing the 1 keeps the fixture from drifting.
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
    """The other half of the pack/unpack contract.

    Rows must be sliced back to the *right* document via `doc_id.shape[-1]`,
    with no cross-contamination.
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
    monkeypatch.setattr(
        "d3text.models.base.embeddings_store", lambda _base_model: None
    )

    m = _embedding_model(stub, fake_base_model)
    batch = [_batch_item(100, 2, token), _batch_item(200, 3, token)]

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

    The write used to be gated on `self.training`, which read as a policy
    reserving the budget for training documents; it is a single module-global
    budget, so the gate only kept validation permanently cold.
    """
    cache = _cpu_cache(monkeypatch, maxsize=8)
    _one_row_per_chunk(monkeypatch)
    monkeypatch.setattr(
        "d3text.models.base.embeddings_store", lambda _base_model: None
    )

    m = _embedding_model(stub, _fake_base_model(hidden=6), training=training)
    batch = [_batch_item(777, 2)]

    m.get_token_embeddings(batch)

    assert cache.get(cpu_cache_key(m.config.base_model, 777)) is not None

    # The second pass is served from the cache: the base model is not re-run.
    def exploding_base_model(input_ids, attention_mask):
        raise AssertionError("cache miss on a document already cached")

    object.__setattr__(m, "base_model", exploding_base_model)
    m.get_token_embeddings(batch)


def test_embed_missing_fills_the_cpu_cache(stub, monkeypatch):
    """A base-model computation must fill the CPU cache, pinned directly
    against `_embed_missing` rather than through `get_token_embeddings`,
    so a change to the resolution order in front of it cannot make this
    vacuous."""
    cache = _cpu_cache(monkeypatch, maxsize=8)
    _one_row_per_chunk(monkeypatch)
    m = _embedding_model(stub, _fake_base_model(hidden=6))
    item = _batch_item(777, 2)
    inputs = [None]

    m._embed_missing([(0, item)], inputs, trunk_trainable=False)

    assert cache.get(cpu_cache_key(m.config.base_model, 777)) is not None


def test_the_cpu_cache_is_not_shared_across_base_models(stub, monkeypatch):
    """Activations belong to the base model that produced them.

    `tune` builds a fresh model per trial from a grid where the base model is
    sweepable, so keyed by document id alone one trial's activations were
    served to the next — a shape error at unequal hidden widths, silent at
    equal.
    """
    ran: list[str] = []

    def base_model_named(name, fill):
        embed = _fake_base_model(hidden=6, fill=fill)

        def forward(input_ids, attention_mask):
            ran.append(name)
            return embed(input_ids, attention_mask)

        return forward

    cache = _cpu_cache(monkeypatch, maxsize=8)
    monkeypatch.setattr(
        "d3text.models.base.embeddings_store", lambda _base_model: None
    )
    _one_row_per_chunk(monkeypatch)

    def model_for(base_model, fill):
        return _embedding_model(
            stub,
            base_model_named(base_model, fill),
            training=True,
            config=ModelConfig(
                model_class="NERClassificationModel", base_model=base_model
            ),
        )

    first = model_for("prajjwal1/bert-mini", 1.0)
    batch = [_batch_item(4242, 2)]
    first_embeddings, _ = first.get_token_embeddings(batch)

    second = model_for("michiyasunaga/BioLinkBERT-base", 2.0)
    second_embeddings, _ = second.get_token_embeddings(batch)

    assert ran == ["prajjwal1/bert-mini", "michiyasunaga/BioLinkBERT-base"]
    assert torch.all(first_embeddings == 1.0)
    assert torch.all(second_embeddings == 2.0)
    assert (
        cache.get(cpu_cache_key("prajjwal1/bert-mini", 4242)) is not None
        and cache.get(cpu_cache_key("michiyasunaga/BioLinkBERT-base", 4242))
        is not None
    )


def test_get_token_embeddings_does_not_write_to_a_full_cache(stub, monkeypatch):
    """A full cache of documents the base model computed rejects a new one
    rather than evicting, which is what keeps the hit rate stable under a
    shuffled sampler."""
    hidden = 6
    cache = _cpu_cache(monkeypatch, maxsize=1)
    cache.set(1, torch.zeros(1, hidden))
    _one_row_per_chunk(monkeypatch)
    monkeypatch.setattr(
        "d3text.models.base.embeddings_store", lambda _base_model: None
    )

    m = _embedding_model(stub, _fake_base_model(hidden), training=True)
    m.get_token_embeddings([_batch_item(2, 1)])

    assert cache.get(2) is None
    assert cache.get(1) is not None


def test_get_token_embeddings_counts_cache_hits_and_misses(stub, monkeypatch):
    """The only way a run can report what fraction of documents it served
    from RAM: both outcomes are counted right where the lookup decides
    which one happened, not inferred from timing whole epochs."""
    cache = _cpu_cache(monkeypatch, maxsize=8)
    _one_row_per_chunk(monkeypatch)
    monkeypatch.setattr(
        "d3text.models.base.embeddings_store", lambda _base_model: None
    )
    monkeypatch.setattr("d3text.models.base.cpu_cache_hits", 0)
    monkeypatch.setattr("d3text.models.base.cpu_cache_misses", 0)

    m = _embedding_model(stub, _fake_base_model(hidden=6))
    cache.set(cpu_cache_key(m.config.base_model, 1), torch.zeros(1, 6))

    m.get_token_embeddings([_batch_item(1, 1), _batch_item(2, 1)])

    assert base_module.cpu_cache_hits == 1
    assert base_module.cpu_cache_misses == 1


def test_the_base_model_output_is_never_copied_to_the_host(stub, monkeypatch):
    """The hidden states are aggregated where the forward produced them.

    On a CPU every assertion about where a tensor sits is vacuous, so this
    spies on the output's `.cpu()`, which catches a copy to the host anywhere.
    """
    copied_to_host: list[str] = []
    embed = _fake_base_model(hidden=6)

    def fake_base_model(input_ids, attention_mask):
        lhs = embed(input_ids, attention_mask).last_hidden_state
        to_host = lhs.cpu

        def spy(*args, **kwargs):
            copied_to_host.append("last_hidden_state")
            return to_host(*args, **kwargs)

        lhs.cpu = spy
        lhs.detach = lambda: lhs
        return types.SimpleNamespace(last_hidden_state=lhs)

    monkeypatch.setattr("d3text.models.base.cpu_embeddings_cache", None)
    monkeypatch.setattr(
        "d3text.models.base.embeddings_store", lambda _base_model: None
    )
    _one_row_per_chunk(monkeypatch)

    m = _embedding_model(stub, fake_base_model)

    m.get_token_embeddings([_batch_item(100, 2)])

    assert copied_to_host == []


@pytest.mark.gpu
def test_every_embedding_source_lands_on_the_model_device(stub, monkeypatch):
    """One batch drawn from all three sources is assembled on one device.

    Cache and store hits are host tensors, so only mixing them with a live
    forward can hand `pad_sequence` two residencies. The aggregation records
    where its windows and masks sit, since host masks index device windows
    without complaint and nothing downstream would tell.
    """
    hidden, token = 4, 64
    cached_doc, stored_doc, fresh_doc = 100, 200, 300
    document_rows = document_token_count(_batch_item(stored_doc, 1, token))

    cache = _cpu_cache(monkeypatch, maxsize=8)
    cache.set(
        cpu_cache_key(
            ModelConfig(model_class="NERClassificationModel").base_model,
            cached_doc,
        ),
        torch.zeros(document_rows, hidden, dtype=torch.float16),
    )

    class FakeStore:
        def get(self, pubmed_id, expected_tokens):
            if pubmed_id != stored_doc:
                return None
            return torch.zeros(expected_tokens, hidden)

    aggregated_on: list[tuple[str, str]] = []
    real_aggregate = aggregate_embeddings

    def recording_aggregate(outs, masks):
        # `document_token_count` measures a document's geometry by aggregating
        # a zero-width tensor on the host; only a real window has a residency
        # this test is about.
        if outs.shape[-1]:
            aggregated_on.append((outs.device.type, masks.device.type))
        return real_aggregate(outs, masks)

    monkeypatch.setattr(
        "d3text.models.base.embeddings_store", lambda _base_model: FakeStore()
    )
    monkeypatch.setattr(
        "d3text.models.base.aggregate_embeddings", recording_aggregate
    )

    m = _embedding_model(
        stub,
        _fake_base_model(hidden, device="cuda"),
        device="cuda",
        amp_dtype=torch.float16,
    )

    embeddings, masks = m.get_token_embeddings(
        [
            _batch_item(cached_doc, 1, token),
            _batch_item(stored_doc, 1, token),
            _batch_item(fresh_doc, 1, token),
        ]
    )

    assert aggregated_on == [("cuda", "cuda")]
    assert embeddings.device.type == "cuda"
    assert masks.device.type == "cuda"

    # The cache's budget is host RAM; a device tensor in it is a VRAM
    # allocation pinned for the life of the process.
    newly_cached = cache.get(cpu_cache_key(m.config.base_model, fresh_doc))
    assert newly_cached is not None
    assert newly_cached.device.type == "cpu"


def test_the_hidden_states_are_freed_before_the_batch_is_padded(
    stub, monkeypatch
):
    """The forward's output is unreachable by the time `pad_sequence` runs.

    A weakref makes the release observable, on a CPU as on a card. Two names
    must go: `iter` unbinds the tensor into views its iterator keeps holding,
    so dropping either alone frees nothing.
    """
    hidden, token = 4, 64
    forward_output: list[weakref.ref] = []
    alive_when_padding: list[bool] = []
    embed = _fake_base_model(hidden)

    def fake_base_model(input_ids, attention_mask):
        hidden_states = embed(input_ids, attention_mask).last_hidden_state
        forward_output.append(weakref.ref(hidden_states))
        # `Tensor.detach` returns a new object and lets its source go, so the
        # tensor to weakref is the one `.detach()` hands back. Reaching that
        # by assigning `hidden_states.detach = lambda: hidden_states` would
        # build a reference cycle only the collector can break, which is
        # precisely what this test must not depend on.
        return types.SimpleNamespace(
            last_hidden_state=types.SimpleNamespace(
                detach=lambda: hidden_states
            )
        )

    real_pad_sequence = torch.nn.utils.rnn.pad_sequence

    def recording_pad_sequence(*args, **kwargs):
        alive_when_padding.append(forward_output[0]() is not None)
        return real_pad_sequence(*args, **kwargs)

    monkeypatch.setattr("d3text.models.base.cpu_embeddings_cache", None)
    monkeypatch.setattr(
        "d3text.models.base.embeddings_store", lambda _base_model: None
    )
    monkeypatch.setattr(
        "d3text.models.base.pad_sequence", recording_pad_sequence
    )

    m = _embedding_model(stub, fake_base_model)

    embeddings, masks = m.get_token_embeddings([_batch_item(100, 2, token)])

    assert alive_when_padding == [False]
    # The release is early, not destructive: the batch it was aggregated into
    # is still the batch the heads get.
    assert embeddings.shape[0] == 1
    assert embeddings.shape[-1] == hidden
    assert masks.shape == embeddings.shape[:2]
    assert masks.all()


@pytest.mark.parametrize("fresh", [True, False], ids=["mixed", "hits-only"])
def test_hits_reach_the_device_only_once_the_hidden_states_are_gone(
    stub, monkeypatch, watch_device_moves, fresh
):
    """No cache or store hit shares the card with the forward's output.

    A hit moved before the forward adds to the phase the step's peak rests
    on, as measured on batches that run the base model's forward. A batch of
    hits alone runs no forward, and must still send every hit to the device.
    """
    hidden, token = 4, 64
    cached_doc, stored_doc, fresh_doc = 100, 200, 300
    rows = document_token_count(_batch_item(stored_doc, 1, token))
    forward_output: list[weakref.ref] = []
    moves: list[tuple[str, str]] = []

    def phase():
        if not forward_output:
            return "before any forward"
        if forward_output[0]() is not None:
            return "beside the hidden states"
        return "after their release"

    def watched(label, tensor):
        return watch_device_moves(
            tensor, lambda: moves.append((label, phase()))
        )

    cache = _cpu_cache(monkeypatch, maxsize=8)
    cache.set(
        cpu_cache_key(
            ModelConfig(model_class="NERClassificationModel").base_model,
            cached_doc,
        ),
        watched("cache hit", torch.full((rows, hidden), 1.0).half()),
    )

    class FakeStore:
        def get(self, pubmed_id, expected_tokens):
            if pubmed_id != stored_doc:
                return None
            # bf16, as the store writes it, so the `amp_dtype` cast is a real
            # copy the watch has to follow.
            return watched(
                "store hit",
                torch.full((expected_tokens, hidden), 2.0).bfloat16(),
            )

    embed = _fake_base_model(hidden, fill=3.0)

    def fake_base_model(input_ids, attention_mask):
        hidden_states = embed(input_ids, attention_mask).last_hidden_state
        forward_output.append(weakref.ref(hidden_states))
        # Handed back through a namespace rather than a patched `.detach`, so
        # no reference cycle leaves the release to the collector.
        return types.SimpleNamespace(
            last_hidden_state=types.SimpleNamespace(
                detach=lambda: hidden_states
            )
        )

    monkeypatch.setattr(
        "d3text.models.base.embeddings_store", lambda _base_model: FakeStore()
    )

    m = _embedding_model(stub, fake_base_model, amp_dtype=torch.float16)
    docs = [cached_doc, stored_doc] + ([fresh_doc] if fresh else [])

    embeddings, masks = m.get_token_embeddings(
        [_batch_item(doc, 1, token) for doc in docs]
    )

    assert len(forward_output) == int(fresh)
    when = "after their release" if fresh else "before any forward"
    assert moves == [("cache hit", when), ("store hit", when)]
    # Late is not lost: every document still lands in its own row.
    assert embeddings.tolist() == [
        [[fill] * hidden] * rows for fill in (1.0, 2.0, 3.0)[: len(docs)]
    ]
    assert masks.all()


@pytest.mark.gpu
def test_the_card_holds_no_hit_while_the_forward_runs(stub, monkeypatch):
    """Read off the allocator: a mixed batch's forward starts beside no hit.

    The ordering test above watches `.to`; the allocator sees any route onto
    the card, and only the forward's own inputs belong on it by then.
    """
    hidden, token = 2048, 64
    cached_doc, stored_doc, fresh_doc = 100, 200, 300
    rows = document_token_count(_batch_item(stored_doc, 1, token))
    hit_bytes = rows * hidden * torch.float16.itemsize
    allocated_at_forward: list[int] = []

    cache = _cpu_cache(monkeypatch, maxsize=8)
    cache.set(
        cpu_cache_key(
            ModelConfig(model_class="NERClassificationModel").base_model,
            cached_doc,
        ),
        torch.zeros(rows, hidden, dtype=torch.float16),
    )

    class FakeStore:
        def get(self, pubmed_id, expected_tokens):
            if pubmed_id != stored_doc:
                return None
            return torch.zeros(expected_tokens, hidden, dtype=torch.bfloat16)

    embed = _fake_base_model(hidden, device="cuda")

    def fake_base_model(input_ids, attention_mask):
        allocated_at_forward.append(torch.cuda.memory_allocated() - baseline)
        return embed(input_ids, attention_mask)

    monkeypatch.setattr(
        "d3text.models.base.embeddings_store", lambda _base_model: FakeStore()
    )

    m = _embedding_model(
        stub, fake_base_model, device="cuda", amp_dtype=torch.float16
    )
    batch = [_batch_item(doc, 1, token) for doc in (cached_doc, stored_doc)]
    batch.append(_batch_item(fresh_doc, 1, token))

    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    embeddings, _ = m.get_token_embeddings(batch)

    assert len(allocated_at_forward) == 1
    assert allocated_at_forward[0] < hit_bytes
    assert embeddings.device.type == "cuda"


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
    stand-in is rejected; `batch_size=None` disables collation.
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

    obj = stub(
        Model,
        compute_losses=fake_compute_losses,
        config=ModelConfig(model_class="NERClassificationModel"),
    )
    obj.run_epoch(
        data=_loader_of_one_batch([object()]),
        step=step,
        epoch=0,
        update=_NoOpUpdate(),
    )

    assert captured["loss"].requires_grad is expect_requires_grad
    assert (captured["loss"].grad_fn is not None) is expect_requires_grad


# --------------------------------------------------------------------------- #
# run_epoch's loss accumulation: sum on-device, read out once                  #
# --------------------------------------------------------------------------- #
def _loader_of_batches(n):
    """A real `DataLoader` yielding `n` placeholder batches, unchanged."""
    return torch.utils.data.DataLoader([object()] * n, batch_size=None)


def test_run_epoch_sums_losses_across_batches(stub):
    """The returned per-epoch losses are the exact sum of what
    `compute_losses` returned each batch, keyed the same way."""
    per_batch = [
        {"entity": 1.0, "class": 2.5},
        {"entity": 3.0, "class": 4.5},
    ]
    calls = iter(per_batch)

    def fake_compute_losses(batch, step, epoch):
        return {k: torch.tensor(v) for k, v in next(calls).items()}

    obj = stub(
        Model,
        compute_losses=fake_compute_losses,
        config=ModelConfig(model_class="NERClassificationModel"),
    )
    losses, n_batches = obj.run_epoch(
        data=_loader_of_batches(2),
        step=Step.VALIDATION,
        epoch=0,
        update=_NoOpUpdate(),
    )

    assert n_batches == 2
    assert losses == {
        "entity": pytest.approx(4.0),
        "class": pytest.approx(7.0),
    }
    assert all(isinstance(v, float) for v in losses.values())


def test_run_epoch_reads_each_loss_off_the_device_once_per_epoch(
    stub, monkeypatch
):
    """The per-batch accumulation must not call `Tensor.item()` — that is the
    blocking device-to-host sync the accumulator exists to avoid until the
    epoch is over.

    Patches `Tensor.item` with a counter, the same technique
    `BatchUpdate._record_grad_norm`'s accumulation relies on being safe from:
    two batches with two loss keys would call `.item()` eight times under the
    old per-batch `.cpu().item()`, and must call it exactly twice here — once
    per key, at epoch end. Only calls made from `base.py` are counted:
    `DataLoader.__iter__` makes its own unrelated `.item()` call generating a
    worker seed, once per epoch regardless of batch or key count, which would
    otherwise inflate every expected total by a constant this test does not
    care about.
    """
    real_item = torch.Tensor.item
    base_module_file = base_module.__file__
    calls = 0

    def counting_item(self):
        nonlocal calls
        caller = sys._getframe(1)
        if caller.f_code.co_filename == base_module_file:
            calls += 1
        return real_item(self)

    monkeypatch.setattr(torch.Tensor, "item", counting_item)

    def fake_compute_losses(batch, step, epoch):
        return {"entity": torch.tensor(1.0), "class": torch.tensor(2.0)}

    obj = stub(
        Model,
        compute_losses=fake_compute_losses,
        config=ModelConfig(model_class="NERClassificationModel"),
    )
    obj.run_epoch(
        data=_loader_of_batches(2),
        step=Step.VALIDATION,
        epoch=0,
        update=_NoOpUpdate(),
    )

    assert calls == 2


# --------------------------------------------------------------------------- #
# Epoch telemetry: loss weights and rates                                      #
# --------------------------------------------------------------------------- #
def test_run_epoch_logs_the_cpu_cache_hit_rate_once_per_pass(
    stub, monkeypatch, caplog
):
    """`run_epoch` is shared by every model class, so it is the one place a
    pass's hit rate can be reported without threading a counter through
    three different `compute_losses` implementations. The counters reset
    once logged, so a later pass reports its own rate, not a running one.

    Uses the real `ByteBudgetCache` rather than the `cacheout.Cache` stand-in
    other tests here substitute: the log line reads `used_bytes`/`max_bytes`,
    which only the real cache carries.
    """
    cache = base_module.ByteBudgetCache(max_bytes=10_000)
    monkeypatch.setattr("d3text.models.base.cpu_embeddings_cache", cache)
    _one_row_per_chunk(monkeypatch)
    monkeypatch.setattr(
        "d3text.models.base.embeddings_store", lambda _base_model: None
    )
    monkeypatch.setattr("d3text.models.base.cpu_cache_hits", 0)
    monkeypatch.setattr("d3text.models.base.cpu_cache_misses", 0)

    m = _embedding_model(stub, _fake_base_model(hidden=6))
    cache.set(cpu_cache_key(m.config.base_model, 1), torch.zeros(1, 6))

    def fake_compute_losses(batch, step, epoch):
        m.get_token_embeddings(batch)
        return {"class": torch.tensor(0.0)}

    object.__setattr__(m, "compute_losses", fake_compute_losses)

    with caplog.at_level(logging.INFO, logger="d3text.models.base"):
        m.run_epoch(
            data=_loader_of_one_batch([_batch_item(1, 1), _batch_item(2, 1)]),
            step=Step.TRAINING,
            epoch=0,
            update=_NoOpUpdate(),
        )

    assert "1/2 hits" in caplog.text
    assert "50.0%" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="d3text.models.base"):
        m.run_epoch(
            data=_loader_of_one_batch([_batch_item(1, 1)]),
            step=Step.TRAINING,
            epoch=1,
            update=_NoOpUpdate(),
        )

    assert "1/1 hits" in caplog.text


def test_epoch_loss_weights_are_empty_for_a_model_that_does_not_ramp(stub):
    """`Model.run_epoch` applies no weight, so nothing should be logged as if
    it had."""
    assert stub(Model, ramp_epochs=4).epoch_loss_weights(0) == {}


def test_epoch_rate_metrics_are_keyed_by_step():
    metrics = epoch_rate_metrics(batches=10, seconds=2.0, step=Step.VALIDATION)
    assert metrics == {
        "validation/epoch_seconds": 2.0,
        "validation/batches_per_second": 5.0,
    }


def test_epoch_rate_metrics_omit_an_undefined_rate():
    """A zero-duration epoch still has a duration worth logging; the rate it
    implies is a division by zero."""
    metrics = epoch_rate_metrics(batches=3, seconds=0.0, step=Step.TRAINING)
    assert metrics == {"training/epoch_seconds": 0.0}


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

    metrics = support_metrics({"class": (gold, collapsed)})

    assert metrics["test/class_predicted_positives"] == 2.0
    assert metrics["test/class_labels_predicted"] == 1.0


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
    """A split can ground no detected span at all; the count is the finding,
    and an accuracy over zero pairs is not."""
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


def test_focal_cross_entropy_downweights_easy_pairs_under_the_clamp_floor():
    """A one-row batch's modulation mass is always <= 1, so
    `clamp(min=1.0)` forces the divisor to exactly 1 under either
    normalisation scheme. This pins the per-pair `(1 - p_t) ** gamma`
    weighting itself, not the mass normalisation — see the growing-N test
    below for that."""
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


def test_focal_cross_entropy_is_not_diluted_by_added_easy_pairs():
    """The loss over K hard pairs must stay close to its own value as easy
    negatives are appended, because mass normalisation divides by those
    negatives' own (near-zero) modulation rather than by their count. A
    plain `.mean()` instead divides by the row count, so it keeps shrinking
    as N grows — the property `clamp(min=1.0)` hides in a one-row batch."""
    gamma = 2.0
    hard = torch.tensor([[0.0, 0.0, 0.0]] * 2)  # K == 2 uninformed pairs
    hard_targets = torch.tensor([2, 2])
    baseline = focal_cross_entropy(hard, hard_targets, gamma=gamma)

    easy = torch.tensor([[-6.0, -6.0, 6.0]])  # p_t ~= 1: already learned
    n_easy = 1000
    preds = torch.cat([hard, easy.repeat(n_easy, 1)])
    targets = torch.cat(
        [hard_targets, torch.full((n_easy,), 2, dtype=torch.int64)]
    )

    diluted = focal_cross_entropy(preds, targets, gamma=gamma)
    assert torch.isclose(diluted, baseline, rtol=1e-3)


# --------------------------------------------------------------------------- #
# label_columns                                                                #
# --------------------------------------------------------------------------- #
def test_label_columns_locates_the_sentinel_and_lists_the_rest():
    index, columns = label_columns(["c0", "OOS", "c1"], "OOS")
    assert index == 1
    assert columns.tolist() == [0, 2]
    assert columns.dtype == torch.int64


def test_label_columns_rejects_a_missing_sentinel():
    with pytest.raises(ValueError):
        label_columns(["c0", "c1"], "OOS")


# --------------------------------------------------------------------------- #
# The precomputed-embeddings store                                             #
# --------------------------------------------------------------------------- #
def _tail_mask(n_chunks, token, tail_real_tokens):
    """An all-ones mask except the last chunk, padded after
    `tail_real_tokens` real tokens.

    The last chunk is where a real document's final window falls short of
    `token` tokens; the other chunks stay full so only the tail's padding is
    under test.
    """
    mask = torch.ones(n_chunks, token, dtype=torch.long)
    mask[-1, tail_real_tokens:] = 0
    return mask


@pytest.mark.parametrize("n_chunks", [1, 2, 5])
@pytest.mark.parametrize("tail_real_tokens", [6, 3, 1])
def test_document_token_count_is_what_the_aggregation_produces(
    n_chunks, tail_real_tokens
):
    """The row count guarding the store must equal the real thing for every
    chunk count, including one — the overlap arithmetic has a separate branch
    for the first sequence and for the tail, and a document of one chunk takes
    both — and for every amount of padding in the final window, since a count
    that ignores the mask would treat that padding as real tokens."""
    token, hidden = 6, 3
    masks = _tail_mask(n_chunks, token, tail_real_tokens)
    aggregated = aggregate_embeddings(
        torch.rand(n_chunks, token, hidden), masks
    )

    assert (
        document_token_count(_batch_item(1, n_chunks, token, mask=masks))
        == (aggregated.shape[0])
    )


def test_a_stored_document_never_reaches_the_base_model(stub, monkeypatch):
    """The whole point of the store: the frozen base model is a pure function
    of the input ids, so a document it has already been run over must not be
    run over again."""

    def base_model_that_must_not_run(input_ids, attention_mask):
        raise AssertionError("the base model ran for a stored document")

    tokens = document_token_count(_batch_item(100, 2))
    stored = torch.rand(tokens, 4)

    class FakeStore:
        def get(self, pubmed_id, expected_tokens):
            assert expected_tokens == tokens
            return stored.to(torch.bfloat16)

    monkeypatch.setattr(
        "d3text.models.base.embeddings_store", lambda _base_model: FakeStore()
    )
    monkeypatch.setattr("d3text.models.base.cpu_embeddings_cache", None)

    m = _embedding_model(
        stub, base_model_that_must_not_run, amp_dtype=torch.float16
    )

    embeddings, masks = m.get_token_embeddings([_batch_item(100, 2)])

    assert tuple(embeddings.shape) == (1, tokens, 4)
    # cast to the live path's dtype, not left as the store's bf16: on a card
    # without bf16 the two differ and the heads see one of them.
    assert embeddings.dtype == torch.float16
    assert masks.all()


def test_a_store_hit_above_fp16_range_raises_naming_the_document(
    stub, monkeypatch
):
    """bf16 carries fp32's exponent range; fp16 caps at 65504. Narrowing a
    store hit above that to fp16 would otherwise turn it into `inf` with no
    error and a NaN loss downstream -- the cast site must catch it instead
    and name the document that overflowed."""

    def base_model_that_must_not_run(input_ids, attention_mask):
        raise AssertionError("the base model ran for a stored document")

    tokens = document_token_count(_batch_item(100, 2))
    stored = torch.ones(tokens, 4)
    stored[0, 0] = 70000.0

    class FakeStore:
        def get(self, pubmed_id, expected_tokens):
            return stored.to(torch.bfloat16)

    monkeypatch.setattr(
        "d3text.models.base.embeddings_store", lambda _base_model: FakeStore()
    )
    monkeypatch.setattr("d3text.models.base.cpu_embeddings_cache", None)

    m = _embedding_model(
        stub, base_model_that_must_not_run, amp_dtype=torch.float16
    )

    with pytest.raises(RuntimeError, match="document 100"):
        m.get_token_embeddings([_batch_item(100, 2)])


def test_a_store_hit_above_fp16_range_is_unguarded_under_bf16(
    stub, monkeypatch
):
    """The guard only matters on a machine without bf16 hardware: a bf16
    `amp_dtype` never narrows the store's own bf16 format, so the same
    out-of-fp16-range value must pass through unchanged rather than being
    flagged."""

    def base_model_that_must_not_run(input_ids, attention_mask):
        raise AssertionError("the base model ran for a stored document")

    tokens = document_token_count(_batch_item(100, 2))
    stored = torch.ones(tokens, 4)
    stored[0, 0] = 70000.0

    class FakeStore:
        def get(self, pubmed_id, expected_tokens):
            return stored.to(torch.bfloat16)

    monkeypatch.setattr(
        "d3text.models.base.embeddings_store", lambda _base_model: FakeStore()
    )
    monkeypatch.setattr("d3text.models.base.cpu_embeddings_cache", None)

    m = _embedding_model(
        stub, base_model_that_must_not_run, amp_dtype=torch.bfloat16
    )

    embeddings, _ = m.get_token_embeddings([_batch_item(100, 2)])

    assert torch.isfinite(embeddings).all()


def test_a_document_the_store_refuses_falls_back_to_the_base_model(
    stub, monkeypatch
):
    """A miss and a row-count mismatch are the same event here — the store
    returns None and the document is embedded live, which is what a run with no
    store configured does for every document."""
    ran = []
    embed = _fake_base_model(hidden=4)

    def fake_base_model(input_ids, attention_mask):
        ran.append(input_ids.shape[0])
        return embed(input_ids, attention_mask)

    class RefusingStore:
        def get(self, pubmed_id, expected_tokens):
            return None

    monkeypatch.setattr(
        "d3text.models.base.embeddings_store",
        lambda _base_model: RefusingStore(),
    )
    monkeypatch.setattr("d3text.models.base.cpu_embeddings_cache", None)

    m = _embedding_model(stub, fake_base_model)

    m.get_token_embeddings([_batch_item(100, 2)])

    assert ran == [2]


def test_the_cpu_cache_is_consulted_before_the_store(stub, monkeypatch):
    """Cheapest source first. A document in RAM must not cost an LMDB read and
    a blosc2 decompress."""
    cache = Cache(maxsize=4)
    item = _batch_item(100, 2)
    # The row count an entry is served under is the document's own: one that
    # disagrees is read as another document's and rejected.
    cached = torch.rand(document_token_count(item), 4)
    cache.set(
        cpu_cache_key(
            ModelConfig(model_class="NERClassificationModel").base_model, 100
        ),
        cached,
    )

    class StoreThatMustNotBeRead:
        def get(self, pubmed_id, expected_tokens):
            raise AssertionError("the store was read for a cached document")

    monkeypatch.setattr("d3text.models.base.cpu_embeddings_cache", cache)
    monkeypatch.setattr(
        "d3text.models.base.embeddings_store",
        lambda _base_model: StoreThatMustNotBeRead(),
    )

    m = _embedding_model(stub, base_model=None)

    embeddings, _ = m.get_token_embeddings([item])

    assert torch.equal(embeddings[0], cached)


def test_no_store_is_configured_by_default(monkeypatch):
    """The store is opt-in: absent the config key, `get_token_embeddings` is
    the function it always was."""
    assert MachineConfig().embeddings_store == {}

    monkeypatch.setattr(
        "d3text.models.base.mconfig",
        types.SimpleNamespace(embeddings_store={}),
    )
    embeddings_store.cache_clear()

    assert embeddings_store("michiyasunaga/BioLinkBERT-base") is None

    embeddings_store.cache_clear()


def _configured_store(tmp_path, monkeypatch, base_model, *, configured_as=None):
    """A one-document store on disk, named by the machine config.

    `configured_as` lets a test point the config's key at a base model other
    than the one the store's provenance names, to simulate the config
    misattributing a store — the key alone decides which run finds it; the
    provenance recorded inside decides whether that run may use it.
    """
    path = tmp_path / "store"
    with lmdb.open(str(path), map_size=2**20) as env:
        write_provenance(env, StoreProvenance(base_model, 512, 20))
        with env.begin(write=True) as transaction:
            transaction.put(b"100", tensor_to_bytes(torch.rand(4, 8)))

    monkeypatch.setattr(
        "d3text.models.base.mconfig",
        types.SimpleNamespace(
            embeddings_store={(configured_as or base_model): str(path)}
        ),
    )
    embeddings_store.cache_clear()


def test_a_store_written_by_another_model_disables_itself(
    tmp_path, monkeypatch, caplog
):
    """The run must lose the store, not the representation space it trains in.

    A store built with another 768-dim encoder answers every `get` with a
    matrix of exactly the right shape, so nothing raises and nothing is logged
    — the loss is merely worse than it should be.
    """
    _configured_store(
        tmp_path,
        monkeypatch,
        "prajjwal1/bert-mini",
        configured_as="michiyasunaga/BioLinkBERT-base",
    )
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


def test_the_cache_and_base_model_path_tests_never_open_a_real_store(
    tmp_path, stub
):
    """The three tests above must describe the cache and base-model path on
    every machine, not just one whose `config.toml` leaves `embeddings_store`
    unset.

    Runs each of them under a `mconfig` naming a store on disk and an
    `EmbeddingsStore` that raises if constructed at all; each test's own
    `monkeypatch.setattr("d3text.models.base.embeddings_store", ...)` must
    intercept the call before that construction is reached. A fresh
    `MonkeyPatch` context per test keeps one test's patch of
    `d3text.models.base.embeddings_store` from surviving into the next and
    masking a missing patch there.
    """

    class StoreMustNotBeConstructed:
        def __init__(self, *args, **kwargs):
            raise AssertionError("a real embeddings store was constructed")

    class _ConfiguredForEveryModel(dict):
        """A store path that answers any base model, like a single global
        path did before `embeddings_store` was keyed by model."""

        def get(self, _key, _default=None):
            return str(tmp_path / "store")

    fake_mconfig = types.SimpleNamespace(
        embeddings_store=_ConfiguredForEveryModel()
    )

    def run_under_a_configured_store(test_fn, *args):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("d3text.models.base.mconfig", fake_mconfig)
            mp.setattr(
                "d3text.models.base.EmbeddingsStore",
                StoreMustNotBeConstructed,
            )
            embeddings_store.cache_clear()
            try:
                test_fn(stub, mp, *args)
            finally:
                embeddings_store.cache_clear()

    run_under_a_configured_store(
        test_get_token_embeddings_unpacks_rows_back_to_each_document
    )
    run_under_a_configured_store(
        test_get_token_embeddings_caches_in_both_train_and_eval, True
    )
    run_under_a_configured_store(
        test_get_token_embeddings_does_not_write_to_a_full_cache
    )
