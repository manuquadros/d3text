"""The CPU embeddings cache is budgeted in bytes, not in documents.

A cached entry is one row per token of a full paper — 14.5 MB on average over
this corpus and 56 MB at the tail — so a budget counted in entries is four
orders of magnitude from what it costs, and the count that reads as modest is
the one that gets the run killed with nothing in the log naming the cache.
These pin the accounting itself: what an entry is charged, that the ceiling is
enforced on the way in, which entries the budget may reclaim and for what, and
that the charge survives the real call site. Some also pin that an entry
outlives the inference mode it was read or computed under.
"""

import types

import torch
from d3text import runtime
from d3text.models.base import (
    BYTES_PER_MB,
    ByteBudgetCache,
    Model,
    Step,
    build_cpu_embeddings_cache,
    cpu_cache_key,
)
from d3text.models.config import ModelConfig
from d3text.training.update import BatchUpdate
from torch.utils.data import DataLoader

BASE_MODEL = ModelConfig(model_class="NERClassificationModel").base_model


def key(doc_id: int) -> tuple[str, int]:
    return cpu_cache_key(BASE_MODEL, doc_id)


def test_the_budget_admits_by_bytes_not_by_entry_count():
    """Three entries of 400 bytes do not fit in 1,000 bytes, whatever their
    number: a budget read as a count admits every one of them."""
    cache = ByteBudgetCache(max_bytes=1000)

    for doc_id in range(3):
        cache.set(key(doc_id), torch.zeros(100, dtype=torch.float32))

    assert cache.get(key(0)) is not None
    assert cache.get(key(1)) is not None
    assert cache.get(key(2)) is None
    assert cache.used_bytes == 800


def test_an_entry_is_charged_its_size_and_not_its_element_count():
    """`numel` alone undercounts a float32 document fourfold, so the cache
    would hold four times the RAM its budget promised."""
    cache = ByteBudgetCache(max_bytes=10**6)

    cache.set(key(1), torch.zeros(64, dtype=torch.float32))
    assert cache.used_bytes == 256

    cache.set(key(2), torch.zeros(64, dtype=torch.bfloat16))
    assert cache.used_bytes == 256 + 128


def test_the_same_element_count_costs_what_its_dtype_costs():
    """The admission decision, not just the tally, turns on element size: 64
    bfloat16 rows fit in 128 bytes and 64 float32 rows do not."""
    narrow = ByteBudgetCache(max_bytes=128)
    narrow.set(key(1), torch.zeros(64, dtype=torch.bfloat16))

    wide = ByteBudgetCache(max_bytes=128)
    wide.set(key(1), torch.zeros(64, dtype=torch.float32))

    assert narrow.get(key(1)) is not None
    assert wide.get(key(1)) is None


def test_an_entry_over_the_remaining_budget_leaves_the_cache_open():
    """The refusal lives in `set`, not at the call site: one document too big
    for what is left must not close the cache to every later, smaller one —
    document size varies tenfold across this corpus."""
    cache = ByteBudgetCache(max_bytes=200)

    cache.set(key(1), torch.zeros(100, dtype=torch.float32))
    assert cache.get(key(1)) is None
    assert cache.used_bytes == 0

    cache.set(key(2), torch.zeros(10, dtype=torch.float32))
    assert cache.get(key(2)) is not None
    assert cache.used_bytes == 40


def test_full_reports_the_byte_ceiling():
    """The call site short-circuits on `full`, so a ceiling it reads too early
    caches nothing and one it reads too late is the overshoot itself."""
    cache = ByteBudgetCache(max_bytes=80)

    cache.set(key(1), torch.zeros(10, dtype=torch.float32))
    assert not cache.full()

    cache.set(key(2), torch.zeros(10, dtype=torch.float32))
    assert cache.full()


def test_re_caching_a_document_is_charged_once():
    """A document written twice holds one tensor, so charging it twice would
    retire budget nothing occupies."""
    cache = ByteBudgetCache(max_bytes=200)

    cache.set(key(1), torch.zeros(10, dtype=torch.float32))
    cache.set(key(1), torch.zeros(10, dtype=torch.float32))

    assert cache.used_bytes == 40


def test_clear_releases_the_budget_the_entries_held():
    cache = ByteBudgetCache(max_bytes=80)
    cache.set(key(1), torch.zeros(20, dtype=torch.float32))
    assert cache.full()

    cache.clear()

    assert cache.used_bytes == 0
    assert not cache.full()
    cache.set(key(2), torch.zeros(20, dtype=torch.float32))
    assert cache.get(key(2)) is not None


def test_a_store_sourced_entry_gives_its_place_to_a_forward_only_one():
    """What the budget buys differs by source: the evicted document can be
    read from disk again, the admitted one can only be recomputed."""
    cache = ByteBudgetCache(max_bytes=80)
    cache.set(key(1), torch.zeros(20, dtype=torch.float32), from_store=True)
    assert cache.full()

    # The call site skips the host copy on a False, so an answer stricter
    # than `set`'s own loses entries the cache would have taken.
    assert cache.would_admit(key(2), 80)
    cache.set(key(2), torch.zeros(20, dtype=torch.float32))

    assert cache.get(key(1)) is None
    assert cache.get(key(2)) is not None
    assert cache.size() == 1
    assert cache.used_bytes == 80


def test_a_forward_only_entry_is_never_evicted_for_a_store_hit():
    """The asymmetry the policy rests on. Promoting a store hit over a
    document whose only other source is a base-model forward trades a disk
    read saved for a forward paid, and both are charged every epoch."""
    cache = ByteBudgetCache(max_bytes=80)
    cache.set(key(1), torch.zeros(20, dtype=torch.float32))

    assert not cache.would_admit(key(2), 80)
    cache.set(key(2), torch.zeros(20, dtype=torch.float32), from_store=True)

    assert cache.get(key(1)) is not None
    assert cache.get(key(2)) is None
    assert cache.used_bytes == 80


def test_a_store_hit_never_takes_another_store_hit_s_place():
    """A promotion displacing a peer costs one read to save one read, and
    under a working set larger than the budget it costs more than that: a
    pass reads its split once and in order, so each promotion would evict
    the entry the next pass is about to ask for and the hit rate would come
    to nothing. The documents admitted first keep their places instead."""
    cache = ByteBudgetCache(max_bytes=80)
    cache.set(key(1), torch.zeros(20, dtype=torch.float32), from_store=True)
    assert cache.full()

    assert not cache.would_admit(key(2), 80, from_store=True)
    cache.set(key(2), torch.zeros(20, dtype=torch.float32), from_store=True)

    assert cache.get(key(1)) is not None
    assert cache.get(key(2)) is None
    assert cache.size() == 1
    assert cache.used_bytes == 80


def test_an_entry_that_cannot_fit_at_all_evicts_nothing():
    """`set` declines before it drops anything: a document larger than the
    whole budget would otherwise empty the cache of everything the store
    served and still not be admitted."""
    cache = ByteBudgetCache(max_bytes=80)
    cache.set(key(1), torch.zeros(20, dtype=torch.float32), from_store=True)

    cache.set(key(2), torch.zeros(40, dtype=torch.float32))

    assert cache.get(key(1)) is not None
    assert cache.get(key(2)) is None
    assert cache.used_bytes == 80


def test_only_as_much_is_evicted_as_the_admission_needs():
    """One document's worth of room costs one document, and the oldest
    admitted pays it: a pass reads its split once, so nothing here is more
    recently used than anything else by the time the room is needed."""
    cache = ByteBudgetCache(max_bytes=120)
    for doc_id in (1, 2, 3):
        cache.set(
            key(doc_id), torch.zeros(10, dtype=torch.float32), from_store=True
        )
    assert cache.full()

    cache.set(key(4), torch.zeros(10, dtype=torch.float32))

    assert cache.get(key(1)) is None
    assert cache.get(key(2)) is not None
    assert cache.get(key(3)) is not None
    assert cache.get(key(4)) is not None
    assert cache.size() == 3
    assert cache.used_bytes == 120


def test_the_configured_megabytes_become_a_byte_ceiling():
    """`cpu_embeddings_cache_mb` is what a reader compares against `free`, so
    the conversion is the whole point of the unit."""
    assert build_cpu_embeddings_cache(2).max_bytes == 2 * BYTES_PER_MB
    assert BYTES_PER_MB == 10**6


def test_a_zero_budget_builds_no_cache():
    assert build_cpu_embeddings_cache(0) is None


def _item(pmid: int, n_chunks: int, token: int = 6) -> dict:
    return {
        "id": torch.tensor(pmid),
        "doc_id": torch.zeros(n_chunks, dtype=torch.uint8),
        "sequence": {
            "input_ids": torch.zeros(n_chunks, token, dtype=torch.long),
            "attention_mask": torch.ones(n_chunks, token, dtype=torch.long),
        },
    }


def test_get_token_embeddings_charges_a_document_its_real_size(
    stub, monkeypatch
):
    """The accounting has to hold at the real call site, not only in isolation.

    The budget here fits the one-chunk document and not the three-chunk one, so
    a cache that counted entries would keep the large document and refuse the
    small one — the exact inversion, and invisible from anything but the bytes.
    """
    hidden = 4

    def fake_base_model(input_ids, attention_mask):
        n_seq, seq_len = input_ids.shape
        return types.SimpleNamespace(
            last_hidden_state=torch.zeros(n_seq, seq_len, hidden)
        )

    cache = ByteBudgetCache(max_bytes=16)
    monkeypatch.setattr("d3text.models.base.cpu_embeddings_cache", cache)
    monkeypatch.setattr(
        "d3text.models.base.embeddings_store", lambda _base_model: None
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
        config=ModelConfig(model_class="NERClassificationModel"),
    )

    m.get_token_embeddings([_item(400, 3), _item(401, 1)])

    # 3 rows x 4 columns x 2 bytes = 24, over the budget; 1 x 4 x 2 = 8, under.
    assert cache.get(key(400)) is None
    assert cache.get(key(401)) is not None
    assert cache.used_bytes == 8


def test_a_declined_write_never_copies_to_the_host(stub, monkeypatch):
    """A document too big for what is left must not pay the device-to-host
    copy before `set` declines it on its own accounting.

    The budget leaves 4 bytes after the first document — not zero, so
    `full()` alone would not short-circuit the second — and 4 is still less
    than the second, same-sized document costs. That gap is exactly where a
    copy-then-decline would happen.
    """
    hidden = 4

    def fake_base_model(input_ids, attention_mask):
        n_seq, seq_len = input_ids.shape
        return types.SimpleNamespace(
            last_hidden_state=torch.zeros(n_seq, seq_len, hidden)
        )

    cache = ByteBudgetCache(max_bytes=12)
    monkeypatch.setattr("d3text.models.base.cpu_embeddings_cache", cache)
    monkeypatch.setattr(
        "d3text.models.base.embeddings_store", lambda _base_model: None
    )
    monkeypatch.setattr(
        "d3text.models.base.aggregate_embeddings",
        lambda outs, masks: outs[:, 0, :],
    )

    original_cpu = torch.Tensor.cpu
    calls: list[int] = []

    def counting_cpu(self, *args, **kwargs):
        calls.append(1)
        return original_cpu(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", counting_cpu)

    m = stub(
        Model,
        device="cpu",
        amp_dtype=torch.bfloat16,
        base_model=fake_base_model,
        config=ModelConfig(model_class="NERClassificationModel"),
    )

    # 1 row x 4 columns x 2 bytes (bfloat16) = 8, under the 12-byte budget.
    m.get_token_embeddings([_item(700, 1)])
    assert cache.get(key(700)) is not None
    assert len(calls) == 1

    # 4 bytes remain — not full, but not enough for another 8-byte
    # document. Must be declined without paying for the copy that used to
    # run before `set`'s own check.
    m.get_token_embeddings([_item(701, 1)])
    assert cache.get(key(701)) is None
    assert len(calls) == 1


HIDDEN = 4


def _fake_base_model(input_ids, attention_mask):
    n_seq, seq_len = input_ids.shape
    return types.SimpleNamespace(
        last_hidden_state=torch.rand(n_seq, seq_len, HIDDEN)
    )


class _OneDocumentStore:
    """A store answering for one document, in the bf16 it writes on disk.

    Records each read, so a test can tell a second pass served from RAM
    from one that went back to the store for bytes that cannot change.
    """

    def __init__(self, doc_id: int) -> None:
        self.doc_id = doc_id
        self.reads: list[int] = []

    def get(self, pubmed_id, expected_tokens):
        if pubmed_id != self.doc_id:
            return None
        self.reads.append(pubmed_id)
        return torch.rand(expected_tokens, HIDDEN, dtype=torch.bfloat16)

    def summary(self) -> str:
        return f"{len(self.reads)} reads"


def _stubbed_model(stub, monkeypatch, store=None, **attrs):
    """A CPU `Model` stub with a fresh cache, and `store` behind it."""
    cache = ByteBudgetCache(max_bytes=10**6)
    monkeypatch.setattr("d3text.models.base.cpu_embeddings_cache", cache)
    monkeypatch.setattr(
        "d3text.models.base.embeddings_store", lambda _base_model: store
    )
    m = stub(
        Model,
        device="cpu",
        amp_dtype=torch.bfloat16,
        base_model=_fake_base_model,
        config=ModelConfig(model_class="NERClassificationModel"),
        **attrs,
    )
    return m, cache


def _assert_trainable_through(cached):
    head = torch.nn.Linear(HIDDEN, 1, dtype=cached.dtype)
    head(cached).sum().backward()

    assert not cached.is_inference()
    assert head.weight.grad is not None


def test_a_document_cached_by_a_validation_pass_can_be_trained_through(
    stub, monkeypatch
):
    """`run_epoch` validates under inference mode, and the cache outlives it.

    On the CPU `.cpu()` hands back the very tensor it is given, so the device
    copy cannot be what takes the entry out of inference mode.
    """
    m, cache = _stubbed_model(
        stub,
        monkeypatch,
        compute_losses=lambda batch, step, epoch: {
            "loss": m.get_token_embeddings(batch)[0].float().sum()
        },
    )
    anchor = torch.nn.Linear(1, 1)
    update = BatchUpdate(
        anchor, torch.optim.SGD(anchor.parameters(), lr=0.1), "cpu"
    )
    loader = DataLoader(
        [[_item(500, 2, token=32)]],
        batch_size=1,
        collate_fn=lambda items: items[0],
    )

    m.run_epoch(loader, Step.VALIDATION, epoch=0, update=update)

    _assert_trainable_through(cache.get(key(500)))


def test_a_compiled_forward_caches_a_tensor_that_can_be_trained_through(
    stub, monkeypatch
):
    """Dynamo ignores an `inference_mode(False)` it captures into a graph.

    So the entry is trainable only while the per-document loop runs eagerly.
    `_write_resolved_embeddings` is decorated `@torch.compiler.disable` for
    exactly that reason, but even without it the loop would still fall back
    to eager on its own: the beartype wrapper on every call in it,
    `@record_function` on the caller, and its `.item()` call are each their
    own graph break, none of them the sole reason this test stays green.
    """
    m, cache = _stubbed_model(stub, monkeypatch)
    torch.nn.Module.__init__(m)
    runtime.exclude_type_checkers_from_dynamo()
    torch._dynamo.reset()
    embed = torch.compile(
        lambda batch: m.get_token_embeddings(batch),
        backend="aot_eager",
        dynamic=True,
    )

    try:
        with torch.inference_mode():
            embed([_item(501, 2, token=32)])
    finally:
        torch._dynamo.reset()

    _assert_trainable_through(cache.get(key(501)))


def test_a_store_hit_is_promoted_to_the_cpu_cache(stub, monkeypatch):
    """Left out of the cache, a stored document paid the store's read and
    decompress on every epoch and every validation pass for bytes that
    cannot change; the base model was spared, the disk never was."""
    store = _OneDocumentStore(600)
    m, cache = _stubbed_model(stub, monkeypatch, store=store)
    batch = [_item(600, 2, token=32)]

    first, _ = m.get_token_embeddings(batch)
    second, _ = m.get_token_embeddings(batch)

    assert store.reads == [600]
    assert cache.get(key(600)) is not None
    # The base model draws at random, so an identical second pass also says
    # no forward stood in for the read that did not happen.
    assert torch.equal(first, second)


def test_a_store_hit_promoted_by_a_validation_pass_is_trainable_through(
    stub, monkeypatch
):
    """`run_epoch` validates under inference mode, and the store's tensor is
    born in the read: cached as it comes back, it is an entry no later
    training pass can run through autograd."""
    m, cache = _stubbed_model(
        stub,
        monkeypatch,
        store=_OneDocumentStore(601),
        compute_losses=lambda batch, step, epoch: {
            "loss": m.get_token_embeddings(batch)[0].float().sum()
        },
    )
    anchor = torch.nn.Linear(1, 1)
    update = BatchUpdate(
        anchor, torch.optim.SGD(anchor.parameters(), lr=0.1), "cpu"
    )
    loader = DataLoader(
        [[_item(601, 2, token=32)]],
        batch_size=1,
        collate_fn=lambda items: items[0],
    )

    m.run_epoch(loader, Step.VALIDATION, epoch=0, update=update)

    _assert_trainable_through(cache.get(key(601)))
