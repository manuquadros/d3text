"""The CPU embeddings cache is budgeted in bytes, not in documents.

A cached entry is one row per token of a full paper — 14.5 MB on average over
this corpus and 56 MB at the tail — so a budget counted in entries is four
orders of magnitude from what it costs, and the count that reads as modest is
the one that gets the run killed with nothing in the log naming the cache.
These pin the accounting itself: what an entry is charged, that the ceiling is
enforced on the way in, and that the charge survives the real call site.
"""

import types

import torch
from d3text.models.base import (
    BYTES_PER_MB,
    ByteBudgetCache,
    Model,
    build_cpu_embeddings_cache,
    cpu_cache_key,
)
from d3text.models.config import ModelConfig

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
