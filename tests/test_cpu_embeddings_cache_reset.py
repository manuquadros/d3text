"""The process-wide CPU embeddings cache must not leak across tests.

`base.cpu_embeddings_cache` is module state keyed by base model and document
id, and fixtures across the suite reuse small integer pmids for unrelated
documents — so on a machine whose `config.toml` enables the cache, one test's
entry could be read back stale by a later one. The first test here enables the
cache in-process and writes such an entry; the second proves it did not
survive. Order matters: pytest collects a file's tests in definition order.
"""

import torch
from d3text.models import base

_LEAKED_KEY = ("prajjwal1/bert-mini", 11)


def test_a_populates_the_process_wide_cache():
    base.cpu_embeddings_cache = base.ByteBudgetCache(max_bytes=10**6)
    base.cpu_embeddings_cache.set(_LEAKED_KEY, torch.ones(4, 8))

    assert base.cpu_embeddings_cache.get(_LEAKED_KEY) is not None


def test_b_cache_does_not_leak_into_a_later_test():
    """`clear` has to release the budget as well as the entries: a cache that
    forgot the bytes it no longer holds is full for the rest of the process."""
    assert base.cpu_embeddings_cache is not None
    assert base.cpu_embeddings_cache.get(_LEAKED_KEY) is None
    assert base.cpu_embeddings_cache.used_bytes == 0
