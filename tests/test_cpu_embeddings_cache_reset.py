"""The process-wide CPU embeddings cache must not leak across tests.

Fixtures reuse small pmids for unrelated documents, so a leaked entry is
read back stale. The tests rely on pytest running them in definition order.
"""

import torch
from d3text.models import base

_LEAKED_KEY = ("prajjwal1/bert-mini", 0, 11)


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
