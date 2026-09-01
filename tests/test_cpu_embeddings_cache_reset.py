"""The process-wide CPU embeddings cache must not leak across tests.

`base.cpu_embeddings_cache` is module state keyed by base model and document
id, and fixtures across the suite reuse small integer pmids for unrelated
documents — so on a machine whose `config.toml` enables the cache, one test's
entry could be read back stale by a later one. The first test here enables the
cache in-process and writes such an entry; the second proves it did not
survive. Order matters: pytest collects a file's tests in definition order.
"""

from cacheout import Cache
from d3text.models import base

_LEAKED_DOC_ID = 11


def test_a_populates_the_process_wide_cache():
    base.cpu_embeddings_cache = Cache(maxsize=10)
    base.cpu_embeddings_cache.set(_LEAKED_DOC_ID, "stale embedding")

    assert base.cpu_embeddings_cache.get(_LEAKED_DOC_ID) == "stale embedding"


def test_b_cache_does_not_leak_into_a_later_test():
    assert base.cpu_embeddings_cache is not None
    assert base.cpu_embeddings_cache.get(_LEAKED_DOC_ID) is None
