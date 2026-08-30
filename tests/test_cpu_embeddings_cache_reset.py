"""Regression coverage for the cross-test leak through the process-wide
CPU embeddings cache in `d3text.models.base`.

`base.cpu_embeddings_cache` is module state keyed by a bare document id, and
several fixtures across the suite reuse small integer pmids for unrelated
documents. On a machine whose `config.toml` enables the cache
(`cpu_embeddings_cache_size` nonzero), an entry one test writes could be read
back, stale, by a later test that happens to reuse the same id — the failure
only reproduced on such a machine, never in a fresh checkout or CI.

These two tests exercise the leak and its fix directly, with no `config.toml`
needed: the first enables the cache in-process (as a real `config.toml`
would) and writes an entry under an id another test's fixtures commonly use;
the second, running after it, proves that entry did not survive into a new
test. Order matters and is relied upon: pytest collects a file's tests in
definition order, and nothing in this suite randomizes it.
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
