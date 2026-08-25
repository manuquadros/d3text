"""`EmbeddingsStore`: reading back what `precompute-embeddings` wrote.

The write half is pinned by `test_embeddings_store.py`. What is new here is
the store's *refusals* — a document it does not hold, and one whose row count
disagrees with the encodings the DataLoader serves. Both must send the caller
back to the base model rather than hand over a matrix that silently does not
line up with the tokens it is about, because nothing downstream can tell the
difference: the shapes are plausible either way and the loss simply gets worse.
"""

import lmdb
import pytest
import torch
from d3text.embeddings_store import (
    EmbeddingsStore,
    bytes_to_tensor,
    tensor_to_bytes,
)


@pytest.fixture
def store_path(tmp_path):
    """An LMDB holding one 12-token document under pubmed id 100."""
    path = tmp_path / "embeddings"
    env = lmdb.open(str(path), map_size=8 * 1024**2)
    with env.begin(write=True) as transaction:
        transaction.put(b"100", tensor_to_bytes(torch.rand(12, 8)))
    env.close()

    return path


def test_a_stored_document_comes_back_as_bfloat16(store_path):
    store = EmbeddingsStore(store_path)

    embeddings = store.get(100, expected_tokens=12)

    assert embeddings is not None
    assert embeddings.shape == (12, 8)
    assert embeddings.dtype == torch.bfloat16
    assert (store.hits, store.misses, store.mismatches) == (1, 0, 0)


def test_the_key_is_the_pubmed_id_however_it_is_spelled(store_path):
    """The corpus disagrees with itself about the type of a pubmed id — int in
    the csv splits, str in the ndjson dump — and the batch item carries a
    Tensor. All three have to reach the same key."""
    store = EmbeddingsStore(store_path)

    assert store.get(100, expected_tokens=12) is not None
    assert store.get("100", expected_tokens=12) is not None


def test_a_document_the_store_does_not_hold_is_a_miss(store_path):
    store = EmbeddingsStore(store_path)

    assert store.get(999, expected_tokens=12) is None
    assert (store.hits, store.misses) == (0, 1)


def test_a_row_count_that_disagrees_with_the_encodings_is_refused(store_path):
    """The failure this exists for: a store built with a different window holds
    a matrix of the wrong length, and the rows would be read as though they
    were the document's tokens."""
    store = EmbeddingsStore(store_path)

    assert store.get(100, expected_tokens=11) is None
    assert (store.hits, store.mismatches) == (0, 1)


def test_the_mismatch_is_warned_about_once(store_path, caplog):
    """Once, not once per document: a mismatched store mismatches on every
    document in a 10,000-document epoch, and a warning repeated that many times
    is one nobody reads."""
    store = EmbeddingsStore(store_path)

    with caplog.at_level("WARNING"):
        for _ in range(3):
            store.get(100, expected_tokens=11)

    warnings = [
        record for record in caplog.records if record.levelname == "WARNING"
    ]
    assert len(warnings) == 1
    assert "precompute-embeddings" in warnings[0].getMessage()
    assert store.mismatches == 3


def test_a_store_that_cannot_be_opened_raises(tmp_path):
    """`embeddings_store()` turns this into a disabled store and a warning; the
    reader itself does not get to decide that."""
    with pytest.raises(lmdb.Error):
        EmbeddingsStore(tmp_path / "nothing-here")


def test_a_memoryview_decodes_to_what_the_bytes_do():
    """The reader hands `bytes_to_tensor` LMDB's mapped page directly. It has
    to decode identically to the copy it used to make, or the saving would be
    bought with a different tensor."""
    packed = tensor_to_bytes(torch.rand(9, 5))

    from_bytes = bytes_to_tensor(packed)
    from_view = bytes_to_tensor(memoryview(packed))

    assert torch.equal(from_bytes, from_view)


def test_the_embeddings_outlive_the_transaction_that_lent_them(store_path):
    """`buffers=True` hands out a view of the mapped page, valid only inside
    its transaction. The decode copies, so the tensor must still be readable —
    and still be right — after the store is closed underneath it."""
    store = EmbeddingsStore(store_path)
    embeddings = store.get(100, expected_tokens=12)
    assert embeddings is not None
    before = embeddings.clone()

    store.close()

    assert torch.equal(embeddings, before)
    assert embeddings.sum().isfinite()
