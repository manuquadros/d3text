"""`EmbeddingsStore`: reading back what `precompute-embeddings` wrote.

The write half is pinned by `test_embeddings_store.py`. What is new here is
the store's *refusals* — a document it does not hold, and one whose row count
disagrees with the encodings the DataLoader serves. Both must send the caller
back to the base model rather than hand over a matrix that silently does not
line up with the tokens it is about, because nothing downstream can tell the
difference: the shapes are plausible either way and the loss simply gets worse.

The third refusal is whole-store and happens at the constructor: a store that
was not written by the run's own base model. Nothing in a matrix says which
encoder produced it, and two encoders of the same hidden size produce matrices
of the same shape, so this one has to be caught from the record the writer
leaves or not at all.
"""

import lmdb
import pytest
import torch
from d3text.embeddings_store import (
    EmbeddingsStore,
    ProvenanceError,
    StoreProvenance,
    bytes_to_tensor,
    tensor_to_bytes,
    write_provenance,
)

BASE_MODEL = "michiyasunaga/BioLinkBERT-base"
PROVENANCE = StoreProvenance(base_model=BASE_MODEL, max_length=512, stride=20)


def _write_store(path, provenance=PROVENANCE, documents=None):
    """An LMDB stamped with `provenance` and holding `documents`."""
    env = lmdb.open(str(path), map_size=8 * 1024**2)
    if provenance is not None:
        write_provenance(env, provenance)
    with env.begin(write=True) as transaction:
        for pubmed_id, embedding in (documents or {}).items():
            transaction.put(str(pubmed_id).encode(), tensor_to_bytes(embedding))
    env.close()

    return path


@pytest.fixture
def store_path(tmp_path):
    """An LMDB holding one 12-token document under pubmed id 100."""
    return _write_store(
        tmp_path / "embeddings", documents={100: torch.rand(12, 8)}
    )


def test_a_stored_document_comes_back_as_bfloat16(store_path):
    store = EmbeddingsStore(store_path, BASE_MODEL)

    embeddings = store.get(100, expected_tokens=12)

    assert embeddings is not None
    assert embeddings.shape == (12, 8)
    assert embeddings.dtype == torch.bfloat16
    assert (store.hits, store.misses, store.mismatches) == (1, 0, 0)


def test_the_key_is_the_pubmed_id_however_it_is_spelled(store_path):
    """The corpus disagrees with itself about the type of a pubmed id — int in
    the csv splits, str in the ndjson dump — and the batch item carries a
    Tensor. All three have to reach the same key."""
    store = EmbeddingsStore(store_path, BASE_MODEL)

    assert store.get(100, expected_tokens=12) is not None
    assert store.get("100", expected_tokens=12) is not None


def test_a_document_the_store_does_not_hold_is_a_miss(store_path):
    store = EmbeddingsStore(store_path, BASE_MODEL)

    assert store.get(999, expected_tokens=12) is None
    assert (store.hits, store.misses) == (0, 1)


def test_a_row_count_that_disagrees_with_the_encodings_is_refused(store_path):
    """The failure this exists for: the store and the encodings were built from
    different text, so the stored matrix holds a different number of rows than
    the document has tokens, and they would be read as though they were the
    document's own."""
    store = EmbeddingsStore(store_path, BASE_MODEL)

    assert store.get(100, expected_tokens=11) is None
    assert (store.hits, store.mismatches) == (0, 1)


def test_the_mismatch_is_warned_about_once(store_path, caplog):
    """Once, not once per document: a mismatched store mismatches on every
    document in a 10,000-document epoch, and a warning repeated that many times
    is one nobody reads."""
    store = EmbeddingsStore(store_path, BASE_MODEL)

    with caplog.at_level("WARNING"):
        for _ in range(3):
            store.get(100, expected_tokens=11)

    warnings = [
        record for record in caplog.records if record.levelname == "WARNING"
    ]
    assert len(warnings) == 1
    # And it must name the only cause a row count can have. The window is not
    # one: the aggregated count is the token count at any `max_length`, so
    # sending the operator to rebuild a 100 GiB store without `--max_length`
    # spends hours on something arithmetically incapable of being the fault.
    message = warnings[0].getMessage()
    assert "different text" in message and "encodings" in message
    assert "not a window mismatch" in message
    assert "--max_length" not in message
    assert store.mismatches == 3


def test_a_store_that_cannot_be_opened_raises(tmp_path):
    """`embeddings_store()` turns this into a disabled store and a warning; the
    reader itself does not get to decide that."""
    with pytest.raises(lmdb.Error):
        EmbeddingsStore(tmp_path / "nothing-here", BASE_MODEL)


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
    store = EmbeddingsStore(store_path, BASE_MODEL)
    embeddings = store.get(100, expected_tokens=12)
    assert embeddings is not None
    before = embeddings.clone()

    store.close()

    assert torch.equal(embeddings, before)
    assert embeddings.sum().isfinite()


def test_a_store_written_by_another_model_is_refused(tmp_path):
    """The failure no shape can catch. Most base encoders worth trying are 768
    wide, so a store built with one and read under another decodes into a
    matrix of exactly the right size holding another model's representation
    space — the documents it holds reach the heads as one model's activations
    and the documents it misses as the run's own."""
    path = _write_store(
        tmp_path / "other",
        provenance=StoreProvenance("google-bert/bert-base-cased", 512, 20),
        documents={100: torch.rand(12, 8)},
    )

    with pytest.raises(ProvenanceError, match="google-bert/bert-base-cased"):
        EmbeddingsStore(path, BASE_MODEL)


def test_a_store_that_does_not_say_who_wrote_it_is_refused(tmp_path):
    """An unstamped store is not a store known to be right; it is a store
    nothing attributes to anything. Reading it would be the same gamble as
    reading one stamped with the wrong model, taken with less information."""
    path = _write_store(
        tmp_path / "unstamped",
        provenance=None,
        documents={100: torch.rand(12, 8)},
    )

    with pytest.raises(ProvenanceError, match="does not record which model"):
        EmbeddingsStore(path, BASE_MODEL)


def test_the_store_a_run_did_write_is_read(store_path):
    """The other half: refusing every store would also refuse the one the run
    is entitled to, and would read as a store that is merely never hit."""
    store = EmbeddingsStore(store_path, BASE_MODEL)

    assert store.provenance == PROVENANCE
    assert store.get(100, expected_tokens=12) is not None
