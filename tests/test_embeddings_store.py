"""The embeddings LMDB is written by one function and read by another.

`precompute-embeddings` compresses each document's embedding matrix on the way
into the store. Nothing in the library reads it back yet, so until something
does, the only thing keeping the byte layout honest is that its inverse exists
and round-trips.
"""

import json
import logging
import struct

import blosc2
import lmdb
import pytest
import torch
from beartype.roar import BeartypeCallHintParamViolation

from d3text.embeddings_store import (
    EmbeddingsStore,
    ProvenanceError,
    StoreProvenance,
    bytes_to_tensor,
    read_provenance,
    tensor_to_bytes,
    write_provenance,
)

BASE_MODEL = "michiyasunaga/BioLinkBERT-base"
PROVENANCE = StoreProvenance(base_model=BASE_MODEL, max_length=512, stride=20)


def test_an_embedding_survives_the_round_trip():
    embedding = torch.tensor([[0.5, -1.25, 2.0], [0.0, 3.5, -0.75]])

    restored = bytes_to_tensor(tensor_to_bytes(embedding))

    assert restored.shape == embedding.shape
    torch.testing.assert_close(restored.float(), embedding)


def test_the_stored_embedding_is_bfloat16():
    """A deliberate, lossy narrowing of the store: these are frozen
    activations, not weights that will be trained further. The dtype is part of
    the contract — a reader that assumed fp32 would read the matrix at twice
    its width."""
    restored = bytes_to_tensor(tensor_to_bytes(torch.rand(4, 8)))

    assert restored.dtype == torch.bfloat16


def test_a_value_past_the_half_precision_range_is_kept():
    """The half of the fp16 -> bf16 trade that is a gain.

    fp16 tops out around 65504 and sent anything beyond it to infinity. bf16
    keeps fp32's exponent, so the range is no longer where the round trip stops
    being exact."""
    restored = bytes_to_tensor(tensor_to_bytes(torch.tensor([[1e6]])))

    assert torch.isfinite(restored).all()
    # Within bf16's ~1-in-256 resolution, which the next test pins directly.
    torch.testing.assert_close(
        restored.float(), torch.tensor([[1e6]]), rtol=1e-2, atol=0.0
    )


def test_precision_past_the_bfloat16_mantissa_is_not_silently_kept():
    """The half of the trade that is a cost, and the reason the store shrank.

    bf16 carries 8 mantissa bits against fp16's 10, so it resolves about 1 part
    in 256. 1.0 and 1.001 are distinct in fp16 and are the same number here."""
    restored = bytes_to_tensor(tensor_to_bytes(torch.tensor([[1.001]])))

    assert restored.float().item() == 1.0


def test_a_non_contiguous_tensor_is_stored_in_its_own_layout():
    """Embeddings reach the store transposed or sliced often enough that the
    bit-pattern `view` would otherwise serialise the wrong buffer — or refuse
    to run at all."""
    embedding = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).T
    assert not embedding.is_contiguous()

    restored = bytes_to_tensor(tensor_to_bytes(embedding))

    torch.testing.assert_close(restored.float(), embedding)


def test_the_bytes_are_what_the_lmdb_holds():
    """The store's on-disk format, not just a self-consistent pair of
    functions: `compress2` records neither shape nor dtype, so the header is
    the only thing that carries them."""
    packed = tensor_to_bytes(torch.ones(2, 3))

    assert isinstance(packed, bytes)
    magic, version, rows, columns = struct.unpack_from("<4sBII", packed)
    assert (magic, version, rows, columns) == (b"D3EB", 1, 2, 3)
    assert len(blosc2.decompress2(packed[13:])) == 2 * 3 * 2


def test_a_blob_from_the_previous_format_is_refused():
    """fp16 and bf16 share an itemsize, so an old `pack_array` blob would
    decompress to the right number of bytes and reinterpret as a plausible
    matrix of garbage. The magic is what turns that into an error."""
    old = blosc2.pack_array(
        torch.rand(4, 8).to(torch.float16).numpy(),
        codec=blosc2.Codec.ZSTD,
        clevel=9,
        filter=blosc2.Filter.BITSHUFFLE,
    )

    with pytest.raises(ValueError, match="not an embeddings-store blob"):
        bytes_to_tensor(old)


def test_a_future_format_version_is_refused():
    packed = tensor_to_bytes(torch.ones(2, 3))
    bumped = struct.pack("<4sB", b"D3EB", 2) + packed[5:]

    with pytest.raises(ValueError, match="version 2 is not readable"):
        bytes_to_tensor(bumped)


def test_a_truncated_blob_is_refused_rather_than_unpacked():
    with pytest.raises(ValueError, match="at least 13 bytes of header"):
        bytes_to_tensor(b"D3E")


def test_only_a_token_feature_matrix_is_storable():
    """The header carries exactly two shape fields, so the annotation on
    `tensor_to_bytes` is load-bearing rather than documentation: a stray batch
    dimension has nowhere to be recorded and must not reach the codec."""
    with pytest.raises(BeartypeCallHintParamViolation):
        tensor_to_bytes(torch.rand(2, 4, 8))


def _store(tmp_path, documents):
    """An `EmbeddingsStore` over an LMDB holding `documents`."""
    path = tmp_path / "store"
    with lmdb.open(str(path), map_size=2**24) as env:
        write_provenance(env, PROVENANCE)
        with env.begin(write=True) as transaction:
            for pubmed_id, embedding in documents.items():
                transaction.put(
                    str(pubmed_id).encode(), tensor_to_bytes(embedding)
                )
    return EmbeddingsStore(path, BASE_MODEL)


def test_a_store_that_answers_nothing_is_distinguishable_from_one_that_does(
    tmp_path, caplog
):
    """The failure the DEC-03 smoke check exists to catch.

    A store keyed on ids this corpus does not use answers every `get` with a
    miss, which is deliberately silent — a miss is also what a document that
    was never embedded looks like. The opening line says only that the path
    opened, so without a line on the first hit there is nothing in a log that
    separates a store being read from one merely configured.
    """
    store = _store(tmp_path, {11: torch.rand(4, 8)})

    with caplog.at_level(logging.INFO, logger="d3text.embeddings_store"):
        assert store.get(22, 4) is None
        assert not [r for r in caplog.records if "served document" in r.message]

        assert store.get(11, 4) is not None
        assert [r for r in caplog.records if "served document" in r.message]


def test_the_served_line_is_logged_once_however_many_documents_are_read(
    tmp_path, caplog
):
    """It is a confirmation, not a running commentary: the training loop asks
    the store for every document of every epoch."""
    store = _store(tmp_path, {11: torch.rand(4, 8), 22: torch.rand(4, 8)})

    with caplog.at_level(logging.INFO, logger="d3text.embeddings_store"):
        store.get(11, 4)
        store.get(22, 4)

    served = [r for r in caplog.records if "served document" in r.message]
    assert len(served) == 1


def test_close_reports_the_hit_rate(tmp_path, caplog):
    """A run that reads half its documents from the store costs the other half
    at the base model's speed and says nothing about it. `close` is the only
    moment that sees the totals, because `embeddings_store()` caches the reader
    for the life of the process and nothing owns it."""
    store = _store(tmp_path, {11: torch.rand(4, 8)})

    store.get(11, 4)  # hit
    store.get(22, 4)  # never embedded
    store.get(11, 99)  # stored against a different window

    assert (store.hits, store.misses, store.mismatches) == (1, 1, 1)

    with caplog.at_level(logging.INFO, logger="d3text.embeddings_store"):
        store.close()

    assert "served 1 of 3 documents (33.3%)" in caplog.text


def test_closing_twice_neither_reports_twice_nor_reopens(tmp_path, caplog):
    """`close` is registered with `atexit` and is also callable by hand; the
    second call must not double-close the environment."""
    store = _store(tmp_path, {11: torch.rand(4, 8)})
    store.get(11, 4)

    store.close()
    caplog.clear()  # the first close reported; only the second one is the test
    with caplog.at_level(logging.INFO, logger="d3text.embeddings_store"):
        store.close()

    assert "served" not in caplog.text


def test_a_store_nobody_asked_reports_nothing_at_close(tmp_path, caplog):
    """Every `evaluate` of a checkpoint whose config names a store opens one;
    a summary line for a store that answered no question would be noise."""
    store = _store(tmp_path, {11: torch.rand(4, 8)})

    caplog.clear()  # the constructor logs that the path opened
    with caplog.at_level(logging.INFO, logger="d3text.embeddings_store"):
        store.close()

    assert "served" not in caplog.text
    assert "never asked" in store.summary()


# --------------------------------------------------------------------------- #
# StoreProvenance                                                              #
# --------------------------------------------------------------------------- #
def test_a_store_reports_the_model_window_and_stride_it_was_written_with(
    tmp_path,
):
    """The three inputs `precompute-embeddings` takes. None of them is
    recoverable from a matrix: the header carries rows and columns, and those
    are the same for every encoder of a given hidden size."""
    with lmdb.open(str(tmp_path / "store"), map_size=2**20) as env:
        write_provenance(env, PROVENANCE)

        assert read_provenance(env) == PROVENANCE


def test_a_store_from_before_provenance_was_recorded_reports_none(tmp_path):
    """Absent, not empty: `None` is what says the store cannot be attributed
    at all, which is a different thing from having been written by a model
    whose name happens to be blank."""
    with lmdb.open(str(tmp_path / "store"), map_size=2**20) as env:
        assert read_provenance(env) is None


def test_a_provenance_record_from_a_future_format_is_refused(tmp_path):
    """A record this build cannot read is not a store it may read anyway: the
    fields it would check the base model against are the ones it cannot
    parse."""
    with lmdb.open(str(tmp_path / "store"), map_size=2**20) as env:
        with env.begin(write=True) as transaction:
            transaction.put(
                b"\x00provenance",
                json.dumps({"format": 99, "base_model": BASE_MODEL}).encode(),
            )

        with pytest.raises(ProvenanceError, match="format"):
            read_provenance(env)


def test_the_provenance_key_is_not_one_a_pubmed_id_can_spell(tmp_path):
    """It shares the keyspace with the documents, so a document able to reach
    it would overwrite the record — or be read as one."""
    with lmdb.open(str(tmp_path / "store"), map_size=2**20) as env:
        write_provenance(env, PROVENANCE)
        with env.begin() as transaction:
            keys = list(transaction.cursor().iternext(values=False))

    (key,) = keys
    assert not key.decode("latin1").isdigit()


def test_a_damaged_provenance_record_is_not_read_as_an_absent_one(tmp_path):
    """`None` sends a writer down the path for a store that has never been
    stamped, which would relabel documents nobody can attribute."""
    with lmdb.open(str(tmp_path / "store"), map_size=2**20) as env:
        with env.begin(write=True) as transaction:
            transaction.put(b"\x00provenance", b"{not json")

        with pytest.raises(ProvenanceError, match="cannot read"):
            read_provenance(env)
