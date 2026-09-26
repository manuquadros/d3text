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
    bytes_to_windowed_tensor,
    read_provenance,
    tensor_to_bytes,
    windowed_tensor_to_bytes,
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


def test_a_windowed_tensor_survives_the_round_trip():
    """The 3-D counterpart of `test_an_embedding_survives_the_round_trip`:
    the layer-boundary codec shares `_pack`/`_unpack` with the
    embeddings-store codec above, and this pins that each still round-trips
    on its own."""
    tensor = torch.tensor(
        [
            [[0.5, -1.25, 2.0], [0.0, 3.5, -0.75]],
            [[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]],
        ]
    )

    restored = bytes_to_windowed_tensor(windowed_tensor_to_bytes(tensor))

    assert restored.shape == tensor.shape
    torch.testing.assert_close(restored.float(), tensor)


def test_the_windowed_bytes_are_what_the_lmdb_holds():
    """The layer-boundary store's on-disk header, pinned the same way as
    `test_the_bytes_are_what_the_lmdb_holds`: a shared `_pack`/`_unpack`
    that drifted the two formats' layouts together would still pass a
    round-trip test, so the wire layout needs its own pin."""
    packed = windowed_tensor_to_bytes(torch.ones(2, 3, 4))

    assert isinstance(packed, bytes)
    magic, version, windows, tokens, features = struct.unpack_from(
        "<4sBIII", packed
    )
    assert (magic, version, windows, tokens, features) == (
        b"D3WL",
        1,
        2,
        3,
        4,
    )
    assert len(blosc2.decompress2(packed[17:])) == 2 * 3 * 4 * 2


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
    """The failure a run's smoke stage exists to catch.

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
    store.get(11, 99)  # stored at a length the encodings disagree with

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


def test_a_store_records_the_precision_its_forward_ran_in(tmp_path):
    """`select_amp_dtype` names a machine, not a dtype, and the precompute's
    own precision has changed once already, so two stores agreeing on model,
    window and stride can still hold forwards computed differently. Nothing
    else on disk tells them apart."""
    provenance = StoreProvenance(
        base_model=BASE_MODEL,
        max_length=512,
        stride=20,
        forward_dtype="torch.bfloat16",
    )
    with lmdb.open(str(tmp_path / "store"), map_size=2**20) as env:
        write_provenance(env, provenance)

        assert read_provenance(env) == provenance


def test_a_record_written_before_the_dtype_field_still_loads(tmp_path):
    """The field is optional on read and must stay so: a ~100 GiB store
    written by a build predating it is on disk, and treating its record as
    unreadable would strand it. An absent value is a complete record from an
    older writer -- reported as `None`, never raised over. The format number
    is deliberately unchanged, since nothing about how the other fields are
    interpreted moved."""
    older_record = {
        "format": 1,
        "base_model": BASE_MODEL,
        "max_length": 512,
        "stride": 20,
    }
    with lmdb.open(str(tmp_path / "store"), map_size=2**20) as env:
        with env.begin(write=True) as transaction:
            transaction.put(
                b"\x00provenance", json.dumps(older_record).encode()
            )

        recorded = read_provenance(env)

    assert recorded == PROVENANCE
    assert recorded.forward_dtype is None


def test_the_dtype_does_not_decide_whether_two_passes_share_a_store():
    """Identity is model, window and stride. The dtype says how the matrices
    were computed, not what they are of, so a store resumed by a build that
    computes it differently is still one store -- and `record_provenance`
    compares on this rather than on equality for exactly that reason."""
    older = StoreProvenance(base_model=BASE_MODEL, max_length=512, stride=20)
    newer = StoreProvenance(
        base_model=BASE_MODEL,
        max_length=512,
        stride=20,
        forward_dtype="torch.bfloat16",
    )

    assert older != newer
    assert older.identity == newer.identity


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


def test_an_unknown_key_in_the_record_does_not_refuse_it(tmp_path):
    """A future build may add a diagnostic field without bumping the format
    number (`forward_dtype` already did this once); reading it back must not
    mistake that extra field for one of this build's own missing fields."""
    record = {
        "format": 1,
        "base_model": BASE_MODEL,
        "max_length": 512,
        "stride": 20,
        "future_diagnostic_field": "whatever",
    }
    with lmdb.open(str(tmp_path / "store"), map_size=2**20) as env:
        with env.begin(write=True) as transaction:
            transaction.put(b"\x00provenance", json.dumps(record).encode())

        assert read_provenance(env) == PROVENANCE


def test_a_numeric_field_written_as_a_string_still_reads_as_an_int(tmp_path):
    """The reader casts a numeric string to int rather than passing it
    straight to the dataclass, where `Positive`/`NonNegative` would refuse
    it as the wrong type."""
    record = {
        "format": 1,
        "base_model": BASE_MODEL,
        "max_length": "512",
        "stride": 20,
    }
    with lmdb.open(str(tmp_path / "store"), map_size=2**20) as env:
        with env.begin(write=True) as transaction:
            transaction.put(b"\x00provenance", json.dumps(record).encode())

        recorded = read_provenance(env)

    assert recorded == PROVENANCE
    assert isinstance(recorded.max_length, int)


def test_an_uncastable_field_raises_provenance_error(tmp_path):
    """A value neither this build nor an older one could have written is a
    record it cannot read, not a raw `int()` failure escaping past it."""
    record = {
        "format": 1,
        "base_model": BASE_MODEL,
        "max_length": "not-a-number",
        "stride": 20,
    }
    with lmdb.open(str(tmp_path / "store"), map_size=2**20) as env:
        with env.begin(write=True) as transaction:
            transaction.put(b"\x00provenance", json.dumps(record).encode())

        with pytest.raises(ProvenanceError):
            read_provenance(env)
