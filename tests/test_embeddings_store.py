"""The embeddings LMDB is written by one function and read by another.

`precompute-embeddings` compresses each document's embedding matrix on the way
into the store. Nothing in the library reads it back yet, so until something
does, the only thing keeping the byte layout honest is that its inverse exists
and round-trips.
"""

import struct

import blosc2
import pytest
import torch
from beartype.roar import BeartypeCallHintParamViolation

from d3text.embeddings_store import bytes_to_tensor, tensor_to_bytes


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
