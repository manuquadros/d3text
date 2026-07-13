"""The embeddings LMDB is written by one function and read by another.

`precompute-embeddings` compresses each document's embedding matrix on the way
into the store. Nothing in the library reads it back yet, so until something
does, the only thing keeping the byte layout honest is that its inverse exists
and round-trips.
"""

import blosc2
import torch

from d3text.embeddings_store import bytes_to_tensor, tensor_to_bytes


def test_an_embedding_survives_the_round_trip():
    embedding = torch.tensor([[0.5, -1.25, 2.0], [0.0, 3.5, -0.75]])

    restored = bytes_to_tensor(tensor_to_bytes(embedding))

    assert restored.shape == embedding.shape
    torch.testing.assert_close(restored.float(), embedding)


def test_the_stored_embedding_is_half_precision():
    """A deliberate, lossy halving of the store: these are frozen activations,
    not weights that will be trained further. The dtype is part of the contract
    — a reader that assumed fp32 would read the matrix at twice its width."""
    restored = bytes_to_tensor(tensor_to_bytes(torch.rand(4, 8)))

    assert restored.dtype == torch.float16


def test_a_value_past_the_half_precision_range_is_not_silently_kept():
    """fp16 tops out around 65504. The round trip is exact only for what fp16
    can represent, and this pins where that stops being true."""
    restored = bytes_to_tensor(tensor_to_bytes(torch.tensor([[1e6]])))

    assert torch.isinf(restored).all()


def test_a_non_contiguous_tensor_is_stored_in_its_own_layout():
    """Embeddings reach the store transposed or sliced often enough that
    `pack_array` would otherwise serialise the wrong buffer."""
    embedding = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).T
    assert not embedding.is_contiguous()

    restored = bytes_to_tensor(tensor_to_bytes(embedding))

    torch.testing.assert_close(restored.float(), embedding)


def test_the_bytes_are_what_the_lmdb_holds():
    """The store's on-disk format, not just a self-consistent pair of
    functions: a reader reaching for blosc2 directly must find an array."""
    packed = tensor_to_bytes(torch.ones(2, 3))

    assert isinstance(packed, bytes)
    assert blosc2.unpack_array(packed).shape == (2, 3)
