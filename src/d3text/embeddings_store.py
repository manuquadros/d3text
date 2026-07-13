"""The codec for the precomputed-embeddings LMDB.

`precompute-embeddings` stores one compressed token-embedding matrix per pubmed
id. `tensor_to_bytes` and `bytes_to_tensor` are the two halves of that store's
contract; keeping them in one place is what makes it a contract rather than two
independent guesses at a byte layout.

The store has no reader in the library yet — nothing in the training path opens
the LMDB — so `bytes_to_tensor` currently exists to prove the bytes that go in
are the bytes that come out. Whatever finally consumes the LMDB should go
through it rather than reach for `blosc2` again.
"""

import typing

import blosc2
import torch
from jaxtyping import Float
from torch import Tensor


def tensor_to_bytes(tensor: Float[Tensor, "token feature"]) -> bytes:
    """Compress `tensor` for storage.

    The cast to fp16 is a deliberate, lossy halving of the store: these are
    frozen base-model activations, not weights that will be trained further.
    `bytes_to_tensor` therefore round-trips the *stored* values exactly, but
    only approximates the fp32 tensor handed in here.
    """
    array = tensor.detach().to(torch.float16).contiguous().cpu().numpy()
    return typing.cast(
        bytes,
        blosc2.pack_array(
            array,
            codec=blosc2.Codec.ZSTD,
            clevel=9,
            filter=blosc2.Filter.BITSHUFFLE,
        ),
    )


def bytes_to_tensor(packed: bytes) -> Float[Tensor, "token feature"]:
    """The stored embedding matrix, as the fp16 tensor it was written as."""
    # `unpack_array` hands back a read-only view; torch refuses to share memory
    # with one, so the copy is not optional.
    return torch.from_numpy(blosc2.unpack_array(packed).copy())
