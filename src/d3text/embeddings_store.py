"""The codec for the precomputed-embeddings LMDB.

`precompute-embeddings` stores one compressed token-embedding matrix per pubmed
id. `tensor_to_bytes` and `bytes_to_tensor` are the two halves of that store's
contract; keeping them in one place is what makes it a contract rather than two
independent guesses at a byte layout.

The store has no reader in the library yet — nothing in the training path opens
the LMDB — so `bytes_to_tensor` currently exists to prove the bytes that go in
are the bytes that come out. Whatever finally consumes the LMDB should go
through it rather than reach for `blosc2` again.

**The stored dtype is bf16, and the codec is zstd level 5 behind a byte
shuffle.** Both were measured rather than chosen
(`scripts/benchmarks/bench_codecs.py`, tabulated in
`design/perf_baseline.md`). Two results drive them:

- These activations are very nearly incompressible losslessly. Every lossless
  combination of codec, filter and level lands between 1.00x and 1.17x, because
  the low mantissa bits are noise no entropy coder can model. Storing bf16
  instead of fp16 spends two of those bits and gets 1.42x, which is 100.8 GiB
  rather than 121.9 for the whole corpus. It is the only near-lossless lever
  there is; the codec knobs are not one.
- `blosc2.pack_array` is 3.8x slower than `compress2` at identical settings,
  and pack_array-at-zstd9 was 72x slower than what is used here.

bf16 costs precision, not range: it keeps fp32's exponent and drops mantissa
bits, so a value fp16 would have overflowed to infinity now survives, while a
value fp16 held exactly may come back rounded. For frozen base-model
activations — read once, never trained further — that is the right side of the
trade, and `test_embeddings_store.py` pins both halves of it.

The blob is a 13-byte header followed by a blosc2 frame. `compress2` stores no
shape or dtype of its own, and numpy has no bfloat16, so the matrix travels as
its int16 bit pattern and the header is what says how to read it back. The
magic number is not decoration: a blob written by the previous fp16
`pack_array` format has the same itemsize as this one, so without a magic to
reject it, it would decode into a plausible matrix of garbage.
"""

import struct
import typing

import blosc2
import numpy
import torch
from jaxtyping import Float
from torch import Tensor

_MAGIC = b"D3EB"
_VERSION = 1
_HEADER = struct.Struct("<4sBII")

_CPARAMS: dict[str, typing.Any] = {
    "codec": blosc2.Codec.ZSTD,
    "clevel": 5,
    "filters": [blosc2.Filter.SHUFFLE],
    "filters_meta": [0],
}


def tensor_to_bytes(tensor: Float[Tensor, "token feature"]) -> bytes:
    """Compress `tensor` for storage.

    The cast to bf16 is a deliberate, lossy narrowing of the store: these are
    frozen base-model activations, not weights that will be trained further.
    `bytes_to_tensor` therefore round-trips the *stored* values exactly, but
    only approximates the fp32 tensor handed in here.
    """
    # `view` reinterprets the buffer, so it needs the bf16 values laid out
    # contiguously first — embeddings reach the store transposed or sliced
    # often enough that this is load-bearing, not defensive.
    array = (
        tensor.detach()
        .to(torch.bfloat16)
        .cpu()
        .contiguous()
        .view(torch.int16)
        .numpy()
    )
    rows, columns = array.shape
    body = typing.cast(bytes, blosc2.compress2(array, **_CPARAMS))

    return _HEADER.pack(_MAGIC, _VERSION, rows, columns) + body


def bytes_to_tensor(packed: bytes) -> Float[Tensor, "token feature"]:
    """The stored embedding matrix, as the bf16 tensor it was written as."""
    if len(packed) < _HEADER.size:
        msg = (
            f"a stored embedding is at least {_HEADER.size} bytes of header; "
            f"got {len(packed)}."
        )
        raise ValueError(msg)

    magic, version, rows, columns = _HEADER.unpack_from(packed)
    if magic != _MAGIC:
        msg = (
            f"not an embeddings-store blob: expected the magic {_MAGIC!r}, got "
            f"{magic!r}. A store written before this format carries a bare "
            f"blosc2 frame of fp16, which shares this format's itemsize and "
            f"would otherwise decode into a matrix of garbage; rebuild it with "
            f"`precompute-embeddings`."
        )
        raise ValueError(msg)
    if version != _VERSION:
        msg = (
            f"embeddings-store format version {version} is not readable by "
            f"this build, which writes version {_VERSION}."
        )
        raise ValueError(msg)

    # `frombuffer` hands back a read-only view; torch refuses to share memory
    # with one, so the copy is not optional.
    raw = numpy.frombuffer(
        blosc2.decompress2(packed[_HEADER.size :]), dtype=numpy.int16
    )

    return torch.from_numpy(raw.reshape(rows, columns).copy()).view(
        torch.bfloat16
    )
