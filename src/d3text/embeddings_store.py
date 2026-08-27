"""The codec for the precomputed-embeddings LMDB.

`precompute-embeddings` stores one compressed token-embedding matrix per pubmed
id. `tensor_to_bytes` and `bytes_to_tensor` are the two halves of that store's
contract; keeping them in one place is what makes it a contract rather than two
independent guesses at a byte layout.

The store also carries one record that is not a document: `StoreProvenance`,
under a reserved key, naming the model, window and stride that produced every
matrix in it. It is written once by `precompute-embeddings` and checked when a
reader opens the store, because the blob header carries only rows and columns
and those are equal between encoders of the same hidden size — the only
mistake the geometry cannot catch is the one worth catching.

`EmbeddingsStore` is the reader, and it is the only one: `get_token_embeddings`
consults it between the CPU cache and the base-model forward. Nothing else may
reach for `blosc2` directly — `blosc2.unpack_array` segfaults on a blob it did
not write rather than raising, so the magic-number check in `bytes_to_tensor` is
what stands between a stale store and a downed process.

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

import dataclasses
import json
import logging
import os
import struct
import typing

import blosc2
import lmdb
import numpy
import torch
from jaxtyping import Float
from torch import Tensor

logger = logging.getLogger(__name__)

_MAGIC = b"D3EB"
_VERSION = 1
_HEADER = struct.Struct("<4sBII")

# A pubmed id is decimal digits, so nothing this store is keyed on can spell a
# key holding a NUL.
_PROVENANCE_KEY = b"\x00provenance"
_PROVENANCE_FORMAT = 1

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


def bytes_to_tensor(
    packed: bytes | memoryview,
) -> Float[Tensor, "token feature"]:
    """The stored embedding matrix, as the bf16 tensor it was written as.

    Takes a `memoryview` as well as `bytes` so a reader under LMDB's
    `buffers=True` need not copy the mapped page in just to be allowed to pass
    it: at ~11 MiB a document that memcpy was a fifth of the read. Nothing
    below needs the copy — `unpack_from`, `decompress2` and `frombuffer` all
    take a buffer.

    What keeps the returned tensor valid once the transaction that lent the
    memory has closed is `decompress2`: it allocates its output, so the mapped
    page leaves the lifetime chain there, before `frombuffer` is reached.
    (An earlier version of this docstring credited the `.copy()` below. It is
    not that — deleting the copy leaves the tensor backed by `decompress2`'s
    own bytes and still valid. The copy is there for the reason its own
    comment gives: torch will not share memory with a read-only view.)
    """
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


class ProvenanceError(RuntimeError):
    """The store cannot be shown to hold this run's own activations.

    Raised rather than warned about because the reader has no safe answer to
    give: the caller decides whether a store it cannot attribute is worth
    running without (`d3text.models.models.embeddings_store` disables it and
    recomputes) or worth stopping for.
    """


@dataclasses.dataclass(frozen=True)
class StoreProvenance:
    """What produced a store's matrices, recorded when it is written.

    The base model is the field that matters: 768 dimensions are 768
    dimensions whichever encoder emitted them, so a store built with one and
    read by another hands the heads a second representation space with no
    shape to fail on. The window and the stride are recorded beside it because
    they are the other two inputs `precompute-embeddings` takes and neither is
    otherwise recoverable from the store.
    """

    base_model: str
    max_length: int
    stride: int


def read_provenance(env: lmdb.Environment) -> StoreProvenance | None:
    """What wrote `env`, or `None` if it does not say.

    `None` is a store written before provenance was recorded, which is not the
    same as a store written by the wrong model and is not distinguishable from
    one either: what it means is that nothing on disk attributes those
    matrices to anything.

    :raises ProvenanceError: if the record is there but this build cannot read
        it — a future format, or a damaged one. Either is a store whose
        matrices are unattributed in the way that matters, and reading it as
        though it were unstamped would hide that behind the friendlier of the
        two diagnoses.
    """
    with env.begin() as transaction:
        raw = transaction.get(_PROVENANCE_KEY)
    if raw is None:
        return None

    try:
        record = json.loads(raw)
        recorded_format = record["format"]
    except (json.JSONDecodeError, TypeError, KeyError) as error:
        msg = f"{env.path()} holds a provenance record this build cannot read."
        raise ProvenanceError(msg) from error

    if recorded_format != _PROVENANCE_FORMAT:
        msg = (
            f"{env.path()} records its provenance in format "
            f"{recorded_format!r}, which this build cannot read; it writes "
            f"and reads format {_PROVENANCE_FORMAT}."
        )
        raise ProvenanceError(msg)

    try:
        return StoreProvenance(
            base_model=str(record["base_model"]),
            max_length=int(record["max_length"]),
            stride=int(record["stride"]),
        )
    except (TypeError, KeyError, ValueError) as error:
        msg = (
            f"{env.path()} records a format-{_PROVENANCE_FORMAT} provenance "
            f"missing a field this build reads: {record!r}."
        )
        raise ProvenanceError(msg) from error


def write_provenance(
    env: lmdb.Environment, provenance: StoreProvenance
) -> None:
    """Stamp `env` with what is writing into it."""
    record = {"format": _PROVENANCE_FORMAT} | dataclasses.asdict(provenance)
    with env.begin(write=True) as transaction:
        transaction.put(
            _PROVENANCE_KEY, json.dumps(record, sort_keys=True).encode()
        )


class EmbeddingsStore:
    """Read-only view of a `precompute-embeddings` LMDB.

    Opened once per process and consulted per document, so the environment is
    opened with `readonly` and without a lock: the store is written by a
    separate command that has long since exited, and a training run must not
    take a writer lock on a 100 GiB file it only reads. `readahead=False`
    matters at that size — the store is far larger than RAM and the documents
    are visited in a shuffled order, so letting the kernel read ahead evicts
    pages that will be wanted again for pages that will not.

    Opening one names the base model the run will feed the matrices to, and a
    store that does not record having been written by that model is refused
    here rather than read. The width is no guard: 768 dimensions are 768
    dimensions whichever encoder produced them, so pointing a run at another
    768-dim encoder's store trains the heads on one model's activations for
    the documents the store holds and another's for the documents it misses —
    two representation spaces inside a batch, with nothing to raise on.

    A `get` verifies the stored matrix against the token count the batch item
    implies and returns `None` when they disagree, because the store and the
    encodings are two recordings of the same text made at different times and
    nothing else compares them: training reads the encodings, the store is
    built from the corpus, and a corpus reader fixed in between leaves the two
    describing different documents. That cannot raise on its own — both row
    counts are plausible — so it is checked here and the document falls back
    to the live forward.

    It does **not** catch a store built with a different token window, though
    an earlier version of this docstring claimed it did. The aggregated row
    count is `sum(L_i) - stride*(N-1)` while `sum(L_i)` is `T + stride*(N-1)`,
    so it comes to `T` for any `max_length` — measured identical at 512, 384,
    256, 128 and 64. Nor would a window mismatch misalign anything:
    `aggregate_embeddings` stitches the windows back into the document's own
    token order, so row *i* is token *i* regardless. What changes is how much
    context each token saw, which is a quality drift no row count can see.
    """

    def __init__(self, path: str | os.PathLike[str], base_model: str) -> None:
        self.path = os.fspath(path)
        self.env = lmdb.open(
            self.path,
            readonly=True,
            lock=False,
            readahead=False,
            max_readers=2048,
        )
        try:
            self.provenance = self._attributed_to(base_model)
        except ProvenanceError:
            self.env.close()
            raise
        self.hits = 0
        self.misses = 0
        self.mismatches = 0
        self._warned = False
        self._served = False
        self._closed = False
        logger.info(
            "Reading precomputed embeddings from %s, written by %s at window "
            "%d, stride %d",
            self.path,
            self.provenance.base_model,
            self.provenance.max_length,
            self.provenance.stride,
        )

    def _attributed_to(self, base_model: str) -> StoreProvenance:
        """The store's provenance, once it is this run's to read.

        :raises ProvenanceError: if the store records no provenance, or
            records another model.
        """
        recorded = read_provenance(self.env)
        if recorded is None:
            msg = (
                f"{self.path} does not record which model wrote it, so its "
                f"matrices cannot be attributed to {base_model}. A store "
                f"built by another encoder of the same width decodes into a "
                f"plausible matrix of the wrong representation space; rebuild "
                f"it with `precompute-embeddings`, which stamps what it "
                f"writes."
            )
            raise ProvenanceError(msg)
        if recorded.base_model != base_model:
            msg = (
                f"{self.path} was written by {recorded.base_model} and this "
                f"run's base model is {base_model}. Their hidden widths may "
                f"agree, in which case nothing downstream would fail: the "
                f"documents the store holds would reach the heads as one "
                f"model's activations and the rest as another's."
            )
            raise ProvenanceError(msg)
        return recorded

    def get(
        self, pubmed_id: int | str, expected_tokens: int
    ) -> Float[Tensor, "token feature"] | None:
        """The stored embeddings for `pubmed_id`, or `None` to compute them.

        `None` covers both ways an attributed store can fail to answer — the
        document was never embedded, or its row count disagrees with the
        encodings — and the caller's response to each is the same one it
        already had: run the base model. A store this run cannot be shown to
        have written never reaches here: it is refused when it is opened.
        """
        with self.env.begin(buffers=True) as transaction:
            blob = transaction.get(str(pubmed_id).encode())
            if blob is None:
                self.misses += 1
                return None
            stored = bytes_to_tensor(blob)

        if stored.shape[0] != expected_tokens:
            self.mismatches += 1
            if not self._warned:
                self._warned = True
                logger.warning(
                    "%s holds %d tokens for document %s where its encodings "
                    "imply %d, so the two were built from different text or "
                    "different windows; this document, and every other that "
                    "disagrees, is being embedded live instead. Which side "
                    "moved is not knowable from here: rebuilding the store "
                    "passing no --max_length fixes a window mismatch, and "
                    "rebuilding the encodings fixes a corpus reader that has "
                    "changed since they were written.",
                    self.path,
                    stored.shape[0],
                    pubmed_id,
                    expected_tokens,
                )
            return None

        if not self._served:
            # The opening line above says only that the path opened. A store
            # keyed on ids this corpus does not use answers every `get` with a
            # miss, which is silent by design and indistinguishable from having
            # no store at all — so the one thing worth saying out loud is that
            # a document actually came back from it.
            self._served = True
            logger.info(
                "%s served document %s from the store", self.path, pubmed_id
            )

        self.hits += 1
        return stored

    def summary(self) -> str:
        """One line of what the store answered, for the end of a run's log."""
        asked = self.hits + self.misses + self.mismatches
        if not asked:
            return f"{self.path} was never asked for a document"
        return (
            f"{self.path} served {self.hits:,} of {asked:,} documents "
            f"({self.hits / asked:.1%}), {self.misses:,} not stored, "
            f"{self.mismatches:,} stored against a different window"
        )

    def close(self) -> None:
        """Close the environment and report what the store answered.

        The summary is logged here rather than at any call site because there
        is none: `embeddings_store()` caches the reader for the life of the
        process and nothing owns it, so `close` — registered with `atexit` —
        is the only moment that sees the totals. A hit rate well under 1.0 is
        the difference between a run that reads the store and one that merely
        opened it, and it costs the whole speed-up without failing.
        """
        if self._closed:
            return
        self._closed = True
        if self.hits + self.misses + self.mismatches:
            logger.info("%s", self.summary())
        self.env.close()
