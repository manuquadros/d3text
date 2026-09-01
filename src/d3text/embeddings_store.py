"""The codec for the precomputed-embeddings LMDB.

`tensor_to_bytes` and `bytes_to_tensor` are the two halves of that store's
contract; keeping them in one place is what makes it a contract rather than two
independent guesses at a byte layout. Nothing else may reach for `blosc2`
directly — `unpack_array` segfaults on a blob it did not write rather than
raising. See the data page of the documentation for the bf16 measurement, the
blob header and the provenance record.
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

    The cast to bf16 is a deliberate, lossy narrowing: these are frozen
    base-model activations, not weights that will be trained further.

    :param tensor: one document's token embeddings.
    :return: the header plus the blosc2 frame to store.
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
    `buffers=True` need not copy the mapped page in; `decompress2` allocates
    its own output, so the mapped page leaves the lifetime chain before
    `frombuffer` is reached.

    :param packed: a blob as `tensor_to_bytes` wrote it.
    :return: the stored matrix.
    :raises ValueError: if the blob carries another format's magic number.
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
    give: the caller decides whether an unattributable store is worth running
    without or worth stopping for.
    """


@dataclasses.dataclass(frozen=True)
class StoreProvenance:
    """What produced a store's matrices, recorded when it is written.

    The base model is the field that matters: 768 dimensions are 768 dimensions
    whichever encoder emitted them. The window and stride are recorded beside
    it because neither is otherwise recoverable from the store.
    """

    base_model: str
    max_length: int
    stride: int


def read_provenance(env: lmdb.Environment) -> StoreProvenance | None:
    """What wrote `env`, or `None` if it does not say.

    `None` means nothing on disk attributes those matrices to anything, which
    is not the same as a store written by the wrong model.

    :param env: the open LMDB environment.
    :return: the recorded provenance, or None if it records none.
    :raises ProvenanceError: if the record is there but this build cannot read
        it, which reading as unstamped would hide behind the friendlier
        diagnosis.
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
    """Stamp `env` with what is writing into it.

    :param env: the open LMDB environment.
    :param provenance: what this run will write.
    """
    record = {"format": _PROVENANCE_FORMAT} | dataclasses.asdict(provenance)
    with env.begin(write=True) as transaction:
        transaction.put(
            _PROVENANCE_KEY, json.dumps(record, sort_keys=True).encode()
        )


class EmbeddingsStore:
    """Read-only view of a `precompute-embeddings` LMDB.

    Opened `readonly` and without a lock, since the writer has long since
    exited and a training run must not lock a 100 GiB file it only reads;
    `readahead=False` because the store is far larger than RAM and the
    documents are visited in shuffled order. Opening one names the base model
    the run will feed the matrices to, and a store not recorded as written by
    it is refused here rather than read.
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

        :raises ProvenanceError: if the store records no provenance, or records
            another model.
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
            # Includes the reserved provenance entry, so a store holding no
            # documents at all reports zero here rather than one: a stamp
            # written before the weights that would have populated it ever
            # loaded looks, without this, exactly like real work that
            # happens to be for another model.
            documents = self.env.stat()["entries"] - 1
            msg = (
                f"{self.path} is stamped for {recorded.base_model} and this "
                f"run's base model is {base_model}; it holds {documents} "
                f"document(s). Their hidden widths may agree, in which case "
                f"nothing downstream would fail: the documents the store "
                f"holds would reach the heads as one model's activations and "
                f"the rest as another's."
            )
            raise ProvenanceError(msg)
        return recorded

    def get(
        self, pubmed_id: int | str, expected_tokens: int
    ) -> Float[Tensor, "token feature"] | None:
        """The stored embeddings for `pubmed_id`, or `None` to compute them.

        `None` covers both ways an attributed store can fail to answer — the
        document was never embedded, or its row count disagrees with the
        encodings — and the caller's response to each is to run the base model.

        :param pubmed_id: the document to read.
        :param expected_tokens: the token count the batch item implies.
        :return: the stored matrix, or None.
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
                    "imply %d, so the two were built from different text; "
                    "this document, and every other that disagrees, is being "
                    "embedded live instead. This is not a window mismatch: "
                    "the aggregated row count comes to the document's token "
                    "count whatever window the store was built at. It is a "
                    "corpus reader that changed between the two builds, so "
                    "rebuild whichever artifact predates that change — the "
                    "encodings are much the cheaper of the two.",
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
        """One line of what the store answered, for the end of a run's log.

        :return: the hit and miss counts as a sentence.
        """
        asked = self.hits + self.misses + self.mismatches
        if not asked:
            return f"{self.path} was never asked for a document"
        return (
            f"{self.path} served {self.hits:,} of {asked:,} documents "
            f"({self.hits / asked:.1%}), {self.misses:,} not stored, "
            f"{self.mismatches:,} stored at a length the encodings disagree "
            f"with"
        )

    def close(self) -> None:
        """Close the environment and report what the store answered.

        Registered with `atexit`, and the only moment that sees the totals:
        nothing owns the reader, which is cached for the life of the process. A
        hit rate well under 1.0 is the difference between a run that reads the
        store and one that merely opened it.
        """
        if self._closed:
            return
        self._closed = True
        if self.hits + self.misses + self.mismatches:
            logger.info("%s", self.summary())
        self.env.close()
