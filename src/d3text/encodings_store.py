"""The `precompute-encodings` LMDB: its document codec and provenance stamps.

Neither the tokenizer, the window nor the stride is recoverable from the stored
arrays, and none of them identifies the ids, which `content_digest`
fingerprints. See the data page of the documentation.
"""

import dataclasses
import hashlib
import json
import logging
import os
import struct
import threading
import warnings
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import TypedDict

import numpy
from numpy.typing import ArrayLike

from d3text import embeddings_store, lmdb_store
from d3text.constraints import NonNegative, Positive

logger = logging.getLogger(__name__)

# Formats 1 and 2 were HDF5 files, refused on sight rather than read.
_PROVENANCE_FORMAT = 3
# Document keys are pubmed ids and corpus-prefixed external ids, so none of
# them starts with a NUL.
_PROVENANCE_KEY = b"\x00provenance"
_CONTENT_DIGEST_KEY = b"\x00content_digest"

_MAGIC = b"D3EN"
_HEADER = struct.Struct("<4sBIII")
"""Magic, codec version, then the planes, windows and tokens per window."""
_ROW_DTYPE = numpy.dtype("<u4")
"""What each of a blob's planes is stored as: ids, mask, offset starts and
offset ends, all of them non-negative and far below 2**32."""

# `precompute-encodings` refuses the same store on its own path, so the
# remedy names another one.
_REBUILD = (
    "Build a new store with `precompute-encodings` under another path, or "
    "delete this one first."
)

_DIGEST_DTYPE = numpy.dtype("<u4")
"""Byte order the ids are hashed in, so one store digests the same anywhere."""


class Encoding(TypedDict):
    """One document's windows, as `BrendaDataset` hands them to a model.

    `input_ids` and `attention_mask` are `[window, token]`, `uint32` and
    `uint8`; `offset_mapping` is `[window, token, 2]` `uint32`, the
    character span each token covers in the document text.
    """

    input_ids: numpy.ndarray
    attention_mask: numpy.ndarray
    offset_mapping: numpy.ndarray


def encoding_to_bytes(encoding: Mapping[str, ArrayLike]) -> bytes:
    """One document's ids, mask and offsets, as the store holds them.

    :param encoding: `input_ids`, `attention_mask` and `offset_mapping`, as
        `Encoding` shapes them; any array-like of those shapes.
    :return: the blob to store.
    :raises ValueError: if the mask or the offsets disagree with the ids'
        shape, which no reader could index consistently.
    """
    ids = numpy.asarray(encoding["input_ids"], dtype=_ROW_DTYPE)
    mask = numpy.asarray(encoding["attention_mask"], dtype=_ROW_DTYPE)
    offsets = numpy.asarray(encoding["offset_mapping"], dtype=_ROW_DTYPE)
    if ids.ndim != 2 or mask.shape != ids.shape:
        msg = (
            f"input_ids of shape {ids.shape} and attention_mask of shape "
            f"{mask.shape}: both must be the same [window, token] shape."
        )
        raise ValueError(msg)
    if offsets.shape != (*ids.shape, 2):
        msg = (
            f"offset_mapping of shape {offsets.shape} for input_ids of shape "
            f"{ids.shape}: it must be {(*ids.shape, 2)}."
        )
        raise ValueError(msg)
    planes = numpy.stack((ids, mask, offsets[..., 0], offsets[..., 1]))
    return embeddings_store.array_to_blob(planes, _HEADER, _MAGIC)


def bytes_to_encoding(packed: bytes | memoryview) -> Encoding:
    """The document `encoding_to_bytes` stored.

    :param packed: a blob as `encoding_to_bytes` wrote it.
    :return: the document's ids, mask and offsets, as writable arrays.
    :raises ValueError: if the blob carries another format's magic or codec
        version.
    """
    ids, mask, starts, ends = embeddings_store.blob_to_array(
        packed, _HEADER, _MAGIC, "an encodings-store", _ROW_DTYPE
    )
    return {
        "input_ids": ids.copy(),
        "attention_mask": mask.astype(numpy.uint8),
        "offset_mapping": numpy.stack((starts, ends), axis=-1),
    }


class EncodingsStore(lmdb_store.LmdbStore):
    """An open encodings store: one LMDB value per document.

    A document is written in one transaction, so it is either there whole or
    absent. Keys are pubmed ids, or `external_key`'s for another corpus.
    What its handles share: `lmdb_store.LmdbStore`.

    :param path: the store's directory; created when opened writable.
    :param writable: whether to open for writing.
    :raises ValueError: if `path` is an encodings store of the older HDF5
        layout.
    :raises RuntimeError: if this process already holds `path` open for
        writing, or open read-only when a writer asks for it, or another
        process holds it open for writing.
    :raises OSError: if, opened writable, the writer lock cannot be taken for
        another reason.
    :raises lmdb.Error: if, opened read-only, no store exists at `path`.
    """

    def __init__(
        self, path: str | os.PathLike[str], *, writable: bool = False
    ) -> None:
        lmdb_store.refuse_hdf5(
            path,
            "an HDF5 encodings store, a layout this build no longer reads; "
            f"it writes an LMDB directory instead. {_REBUILD}",
        )
        self._in_pass = False
        super().__init__(path, writable=writable)

    def get(self, key: str) -> Encoding | None:
        """The document stored under `key`, or None if there is none.

        :param key: a pubmed id, or an `external_key`.
        :return: the document's ids, mask and offsets.
        """
        with self.env.begin(buffers=True) as transaction:
            blob = transaction.get(key.encode())
            return None if blob is None else bytes_to_encoding(blob)

    def windows(self, key: str) -> int | None:
        """How many windows the document under `key` holds, off its header.

        :param key: a pubmed id, or an `external_key`.
        :return: the window count, or None if there is no such document.
        """
        with self.env.begin(buffers=True) as transaction:
            blob = transaction.get(key.encode())
            if blob is None:
                return None
            magic, _, _, windows, _ = _HEADER.unpack_from(blob)
        if magic != _MAGIC:
            msg = f"{key} in {self.path} is not an encodings-store blob."
            raise ValueError(msg)
        return int(windows)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self.windows(key) is not None

    def put(self, key: str, encoding: Mapping[str, ArrayLike]) -> None:
        """Store `encoding` under `key`, replacing any document there.

        :param key: a pubmed id, or an `external_key`.
        :param encoding: the document, as `encoding_to_bytes` takes it.
        :raises ValueError: if its arrays disagree on shape; nothing is
            written then.
        """
        self._put_raw(key.encode(), encoding_to_bytes(encoding))


@dataclasses.dataclass(frozen=True)
class EncodingsProvenance:
    """What produced a store's token ids, recorded when it is written.

    The base model is what a reader ultimately cares about; `max_length` and
    `stride` are recorded beside it because they are the other two inputs
    `precompute-encodings` takes and neither is otherwise recoverable.
    """

    base_model: str
    max_length: Positive
    stride: NonNegative


def read_provenance(store: EncodingsStore) -> EncodingsProvenance | None:
    """What wrote `store`, or `None` if it does not say.

    :param store: an open encodings store.
    :return: the recorded provenance, or None if it records none.
    :raises ValueError: if the record is there but this build cannot read
        it, which reading as unstamped would hide.
    """
    raw = store._get_raw(_PROVENANCE_KEY)
    if raw is None:
        return None

    try:
        record = json.loads(raw)
        recorded_format = record["format"]
    except (json.JSONDecodeError, TypeError, KeyError) as error:
        msg = f"{store.path} holds a provenance record this build cannot read."
        raise ValueError(msg) from error

    if recorded_format != _PROVENANCE_FORMAT:
        msg = (
            f"{store.path} records its provenance in format "
            f"{recorded_format!r}, which this build cannot read; it writes "
            f"and reads format {_PROVENANCE_FORMAT}. {_REBUILD}"
        )
        raise ValueError(msg)

    return EncodingsProvenance(
        base_model=str(record["base_model"]),
        max_length=int(record["max_length"]),
        stride=int(record["stride"]),
    )


def write_provenance(
    store: EncodingsStore, provenance: EncodingsProvenance
) -> None:
    """Stamp `store` with `provenance`.

    :param store: an open, writable encodings store.
    :param provenance: what is writing into it.
    """
    record = {"format": _PROVENANCE_FORMAT} | dataclasses.asdict(provenance)
    store._put_raw(_PROVENANCE_KEY, json.dumps(record, sort_keys=True).encode())


def record_provenance(
    store: EncodingsStore, provenance: EncodingsProvenance
) -> None:
    """Stamp `store` with what this run is about to write into it.

    A store recording another geometry is refused, and so is one holding
    documents while recording none, since nothing attributes them to this
    run's.

    :param store: an open, writable encodings store.
    :param provenance: what this run will write.
    :raises ValueError: if the store records another geometry, or records
        none over documents it already holds.
    """
    recorded = read_provenance(store)
    if recorded is not None and recorded != provenance:
        msg = (
            f"{store.path} was written by {recorded.base_model} at "
            f"window {recorded.max_length}, stride {recorded.stride}, and "
            f"this run writes {provenance.base_model} at window "
            f"{provenance.max_length}, stride {provenance.stride}. One store "
            f"holding both is one no reader can tell apart. Build this into "
            f"a store of its own."
        )
        raise ValueError(msg)

    if recorded is None and store.keys():
        msg = (
            f"{store.path} holds documents but does not record which model, "
            f"window or stride tokenized them, so they cannot be attributed "
            f"to {provenance.base_model}. Build into a store of its own."
        )
        raise ValueError(msg)

    write_provenance(store, provenance)


def check_provenance(
    path: str | os.PathLike[str], base_model: str, expected_stride: int
) -> None:
    """Refuse an encodings store this run cannot read as it was written.

    `max_length` is deliberately not compared: windows are stitched off the
    attention mask, so a shorter window still reconstructs each document
    token-for-token.

    :param path: an encodings store known to exist.
    :param base_model: the model this run will feed the ids to.
    :param expected_stride: the stride this run's windows are merged under.
    :raises ValueError: if the store records no provenance, another base
        model — the ids come from another vocabulary, which is a confident
        wrong answer rather than a shape error — or another stride than
        `expected_stride`; or if it is of the older HDF5 layout.
    """
    with EncodingsStore(path) as store:
        recorded = read_provenance(store)

    if recorded is None:
        msg = (
            f"{path} does not record which model or stride tokenized it, so "
            f"its ids cannot be attributed to {base_model}. {_REBUILD}"
        )
        raise ValueError(msg)

    if recorded.base_model != base_model:
        msg = (
            f"{path} was tokenized by {recorded.base_model} and this run's "
            f"base model is {base_model}. Their input ids come from "
            f"different vocabularies, so the embedding layer would read "
            f"every id under the wrong one; rebuild the encodings with "
            f"`precompute-encodings`."
        )
        raise ValueError(msg)

    if recorded.stride != expected_stride:
        msg = (
            f"{path} was tokenized with a stride of {recorded.stride} and "
            f"this run merges its windows at {expected_stride}. Every seam "
            f"would be stitched at the wrong offset — tokens duplicated or "
            f"dropped once per window, with the row count and every shape "
            f"still plausible; rebuild the encodings with "
            f"`precompute-encodings`."
        )
        raise ValueError(msg)


_EXTERNAL_KEY_SEPARATOR = ":"


def external_key(corpus: str, document: str) -> str:
    """The key `precompute-encodings` writes an external-corpus document under.

    S800 and enzymeNER's own document ids (S800's file stem, enzymeNER's
    `sentence_id`) share this store's key space with BRENDA's bare pubmed
    ids, so they are corpus-prefixed rather than written as-is.

    :param corpus: the corpus that produced `document` (`"s800"` or
        `"enzymener"`).
    :param document: the corpus's own document id — the same string gold
        `ExternalMention.document` carries.
    :return: the key precompute-encodings stores it under.
    """
    return f"{corpus}{_EXTERNAL_KEY_SEPARATOR}{document}"


def external_document(key: str) -> tuple[str, str] | None:
    """Split a store key back into its corpus and gold document id.

    The inverse of `external_key` — what a reader needs to match a
    predicted span's document (an encodings-store key) against gold
    `ExternalMention.document`, which carries no corpus prefix. Splits on
    the first colon only: enzymeNER's own document id already contains one
    (`"PMC1233920:M01009"`).

    :param key: a key from the encodings store.
    :return: `(corpus, document)` if `key` is corpus-prefixed, else `None`
        for a bare pubmed id.
    """
    corpus, sep, document = key.partition(_EXTERNAL_KEY_SEPARATOR)
    return (corpus, document) if sep else None


_external_document_ids: dict[str, int] = {}
"""Every external-corpus key this process has minted a document id for."""

_MINT = threading.Lock()
"""Held across the mint below, whose read of the map decides the next id."""


def external_document_id(key: str) -> int:
    """A document id for `key`, unique in this process.

    `Model.get_token_embeddings` caches on this id for the life of the
    process, so it is minted per key, not counted off a store's key order.
    It is below zero, where no pubmed id reaches; zero is never minted.

    :param key: a store key, as `external_key` spells it.
    :return: the id, and the same one on every later call for `key`.
    """
    # The next id is read off the map this call writes: unlocked, two
    # threads could read one length and hand two keys one id.
    with _MINT:
        return _external_document_ids.setdefault(
            key, -(len(_external_document_ids) + 1)
        )


def content_digest(store: EncodingsStore) -> str:
    """A fingerprint of the token ids `store` holds, keyed by document.

    Sorted and at a fixed byte order, so one store digests the same anywhere.
    It decompresses every document, so the writer stamps it once.

    :param store: an open encodings store.
    :return: the hex SHA-256 of its documents and their token ids.
    """
    digest = hashlib.sha256()
    for key in store.keys():
        encoding = store.get(key)
        if encoding is None:
            continue
        ids = encoding["input_ids"]
        digest.update(f"{key}\t{ids.shape}\n".encode("utf8"))
        digest.update(ids.astype(_DIGEST_DTYPE, copy=False).tobytes())
    return digest.hexdigest()


def read_content_digest(store: EncodingsStore) -> str | None:
    """The fingerprint of the token ids `store` was stamped with.

    :param store: an open encodings store.
    :return: the recorded digest, or None where no writing pass has run to
        its end since the store last changed.
    """
    raw = store._get_raw(_CONTENT_DIGEST_KEY)
    return None if raw is None else str(json.loads(raw))


def stamp_content_digest(store: EncodingsStore) -> str:
    """Fingerprint what `store` now holds and record it.

    The digest is a property of the whole store rather than of the documents
    one run wrote, so a resume replaces it rather than extending it. Call it
    through `writing_pass`, which is what keeps the recorded value from
    outliving the ids it was taken over.

    :param store: an open, writable encodings store.
    :return: the digest recorded.
    """
    digest = content_digest(store)
    store._put_raw(_CONTENT_DIGEST_KEY, json.dumps(digest).encode())
    return digest


@contextmanager
def writing_pass(store: EncodingsStore) -> Iterator[None]:
    """Bracket a pass that writes token ids into `store`.

    Drops the digest on the way in and restates it only on a clean exit, so
    an interrupted pass leaves the store unstamped, not falsely stamped.
    Re-entering a pass on the same handle is refused; a second handle is
    refused when it opens.

    :param store: an open, writable encodings store.
    :raises RuntimeError: if a pass is already open on `store`.
    """
    if store._in_pass:
        msg = (
            f"a writing pass is already open on {store.path}; a second one "
            f"would fingerprint the ids on its own exit and leave the first "
            f"pass writing under a stamp that no longer describes them. "
            f"Write the store in one pass."
        )
        raise RuntimeError(msg)

    store._put_raw(_CONTENT_DIGEST_KEY, None)
    store._in_pass = True
    try:
        yield
    finally:
        store._in_pass = False

    # Outside the `finally` on purpose: a pass that did not reach its end has
    # to leave the store unstamped.
    logger.info("Token ids fingerprinted as %s", stamp_content_digest(store))


def store_content_digest(path: str | os.PathLike[str] | None) -> str | None:
    """The content digest recorded by the encodings store at `path`.

    Reads the one stamp, not the ids. A path naming nothing reads as no
    digest rather than raising, leaving a mistyped path to the code that
    needs the ids.

    :param path: an encodings store, or an empty or absent path.
    :return: the recorded digest, or None where there is no store to read or
        it records none.
    :raises ValueError: if `path` is an encodings store of the older HDF5
        layout.
    """
    if not path or not os.path.exists(path):
        return None

    with EncodingsStore(path) as store:
        return read_content_digest(store)


def encodings_provenance(recorded: str | None, current: str | None) -> str:
    """Say whether this run's token ids are the ones a checkpoint trained on.

    A corpus re-tokenized under a newer tokenizer revision, or a corrected
    `document_text`, gives the heads different inputs for the same document
    than the checkpoint trained on, without making those inputs wrong.

    :param recorded: the digest the checkpoint carries, if any.
    :param current: the digest of the store this run reads, if any.
    :return: `"matched"`, `"unrecorded"`, `"unstamped"` or `"mismatched"`.
    """
    if recorded is not None and recorded == current:
        return "matched"

    if recorded is None:
        warnings.warn(
            "this checkpoint records no encodings digest, so nothing says "
            "which tokenization produced the inputs it was trained on; these "
            "token ids are that run's only if the store has not been "
            "rebuilt since.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "unrecorded"

    if current is None:
        warnings.warn(
            f"this checkpoint was trained on encodings {recorded[:12]} and "
            "the store this run reads carries no digest of its own, so "
            "whether it holds the same token ids cannot be established; "
            "rebuild it with `precompute-encodings` to stamp it.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "unstamped"

    warnings.warn(
        f"this checkpoint was trained on encodings {recorded[:12]} but this "
        f"run reads {current[:12]}; the two files hold different token ids "
        "for the same documents, so these are not the inputs that run "
        "trained on.",
        RuntimeWarning,
        stacklevel=2,
    )
    return "mismatched"


__all__ = [
    "Encoding",
    "EncodingsProvenance",
    "EncodingsStore",
    "bytes_to_encoding",
    "check_provenance",
    "content_digest",
    "encoding_to_bytes",
    "encodings_provenance",
    "external_document",
    "external_document_id",
    "external_key",
    "read_content_digest",
    "read_provenance",
    "record_provenance",
    "stamp_content_digest",
    "store_content_digest",
    "write_provenance",
    "writing_pass",
]
