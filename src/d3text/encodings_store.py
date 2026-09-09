"""Provenance stamp for the `precompute-encodings` HDF5.

Neither the tokenizer, the window nor the stride is recoverable from the stored
arrays: a mismatched tokenizer yields an array of exactly the right shape over
the wrong vocabulary, and the aggregated row count comes to the document's
token count under any window. Nor is any of the three enough to identify the
ids themselves, which is what `content_digest` fingerprints. See the data page
of the documentation.
"""

import dataclasses
import hashlib
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager

import h5py
import numpy
from d3text.constraints import NonNegative, Positive

logger = logging.getLogger(__name__)

_FORMAT_ATTRIBUTE = "d3text_encodings_format"
_PROVENANCE_FORMAT = 1
_BASE_MODEL_ATTRIBUTE = "base_model"
_MAX_LENGTH_ATTRIBUTE = "max_length"
_STRIDE_ATTRIBUTE = "stride"
# Optional within format 1 rather than a format of its own: bumping would
# refuse every store already on disk, and a reader that does not find this
# attribute is in exactly the position it was in before there was one.
_CONTENT_DIGEST_ATTRIBUTE = "content_digest"

_INPUT_IDS_DATASET = "input_ids"
_DIGEST_DTYPE = numpy.dtype("<u4")
"""Byte order the ids are hashed in, so one file digests the same anywhere."""

_open_passes: set[str] = set()
"""Resolved paths of the stores a `writing_pass` is currently open on."""


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


def read_provenance(store: h5py.File) -> EncodingsProvenance | None:
    """What wrote `store`, or `None` if it does not say.

    `None` means nothing on disk attributes those token ids to anything, which
    is not the same as a store written by the wrong model and is not
    distinguishable from one either.

    :param store: an open encodings file.
    :return: the recorded provenance, or None if it records none.
    :raises ValueError: if the store is stamped with a format this build does
        not read.
    """
    if _FORMAT_ATTRIBUTE not in store.attrs:
        return None

    recorded_format = int(store.attrs[_FORMAT_ATTRIBUTE])
    if recorded_format != _PROVENANCE_FORMAT:
        msg = (
            f"{store.filename} records its provenance in format "
            f"{recorded_format!r}, which this build cannot read; it writes "
            f"and reads format {_PROVENANCE_FORMAT}."
        )
        raise ValueError(msg)

    return EncodingsProvenance(
        base_model=str(store.attrs[_BASE_MODEL_ATTRIBUTE]),
        max_length=int(store.attrs[_MAX_LENGTH_ATTRIBUTE]),
        stride=int(store.attrs[_STRIDE_ATTRIBUTE]),
    )


def write_provenance(store: h5py.File, provenance: EncodingsProvenance) -> None:
    """Stamp `store`'s root attributes with `provenance`.

    :param store: an open, writable encodings file.
    :param provenance: what is writing into it.
    """
    store.attrs[_FORMAT_ATTRIBUTE] = _PROVENANCE_FORMAT
    store.attrs[_BASE_MODEL_ATTRIBUTE] = provenance.base_model
    store.attrs[_MAX_LENGTH_ATTRIBUTE] = provenance.max_length
    store.attrs[_STRIDE_ATTRIBUTE] = provenance.stride


def record_provenance(
    store: h5py.File, provenance: EncodingsProvenance
) -> None:
    """Stamp `store` with what this run is about to write into it.

    Appending under another geometry is refused outright: the resulting mixture
    is indistinguishable from a store that agrees with itself. An unstamped
    store that already holds documents is warned about and stamped rather than
    refused, since every file written before the stamp existed is one — the
    opposite call from the LMDB store, which is two orders of magnitude larger
    to rebuild.

    :param store: an open, writable encodings file.
    :param provenance: what this run will write.
    :raises ValueError: if the store records another geometry.
    """
    recorded = read_provenance(store)
    if recorded == provenance:
        return

    if recorded is not None:
        msg = (
            f"{store.filename} was written by {recorded.base_model} at "
            f"window {recorded.max_length}, stride {recorded.stride}, and "
            f"this run writes {provenance.base_model} at window "
            f"{provenance.max_length}, stride {provenance.stride}. One file "
            f"holding both is one no reader can tell apart. Build this into "
            f"a store of its own."
        )
        raise ValueError(msg)

    if len(store.keys()):
        logger.warning(
            "%s holds documents but does not record which model, window or "
            "stride tokenized them; stamping it as %s at window %d, stride "
            "%d now. The groups already there stay unattributed until the "
            "store is rebuilt.",
            store.filename,
            provenance.base_model,
            provenance.max_length,
            provenance.stride,
        )

    write_provenance(store, provenance)


def stored_ids(member: object) -> h5py.Dataset | None:
    """The token ids a member of an encodings store holds, or `None`.

    A pass killed between `create_group` and the `create_dataset` that follows
    it leaves a keyed group holding no ids, and a resume skips a key already
    present, so it stays there. Every reader of the store has to tolerate one;
    they ask here rather than each spelling the test, because two spellings of
    it are how a reader and the digest came to disagree.

    :param member: a member of an encodings store, as `h5py.File.get` returns
        it — a group, something else, or None where the key is absent.
    :return: the member's `input_ids`, or None where there are none to read.
    """
    if not isinstance(member, h5py.Group) or _INPUT_IDS_DATASET not in member:
        return None
    return member[_INPUT_IDS_DATASET]


def content_digest(store: h5py.File) -> str:
    """A fingerprint of the token ids `store` holds, keyed by document.

    Sorted, and read at a fixed byte order and shape, so one file digests the
    same in any process on any machine. Computing it decompresses every id in
    the store, which is why the writer computes it once and stamps the result.

    A group holding no ids is passed over, so it digests as though it were not
    there. It is served by no reader, so a store carrying one is the same store
    to everything downstream as the same file without it, and a fingerprint
    that separated them would report a difference that changes no number.

    :param store: an open encodings file.
    :return: the hex SHA-256 of its documents and their token ids.
    """
    digest = hashlib.sha256()
    for key in sorted(store):
        stored = stored_ids(store.get(key))
        if stored is None:
            continue
        ids = stored[:]
        digest.update(f"{key}\t{ids.shape}\n".encode("utf8"))
        digest.update(ids.astype(_DIGEST_DTYPE, copy=False).tobytes())
    return digest.hexdigest()


def read_content_digest(store: h5py.File) -> str | None:
    """The fingerprint of the token ids `store` was stamped with.

    :param store: an open encodings file.
    :return: the recorded digest, or None for a store written before the stamp
        existed, which is every store the geometry stamp already warns about.
    """
    if _CONTENT_DIGEST_ATTRIBUTE not in store.attrs:
        return None

    recorded = store.attrs[_CONTENT_DIGEST_ATTRIBUTE]
    return (
        recorded.decode("utf8")
        if isinstance(recorded, bytes)
        else str(recorded)
    )


def stamp_content_digest(store: h5py.File) -> str:
    """Fingerprint what `store` now holds and record it on its root.

    The digest is a property of the whole file rather than of the documents
    one run wrote, so a resume replaces it rather than extending it. Call it
    through `writing_pass`, which is what keeps the recorded value from
    outliving the ids it was taken over.

    :param store: an open, writable encodings file.
    :return: the digest recorded.
    """
    digest = content_digest(store)
    store.attrs[_CONTENT_DIGEST_ATTRIBUTE] = digest
    return digest


@contextmanager
def writing_pass(store: h5py.File) -> Iterator[None]:
    """Bracket a pass that writes token ids into `store`.

    The digest fingerprints the file's own contents, so the first group a pass
    writes falsifies it. Dropping it on the way in and restating it only on
    the way out is what makes an interrupted pass read as unstamped: an
    interrupt propagates out of the enclosing `with h5py.File(...)`, which
    closes the file *cleanly*, so a digest merely restated at the end would
    survive over ids it no longer describes — a stamp asserting agreement no
    file supports, which is worse than no stamp at all.

    Nesting is refused rather than counted: an inner pass restates the digest
    on its own exit and the outer one goes on writing under it, which is this
    bracket's own failure one level up. The guard is keyed to the file rather
    than to the handle, since a second handle onto the same path is a second
    writer into the same ids. It fires before anything is written and the
    outer pass it aborts stamps nothing, so a refused nesting leaves the store
    unstamped, not falsely stamped. Concurrent *processes* are not its
    business; that is what HDF5's own file lock is for.

    :param store: an open, writable encodings file.
    :raises RuntimeError: if a pass is already open on the same store.
    """
    path = os.path.realpath(store.filename)
    if path in _open_passes:
        msg = (
            f"a writing pass is already open on {path}; a second one would "
            f"fingerprint the ids on its own exit and leave the first pass "
            f"writing under a stamp that no longer describes them. Write the "
            f"store in one pass."
        )
        raise RuntimeError(msg)

    if _CONTENT_DIGEST_ATTRIBUTE in store.attrs:
        del store.attrs[_CONTENT_DIGEST_ATTRIBUTE]

    # The deletion has to reach the file before the first group does: a pass
    # killed outright leaves whatever HDF5 has flushed, and HDF5 flushes its
    # metadata cache in no particular order.
    store.flush()

    _open_passes.add(path)
    try:
        yield
    finally:
        _open_passes.discard(path)

    # Outside the `finally` on purpose: a pass that did not reach its end has
    # to leave the store unstamped.
    logger.info("Token ids fingerprinted as %s", stamp_content_digest(store))


def store_content_digest(path: str | os.PathLike[str] | None) -> str | None:
    """The content digest recorded by the encodings store at `path`.

    Reads the one attribute, so a run recording or comparing which
    tokenization it read pays nothing for the ids themselves.

    A path that names no file reads as no digest rather than raising, which
    is the call `BrendaDataset._check_encodings_provenance` already makes
    about the same file: both are read before the dataset opens it, and a
    mistyped path is worth hearing about from the code that needs the ids.

    :param path: an encodings store, or an empty or absent path.
    :return: the recorded digest, or None where there is no file to read or it
        records none.
    """
    if not path or not os.path.exists(path):
        return None

    with h5py.File(path, "r") as store:
        return read_content_digest(store)


__all__ = [
    "EncodingsProvenance",
    "content_digest",
    "read_content_digest",
    "read_provenance",
    "record_provenance",
    "stamp_content_digest",
    "store_content_digest",
    "stored_ids",
    "write_provenance",
    "writing_pass",
]
