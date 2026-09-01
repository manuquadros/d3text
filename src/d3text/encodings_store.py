"""Provenance stamp for the `precompute-encodings` HDF5.

Neither the tokenizer, the window nor the stride is recoverable from the stored
arrays: a mismatched tokenizer yields an array of exactly the right shape over
the wrong vocabulary, and the aggregated row count comes to the document's
token count under any window. See the data page of the documentation.
"""

import dataclasses
import logging

import h5py

logger = logging.getLogger(__name__)

_FORMAT_ATTRIBUTE = "d3text_encodings_format"
_PROVENANCE_FORMAT = 1
_BASE_MODEL_ATTRIBUTE = "base_model"
_MAX_LENGTH_ATTRIBUTE = "max_length"
_STRIDE_ATTRIBUTE = "stride"


@dataclasses.dataclass(frozen=True)
class EncodingsProvenance:
    """What produced a store's token ids, recorded when it is written.

    The base model is what a reader ultimately cares about; `max_length` and
    `stride` are recorded beside it because they are the other two inputs
    `precompute-encodings` takes and neither is otherwise recoverable.
    """

    base_model: str
    max_length: int
    stride: int


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


__all__ = [
    "EncodingsProvenance",
    "read_provenance",
    "record_provenance",
    "write_provenance",
]
