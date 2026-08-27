"""Provenance stamp for the `precompute-encodings` HDF5.

`precompute-encodings` tokenizes every document once, under one base model's
tokenizer, one `max_length` and one `stride`, and stores the resulting token
ids keyed by pubmed id. None of the three is recoverable from the stored
arrays: a mismatched tokenizer produces a `input_ids` array of exactly the
same shape and dtype as the right one, only over the wrong vocabulary, and
`d3text.embeddings_store.EmbeddingsStore` already showed that the aggregated
row count is `T` for any window and stride — so a document that was split at
one window and later resumed at another leaves no trace a shape check can
catch either.

`record_provenance` is the write-side guard, called once per run before any
group is written: it refuses to add to a store that already recorded another
geometry, and refuses to add to a store that holds documents but recorded
none, for the same reason `d3text.cli.precompute_embeddings.record_provenance`
refuses both — the mixture, once written, is indistinguishable from a store
that agrees with itself.

`read_provenance` is what a reader — `d3text.data.data.BrendaDataset`, when it
is told which base model it is about to feed — checks its own configuration
against, the same way `EmbeddingsStore._attributed_to` does for the LMDB
store.
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

    The base model is what a reader ultimately cares about — a wrong
    tokenizer hands the embedding layer ids from another vocabulary, which is
    a silent wrong answer rather than a shape error. `max_length` and
    `stride` are recorded beside it because they are the other two inputs
    `precompute-encodings` takes and neither is otherwise recoverable from the
    store: the aggregated row count comes to the document's token count under
    any window or stride.
    """

    base_model: str
    max_length: int
    stride: int


def read_provenance(store: h5py.File) -> EncodingsProvenance | None:
    """What wrote `store`, or `None` if it does not say.

    `None` is a store written before provenance was recorded, which is not
    the same as a store written by the wrong model or window and is not
    distinguishable from one either: what it means is that nothing on disk
    attributes those token ids to anything.

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
    """Stamp `store`'s root attributes with `provenance`."""
    store.attrs[_FORMAT_ATTRIBUTE] = _PROVENANCE_FORMAT
    store.attrs[_BASE_MODEL_ATTRIBUTE] = provenance.base_model
    store.attrs[_MAX_LENGTH_ATTRIBUTE] = provenance.max_length
    store.attrs[_STRIDE_ATTRIBUTE] = provenance.stride


def record_provenance(
    store: h5py.File, provenance: EncodingsProvenance
) -> None:
    """Stamp `store` with what this run is about to write into it.

    A pass that appends to a store built under another model, window or
    stride produces one HDF5 file holding two kinds of token id that nothing
    downstream can separate: every group has the same shape and dtype
    regardless of which geometry produced it. That is refused outright — the
    only place the mixture can still be prevented.

    An unstamped store that already holds documents predates this stamp
    existing at all, and every encodings file `precompute-encodings` had ever
    written is exactly that on this build's first run against it. Refusing
    those outright would turn every one of them unresumable in one release;
    warning and stamping trades the guarantee for continuity instead — the
    groups already there stay unattributed, but the run proceeds and every
    group from here on is. This is the opposite call from
    `d3text.cli.precompute_embeddings.record_provenance`'s LMDB, which is
    two orders of magnitude larger to rebuild and refuses instead.
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
