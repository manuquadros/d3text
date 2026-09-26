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
from typing import Self

import blosc2
import lmdb
import numpy
import torch
from jaxtyping import Float
from torch import Tensor
from d3text.constraints import NonNegative, Positive

logger = logging.getLogger(__name__)

_MAGIC = b"D3EB"
_VERSION = 1
_HEADER = struct.Struct("<4sBII")

# A pubmed id is decimal digits, so nothing this store is keyed on can spell a
# key holding a NUL.
_PROVENANCE_KEY = b"\x00provenance"
_PROVENANCE_FORMAT = 1

# The whole corpus measures 100.8 GiB through this store's codec, so the 100 GiB
# this used to reserve ran out near the end of a full pass. On Linux `map_size`
# reserves address space rather than allocating it, and LMDB writes the file
# sparsely, so the headroom costs nothing until the pages are written.
DEFAULT_MAP_SIZE_GIB = 256.0

_CPARAMS: dict[str, typing.Any] = {
    "codec": blosc2.Codec.ZSTD,
    "clevel": 5,
    "filters": [blosc2.Filter.SHUFFLE],
    "filters_meta": [0],
}


def _compress(tensor: Tensor) -> tuple[bytes, tuple[int, ...]]:
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
    return typing.cast(bytes, blosc2.compress2(array, **_CPARAMS)), array.shape


def _decompress(body: bytes | memoryview, shape: tuple[int, ...]) -> Tensor:
    # `frombuffer` hands back a read-only view; torch refuses to share memory
    # with one, so the copy is not optional.
    raw = numpy.frombuffer(blosc2.decompress2(body), dtype=numpy.int16)
    return torch.from_numpy(raw.reshape(shape).copy()).view(torch.bfloat16)


def _pack(header: struct.Struct, magic: bytes, *shape: int) -> bytes:
    return header.pack(magic, _VERSION, *shape)


def _unpack(
    packed: bytes | memoryview,
    header: struct.Struct,
    magic: bytes,
    name: str,
    note: str = "",
) -> tuple[int, ...]:
    """The blob's shape, once its header's magic and version check out."""
    if len(packed) < header.size:
        msg = (
            f"{name} blob is at least {header.size} bytes of header; got "
            f"{len(packed)}."
        )
        raise ValueError(msg)

    unpacked = header.unpack_from(packed)
    got_magic, version = unpacked[0], unpacked[1]
    if got_magic != magic:
        msg = (
            f"not {name} blob: expected the magic {magic!r}, got "
            f"{got_magic!r}.{note}"
        )
        raise ValueError(msg)
    if version != _VERSION:
        # `name` (e.g. "an embeddings-store") is one noun for both
        # messages: whole above, article-stripped here.
        bare_name = name.split(" ", 1)[1]
        msg = (
            f"{bare_name} format version {version} is not readable by this "
            f"build, which writes version {_VERSION}."
        )
        raise ValueError(msg)

    return unpacked[2:]


def tensor_to_bytes(tensor: Float[Tensor, "token feature"]) -> bytes:
    """Compress `tensor` for storage.

    The cast to bf16 is a deliberate, lossy narrowing: these are frozen
    base-model activations, not weights that will be trained further.

    :param tensor: one document's token embeddings.
    :return: the header plus the blosc2 frame to store.
    """
    body, shape = _compress(tensor)
    return _pack(_HEADER, _MAGIC, *shape) + body


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
    shape = _unpack(
        packed,
        _HEADER,
        _MAGIC,
        "an embeddings-store",
        note=(
            " A store written before this format carries a bare blosc2 "
            "frame of fp16, which shares this format's itemsize and would "
            "otherwise decode into a matrix of garbage; rebuild it with "
            "`precompute-embeddings`."
        ),
    )
    return _decompress(packed[_HEADER.size :], shape)


_WINDOW_MAGIC = b"D3WL"
_WINDOW_HEADER = struct.Struct("<4sBIII")


def windowed_tensor_to_bytes(
    tensor: Float[Tensor, "window token feature"],
) -> bytes:
    """Compress `tensor` for storage in a layer-boundary store.

    The 3-D counterpart of `tensor_to_bytes`: a layer-boundary store holds
    one row of hidden states per window, not one aggregated row per
    document, because the top encoder layers a cache hit resumes into
    attend only within a window.

    :param tensor: one document's per-window hidden states at the layer
        boundary.
    :return: the header plus the blosc2 frame to store.
    """
    body, shape = _compress(tensor)
    return _pack(_WINDOW_HEADER, _WINDOW_MAGIC, *shape) + body


def bytes_to_windowed_tensor(
    packed: bytes | memoryview,
) -> Float[Tensor, "window token feature"]:
    """The stored per-window hidden states, as the bf16 tensor they were
    written as.

    :param packed: a blob as `windowed_tensor_to_bytes` wrote it.
    :return: the stored `[window, token, feature]` tensor.
    :raises ValueError: if the blob carries another format's magic number.
    """
    shape = _unpack(
        packed, _WINDOW_HEADER, _WINDOW_MAGIC, "a layer-boundary store"
    )
    return _decompress(packed[_WINDOW_HEADER.size :], shape)


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

    `forward_dtype` is recorded because `select_amp_dtype` names a machine
    rather than a dtype: two stores agreeing on all three fields above can
    still hold forwards computed in different precisions, because the cards
    that built them differ. A store written by a build that predates this
    field holds fp16 whatever built it. It is diagnostic only — nothing
    reads it to decide anything, the difference being seed-sized — and
    `None` means the writer recorded none.
    """

    base_model: str
    max_length: Positive
    stride: NonNegative
    forward_dtype: str | None = None

    @property
    def identity(self) -> tuple[str, Positive, NonNegative]:
        """The fields deciding whether two passes belong in one store.

        `forward_dtype` is deliberately not among them: it says how the
        activations were computed, not what they are of, so a store resumed
        by a build that computes them differently is still one store.

        :return: the base model, the window and the stride.
        """
        return (self.base_model, self.max_length, self.stride)


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
            # Read with `get`, so a record written before this field existed
            # stays a complete format-1 record rather than becoming one this
            # build refuses. The format number says how to interpret a
            # record, and an absent diagnostic field changes that for none
            # of the fields above; bumping it would strand every store
            # already on disk to gain nothing.
            forward_dtype=(
                None
                if record.get("forward_dtype") is None
                else str(record["forward_dtype"])
            ),
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


@dataclasses.dataclass(frozen=True)
class LayerBoundaryProvenance:
    """What produced a layer-boundary store's cached prefixes.

    The same fields `StoreProvenance` records, plus `frozen_layers`: the
    number of leading encoder layers the store's rows were computed
    through. Two runs of the same base model at different
    `unfrozen_top_layers` split the trunk at different layers, and a store
    built for one boundary read at another would hand the wrong prefix to
    the top layers without either side raising.
    """

    base_model: str
    max_length: Positive
    stride: NonNegative
    frozen_layers: NonNegative
    forward_dtype: str | None = None

    @property
    def identity(self) -> tuple[str, Positive, NonNegative, NonNegative]:
        """The fields deciding whether two passes belong in one store.

        :return: the base model, the window, the stride and the layer
            boundary.
        """
        return (
            self.base_model,
            self.max_length,
            self.stride,
            self.frozen_layers,
        )


_LAYER_PROVENANCE_KEY = b"\x00layer_provenance"
_LAYER_PROVENANCE_FORMAT = 1


def read_layer_provenance(
    env: lmdb.Environment,
) -> LayerBoundaryProvenance | None:
    """What wrote `env`'s layer-boundary rows, or `None` if it does not say.

    :param env: the open LMDB environment.
    :return: the recorded provenance, or None if it records none.
    :raises ProvenanceError: if the record is there but this build cannot
        read it.
    """
    with env.begin() as transaction:
        raw = transaction.get(_LAYER_PROVENANCE_KEY)
    if raw is None:
        return None

    try:
        record = json.loads(raw)
        recorded_format = record["format"]
    except (json.JSONDecodeError, TypeError, KeyError) as error:
        msg = (
            f"{env.path()} holds a layer-boundary provenance record this "
            f"build cannot read."
        )
        raise ProvenanceError(msg) from error

    if recorded_format != _LAYER_PROVENANCE_FORMAT:
        msg = (
            f"{env.path()} records its layer-boundary provenance in format "
            f"{recorded_format!r}, which this build cannot read; it writes "
            f"and reads format {_LAYER_PROVENANCE_FORMAT}."
        )
        raise ProvenanceError(msg)

    try:
        return LayerBoundaryProvenance(
            base_model=str(record["base_model"]),
            max_length=int(record["max_length"]),
            stride=int(record["stride"]),
            frozen_layers=int(record["frozen_layers"]),
            forward_dtype=(
                None
                if record.get("forward_dtype") is None
                else str(record["forward_dtype"])
            ),
        )
    except (TypeError, KeyError, ValueError) as error:
        msg = (
            f"{env.path()} records a format-{_LAYER_PROVENANCE_FORMAT} "
            f"layer-boundary provenance missing a field this build reads: "
            f"{record!r}."
        )
        raise ProvenanceError(msg) from error


def write_layer_provenance(
    env: lmdb.Environment, provenance: LayerBoundaryProvenance
) -> None:
    """Stamp `env` with what is writing into it.

    :param env: the open LMDB environment.
    :param provenance: what this run will write.
    """
    record = {"format": _LAYER_PROVENANCE_FORMAT} | dataclasses.asdict(
        provenance
    )
    with env.begin(write=True) as transaction:
        transaction.put(
            _LAYER_PROVENANCE_KEY, json.dumps(record, sort_keys=True).encode()
        )


class LayerBoundaryStore:
    """Read-only view of a layer-boundary LMDB `precompute-embeddings` writes.

    Holds one row of hidden states per window at the boundary between a
    partially-trainable trunk's frozen and trainable encoder layers, keyed
    by document id like `EmbeddingsStore`. Opening one names the base model
    and the layer boundary the run will resume from, and a store not
    recorded as written by that exact pair is refused here rather than read
    — see `LayerBoundaryProvenance`.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        base_model: str,
        frozen_layers: NonNegative,
        max_length: Positive,
    ) -> None:
        self.path = os.fspath(path)
        self.env = lmdb.open(
            self.path,
            readonly=True,
            lock=False,
            readahead=False,
            max_readers=2048,
        )
        try:
            self.provenance = self._attributed_to(
                base_model, frozen_layers, max_length
            )
        except ProvenanceError:
            self.env.close()
            raise
        # `_resolve_layer_boundary_cached` reads a batch's hits from a
        # single background thread so the trainable top layers can replay
        # one item while the next is decompressing; blosc2 holds the GIL
        # during decompress unless told not to, which would serialize that
        # thread behind this process's own kernel launches. The flag is
        # process-global, so it also releases the GIL for `EmbeddingsStore`
        # decompression, which is harmless: nothing there depends on
        # holding it.
        blosc2.set_releasegil(True)
        self.hits = 0
        self.misses = 0
        self.mismatches = 0
        self._warned = False
        self._served = False
        self._closed = False
        _opened.append(self)
        logger.info(
            "Reading precomputed layer-boundary prefixes from %s, written "
            "by %s at window %d, stride %d, frozen through layer %d",
            self.path,
            self.provenance.base_model,
            self.provenance.max_length,
            self.provenance.stride,
            self.provenance.frozen_layers,
        )

    def _attributed_to(
        self,
        base_model: str,
        frozen_layers: NonNegative,
        max_length: Positive,
    ) -> LayerBoundaryProvenance:
        """The store's provenance, once it is this run's boundary to read.

        Checked once, at open, rather than per lookup: a store attributed
        to the wrong model or the wrong boundary is wrong for every
        document it holds, not just the one a particular call happens to
        ask for first.

        :param base_model: the base model this run trains.
        :param frozen_layers: the number of leading encoder layers this
            run's `unfrozen_top_layers` freezes.
        :param max_length: the window this run's encodings, and its live
            forward fallback, are cut at.
        :raises ProvenanceError: if the store records no provenance, or
            records another model, another layer boundary, or another
            window.
        """
        recorded = read_layer_provenance(self.env)
        if recorded is None:
            msg = (
                f"{self.path} does not record which model or layer "
                f"boundary wrote it, so its rows cannot be attributed to "
                f"{base_model} frozen through layer {frozen_layers}. "
                f"Rebuild it with `precompute-embeddings`, which stamps "
                f"what it writes."
            )
            raise ProvenanceError(msg)
        if (
            recorded.base_model != base_model
            or recorded.frozen_layers != frozen_layers
        ):
            documents = self.env.stat()["entries"] - 1
            msg = (
                f"{self.path} is stamped for {recorded.base_model} frozen "
                f"through layer {recorded.frozen_layers}, and this run's "
                f"base model is {base_model} frozen through layer "
                f"{frozen_layers}; it holds {documents} document(s). A "
                f"prefix cached at another boundary is a valid tensor of "
                f"the right shape for the wrong layer, so nothing "
                f"downstream would fail loudly if it were read anyway."
            )
            raise ProvenanceError(msg)
        if recorded.max_length != max_length:
            documents = self.env.stat()["entries"] - 1
            msg = (
                f"{self.path} is stamped at window {recorded.max_length}, "
                f"and this run's encodings are cut at {max_length}; it "
                f"holds {documents} document(s). A one-window document "
                f"passes the window-count check `get` runs regardless of "
                f"which width each was actually embedded at, so a "
                f"wrong-window store would not be caught there. Rebuild it "
                f"with `precompute-embeddings`, which now always writes at "
                f"`utils.WINDOW_LENGTH`."
            )
            raise ProvenanceError(msg)
        return recorded

    def get(
        self, pubmed_id: int | str, expected_windows: Positive
    ) -> Float[Tensor, "window token feature"] | None:
        """The stored per-window prefix for `pubmed_id`, or `None` to run it.

        :param pubmed_id: the document to read.
        :param expected_windows: the window count the batch item implies.
        :return: the stored `[window, token, feature]` tensor, or None.
        """
        with self.env.begin(buffers=True) as transaction:
            blob = transaction.get(str(pubmed_id).encode())
            if blob is None:
                self.misses += 1
                return None
            stored = bytes_to_windowed_tensor(blob)

        if stored.shape[0] != expected_windows:
            self.mismatches += 1
            if not self._warned:
                self._warned = True
                logger.warning(
                    "%s holds %d windows for document %s where its "
                    "encodings imply %d, so the two were built from "
                    "different text; this document, and every other that "
                    "disagrees, is being embedded live instead.",
                    self.path,
                    stored.shape[0],
                    pubmed_id,
                    expected_windows,
                )
            return None

        if not self._served:
            self._served = True
            logger.info(
                "%s served document %s from the layer-boundary store",
                self.path,
                pubmed_id,
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
            f"{self.mismatches:,} stored at a window count the encodings "
            f"disagree with"
        )

    def close(self) -> None:
        """Close the environment and report what the store answered.

        Registered with `atexit`, mirroring `EmbeddingsStore.close`.
        """
        if self._closed:
            return
        self._closed = True
        if self.hits + self.misses + self.mismatches:
            logger.info("%s", self.summary())
        self.env.close()


class EmbeddingsStore:
    """A `precompute-embeddings` LMDB, read-only unless this run is building it.

    An existing store is opened `readonly` and without a lock, since the
    writer has long since exited and a training run must not lock a 100 GiB
    file it only reads. One made by `create` is opened writable instead, and
    `put` fills it with the documents the run embeds itself. Either way
    `readahead=False`, because the store is far larger than RAM and the
    documents are visited in shuffled order. Opening one names the base model
    the run will feed the matrices to, and a store not recorded as written by
    it is refused here rather than read.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        base_model: str,
        max_length: Positive,
        *,
        writable: bool = False,
    ) -> None:
        self.path = os.fspath(path)
        self.env = (
            lmdb.open(
                self.path,
                map_size=int(DEFAULT_MAP_SIZE_GIB * 1024**3),
                readahead=False,
                max_readers=2048,
                # A commit per document would otherwise be an fsync per
                # document. Durability is only lost to a machine crash, not a
                # killed process, and `close` syncs.
                sync=False,
            )
            if writable
            else lmdb.open(
                self.path,
                readonly=True,
                lock=False,
                readahead=False,
                max_readers=2048,
            )
        )
        try:
            self.provenance = self._attributed_to(base_model, max_length)
        except ProvenanceError:
            self.env.close()
            raise
        self.writable = writable
        self.written = 0
        self.hits = 0
        self.misses = 0
        self.mismatches = 0
        self._warned = False
        self._served = False
        self._closed = False
        _opened.append(self)
        logger.info(
            "Reading precomputed embeddings from %s, written by %s at window "
            "%d, stride %d",
            self.path,
            self.provenance.base_model,
            self.provenance.max_length,
            self.provenance.stride,
        )

    @classmethod
    def create(
        cls, path: str | os.PathLike[str], provenance: StoreProvenance
    ) -> Self:
        """Make an empty store at `path`, stamped, and open it for writing.

        :param path: where the store goes; missing parent directories are
            made too.
        :param provenance: what the documents `put` into it are computed by.
        :return: the new store, open for reading and writing.
        """
        os.makedirs(path, exist_ok=True)
        with lmdb.open(os.fspath(path)) as env:
            write_provenance(env, provenance)
        return cls(
            path, provenance.base_model, provenance.max_length, writable=True
        )

    def put(
        self, pubmed_id: int | str, embedding: Float[Tensor, "token feature"]
    ) -> None:
        """Store `embedding` as `pubmed_id`'s, in a store opened writable.

        A write that fails is warned about once and ends the writing, not the
        run: what the store already holds is still read, and every document it
        lacks is embedded live, as for any miss.

        :param pubmed_id: the document the embedding belongs to.
        :param embedding: the document's aggregated token embeddings.
        :raises RuntimeError: if the store was opened read-only.
        """
        if not self.writable:
            msg = f"{self.path} is open read-only; nothing can be put into it."
            raise RuntimeError(msg)
        try:
            with self.env.begin(write=True) as transaction:
                transaction.put(
                    str(pubmed_id).encode(), tensor_to_bytes(embedding)
                )
        except lmdb.Error as error:
            self.writable = False
            logger.warning(
                "Cannot write document %s into the embeddings store at %s "
                "(%s); it stops growing here, and every document it does not "
                "hold keeps being embedded live. `precompute-embeddings` "
                "resumes it, skipping what it already holds.",
                pubmed_id,
                self.path,
                error,
            )
            return
        self.written += 1

    def _attributed_to(
        self, base_model: str, max_length: Positive
    ) -> StoreProvenance:
        """The store's provenance, once it is this run's to read.

        :raises ProvenanceError: if the store records no provenance, or
            records another model or another window.
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
        if recorded.max_length != max_length:
            documents = self.env.stat()["entries"] - 1
            msg = (
                f"{self.path} is stamped at window {recorded.max_length}, "
                f"and this run's encodings are cut at {max_length}; it "
                f"holds {documents} document(s). The aggregated row count "
                f"this store is checked against comes to the document's "
                f"token count under any window, so a document built at the "
                f"wrong one would be read as a hit rather than caught by "
                f"that check. Rebuild it with `precompute-embeddings`, "
                f"which now always writes at `utils.WINDOW_LENGTH`."
            )
            raise ProvenanceError(msg)
        return recorded

    def get(
        self, pubmed_id: int | str, expected_tokens: Positive
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

        The recorded forward precision rides along because this line is where
        a reader comparing two runs afterwards will look for what made them
        differ.

        :return: the hit and miss counts as a sentence.
        """
        dtype = self.provenance.forward_dtype
        computed = (
            f"forwards computed in {dtype}"
            if dtype
            else "forward precision not recorded"
        )
        asked = self.hits + self.misses + self.mismatches
        if not asked:
            return f"{self.path} was never asked for a document; {computed}"
        return (
            f"{self.path} served {self.hits:,} of {asked:,} documents "
            f"({self.hits / asked:.1%}), {self.misses:,} not stored, "
            f"{self.mismatches:,} stored at a length the encodings disagree "
            f"with, {self.written:,} written by this process; {computed}"
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
        if self.written:
            self.env.sync()
        self.env.close()


# Every store opened in this process. Nothing owns a reader — `models.base`
# caches it for the life of the process — so a caller asking what a store
# answered has nothing to ask, and this is the list it asks instead.
_opened: list[EmbeddingsStore | LayerBoundaryStore] = []


def lookup_totals() -> tuple[NonNegative, NonNegative]:
    """How many documents the stores opened so far served, and were asked for.

    Cumulative over the process, as the stores are: one run's own share is the
    difference between this pair read when it began and read when it ended,
    which is what a sweep running many runs in one process has to subtract.

    :return: the documents served from a store, and the documents looked up in
        one.
    """
    return (
        sum(store.hits for store in _opened),
        sum(store.hits + store.misses + store.mismatches for store in _opened),
    )
