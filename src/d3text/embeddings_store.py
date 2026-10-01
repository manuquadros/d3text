"""The codec for the precomputed-embeddings LMDB, and its blob format.

`tensor_to_bytes` and `bytes_to_tensor` are the two halves of that store's
contract; `array_to_blob` and `blob_to_array` lend the same blob format to
another store. Nothing else may reach for `blosc2` directly — `unpack_array`
segfaults on a blob it did not write rather than raising. See the data page
of the documentation for the codec choice and the provenance record.
"""

import dataclasses
import json
import logging
import os
import shutil
import struct
import typing
from collections.abc import Iterable
from typing import Self

import blosc2
import lmdb
import numpy
import torch
from jaxtyping import Float
from torch import Tensor
from d3text.constraints import NonNegative, Positive
from d3text.lmdb_store import DEFAULT_MAP_SIZE_GIB

logger = logging.getLogger(__name__)

_MAGIC = b"D3EB"
_VERSION = 1
_HEADER = struct.Struct("<4sBII")

# A pubmed id is decimal digits, so nothing this store is keyed on can spell a
# key holding a NUL.
_PROVENANCE_KEY = b"\x00provenance"
# Format 1 kept one cut of the trunk per env, its rows in the main database;
# format 2 keeps each cut in a named sub-database. A format-1 env is refused
# rather than read as an env holding no sub-database at all.
_PROVENANCE_FORMAT = 2
# The stamp of the older layer-boundary env, kept only to name it on refusal.
_LEGACY_LAYER_PROVENANCE_KEY = b"\x00layer_provenance"

_CPARAMS: dict[str, typing.Any] = {
    "codec": blosc2.Codec.ZSTD,
    "clevel": 1,
    "filters": [blosc2.Filter.SHUFFLE],
    "filters_meta": [0],
}


def _compress(
    tensor: Tensor, *, compress: bool
) -> tuple[bytes, tuple[int, ...]]:
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
    return _frame(array, compress=compress), array.shape


def _frame(array: numpy.ndarray, *, compress: bool) -> bytes:
    # Level 0 is still a blosc2 frame, only a memcpy inside it, so the reader
    # needs no flag to tell the two apart.
    cparams = _CPARAMS if compress else {**_CPARAMS, "clevel": 0}
    return typing.cast(bytes, blosc2.compress2(array, **cparams))


def _unframe(body: bytes | memoryview, dtype: numpy.dtype) -> numpy.ndarray:
    return numpy.frombuffer(blosc2.decompress2(body), dtype=dtype)


def _decompress(body: bytes | memoryview, shape: tuple[int, ...]) -> Tensor:
    # `frombuffer` hands back a read-only view; torch refuses to share memory
    # with one, so the copy is not optional.
    raw = _unframe(body, numpy.dtype(numpy.int16))
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


def array_to_blob(
    array: numpy.ndarray,
    header: struct.Struct,
    magic: bytes,
    *,
    compress: bool = True,
) -> bytes:
    """`array` as a blob another d3text store can hold: header, then frame.

    The header is `magic`, this codec's version and `array`'s shape; the
    dtype is not recorded, so the reader names it.

    :param array: the values to store, contiguous.
    :param header: the header layout: magic, version, then one field per
        dimension of `array`.
    :param magic: the four bytes naming the store the blob belongs to.
    :param compress: whether the frame is zstd-compressed or stored raw.
    :return: the header plus the blosc2 frame.
    """
    return _pack(header, magic, *array.shape) + _frame(array, compress=compress)


def blob_to_array(
    packed: bytes | memoryview,
    header: struct.Struct,
    magic: bytes,
    name: str,
    dtype: numpy.dtype,
    note: str = "",
) -> numpy.ndarray:
    """The array `array_to_blob` stored, read-only, at its recorded shape.

    :param packed: a blob as `array_to_blob` wrote it.
    :param header: the header layout it was written with.
    :param magic: the magic it must carry.
    :param name: the blob's kind with its article (e.g. "an encodings-store"),
        for the refusal message.
    :param dtype: the dtype it was written in.
    :param note: appended to the message refusing another magic.
    :return: the stored array, a read-only view of the decompressed buffer.
    :raises ValueError: if the blob is shorter than its header, or carries
        another magic or codec version.
    """
    shape = _unpack(packed, header, magic, name, note)
    return _unframe(packed[header.size :], dtype).reshape(shape)


def tensor_to_bytes(
    tensor: Float[Tensor, "token feature"], *, compress: bool = True
) -> bytes:
    """Compress `tensor` for storage.

    The cast to bf16 is a deliberate, lossy narrowing: these are frozen
    base-model activations, not weights that will be trained further.

    :param tensor: one document's token embeddings.
    :param compress: whether the frame is zstd-compressed or stored raw.
    :return: the header plus the blosc2 frame to store.
    """
    body, shape = _compress(tensor, compress=compress)
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
    tensor: Float[Tensor, "window token feature"], *, compress: bool = True
) -> bytes:
    """Compress `tensor` for storage in a layer-boundary store.

    The 3-D counterpart of `tensor_to_bytes`: a layer-boundary store holds
    one row of hidden states per window, not one aggregated row per
    document, because the top encoder layers a cache hit resumes into
    attend only within a window.

    :param tensor: one document's per-window hidden states at the layer
        boundary.
    :param compress: whether the frame is zstd-compressed or stored raw.
    :return: the header plus the blosc2 frame to store.
    """
    body, shape = _compress(tensor, compress=compress)
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
    """What produced an env's rows, recorded once for all its sub-databases.

    None of the fields is recoverable from the matrices. `forward_dtype` is
    diagnostic only, since `select_amp_dtype` names a machine rather than a
    dtype; `None` means the writer recorded none, and such a store holds
    fp16.
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


# The sub-databases of a base model's env. A boundary is named by its
# unfrozen top-layer count, the number a training config sets, rather than by
# the frozen count, which depends on the encoder's depth.
AGGREGATED = "aggregated"
_BOUNDARY_PREFIX = "unfrozen_"

# LMDB sizes its table of named databases when the env opens.
MAX_SUB_DATABASES = 128

_REBUILD = (
    "Rebuild it with `precompute-embeddings`, which writes one env per base "
    "model with one sub-database per cut of the trunk."
)


def boundary_name(unfrozen_top_layers: NonNegative) -> str:
    """The sub-database holding the prefixes a run at this depth resumes.

    :param unfrozen_top_layers: how many top encoder layers are left to run
        over the stored rows.
    :return: the sub-database's name.
    """
    return f"{_BOUNDARY_PREFIX}{unfrozen_top_layers}"


def unfrozen_counts(names: Iterable[str]) -> list[int]:
    """The unfrozen counts of the boundaries among `names`, fewest first.

    :param names: sub-database names, as `sub_databases` lists them.
    :return: each boundary's unfrozen top-layer count, ascending.
    """
    return sorted(
        int(name.removeprefix(_BOUNDARY_PREFIX))
        for name in names
        if name.startswith(_BOUNDARY_PREFIX)
        and name.removeprefix(_BOUNDARY_PREFIX).isdigit()
    )


def read_provenance(env: lmdb.Environment) -> StoreProvenance | None:
    """What wrote `env`, or `None` if it does not say.

    `None` means nothing on disk attributes those matrices to anything, which
    is not the same as a store written by the wrong model.

    :param env: the open LMDB environment.
    :return: the recorded provenance, or None if it records none.
    :raises ProvenanceError: if the record is there but this build cannot read
        it, which reading as unstamped would hide behind the friendlier
        diagnosis; if a field is out of range; or if `env` is a layer-boundary
        store of the older one-cut-per-env layout.
    """
    with env.begin() as transaction:
        raw = transaction.get(_PROVENANCE_KEY)
        legacy = transaction.get(_LEGACY_LAYER_PROVENANCE_KEY) is not None
    if raw is None:
        if legacy:
            msg = (
                f"{env.path()} is a layer-boundary store in the older "
                f"one-cut-per-env layout, which this build does not read. "
                f"{_REBUILD}"
            )
            raise ProvenanceError(msg)
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
            f"and reads format {_PROVENANCE_FORMAT}. {_REBUILD}"
        )
        raise ProvenanceError(msg)

    try:
        base_model = str(record["base_model"])
        max_length = int(record["max_length"])
        stride = int(record["stride"])
        # Optional: an absent diagnostic field changes how no other field
        # reads, and a bump would strand every store.
        forward_dtype = (
            None
            if record.get("forward_dtype") is None
            else str(record["forward_dtype"])
        )
    except (TypeError, KeyError, ValueError) as error:
        msg = (
            f"{env.path()} records a format-{_PROVENANCE_FORMAT} provenance "
            f"missing a field this build reads, or holding one it cannot "
            f"cast: {record!r}."
        )
        raise ProvenanceError(msg) from error

    # beartype is optional (see `d3text.constraints`), so without it
    # `StoreProvenance` would accept these values silently.
    if max_length < 1:
        msg = f"{env.path()} records max_length={max_length}; it must be >= 1."
        raise ProvenanceError(msg)
    if stride < 0:
        msg = f"{env.path()} records stride={stride}; it must be >= 0."
        raise ProvenanceError(msg)
    return StoreProvenance(
        base_model=base_model,
        max_length=max_length,
        stride=stride,
        forward_dtype=forward_dtype,
    )


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


# One handle per env per process, keyed by real path: LMDB does not support
# opening an env twice in one process, and every store over one base model
# now opens the same env.
_envs: dict[str, lmdb.Environment] = {}


def _open_env(path: str, writable: bool) -> lmdb.Environment:
    """The process's handle on a store's LMDB, opened on first use.

    A writable open takes LMDB's locks. A read-only one is unlocked, and
    py-lmdb's `Environment` requires of `lock=False` that no reader use an old
    transaction while a writer is active, so it is for a store the run is not
    to write.
    """
    key = os.path.realpath(path)
    env = _envs.get(key)
    if env is not None:
        if writable and env.flags()["readonly"]:
            msg = (
                f"{path} is already open read-only in this process, so no "
                f"sub-database can be written into it."
            )
            raise lmdb.ReadonlyError(msg)
        return env
    if writable:
        env = lmdb.open(
            path,
            map_size=int(DEFAULT_MAP_SIZE_GIB * 1024**3),
            readahead=True,
            max_readers=2048,
            max_dbs=MAX_SUB_DATABASES,
            # Each commit still flushes its data, but not the meta page:
            # py-lmdb's `metasync` docstring says this "maintains database
            # integrity, but a system crash may undo the last committed
            # transaction". `sync=False` would instead risk corrupting it.
            metasync=False,
        )
    else:
        env = lmdb.open(
            path,
            readonly=True,
            lock=False,
            readahead=True,
            max_readers=2048,
            max_dbs=MAX_SUB_DATABASES,
        )
    _envs[key] = env
    return env


def _release(env: lmdb.Environment) -> None:
    """Close `env` unless a store still open reads it."""
    if any(store.env is env and not store.closed for store in _opened):
        return
    for key in [key for key, cached in _envs.items() if cached is env]:
        del _envs[key]
    env.close()


def sub_databases(path: str | os.PathLike[str]) -> frozenset[str]:
    """The named sub-databases the env at `path` holds.

    :param path: the env's directory.
    :return: the names, without the env's own provenance record.
    :raises lmdb.Error: if no env can be opened at `path`.
    """
    env = _open_env(os.fspath(path), writable=False)
    try:
        return _names(env)
    finally:
        _release(env)


def _names(env: lmdb.Environment) -> frozenset[str]:
    """The keys of `env`'s main database that are not a stamp."""
    with env.begin() as transaction:
        keys = list(transaction.cursor().iternext(keys=True, values=False))
    return frozenset(
        key.decode(errors="replace")
        for key in keys
        if not key.startswith(b"\x00")
    )


def _documents(env: lmdb.Environment) -> int:
    """The most documents any one sub-database of `env` holds."""
    names = _names(env)
    ours = [boundary_name(count) for count in unfrozen_counts(names)]
    ours += [AGGREGATED] if AGGREGATED in names else []
    with env.begin() as transaction:
        return max(
            (
                transaction.stat(
                    env.open_db(name.encode(), txn=transaction, create=False)
                )["entries"]
                for name in ours
            ),
            default=0,
        )


# blosc2's chunk-header flag for a frame stored by memcpy. A frame records
# neither its codec's level nor whether that level was 0: `_compress` at level
# 0 memcpys, but so does any level on a block that would not shrink.
_BLOSC_FLAGS_OFFSET = 2
_BLOSC_MEMCPYED = 0x02

_FRAME_START = {
    _MAGIC: _HEADER.size,
    _WINDOW_MAGIC: _WINDOW_HEADER.size,
}


@dataclasses.dataclass(frozen=True)
class SubDatabaseInfo:
    """What one sub-database holds, read off its frame headers.

    :param name: the sub-database's name.
    :param documents: how many documents it stores.
    :param raw_frames: frames blosc2 stored uncompressed (memcpyed).
    :param compressed_frames: frames blosc2 stored compressed.
    :param unknown_frames: values carrying neither blob magic.
    :param decompressed_bytes: the frames' payload once decompressed.
    :param compressed_bytes: the frames' size as stored.
    :param disk_bytes: the LMDB pages the sub-database occupies.
    """

    name: str
    documents: NonNegative
    raw_frames: NonNegative
    compressed_frames: NonNegative
    unknown_frames: NonNegative
    decompressed_bytes: NonNegative
    compressed_bytes: NonNegative
    disk_bytes: NonNegative

    @property
    def ratio(self) -> float | None:
        """Decompressed over stored bytes, or None if nothing is stored.

        :return: the compression ratio.
        """
        if not self.compressed_bytes:
            return None
        return self.decompressed_bytes / self.compressed_bytes


# What `describe` calls the main database when an older layout keeps the rows
# there rather than in named sub-databases.
MAIN_DATABASE = "(main database)"


@dataclasses.dataclass(frozen=True)
class StoreDescription:
    """What an env records about itself and holds in each database.

    :param provenance: the env's provenance, or None if it records none or
        records one this build cannot read.
    :param provenance_error: why the provenance could not be read, if so.
    :param databases: one record per database holding rows, by name.
    """

    provenance: StoreProvenance | None
    provenance_error: str | None
    databases: list[SubDatabaseInfo]


def describe(path: str | os.PathLike[str]) -> StoreDescription:
    """What the env at `path` records and each of its databases holds.

    Reads every value's headers but decompresses nothing, so its cost is one
    page touched per document. Unlike opening a store, it describes an env
    of an older layout too, rows in the main database, rather than refusing
    it.

    :param path: the env's directory.
    :return: the env's provenance and one record per database holding rows.
    :raises lmdb.Error: if no env can be opened at `path`.
    """
    env = _open_env(os.fspath(path), writable=False)
    try:
        try:
            provenance, error = read_provenance(env), None
        except ProvenanceError as refused:
            provenance, error = None, str(refused)
        if _rows_in_main(env):
            databases = [_describe_one(env, None)]
        else:
            databases = [
                _describe_one(env, name) for name in sorted(_names(env))
            ]
        return StoreDescription(provenance, error, databases)
    finally:
        _release(env)


def _rows_in_main(env: lmdb.Environment) -> bool:
    """Whether `env`'s main database holds blobs, not sub-database names."""
    with env.begin(buffers=True) as transaction:
        for key, value in transaction.cursor():
            if not bytes(key).startswith(b"\x00"):
                return bytes(value[:4]) in _FRAME_START
    return False


def _describe_one(env: lmdb.Environment, name: str | None) -> SubDatabaseInfo:
    """Tally the frames of sub-database `name`, or of the main database."""
    documents = raw = compressed = unknown = 0
    decompressed_bytes = compressed_bytes = 0
    with env.begin(buffers=True) as transaction:
        key_name = None if name is None else name.encode()
        db = env.open_db(key_name, txn=transaction, create=False)
        stat = transaction.stat(db)
        for key, value in transaction.cursor(db):
            if bytes(key).startswith(b"\x00"):
                continue
            documents += 1
            start = _FRAME_START.get(bytes(value[:4]))
            if start is None:
                unknown += 1
                continue
            frame = value[start:]
            nbytes, cbytes, _ = blosc2.get_cbuffer_sizes(frame)
            decompressed_bytes += nbytes
            compressed_bytes += cbytes
            if frame[_BLOSC_FLAGS_OFFSET] & _BLOSC_MEMCPYED:
                raw += 1
            else:
                compressed += 1
    pages = stat["branch_pages"] + stat["leaf_pages"] + stat["overflow_pages"]
    return SubDatabaseInfo(
        name=MAIN_DATABASE if name is None else name,
        documents=documents,
        raw_frames=raw,
        compressed_frames=compressed,
        unknown_frames=unknown,
        decompressed_bytes=decompressed_bytes,
        compressed_bytes=compressed_bytes,
        disk_bytes=pages * stat["psize"],
    )


class _SubDatabaseStore:
    """One named sub-database of a base model's env.

    Read-only unless the opener asks for `writable`; opened writable, the
    sub-database is created if the env lacks it. The env's provenance is
    checked once, at open: a store attributed to the wrong model or window
    is wrong for every document it holds, not just the one a particular call
    happens to ask for first.

    `min_free_gib` is the free space, on the env's filesystem, that `_put`
    keeps: a blob that would take it below that ends the writing. Zero, until
    the opener sets one, is no floor.
    """

    min_free_gib: float = 0.0

    def __init__(
        self,
        path: str | os.PathLike[str],
        base_model: str,
        max_length: Positive,
        name: str,
        *,
        writable: bool,
    ) -> None:
        self.path = os.fspath(path)
        self.env = _open_env(self.path, writable)
        try:
            self.provenance = self._attributed_to(base_model, max_length)
            self.db = self.env.open_db(name.encode(), create=writable)
        except (ProvenanceError, lmdb.Error):
            _release(self.env)
            raise
        self.writable = writable
        self.written = 0
        self.hits = 0
        self.misses = 0
        self.mismatches = 0
        self.closed = False
        self._warned = False
        self._served = False
        _opened.append(self)

    @staticmethod
    def _stamped(
        path: str | os.PathLike[str], provenance: StoreProvenance
    ) -> lmdb.Environment:
        """Make the env at `path` if needed and stamp it if it is new.

        An env holding anything already keeps whatever it records, for the
        constructor to accept or refuse.
        """
        os.makedirs(path, exist_ok=True)
        env = _open_env(os.fspath(path), writable=True)
        try:
            if read_provenance(env) is None and not env.stat()["entries"]:
                write_provenance(env, provenance)
        except ProvenanceError:
            _release(env)
            raise
        return env

    def _attributed_to(
        self, base_model: str, max_length: Positive
    ) -> StoreProvenance:
        """The env's provenance, once it is this run's to read.

        :raises ProvenanceError: if the env records no provenance, or
            records another model or another window.
        """
        recorded = read_provenance(self.env)
        if recorded is None:
            msg = (
                f"{self.path} does not record which model wrote it, so its "
                f"matrices cannot be attributed to {base_model}. A store "
                f"built by another encoder of the same width decodes into a "
                f"plausible matrix of the wrong representation space; "
                f"{_REBUILD[0].lower()}{_REBUILD[1:]}"
            )
            raise ProvenanceError(msg)
        if recorded.base_model != base_model:
            msg = (
                f"{self.path} is stamped for {recorded.base_model} and this "
                f"run's base model is {base_model}; it holds "
                f"{_documents(self.env)} document(s). Their hidden widths may "
                f"agree, in which case nothing downstream would fail: the "
                f"documents the store holds would reach the heads as one "
                f"model's activations and the rest as another's."
            )
            raise ProvenanceError(msg)
        if recorded.max_length != max_length:
            msg = (
                f"{self.path} is stamped at window {recorded.max_length}, "
                f"and this run's encodings are cut at {max_length}; it holds "
                f"{_documents(self.env)} document(s). Neither "
                f"an aggregated row count nor a window count catches a "
                f"document embedded at the wrong window, so the store is "
                f"refused whole. {_REBUILD}"
            )
            raise ProvenanceError(msg)
        return recorded

    def _put(self, pubmed_id: int | str, blob: bytes) -> None:
        """Store `blob` under `pubmed_id`; a refused write ends the writing.

        A write that fails, or whose blob is larger than the free space
        left above `min_free_gib`, is warned about once and ends the writing,
        not the run: what the store already holds is still read, and every
        document it lacks is computed live, as for any miss.
        """
        if not self.writable:
            msg = f"{self.path} is open read-only; nothing can be put into it."
            raise RuntimeError(msg)
        try:
            free = shutil.disk_usage(self.path).free
            if free - len(blob) < self.min_free_gib * 1024**3:
                self._stop_writing(
                    pubmed_id,
                    f"{free / 1024**3:.1f} GiB free; {len(blob)} more bytes "
                    f"would cross the {self.min_free_gib:g} GiB floor",
                )
                return
            with self.env.begin(write=True, db=self.db) as transaction:
                transaction.put(str(pubmed_id).encode(), blob)
        except (lmdb.Error, OSError) as error:
            self._stop_writing(pubmed_id, error)
            return
        self.written += 1

    def _stop_writing(self, pubmed_id: int | str, reason: object) -> None:
        self.writable = False
        logger.warning(
            "Cannot write document %s into %s (%s); it stops growing "
            "here, and every document it does not hold keeps being "
            "computed live. `precompute-embeddings` resumes it, "
            "skipping what it already holds.",
            pubmed_id,
            self.path,
            reason,
        )

    def _mismatched(
        self,
        pubmed_id: int | str,
        counts: tuple[int, int],
        unit: str,
        note: str = "",
    ) -> None:
        """Count a row or window count the encodings disagree with."""
        self.mismatches += 1
        if not self._warned:
            self._warned = True
            logger.warning(
                "%s holds %d %s for document %s where its encodings imply %d, "
                "so the two were built from different text; this document, "
                "and every other that disagrees, is being embedded live "
                "instead.%s",
                self.path,
                counts[0],
                unit,
                pubmed_id,
                counts[1],
                note,
            )

    def _hit(self, pubmed_id: int | str) -> None:
        if not self._served:
            # A store keyed on ids this corpus does not use misses silently,
            # like no store at all; say so once a document is actually served.
            self._served = True
            logger.info(
                "%s served document %s from the store", self.path, pubmed_id
            )
        self.hits += 1

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
            f"{self.mismatches:,} stored at a shape the encodings disagree "
            f"with, {self.written:,} written by this process"
        )

    def close(self) -> None:
        """Close the environment and report what the store answered.

        Registered with `atexit`, and the only moment that sees the totals:
        nothing owns the reader, which is cached for the life of the process. A
        hit rate well under 1.0 is the difference between a run that reads the
        store and one that merely opened it. The env itself stays open while
        another store in this process still reads it.
        """
        if self.closed:
            return
        self.closed = True
        if self.hits + self.misses + self.mismatches:
            logger.info("%s", self.summary())
        if self.written:
            self.env.sync()
        _release(self.env)


class LayerBoundaryStore(_SubDatabaseStore):
    """One boundary's sub-database: per-window hidden states, keyed by id.

    The rows are what the frozen bottom of the trunk leaves each window
    with, for a run that trains the top `unfrozen_top_layers` encoder
    layers. `precompute-embeddings` writes them; `create` adds the
    sub-database to an env (making the env if need be) for a training run
    to fill through `put`.

    :param path: the base model's env.
    :param base_model: the base model the env must record.
    :param unfrozen_top_layers: the boundary, as the number of top encoder
        layers left to run over the stored rows.
    :param max_length: the window the env must record.
    :param writable: whether to open for writing, creating the
        sub-database if the env lacks it.
    :raises ProvenanceError: if the env records another model or window, or
        none, or is in the older one-cut-per-env layout.
    :raises lmdb.NotFoundError: if, opened read-only, the env holds no
        sub-database for this boundary.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        base_model: str,
        unfrozen_top_layers: NonNegative,
        max_length: Positive,
        *,
        writable: bool = False,
    ) -> None:
        super().__init__(
            path,
            base_model,
            max_length,
            boundary_name(unfrozen_top_layers),
            writable=writable,
        )
        self.unfrozen_top_layers = unfrozen_top_layers
        # `_resolve_layer_boundary_cached` decompresses on a background
        # thread; blosc2 holding the GIL would serialize it behind kernel
        # launches. Process-global, harmless for `EmbeddingsStore` too.
        blosc2.set_releasegil(True)
        logger.info(
            "Reading precomputed layer-boundary prefixes from %s, written "
            "by %s at window %d, stride %d, %d top layer(s) left to run",
            self.path,
            self.provenance.base_model,
            self.provenance.max_length,
            self.provenance.stride,
            unfrozen_top_layers,
        )

    @classmethod
    def create(
        cls,
        path: str | os.PathLike[str],
        provenance: StoreProvenance,
        unfrozen_top_layers: NonNegative,
    ) -> Self:
        """Add this boundary's sub-database to the env at `path`, for writing.

        :param path: the env; it and missing parent directories are made if
            absent, and a new env is stamped with `provenance`.
        :param provenance: what the prefixes `put` into it are computed by.
        :param unfrozen_top_layers: the boundary the prefixes are cut at.
        :return: the store, open for reading and writing.
        :raises ProvenanceError: if the env already records another model or
            window, or holds rows and records nothing.
        """
        env = cls._stamped(path, provenance)
        try:
            return cls(
                path,
                provenance.base_model,
                unfrozen_top_layers,
                provenance.max_length,
                writable=True,
            )
        finally:
            _release(env)

    def put(
        self,
        pubmed_id: int | str,
        prefix: Float[Tensor, "window token feature"],
    ) -> None:
        """Store `prefix` as `pubmed_id`'s, in a store opened writable.

        A write that fails is warned about once and ends the writing, not the
        run.

        :param pubmed_id: the document the prefix belongs to.
        :param prefix: the document's per-window hidden states at the layer
            boundary.
        :raises RuntimeError: if the store was opened read-only.
        """
        self._put(pubmed_id, windowed_tensor_to_bytes(prefix))

    def get(
        self, pubmed_id: int | str, expected_windows: Positive
    ) -> Float[Tensor, "window token feature"] | None:
        """The stored per-window prefix for `pubmed_id`, or `None` to run it.

        :param pubmed_id: the document to read.
        :param expected_windows: the window count the batch item implies.
        :return: the stored `[window, token, feature]` tensor, or None.
        """
        with self.env.begin(db=self.db, buffers=True) as transaction:
            blob = transaction.get(str(pubmed_id).encode())
            if blob is None:
                self.misses += 1
                return None
            stored = bytes_to_windowed_tensor(blob)

        if stored.shape[0] != expected_windows:
            self._mismatched(
                pubmed_id, (stored.shape[0], expected_windows), "windows"
            )
            return None
        self._hit(pubmed_id)
        return stored


class EmbeddingsStore(_SubDatabaseStore):
    """The aggregated sub-database: one row per token of each document.

    Optional in an env: a run with the whole trunk frozen can derive these
    rows from a stored boundary instead. The data page of the documentation
    explains the lock and readahead choices.

    :param path: the base model's env.
    :param base_model: the base model the env must record.
    :param max_length: the window the env must record.
    :param writable: whether to open for writing, creating the
        sub-database if the env lacks it.
    :raises ProvenanceError: if the env records another model or window, or
        none, or is in the older one-cut-per-env layout.
    :raises lmdb.NotFoundError: if, opened read-only, the env holds no
        aggregated sub-database.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        base_model: str,
        max_length: Positive,
        *,
        writable: bool = False,
    ) -> None:
        super().__init__(
            path, base_model, max_length, AGGREGATED, writable=writable
        )
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
        """Add the aggregated sub-database to the env at `path`, for writing.

        :param path: the env; it and missing parent directories are made if
            absent, and a new env is stamped with `provenance`.
        :param provenance: what the documents `put` into it are computed by.
        :return: the store, open for reading and writing.
        :raises ProvenanceError: if the env already records another model or
            window, or holds rows and records nothing.
        """
        env = cls._stamped(path, provenance)
        try:
            return cls(
                path,
                provenance.base_model,
                provenance.max_length,
                writable=True,
            )
        finally:
            _release(env)

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
        self._put(pubmed_id, tensor_to_bytes(embedding))

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
        with self.env.begin(db=self.db, buffers=True) as transaction:
            blob = transaction.get(str(pubmed_id).encode())
            if blob is None:
                self.misses += 1
                return None
            stored = bytes_to_tensor(blob)

        if stored.shape[0] != expected_tokens:
            self._mismatched(
                pubmed_id,
                (stored.shape[0], expected_tokens),
                "tokens",
                " This is not a window mismatch: the aggregated row count "
                "comes to the document's token count whatever window the "
                "store was built at. It is a corpus reader that changed "
                "between the two builds, so rebuild whichever artifact "
                "predates that change — the encodings are much the cheaper "
                "of the two.",
            )
            return None
        self._hit(pubmed_id)
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
        return f"{super().summary()}; {computed}"


# Every store opened in this process. Nothing owns a reader — `models.base`
# caches it for the life of the process — so a caller asking what a store
# answered has nothing to ask, and this is the list it asks instead.
_opened: list[_SubDatabaseStore] = []


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
