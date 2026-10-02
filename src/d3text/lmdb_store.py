"""What the encodings and token-label LMDBs share: one handle per document.

Each store keeps one value per document under its key, and its stamps under
keys starting with a NUL byte, which no document key does.
"""

import dataclasses
import fcntl
import gc
import os
import shutil
import weakref
from types import TracebackType
from typing import Self

import lmdb

# A full embeddings env holds every ~100 GiB corpus cut of its trunk, so has
# outgrown 256 GiB; `map_size` reserves address space only and LMDB writes
# sparsely, so headroom costs nothing.
DEFAULT_MAP_SIZE_GIB = 1024.0

_HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"

_COMPACTING = "compacting"
"""The scratch directory `compact` copies a store into, inside that store."""

_WRITER_LOCK = "writer.lock"
"""The file a writer, or `compact`, holds an exclusive `flock` on."""


@dataclasses.dataclass
class _SharedEnv:
    """One store's environment, and what this process's handles do with it."""

    env: lmdb.Environment
    lock: int | None
    """The writer lock a writable env holds from before it maps the store
    until it closes, so no compaction renames the file under its map."""
    directory: tuple[int, int]
    """The store directory's `(st_dev, st_ino)` when `env` opened it; its
    data file's would not do, since `compact` replaces that file."""
    pid: int = dataclasses.field(default_factory=os.getpid)
    """The process that opened `env`."""
    users: int = 0
    writing: bool = False


_shared: dict[str, _SharedEnv] = {}
"""This process's open environments, keyed by the store's real path."""
_shared_pid = os.getpid()


def _release(key: str, shared: _SharedEnv, writing: bool) -> None:
    """Drop one handle's use of `shared`, closing it when no handle is left.

    `LmdbStore.close` runs this, and so does the store's finaliser when the
    store is collected unclosed or the interpreter exits.
    """
    if shared.pid != os.getpid():
        # Inherited across a fork: `_acquire` closes the child's copy, and
        # may already have.
        return
    if writing:
        shared.env.sync()
        shared.writing = False
    shared.users -= 1
    if not shared.users:
        if _shared.get(key) is shared:
            del _shared[key]
        shared.env.close()
        if shared.lock is not None:
            os.close(shared.lock)


def _directory(path: str) -> tuple[int, int] | None:
    try:
        status = os.stat(path)
    except FileNotFoundError:
        return None
    return status.st_dev, status.st_ino


def _refusal(shared: _SharedEnv, path: str, writable: bool) -> str | None:
    """Why `path` cannot be opened over `shared`, or None if it can."""
    if _directory(path) != shared.directory:
        return (
            f"{path} has been deleted or replaced since this process opened "
            f"it, and a handle on the old store is still open; close it "
            f"before reopening the path."
        )
    if writable and shared.lock is None:
        return (
            f"{path} is already open read-only in this process, so it "
            f"cannot be written through another handle."
        )
    if writable and shared.writing:
        return (
            f"{path} is already open for writing in this process; a second "
            f"writer's stamps would describe documents the first is still "
            f"writing. Write the store through one handle."
        )
    return None


def _open(path: str, writable: bool) -> _SharedEnv:
    lock = None
    if not writable:
        env = lmdb.open(path, readonly=True, lock=False)
    else:
        lock = _lock_writer(path)
        try:
            env = lmdb.open(
                path,
                map_size=int(DEFAULT_MAP_SIZE_GIB * 1024**3),
                # Each commit still flushes its data, not the meta page:
                # py-lmdb's `metasync` docstring says this "maintains
                # database integrity, but a system crash may undo the
                # last committed transaction". `sync=False` would
                # instead let a system crash corrupt the database.
                metasync=False,
            )
        except BaseException:
            os.close(lock)
            raise
    status = os.stat(path)
    return _SharedEnv(env, lock, (status.st_dev, status.st_ino))


def _acquire(key: str, path: str, writable: bool) -> _SharedEnv:
    """This process's environment for the store at `path`, one user more."""
    global _shared_pid
    if _shared_pid != os.getpid():
        # A parent's environments must not be used across the fork, and
        # `lmdb` refuses to open a path again until this process's copy is
        # closed; closing it unmaps the child's view only.
        for inherited in _shared.values():
            inherited.env.close()
            if inherited.lock is not None:
                os.close(inherited.lock)
                inherited.lock = None
        _shared.clear()
        _shared_pid = os.getpid()

    while True:
        shared = _shared.get(key)
        if shared is None:
            shared = _shared[key] = _open(path, writable)
            shared.users += 1
            shared.writing = writable
            return shared
        # Pinned before anything that can run the cyclic collector, whose
        # finalisers would otherwise close the env once its other users go.
        shared.users += 1
        if _shared.get(key) is not shared:
            shared.users -= 1
            continue
        try:
            refusal = _refusal(shared, path, writable)
            if refusal is not None:
                # The handles in the way may be garbage in a reference
                # cycle, whose finalisers wait for the cyclic collector.
                gc.collect()
                if shared.users > 1:
                    refusal = _refusal(shared, path, writable)
        except BaseException:
            _release(key, shared, writing=False)
            raise
        if refusal is None:
            shared.writing = shared.writing or writable
            return shared
        stale = shared.users == 1
        _release(key, shared, writing=False)
        if not stale:
            raise RuntimeError(refusal)


def _lock_writer(path: str) -> int:
    """Hold `path`'s writer lock, refusing a store another process writes."""
    descriptor = os.open(
        os.path.join(path, _WRITER_LOCK), os.O_RDWR | os.O_CREAT, 0o644
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException as error:
        os.close(descriptor)
        if not isinstance(error, BlockingIOError):
            raise
        msg = (
            f"{path} is open for writing in another process; LMDB would "
            f"interleave the two runs' writes, and a compaction would lose "
            f"the writer's later ones. Wait for that process to finish."
        )
        raise RuntimeError(msg) from None
    return descriptor


def refuse_hdf5(path: str | os.PathLike[str], refusal: str) -> None:
    """Refuse `path` if it is a file of the HDF5 layout the stores replaced.

    :param path: where a store is about to be opened.
    :param refusal: what to say after the path: which store the file was,
        and how to rebuild it.
    :raises ValueError: if `path` is a file starting with HDF5's signature.
    """
    if not os.path.isfile(path):
        return
    with open(path, "rb") as handle:
        signature = handle.read(len(_HDF5_SIGNATURE))
    if signature == _HDF5_SIGNATURE:
        raise ValueError(f"{os.fspath(path)} is {refusal}")


class LmdbStore:
    """An open store: one LMDB value per document, written in one transaction.

    Handles on one store in one process share its LMDB environment, which
    LMDB refuses to open twice in a process; a process forked after opening
    one opens its own. Readers open it without LMDB's lock; a writable
    environment holds the store's writer lock until its last handle closes.

    :param path: the store's directory; created when opened writable.
    :param writable: whether to open for writing.
    :raises RuntimeError: if this process already holds `path` open for
        writing, or open read-only when a writer asks for it, or still holds
        a handle on a store since deleted or replaced at `path`, or another
        process holds it open for writing.
    :raises OSError: if, opened writable, the writer lock cannot be taken for
        another reason.
    :raises lmdb.Error: if, opened read-only, no store exists at `path`.
    """

    def __init__(
        self, path: str | os.PathLike[str], *, writable: bool = False
    ) -> None:
        self.path = os.fspath(path)
        if writable:
            os.makedirs(self.path, exist_ok=True)
        key = os.path.realpath(self.path)
        shared = _acquire(key, self.path, writable)
        self.env = shared.env
        # An owner dropped unclosed, or the interpreter exiting, still
        # releases its use of the shared environment.
        self._finaliser = weakref.finalize(
            self, _release, key, shared, writable
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Sync what was written and release this handle on the store."""
        self._finaliser()

    def _get_raw(self, key: bytes) -> bytes | None:
        with self.env.begin() as transaction:
            value = transaction.get(key)
        return None if value is None else bytes(value)

    def _put_raw(self, key: bytes, value: bytes | None) -> None:
        """Store `value` under `key` in one transaction; None deletes it."""
        with self.env.begin(write=True) as transaction:
            if value is None:
                transaction.delete(key)
            else:
                transaction.put(key, value)

    def clear(self) -> None:
        """Drop every document and stamp, in one transaction."""
        with self.env.begin(write=True) as transaction:
            transaction.drop(self.env.open_db(txn=transaction), delete=False)

    def keys(self) -> list[str]:
        """Every document key, in sorted order, without the stamps.

        :return: the keys.
        """
        with self.env.begin() as transaction:
            keys = transaction.cursor().iternext(keys=True, values=False)
            return [
                bytes(key).decode()
                for key in keys
                if not bytes(key).startswith(b"\x00")
            ]

    def delete(self, key: str) -> None:
        """Remove the document under `key`, if there is one.

        :param key: the document's key.
        """
        self._put_raw(key.encode(), None)


def _fsync_path(path: str, flags: int = os.O_RDONLY) -> None:
    """Flush the file or directory at `path` to disk."""
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def compact(path: str | os.PathLike[str]) -> None:
    """Rewrite the store at `path` without the free pages it has accrued.

    LMDB writes a replaced value to new pages and keeps the old ones in the
    file as free space. The compacted copy is fsynced, then replaces the
    data file in one rename, and the directory is fsynced after it, so a
    crash or power loss leaves either the old file or the whole copy.

    :param path: a store no handle in this process holds open.
    :raises RuntimeError: if this process holds `path` open, or another
        process holds it open for writing; either's writes would land in
        the file this replaces.
    """
    path = os.fspath(path)
    realpath = os.path.realpath(path)
    if realpath in _shared and _shared_pid == os.getpid():
        # A store's handle in a garbage reference cycle stays in _shared until
        # the cyclic collector runs its finaliser; collect garbage before
        # refusing, same as _acquire does.
        gc.collect()
        if realpath in _shared:
            msg = f"{path} is open in this process; close it before compacting."
            raise RuntimeError(msg)

    lock = _lock_writer(path)
    try:
        scratch = os.path.join(path, _COMPACTING)
        shutil.rmtree(scratch, ignore_errors=True)
        os.mkdir(scratch)
        env = lmdb.open(path, readonly=True, lock=False)
        try:
            env.copy(scratch, compact=True)
        finally:
            env.close()
        copy = os.path.join(scratch, "data.mdb")
        _fsync_path(copy)
        os.replace(copy, os.path.join(path, "data.mdb"))
        os.rmdir(scratch)
        _fsync_path(path, os.O_RDONLY | os.O_DIRECTORY)
    finally:
        os.close(lock)


__all__ = ["DEFAULT_MAP_SIZE_GIB", "LmdbStore", "compact", "refuse_hdf5"]
