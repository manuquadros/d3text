"""The exclusion a writable LMDB store holds against other processes."""

import errno
import gc
import multiprocessing
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from multiprocessing.connection import Connection
from pathlib import Path

import pytest

from d3text import lmdb_store

_HELD_WRITER = """
import sys
from d3text import lmdb_store

with lmdb_store.LmdbStore(sys.argv[1], writable=True) as store:
    with store.env.begin(write=True) as transaction:
        transaction.put(b"before", b"1")
    print("ready", flush=True)
    sys.stdin.readline()
    with store.env.begin(write=True) as transaction:
        transaction.put(b"after", b"1")
"""


@pytest.fixture
def held_writer(tmp_path) -> Iterator[tuple[str, subprocess.Popen[str]]]:
    """A store another process holds open for writing, and that process."""
    path = str(tmp_path / "store")
    writer = subprocess.Popen(
        [sys.executable, "-c", _HELD_WRITER, path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert writer.stdout is not None
        assert writer.stdout.readline() == "ready\n"
        yield path, writer
    finally:
        writer.kill()
        writer.wait()


def _finish(writer: subprocess.Popen[str]) -> None:
    writer.communicate("go\n", timeout=30)
    assert writer.returncode == 0


def test_a_second_writer_is_refused_while_another_process_writes(
    held_writer,
) -> None:
    """Two writing processes would each stamp over what the other wrote.

    A reader still opens the store: it takes no part in the exclusion.
    """
    path, writer = held_writer

    with pytest.raises(RuntimeError, match="another process"):
        lmdb_store.LmdbStore(path, writable=True)
    with lmdb_store.LmdbStore(path) as reader:
        assert reader.keys() == ["before"]

    _finish(writer)


def test_compacting_a_store_another_process_writes_is_refused(
    held_writer,
) -> None:
    """A compaction under a live writer would lose its later writes.

    The writer keeps its map on the data file the compaction renames away,
    so it would commit into a file nobody opens again and exit cleanly.
    """
    path, writer = held_writer

    with pytest.raises(RuntimeError, match="another process"):
        lmdb_store.compact(path)
    _finish(writer)

    with lmdb_store.LmdbStore(path) as reader:
        assert reader.keys() == ["after", "before"]


_WRITER_PAUSED_AFTER_MAPPING = """
import sys
import lmdb
from d3text import lmdb_store

real_open = lmdb.open


def open_then_pause(*args, **kwargs):
    env = real_open(*args, **kwargs)
    if not kwargs.get("readonly"):
        print("mapped", flush=True)
        sys.stdin.readline()
    return env


lmdb.open = open_then_pause
with lmdb_store.LmdbStore(sys.argv[1], writable=True) as store:
    with store.env.begin(write=True) as transaction:
        transaction.put(b"late", b"1")
"""


def _seed(path: str) -> None:
    with lmdb_store.LmdbStore(path, writable=True) as store:
        with store.env.begin(write=True) as transaction:
            transaction.put(b"before", b"1")


def _compact_unless_refused(path: str) -> None:
    try:
        lmdb_store.compact(path)
    except RuntimeError:
        pass


def test_a_compaction_while_a_writer_maps_the_store_loses_no_write(
    tmp_path,
) -> None:
    """A writer that maps the data file before taking the lock is exposed.

    A compaction run in that gap renames the file away under the writer,
    whose commits then land in the unlinked copy. Either the compaction is
    refused or the write survives; the writer pauses right after mapping.
    """
    path = str(tmp_path / "store")
    _seed(path)
    writer = subprocess.Popen(
        [sys.executable, "-c", _WRITER_PAUSED_AFTER_MAPPING, path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert writer.stdout is not None
        assert writer.stdout.readline() == "mapped\n"
        _compact_unless_refused(path)
        _finish(writer)
    finally:
        writer.kill()
        writer.wait()

    with lmdb_store.LmdbStore(path) as reader:
        assert reader.keys() == ["before", "late"]


_COMPACT = """
import sys
from d3text import lmdb_store

try:
    lmdb_store.compact(sys.argv[1])
except RuntimeError:
    pass
"""


def test_a_reader_keeping_the_writable_env_alive_keeps_it_locked(
    tmp_path,
) -> None:
    """Closing the writer while a reader shares its env must keep the lock.

    The env is still mapped on the data file, and a later writable handle in
    this process reuses it; a compaction in between would strand its writes.
    """
    path = str(tmp_path / "store")
    writer = lmdb_store.LmdbStore(path, writable=True)
    reader = lmdb_store.LmdbStore(path)
    writer.close()
    try:
        subprocess.run([sys.executable, "-c", _COMPACT, path], check=True)
        with lmdb_store.LmdbStore(path, writable=True) as again:
            with again.env.begin(write=True) as transaction:
                transaction.put(b"after", b"1")
    finally:
        reader.close()

    with lmdb_store.LmdbStore(path) as fresh:
        assert fresh.keys() == ["after"]


def _reset_after_fork(
    path: str,
    inherited_reader: lmdb_store.LmdbStore,
    connection: Connection,
) -> None:
    try:
        with lmdb_store.LmdbStore(path):
            pass
        connection.send("reset")
        assert connection.recv() == "close"
        inherited_reader.close()
    except BaseException as error:
        connection.send(f"{type(error).__name__}: {error}")
    else:
        connection.send("closed")
    finally:
        connection.close()


def test_fork_reset_drops_only_the_child_writer_lock(tmp_path: Path) -> None:
    """A fork reset must not retain or double-close its writer-lock fd."""
    path = str(tmp_path / "store")
    writer = lmdb_store.LmdbStore(path, writable=True)
    inherited_reader = lmdb_store.LmdbStore(path)
    writer.close()

    context = multiprocessing.get_context("fork")
    parent_connection, child_connection = context.Pipe()
    child = context.Process(
        target=_reset_after_fork,
        args=(path, inherited_reader, child_connection),
    )
    child.start()
    child_connection.close()
    close_sent = False
    try:
        assert parent_connection.poll(30)
        assert parent_connection.recv() == "reset"
        compact = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from d3text import lmdb_store; "
                "lmdb_store.compact(sys.argv[1])",
                path,
            ],
            capture_output=True,
            text=True,
        )
        assert compact.returncode != 0
        assert "another process" in compact.stderr

        inherited_reader.close()
        lmdb_store.compact(path)
        assert child.is_alive()

        parent_connection.send("close")
        close_sent = True
        assert parent_connection.poll(30)
        assert parent_connection.recv() == "closed"
        child.join(timeout=30)
        assert child.exitcode == 0
    finally:
        inherited_reader.close()
        if child.is_alive() and not close_sent:
            parent_connection.send("close")
        child.join(timeout=30)
        if child.is_alive():
            child.terminate()
            child.join()
        parent_connection.close()


def test_a_failed_writer_lock_leaks_neither_descriptor_nor_env(
    tmp_path, monkeypatch
) -> None:
    """A lock error other than contention must still undo the open."""
    path = str(tmp_path / "store")
    _seed(path)

    def no_locks(descriptor: int, operation: int) -> None:
        raise OSError(errno.ENOLCK, "no locks available")

    monkeypatch.setattr(lmdb_store.fcntl, "flock", no_locks)
    before = sorted(os.listdir("/proc/self/fd"))
    with pytest.raises(OSError, match="no locks available"):
        lmdb_store.LmdbStore(path, writable=True)

    assert sorted(os.listdir("/proc/self/fd")) == before
    assert os.path.realpath(path) not in lmdb_store._shared


def test_a_writable_store_syncs_its_data_on_every_commit(
    tmp_path: Path,
) -> None:
    """A writable store must survive a machine crash during a precompute:
    py-lmdb's `sync=False` lets a system crash corrupt the database, while
    `metasync=False` can only undo the last committed transaction."""
    path = str(tmp_path / "store")
    with lmdb_store.LmdbStore(path, writable=True) as store:
        flags = store.env.flags()
        assert (flags["sync"], flags["metasync"]) == (True, False)


def test_compact_fsyncs_the_copy_before_and_the_directory_after_the_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Power loss after a rename of an unsynced copy leaves a torn store.

    A crash cannot be simulated in-process, so this pins the ordering the
    durability rests on: the copy reaches disk before it takes the data
    file's name, and the directory entry reaches disk after.
    """
    path = str(tmp_path / "store")
    with lmdb_store.LmdbStore(path, writable=True) as store:
        with store.env.begin(write=True) as transaction:
            transaction.put(b"key", b"value")

    events: list[tuple[str, str]] = []
    real_fsync, real_replace = os.fsync, os.replace

    def spy_fsync(fd: int) -> None:
        events.append(("fsync", os.readlink(f"/proc/self/fd/{fd}")))
        real_fsync(fd)

    def spy_replace(src: str, dst: str) -> None:
        events.append(("replace", os.fspath(src)))
        real_replace(src, dst)

    monkeypatch.setattr(os, "fsync", spy_fsync)
    monkeypatch.setattr(os, "replace", spy_replace)
    lmdb_store.compact(path)

    copy = os.path.join(path, "compacting", "data.mdb")
    replaced = events.index(("replace", copy))
    assert ("fsync", copy) in events[:replaced]
    assert ("fsync", os.path.realpath(path)) in events[replaced:]
    with lmdb_store.LmdbStore(path) as store:
        with store.env.begin() as transaction:
            assert transaction.get(b"key") == b"value"


def _seeded(path: str, key: bytes) -> None:
    with lmdb_store.LmdbStore(path, writable=True) as store:
        with store.env.begin(write=True) as transaction:
            transaction.put(key, b"1")


def test_an_unclosed_reader_dropped_does_not_block_a_writer(
    tmp_path: Path,
) -> None:
    """Dropping a reader unclosed releases its use of the shared env."""
    path = str(tmp_path / "store")
    _seeded(path, b"seed")
    reader = lmdb_store.LmdbStore(path)
    del reader

    _seeded(path, b"after")
    with lmdb_store.LmdbStore(path) as store:
        assert store.keys() == ["after", "seed"]


@pytest.mark.parametrize("writable", [False, True])
def test_a_handle_left_in_a_reference_cycle_does_not_block_a_writer(
    tmp_path: Path, writable: bool
) -> None:
    """A handle only the cyclic collector frees is released before refusing.

    Automatic collection is off, so nothing but the open itself can run the
    collector that finalises the handle.
    """
    path = str(tmp_path / "store")
    _seeded(path, b"seed")
    cycle: list[object] = [lmdb_store.LmdbStore(path, writable=writable)]
    cycle.append(cycle)
    gc.disable()
    try:
        del cycle
        _seeded(path, b"after")
    finally:
        gc.enable()
    with lmdb_store.LmdbStore(path) as store:
        assert store.keys() == ["after", "seed"]


def test_reopening_a_replaced_store_under_a_live_reader_is_refused(
    tmp_path: Path,
) -> None:
    """A store replaced on disk is not served through the old one's env,
    which would hand the new opener the deleted store's data."""
    path = str(tmp_path / "store")
    replacement = str(tmp_path / "replacement")
    _seeded(path, b"old")
    _seeded(replacement, b"new")
    reader = lmdb_store.LmdbStore(path)
    try:
        shutil.rmtree(path)
        shutil.copytree(replacement, path)
        with pytest.raises(RuntimeError, match="has been deleted or replaced"):
            lmdb_store.LmdbStore(path)
    finally:
        reader.close()


def test_gc_collecting_during_acquire_does_not_close_the_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A collection inside the reuse check does not close the env reused.

    `_directory` is patched to collect, so the only other user of the env, a
    reader held by a garbage cycle, is finalised while the open decides.
    """
    path = str(tmp_path / "store")
    _seeded(path, b"seed")
    cycle: list[object] = [lmdb_store.LmdbStore(path)]
    cycle.append(cycle)
    gc.disable()
    try:
        del cycle
    finally:
        gc.enable()
    real_directory = lmdb_store._directory

    def directory_with_collect(p: str) -> tuple[int, int] | None:
        gc.collect()
        return real_directory(p)

    monkeypatch.setattr(lmdb_store, "_directory", directory_with_collect)
    with lmdb_store.LmdbStore(path) as store:
        assert store._get_raw(b"seed") == b"1"

    with lmdb_store.LmdbStore(path, writable=True) as store:
        with store.env.begin(write=True) as transaction:
            transaction.put(b"after", b"1")


def test_a_failed_reuse_open_gives_back_its_use_of_the_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An open that raises while checking the shared env leaves no user on it,
    which would otherwise refuse a later writer of the store."""
    path = str(tmp_path / "store")
    _seeded(path, b"seed")
    reader = lmdb_store.LmdbStore(path)
    real_directory = lmdb_store._directory

    def denied(p: str) -> tuple[int, int] | None:
        monkeypatch.setattr(lmdb_store, "_directory", real_directory)
        raise PermissionError(p)

    monkeypatch.setattr(lmdb_store, "_directory", denied)
    with pytest.raises(PermissionError):
        lmdb_store.LmdbStore(path)
    reader.close()

    _seeded(path, b"after")
