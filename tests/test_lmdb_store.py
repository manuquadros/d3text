"""The exclusion a writable LMDB store holds against other processes."""

import errno
import multiprocessing
import os
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
