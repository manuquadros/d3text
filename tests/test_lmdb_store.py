"""The exclusion a writable LMDB store holds against other processes."""

import errno
import os
import subprocess
import sys
from collections.abc import Iterator

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
