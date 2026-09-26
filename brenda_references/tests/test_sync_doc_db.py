"""Regression tests for `sync_doc_db`: the dropped `store_strains` await,
and its worker-pool concurrency over the streamed BRENDA cursor.
"""

import asyncio

import pytest
from apiadapters.straininfo import AsyncStrainInfoAdapter
from apiadapters.straininfo.straininfo import StrainRef
from tinydb import TinyDB
from tinydb.storages import JSONStorage
from tinydb.table import Document as TDBDocument

from brenda_references import brenda_references as bref
from brenda_references import db as db_module
from brenda_references.docdb import BrendaDocDB

STRAIN = StrainRef(id=42, name="ATCC 1234")


class _FakeReference:
    reference_id = 1

    def model_dump(self) -> dict:
        return {
            "authors": "",
            "title": "",
            "journal": "",
            "volume": "",
            "pages": "",
            "year": 2000,
            "pubmed_id": "",
            "path": "",
        }


class _FakeBRENDA:
    """Stands in for `db.BRENDA`, serving one reference with one strain."""

    async def __aenter__(self) -> "_FakeBRENDA":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    def count_references(self) -> int:
        return 1

    def references(self) -> list[_FakeReference]:
        return [_FakeReference()]

    def enzyme_relations(self, reference_id: int) -> dict:
        return {
            "enzymes": set(),
            "bacteria": set(),
            "strains": {STRAIN},
            "other_organisms": set(),
            "triples": {},
        }


class _FakeNCBI:
    """Never actually called: the fake reference has no pubmed_id."""

    async def __aenter__(self) -> "_FakeNCBI":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


def test_sync_doc_db_stores_strains(tmp_path, monkeypatch) -> None:
    """A strain attached to a document must survive `sync_doc_db` intact.

    Regression test, two invariants pinned together: `store_strains` is a
    coroutine and was being called without `await`, so the buffer was never
    updated and no strain was ever written to the store; and the row must
    carry its full `model_dump()`, `id` included (`None` here, since the
    strain is unresolved) and keyed under the BRENDA id, not dropped by a
    sink that only writes a partial record. Driven with `asyncio.run`, like
    `main()` itself, rather than `pytest.mark.asyncio`, which needs a
    plugin not present in every environment this suite runs in.
    """
    docdb_path = tmp_path / "documents.json"
    monkeypatch.setattr(bref, "documents_path", lambda: docdb_path)
    monkeypatch.setattr(db_module, "BRENDA", _FakeBRENDA)
    monkeypatch.setattr(bref, "AsyncNCBIAdapter", _FakeNCBI)

    async def fake_get_strain_ids(self, query):
        return []

    async def fake_get_strain_data(self, query):
        return ()

    monkeypatch.setattr(
        AsyncStrainInfoAdapter, "get_strain_ids", fake_get_strain_ids
    )
    monkeypatch.setattr(
        AsyncStrainInfoAdapter, "get_strain_data", fake_get_strain_data
    )

    asyncio.run(bref.sync_doc_db())

    with BrendaDocDB(path=str(docdb_path)) as docdb:
        stored = docdb.strains.get(doc_id=STRAIN.id)
        assert stored is not None
        assert "id" in stored
        assert stored["id"] is None
        assert stored["designations"] == ["ATCC 1234"]


class _ManyReference:
    """A fake reference with a unique id and a non-empty `pubmed_id`.

    A non-empty `pubmed_id` routes `expand_doc` through `AsyncNCBIAdapter`,
    which is what the tests below need to observe.
    """

    def __init__(self, reference_id: int) -> None:
        self.reference_id = reference_id

    def model_dump(self) -> dict:
        return {
            "authors": "",
            "title": "",
            "journal": "",
            "volume": "",
            "pages": "",
            "year": 2000,
            "pubmed_id": str(self.reference_id),
            "path": "",
        }


def _no_relations(reference_id: int) -> dict:
    return {
        "enzymes": set(),
        "bacteria": set(),
        "strains": set(),
        "other_organisms": set(),
        "triples": {},
    }


class _ManyBRENDA:
    """Stands in for `db.BRENDA`, serving `n` references with no relations."""

    def __init__(self, n: int) -> None:
        self.n = n

    async def __aenter__(self) -> "_ManyBRENDA":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    def count_references(self) -> int:
        return self.n

    def references(self) -> list[_ManyReference]:
        return [_ManyReference(i) for i in range(self.n)]

    def enzyme_relations(self, reference_id: int) -> dict:
        return _no_relations(reference_id)


class _CountingBRENDA:
    """Like `_ManyBRENDA`, but its `references()` is a generator that
    records each pull in `pulled` as it happens, not up front.
    """

    def __init__(self, n: int, pulled: list[int]) -> None:
        self.n = n
        self._pulled = pulled

    async def __aenter__(self) -> "_CountingBRENDA":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    def count_references(self) -> int:
        return self.n

    def references(self):
        for i in range(self.n):
            self._pulled.append(i)
            yield _ManyReference(i)

    def enzyme_relations(self, reference_id: int) -> dict:
        return _no_relations(reference_id)


class _TrackingNCBI:
    """Records how many `article_ids` calls are in flight at once, and how
    many references had been pulled from `pulled` before its first call.
    """

    def __init__(self, pulled: list[int], delay: float = 0.01) -> None:
        self.delay = delay
        self._pulled = pulled
        self._in_flight = 0
        self.max_in_flight = 0
        self.pulled_before_first_call: int | None = None
        self.calls: list[str] = []

    async def __aenter__(self) -> "_TrackingNCBI":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def article_ids(self, pubmed_id: str) -> dict:
        if self.pulled_before_first_call is None:
            self.pulled_before_first_call = len(self._pulled)
        self._in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self._in_flight)
        await asyncio.sleep(self.delay)
        self.calls.append(pubmed_id)
        self._in_flight -= 1
        return {}

    async def is_pmc_open(self, pmc_id: str | None) -> bool:
        return False


def test_sync_doc_db_admits_at_most_one_worker_batch_ahead(
    tmp_path, monkeypatch
) -> None:
    """Workers pull references one at a time, not the whole cursor upfront.

    A task-per-reference shape (one `TaskGroup.create_task` per reference,
    built by a `for reference in brenda.references():` loop with no
    `await`) drains the entire streamed cursor before any task runs, so
    `pulled_before_first_call` would equal `n_refs`. The worker-pool shape
    spawns only `_MAX_CONCURRENT_DOCUMENTS` tasks, each of which pulls a
    single reference before its own first `await`, so at most one batch of
    workers is ever pulled ahead of processing. RED against the
    task-per-reference shape with `n_refs` well above the worker count.
    """
    docdb_path = tmp_path / "documents.json"
    n_refs = 40
    pulled: list[int] = []
    monkeypatch.setattr(bref, "documents_path", lambda: docdb_path)
    monkeypatch.setattr(
        db_module, "BRENDA", lambda: _CountingBRENDA(n_refs, pulled)
    )
    ncbi = _TrackingNCBI(pulled)
    monkeypatch.setattr(bref, "AsyncNCBIAdapter", lambda: ncbi)

    asyncio.run(bref.sync_doc_db())

    assert ncbi.pulled_before_first_call is not None
    assert ncbi.pulled_before_first_call <= bref._MAX_CONCURRENT_DOCUMENTS + 1
    assert len(ncbi.calls) == n_refs


def test_sync_doc_db_bounds_concurrent_documents(tmp_path, monkeypatch) -> None:
    """More than one document expands at once, never more than the cap.

    Against the old sequential `for reference in brenda.references(): await
    add_document(...)` loop, `max_in_flight` never rises above 1.
    """
    docdb_path = tmp_path / "documents.json"
    n_refs = 24
    pulled: list[int] = []
    monkeypatch.setattr(bref, "documents_path", lambda: docdb_path)
    monkeypatch.setattr(db_module, "BRENDA", lambda: _ManyBRENDA(n_refs))
    ncbi = _TrackingNCBI(pulled)
    monkeypatch.setattr(bref, "AsyncNCBIAdapter", lambda: ncbi)

    asyncio.run(bref.sync_doc_db())

    assert ncbi.max_in_flight > 1
    assert ncbi.max_in_flight <= bref._MAX_CONCURRENT_DOCUMENTS
    assert len(ncbi.calls) == n_refs


def test_sync_doc_db_skips_existing_and_processes_new_once(
    tmp_path, monkeypatch
) -> None:
    """Workers still skip a stored reference and touch every new one once.

    Seeds one reference already in the database; `sync_doc_db` must not
    call NCBI for it, and must call NCBI exactly once for every other
    reference, however the workers interleave.
    """
    docdb_path = tmp_path / "documents.json"
    n_refs = 5
    existing_doc = {
        "authors": "",
        "title": "",
        "journal": "",
        "volume": "",
        "pages": "",
        "year": 2000,
        "pubmed_id": "",
        "path": "",
    }
    with TinyDB(docdb_path, storage=JSONStorage) as seed_db:
        seed_db.table("documents").insert(TDBDocument(existing_doc, doc_id=0))

    pulled: list[int] = []
    monkeypatch.setattr(bref, "documents_path", lambda: docdb_path)
    monkeypatch.setattr(db_module, "BRENDA", lambda: _ManyBRENDA(n_refs))
    ncbi = _TrackingNCBI(pulled)
    monkeypatch.setattr(bref, "AsyncNCBIAdapter", lambda: ncbi)

    asyncio.run(bref.sync_doc_db())

    assert sorted(ncbi.calls, key=int) == [str(i) for i in range(1, n_refs)]
    with BrendaDocDB(path=str(docdb_path)) as docdb:
        assert {doc.doc_id for doc in docdb.documents} == set(range(n_refs))


class _FailingNCBI:
    """Raises on the first `article_ids` call; a worker's own subsequent
    calls succeed, so the error must still surface from `sync_doc_db`.
    """

    def __init__(self) -> None:
        self.calls = 0

    async def __aenter__(self) -> "_FailingNCBI":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def article_ids(self, pubmed_id: str) -> dict:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("boom")
        await asyncio.sleep(0.01)
        return {}

    async def is_pmc_open(self, pmc_id: str | None) -> bool:
        return False


def test_sync_doc_db_propagates_a_reference_error(
    tmp_path, monkeypatch
) -> None:
    """An error fetching one document must not vanish silently.

    `asyncio.TaskGroup` cancels the sibling workers and re-raises inside an
    `ExceptionGroup`; `group_contains` checks the group's exceptions without
    pinning the group's own shape.
    """
    docdb_path = tmp_path / "documents.json"
    monkeypatch.setattr(bref, "documents_path", lambda: docdb_path)
    monkeypatch.setattr(db_module, "BRENDA", lambda: _ManyBRENDA(5))
    monkeypatch.setattr(bref, "AsyncNCBIAdapter", _FailingNCBI)

    with pytest.raises(ExceptionGroup) as excinfo:
        asyncio.run(bref.sync_doc_db())

    assert excinfo.group_contains(RuntimeError)
