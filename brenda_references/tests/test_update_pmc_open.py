"""Regression test for `update_pmc_open` opening the document database."""

import asyncio

from tinydb import TinyDB
from tinydb.storages import JSONStorage

from scripts import update_pmc_open


class _FakeNCBI:
    """Stands in for `AsyncNCBIAdapter`; no network involved."""

    async def __aenter__(self) -> "_FakeNCBI":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def is_pmc_open(self, pmcid: str | None) -> bool:
        return True


def test_run_opens_the_database_and_updates_pmc_open(
    tmp_path, monkeypatch
) -> None:
    """`run` must open the document database and reach the update.

    `run` used to build `AIOTinyDB` with plain `tinydb.middlewares
    .CachingMiddleware` wrapping `tinydb.storages.JSONStorage` — neither
    async-aware, so `AIOTinyDB.__aenter__` raised `AttributeError` before a
    single document was read. Pins the fix: opening a one-document database
    through the script's real `run()` must reach the `pmc_open` update.
    """
    docdb_path = tmp_path / "documents.json"
    with TinyDB(docdb_path, storage=JSONStorage) as seed_db:
        doc_id = seed_db.table("documents").insert(
            {"pmc_id": "PMC123", "pmc_open": False}
        )

    monkeypatch.setitem(update_pmc_open.config, "documents", str(docdb_path))
    monkeypatch.setattr(update_pmc_open, "AsyncNCBIAdapter", _FakeNCBI)

    asyncio.run(update_pmc_open.run())

    with TinyDB(docdb_path, storage=JSONStorage) as result_db:
        stored = result_db.table("documents").get(doc_id=doc_id)
        assert stored["pmc_open"] is True
