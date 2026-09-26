"""Regression test for `store_in_db` clobbering untouched fields."""

import asyncio

from aiotinydb import AIOTinyDB
from aiotinydb.storage import AIOJSONStorage
from brenda_references.utils import CachingMiddleware
from d3types import Document
from scripts.retrieve_text import MISSING_FULLTEXT_QUERY, store_in_db


def _doc(**overrides: object) -> Document:
    fields: dict[str, object] = {
        "authors": "",
        "title": "",
        "journal": "",
        "volume": "",
        "pages": "",
        "year": 2000,
        "pubmed_id": "1",
        "path": "",
    }
    fields.update(overrides)
    return Document(**fields)  # type: ignore[arg-type]


def test_store_in_db_does_not_revert_a_field_stored_by_an_earlier_pass(
    tmp_path,
) -> None:
    """A field written earlier in the run must survive a later, unrelated write.

    `retrieve_text.run` fetches full text first and stores it, then fetches
    abstracts from a snapshot taken before either fetch ran and stores
    those. `store_in_db` used to write that snapshot's whole
    `model_dump()`, so a document needing both fields had its just-stored
    `fulltext` reverted to the snapshot's `None` the moment the abstract
    was written back.
    """
    docdb_path = tmp_path / "documents.json"

    async def scenario() -> dict:
        async with AIOTinyDB(
            docdb_path, storage=CachingMiddleware(AIOJSONStorage)
        ) as docdb:
            doc_id = docdb.table("documents").insert(
                _doc(fulltext="already fetched").model_dump()
            )

            # Snapshot as it stood before the full-text fetch: fulltext
            # is still unset.
            stale_snapshot = _doc(fulltext=None, abstract="new abstract")
            await store_in_db(
                field="abstract",
                items={doc_id: stale_snapshot},
                docdb=docdb,
            )

            return docdb.table("documents").get(doc_id=doc_id)

    stored = asyncio.run(scenario())

    assert stored["abstract"] == "new abstract"
    assert stored["fulltext"] == "already fetched"


def test_missing_fulltext_query_selects_a_null_fulltext(tmp_path) -> None:
    """A document with ``fulltext: null`` must still be selected for retrieval.

    `sync-doc-db` inserts every document with `fulltext` present and set
    to `None` (the `Document` default), never with the key absent. The
    query used to test only for the key missing or the empty string, so
    it skipped every such document.
    """
    docdb_path = tmp_path / "documents.json"

    async def scenario() -> list:
        async with AIOTinyDB(
            docdb_path, storage=CachingMiddleware(AIOJSONStorage)
        ) as docdb:
            docdb.table("documents").insert(
                _doc(pmc_id="PMC1", pmc_open=True, fulltext=None).model_dump()
            )
            return docdb.table("documents").search(MISSING_FULLTEXT_QUERY)

    matches = asyncio.run(scenario())

    assert len(matches) == 1
