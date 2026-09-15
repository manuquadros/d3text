"""Regression test for the dropped `store_strains` await in `sync_doc_db`."""

import asyncio

from apiadapters.straininfo import AsyncStrainInfoAdapter
from apiadapters.straininfo.straininfo import StrainRef

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
    """A strain attached to a document must survive `sync_doc_db`.

    Regression test: `store_strains` is a coroutine and was being called
    without `await`, so the buffer was never updated and no strain was ever
    written to the store. Driven with `asyncio.run`, like `main()` itself,
    rather than `pytest.mark.asyncio`, which needs a plugin not present in
    every environment this suite runs in.
    """
    docdb_path = tmp_path / "documents.json"
    monkeypatch.setitem(bref.config, "documents", docdb_path)
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
        assert docdb.strains.contains(doc_id=STRAIN.id)
