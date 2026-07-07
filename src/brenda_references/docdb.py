"""Module providing queries into the document database."""

from collections.abc import Mapping, MutableMapping
from types import TracebackType
from typing import Any, Iterable, Self, Set, cast

from apiadapters.ncbi.parser import is_scanned
from d3types import Document, Strain
from lpsn_interface import lpsn_id, lpsn_parent, lpsn_synonyms
from tinydb import Query, TinyDB, where
from tinydb.middlewares import CachingMiddleware
from tinydb.storages import JSONStorage, MemoryStorage
from tinydb.table import Document as TDocument

from brenda_references.config import config


class BrendaDocDB:
    def __init__(self, path: str | None = None, storage: str = "json") -> None:
        self._path = path or config["documents"]

        if storage == "memory":
            self._db: TinyDB = TinyDB(storage=CachingMiddleware(MemoryStorage))
        else:
            self._db = TinyDB(
                self._path, storage=CachingMiddleware(JSONStorage)
            )

        self.documents = self._db.table("documents")
        self.bacteria = self._db.table("bacteria")
        self.strains = self._db.table("strains")

    def __enter__(self) -> Self:
        self._db.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._db.__exit__()

    def as_dict(self) -> dict[str, dict[str, Any]] | None:
        return self._db.storage.read()

    def fulltext_articles(self) -> tuple[TDocument, ...]:
        """Retrieve documents from the database with full text available."""
        fulltext = self._db.table("documents").search(
            where("fulltext").exists() & (where("fulltext") != "")
        )
        return tuple(
            filter(lambda doc: not is_scanned(doc["fulltext"]), fulltext)
        )

    def insert(self, table: str, record: Mapping) -> int | None:
        """Insert `record` in `table` and return its id."""
        try:
            return self._db.table(table).insert(record)
        except ValueError:
            return None

    def get_record(self, table: str, doc_id: int) -> TDocument | None:
        """Return doc at `doc_id` on `table`."""
        return self._db.table(table).get(doc_id=doc_id)

    def get_reference(self, doc_id: int) -> TDocument:
        """Return the reference at `doc_id`."""
        return self.documents.get(doc_id=doc_id)

    @property
    def references(self) -> tuple[TDocument, ...]:
        """Retrieve all documents from the database."""
        return tuple(self._db.table("documents"))

    def get_strain(self, _id: str | int) -> TDocument | None:
        """Retrieve strain record from the document database."""
        return cast(TDocument, self._db.table("strains").get(doc_id=int(_id)))

    def get_bacteria(self, _id: str | int) -> TDocument | None:
        """Retrieve bacteria record from `self`"""
        return cast(TDocument, self._db.table("bacteria").get(doc_id=int(_id)))

    def bacteria_by_name(self, query: str) -> TDocument | None:
        """Return a bacteria record with `query` in its designations"""
        table = self._db.table("bacteria")
        match = table.get(
            (where("organism") == query)
            | (where("synonyms").test(lambda syns: query in syns))
        )

        if match is not None:
            return cast(TDocument, match)

        return None

    def strain_by_designation(self, query: str) -> TDocument | None:
        """Return a strain record with `query` among its designations."""
        match = self.strains.get(
            (Query().taxon.name == query)
            | (Query().cultures.any(Query().strain_number == query))
            | (Query().designations.test(lambda names: query in names))
        )

        if match is not None:
            return cast(TDocument, match)

        return None

    def update_record(
        self, table: str, fields: dict[str, Any], doc_id: int
    ) -> None:
        """Update `doc_id` according to `fields`."""
        tbl = self._db.table(table)
        tbl.update(fields=fields, doc_ids=[doc_id])

    def __add_bacteria_record(
        self, organism: str, synonyms: frozenset[str]
    ) -> int:
        """Store a new bacteria record and return its doc_id."""
        table = self.bacteria

        doc_id = table.insert(
            {"organism": organism, "synonyms": list(synonyms)}
        )

        return doc_id

    def add_synonyms(
        self, table: str, doc_id: int, synonyms: Iterable[str]
    ) -> None:
        """Add `synonyms` to the synonym set of the `doc_id` record."""

        def add(synset_field: str, synonyms: Iterable[str]):
            def transform(doc: MutableMapping):
                synset = set(doc[synset_field])
                synset.update(synonyms)
                doc[synset_field] = list(synset)

            return transform

        synset_field = {"bacteria": "synonyms", "strains": "designations"}

        getattr(self, table).update(
            add(synset_field[table], synonyms),
            doc_ids=[doc_id],
        )

    def add_bac_synonyms(self, doc_id: int, synonyms: Set[str]) -> None:
        """Add `synonyms` to the synonym set of the `doc_id` record."""
        self.add_synonyms(table="bacteria", doc_id=doc_id, synonyms=synonyms)

    def add_strain_synonyms(self, doc_id: int, synonyms: Set[str]) -> None:
        self.add_synonyms(table="strains", doc_id=doc_id, synonyms=synonyms)

    def insert_bacteria_record(self, query: str) -> int:
        """Return the id of a bacteria record if it exists or of a new one."""
        match = self.bacteria_by_name(query)

        if isinstance(match, TDocument):
            return match.doc_id

        _lpsn_id = lpsn_id(query)
        synonyms: frozenset[str] = frozenset()

        if _lpsn_id:
            _lpsn_parent = lpsn_parent(_lpsn_id)

            if _lpsn_parent:
                parent_id, organism = _lpsn_parent

                # Check if there is already a record for the parent LPSN.
                # In that case, we just add `query` to its synonym set.
                parent_record = self.bacteria_by_name(organism)
                if parent_record is not None:
                    self.add_bac_synonyms(
                        doc_id=parent_record.doc_id, synonyms={query}
                    )
                    return parent_record.doc_id

                synonyms = (
                    lpsn_synonyms(_lpsn_id) | lpsn_synonyms(parent_id) | {query}
                )
            else:
                return self.__add_bacteria_record(
                    organism=query, synonyms=lpsn_synonyms(_lpsn_id)
                )

        return self.__add_bacteria_record(organism=query, synonyms=synonyms)
