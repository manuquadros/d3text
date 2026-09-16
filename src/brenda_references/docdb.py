"""Module providing queries into the document database."""

import logging
from collections.abc import Callable, Iterable, Mapping, MutableMapping
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self

from apiadapters.ncbi.parser import is_scanned
from d3types import Document, Strain
from lpsn_interface import lpsn_id, lpsn_parent, lpsn_synonyms
from tinydb import Query, TinyDB, where
from tinydb.middlewares import CachingMiddleware
from tinydb.storages import JSONStorage, MemoryStorage
from tinydb.table import Document as TDocument
from tinydb.table import Table

from brenda_references.config import config

logger = logging.getLogger(__name__)


class BrendaDocDB:
    """Query and update the TinyDB document database of BRENDA references."""

    def __init__(
        self,
        path: str | None = None,
        storage: Literal["json", "memory"] = "json",
        create: bool = False,
    ) -> None:
        """Open the JSON document database.

        :param path: Database path; defaults to `config["documents"]`.
        :param storage: `"json"` for the on-disk TinyDB, or `"memory"`.
        :param create: Allow creating `path` if it does not exist yet.
        :raises FileNotFoundError: `path` does not exist and `create` is
            false — `JSONStorage` otherwise touches a missing path into an
            empty, valid-looking database instead of failing.
        :raises ValueError: `storage` is neither `"json"` nor `"memory"` —
            catches a typo that would otherwise silently open the on-disk
            corpus for writing.
        """
        self._path = path or config["documents"]

        if storage == "memory":
            self._db: TinyDB = TinyDB(storage=CachingMiddleware(MemoryStorage))
        elif storage == "json":
            if not create and not Path(self._path).exists():
                raise FileNotFoundError(
                    f"Document database not found at {self._path!r} "
                    "(config key 'documents'); pass create=True to create "
                    "a new one there."
                )
            self._db = TinyDB(
                self._path, storage=CachingMiddleware(JSONStorage)
            )
        else:
            raise ValueError(
                f"storage must be 'json' or 'memory', got {storage!r}"
            )

        self.documents = self._db.table("documents")
        self.bacteria = self._db.table("bacteria")
        self.strains = self._db.table("strains")

    def __enter__(self) -> Self:
        """Enter the underlying TinyDB's context, flushing on exit."""
        self._db.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close the underlying TinyDB, flushing any cached writes."""
        self._db.__exit__()

    def as_dict(self) -> dict[str, dict[str, Any]] | None:
        """Return the whole database as nested dicts, keyed by table."""
        return self._db.storage.read()

    def fulltext_articles(self) -> tuple[TDocument, ...]:
        """Retrieve documents from the database with full text available."""
        fulltext = self.documents.search(
            where("fulltext").exists() & (where("fulltext") != "")
        )

        def is_parseable(doc: TDocument) -> bool:
            text = doc["fulltext"]
            if not isinstance(text, str) or not text.startswith("<"):
                logger.warning(
                    "Skipping document %s: fulltext is not parseable XML",
                    doc.doc_id,
                )
                return False

            return True

        return tuple(
            filter(
                lambda doc: is_parseable(doc)
                and not is_scanned(doc["fulltext"]),
                fulltext,
            )
        )

    def insert(self, table: str, record: Mapping[str, Any]) -> int:
        """Insert `record` in `table` and return its id.

        :param table: Name of the table to insert into.
        :param record: The record to store.
        :return: The inserted record's doc_id.
        :raises ValueError: `record` is not a mapping, or its `doc_id`
            (when it is a TinyDB `Document`) already exists in `table` —
            raised by TinyDB and left to propagate rather than reported as
            an indistinguishable `None`.
        """
        return self._db.table(table).insert(record)

    def get_record(self, table: str, doc_id: int) -> TDocument | None:
        """Return doc at `doc_id` on `table`."""
        return self._db.table(table).get(doc_id=doc_id)

    def get_reference(self, doc_id: int) -> TDocument | None:
        """Return the reference at `doc_id`.

        :param doc_id: Document ID to look up.
        :return: The document, or `None` if `doc_id` is not present.
        """
        return self.documents.get(doc_id=doc_id)

    @property
    def references(self) -> tuple[TDocument, ...]:
        """Retrieve all documents from the database."""
        return tuple(self.documents)

    def get_strain(self, _id: str | int) -> TDocument | None:
        """Retrieve strain record from the document database."""
        return self.strains.get(doc_id=int(_id))

    def get_bacteria(self, _id: str | int) -> TDocument | None:
        """Retrieve bacteria record from `self`"""
        return self.bacteria.get(doc_id=int(_id))

    def bacteria_by_name(self, query: str) -> TDocument | None:
        """Return a bacteria record with `query` in its designations"""
        return self.bacteria.get(
            (where("organism") == query)
            | (
                where("synonyms").test(
                    lambda syns: isinstance(syns, list) and query in syns
                )
            )
        )

    def strain_by_designation(self, query: str) -> TDocument | None:
        """Return a strain record with `query` among its designations."""
        return self.strains.get(
            (Query().taxon.name == query)
            | (Query().cultures.any(Query().strain_number == query))
            | (
                Query().designations.test(
                    lambda names: isinstance(names, list) and query in names
                )
            )
        )

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
            {"organism": organism, "synonyms": sorted(synonyms)}
        )

        return doc_id

    def add_synonyms(
        self, table: str, doc_id: int, synonyms: Iterable[str]
    ) -> None:
        """Add `synonyms` to the synonym set of the `doc_id` record."""

        def add(
            synset_field: str, synonyms: Iterable[str]
        ) -> Callable[[MutableMapping[str, Any]], None]:
            def transform(doc: MutableMapping[str, Any]) -> None:
                synset = set(doc[synset_field])
                synset.update(synonyms)
                doc[synset_field] = sorted(synset)

            return transform

        tables: dict[str, tuple[Table, str]] = {
            "bacteria": (self.bacteria, "synonyms"),
            "strains": (self.strains, "designations"),
        }
        tbl, synset_field = tables[table]

        tbl.update(add(synset_field, synonyms), doc_ids=[doc_id])

    def add_bac_synonyms(self, doc_id: int, synonyms: set[str]) -> None:
        """Add `synonyms` to the synonym set of the `doc_id` record."""
        self.add_synonyms(table="bacteria", doc_id=doc_id, synonyms=synonyms)

    def add_strain_synonyms(self, doc_id: int, synonyms: set[str]) -> None:
        """Add `synonyms` to the designations of the `doc_id` strain."""
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
