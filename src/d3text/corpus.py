"""Reading the raw document corpus.

`document_text` is the single place a corpus row becomes a string, so the two
precompute stages cannot describe the same document differently. Deliberately a
leaf: importing `d3text.data` would drag the whole BRENDA stack in to read csv
and json rows that need none of it. See the data page of the documentation.
"""

import ast
import dataclasses
import logging
import pathlib
import threading
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import nltk.redos
import polars as pl
import xmlparser

from d3text.schema import BRENDA_SCHEMA, Schema

logger = logging.getLogger(__name__)

_REDOS_EXEMPTION = threading.Lock()

# The corpus disagrees with itself about the type of a pubmed id: the csv
# splits store it as an integer, the PMC ndjson dump as a string. Both readers
# below preserve whatever they were given; callers stringify it.
PubmedId = int | str

_SEPARATOR = "\n"

# The one entity type whose names live nowhere but this column; see
# `CorpusDocument` and `surface_forms.other_organism_forms`.
_OTHER_ORGANISMS = "other_organisms"


def _present(value: str | float | None) -> str:
    """The cell's text, or `""` where the cell is empty.

    Anything that is not a `str` is a missing value. The obvious spelling,
    `str(value) or ""`, is the bug this replaces: `str(nan)` is the *truthy*
    string `"nan"`, so the fallback never fired.
    """
    return value if isinstance(value, str) else ""


def _remove_tags(markup: str) -> str:
    """`xmlparser.remove_tags`, exempted from nltk's ReDoS guard.

    The pattern is `xmlparser`'s own constant and strips linearly, so a guard
    that fires here is timing the host, and a write-back stall during an 80 GiB
    pass would end a multi-hour run over a five-millisecond match. The
    exemption is granted per call and restored on the way out, so every
    caller-supplied pattern elsewhere keeps its guard; the lock is for that
    restore, not for the match.
    """
    with _REDOS_EXEMPTION:
        previous = nltk.redos.DEFAULT_TIMEOUT
        nltk.redos.DEFAULT_TIMEOUT = None
        try:
            return xmlparser.remove_tags(markup)
        finally:
            nltk.redos.DEFAULT_TIMEOUT = previous


def document_text(
    abstract: str | float | None, fulltext: str | float | None
) -> str:
    """One document's text: its abstract, then its body, with XML tags removed.

    Both halves arrive as JATS markup, which is not part of the language the
    model is meant to read. Whitespace is not content either: a body wrapping
    nothing but newlines strips to a *truthy* string of indentation that every
    caller's `if not text` check would wave through.

    :param abstract: the row's abstract cell.
    :param fulltext: the row's body cell.
    :return: the joined, stripped text, empty if the row has neither.
    """
    parts = [part for part in (_present(abstract), _present(fulltext)) if part]
    text = _remove_tags(_SEPARATOR.join(parts))
    return text if text.strip() else ""


def _scan(path: pathlib.Path) -> pl.LazyFrame:
    if path.suffix == ".csv":
        return pl.scan_csv(path)
    if path.suffix == ".json":
        # Line-delimited; the PMC dump calls the body what the csv splits call
        # the fulltext.
        return pl.scan_ndjson(path).rename({"body": "fulltext"})

    msg = f"{path} has an unrecognized file format."
    raise ValueError(msg)


def _slices(lazy: pl.LazyFrame, batch_size: int) -> Iterator[tuple[Any, ...]]:
    """`lazy`'s rows, read `batch_size` at a time in a single pass.

    `collect_batches` rather than repeated `slice(...).collect()`, which would
    parse the file from the top for every batch: CSV and NDJSON have no random
    access, so a scan cannot seek.
    """
    for chunk in lazy.collect_batches(chunk_size=batch_size):
        yield from chunk.iter_rows()


class CorpusStream[RowT](Iterator[RowT]):
    """A corpus iterator that counts the rows it dropped.

    The streaming functions return a row count and then an iterator that may
    drop a row, so the two can disagree; without `dropped` the stream shrinks
    silently.
    """

    def __init__(
        self, generate: Callable[["CorpusStream[RowT]"], Iterator[RowT]]
    ) -> None:
        self.dropped = 0
        self._rows = generate(self)

    def __iter__(self) -> "CorpusStream[RowT]":
        return self

    def __next__(self) -> RowT:
        return next(self._rows)


def _text_or_drop(
    stream: CorpusStream[Any],
    pubmed_id: PubmedId,
    abstract: str | float | None,
    fulltext: str | float | None,
    path: pathlib.Path,
) -> str | None:
    """The row's text, or `None` once the row is tallied as dropped.

    One row must not cost a multi-hour pass, and a document that cannot be
    stripped is one the consumer already knows how to be without. The catch is
    kept around the one call because nltk raises the *builtin* `TimeoutError`,
    which is an `OSError`, and a real I/O timeout elsewhere must still end the
    stream loudly.
    """
    try:
        return document_text(abstract, fulltext)
    except TimeoutError:
        stream.dropped += 1
        logger.warning(
            "the ReDoS guard abandoned stripping the markup of document %s; "
            "dropping it from %s",
            pubmed_id,
            path,
        )
        return None


def _report_drops(
    stream: CorpusStream[Any], total: int, path: pathlib.Path
) -> None:
    if stream.dropped:
        logger.warning(
            "%d of %d rows of %s were dropped because their markup "
            "could not be stripped in time",
            stream.dropped,
            total,
            path,
        )


def stream_rows(
    path: pathlib.Path, batch_size: int
) -> tuple[int, CorpusStream[tuple[PubmedId, str]]]:
    """The corpus's row count, and its `(pubmed_id, text)` pairs in slices.

    :param path: the corpus file to read.
    :param batch_size: rows per slice.
    :return: the file's row count, and a stream that may fall short of it by
        the rows it dropped.
    """
    lazy = _scan(path).select(
        pl.col("pubmed_id"), pl.col("abstract"), pl.col("fulltext")
    )
    total: int = lazy.select(pl.len()).collect().item()

    def rows(
        stream: CorpusStream[tuple[PubmedId, str]],
    ) -> Iterator[tuple[PubmedId, str]]:
        for row in _slices(lazy, batch_size):
            pubmed_id, abstract, fulltext = row
            text = _text_or_drop(stream, pubmed_id, abstract, fulltext, path)
            if text is None:
                continue
            yield pubmed_id, text
        _report_drops(stream, total, path)

    return total, CorpusStream(rows)


@dataclasses.dataclass(frozen=True, slots=True)
class CorpusDocument:
    """One corpus row: its text, and what it was annotated with.

    `other_organisms` is carried separately because it is the one namespace
    whose *names* exist nowhere else — the BRENDA dump has no table for them,
    so an index over that namespace can only be built by pooling this column.
    """

    pubmed_id: PubmedId
    text: str
    entity_ids: frozenset[str]
    other_organisms: Mapping[str, str]


def _entity_columns(
    lazy: pl.LazyFrame, schema: Schema
) -> tuple[tuple[str, str], ...]:
    """The schema's entity columns that `lazy` actually carries.

    The PMC noise dump has none of them, and a document with no gold entities
    is exactly what unannotated text means, so a missing column contributes an
    empty set rather than raising.
    """
    present = set(lazy.collect_schema().names())
    return tuple(
        (entity_type.name, entity_type.prefix)
        for entity_type in schema.entity_types
        if entity_type.name in present
    )


def _cell(value: object) -> list[Any] | dict[str, Any] | None:
    """A split frame's entity cell, parsed.

    The splits store Python `repr`s, not JSON. The element type is not knowable
    statically: a list holds numeric IDs, a mapping holds IDs against names,
    and the columns disagree about which.

    :raises ValueError: if a non-empty cell parses to neither shape, which
        means the file is not a split frame.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    parsed = ast.literal_eval(value)
    if isinstance(parsed, (list, dict)):
        return parsed
    msg = f"an entity cell holds a list or a mapping; got {parsed!r}"
    raise ValueError(msg)


def _cell_ids(value: object, prefix: str) -> set[str]:
    """The prefixed IDs of an entity cell.

    Iterating covers both shapes the columns take, since iterating a mapping
    yields its keys.
    """
    parsed = _cell(value)
    if parsed is None:
        return set()
    return {f"{prefix}{int(identifier)}" for identifier in parsed}


def _cell_names(value: object) -> dict[str, str]:
    """An entity cell's ID -> name mapping, or nothing if it has no names."""
    parsed = _cell(value)
    if not isinstance(parsed, dict):
        return {}
    return {
        str(identifier): name
        for identifier, name in parsed.items()
        if isinstance(name, str) and name
    }


def stream_documents(
    path: pathlib.Path,
    batch_size: int,
    schema: Schema = BRENDA_SCHEMA,
) -> tuple[int, CorpusStream[CorpusDocument]]:
    """The corpus's row count, and its rows with their gold entity sets.

    :param path: the corpus file to read.
    :param batch_size: rows per slice.
    :param schema: names the entity columns and their ID prefixes.
    :return: the file's row count, and a stream of annotated documents.
    """
    lazy = _scan(path)
    columns = _entity_columns(lazy, schema)
    lazy = lazy.select(
        pl.col("pubmed_id"),
        pl.col("abstract"),
        pl.col("fulltext"),
        *(pl.col(name) for name, _ in columns),
    )
    total: int = lazy.select(pl.len()).collect().item()

    def documents(
        stream: CorpusStream[CorpusDocument],
    ) -> Iterator[CorpusDocument]:
        for row in _slices(lazy, batch_size):
            pubmed_id, abstract, fulltext = row[:3]
            text = _text_or_drop(stream, pubmed_id, abstract, fulltext, path)
            if text is None:
                continue
            cells = dict(zip((name for name, _ in columns), row[3:]))
            entity_ids = frozenset(
                identifier
                for name, prefix in columns
                for identifier in _cell_ids(cells[name], prefix)
            )
            yield CorpusDocument(
                pubmed_id=pubmed_id,
                text=text,
                entity_ids=entity_ids,
                other_organisms=_cell_names(cells.get(_OTHER_ORGANISMS)),
            )
        _report_drops(stream, total, path)

    return total, CorpusStream(documents)


def other_organism_names(
    path: pathlib.Path, batch_size: int
) -> Iterator[Mapping[str, str]]:
    """Each row's inline other-organism names, and nothing else.

    A separate pass because the surface-form index needs these before any
    document can be labelled, and `stream_documents` would strip a gigabyte of
    JATS markup to hand back a column already in the file.

    :param path: the corpus file to read.
    :param batch_size: rows per slice.
    :return: one id -> name mapping per row; nothing if the column is absent.
    """
    lazy = _scan(path)
    if _OTHER_ORGANISMS not in lazy.collect_schema().names():
        return
    lazy = lazy.select(pl.col(_OTHER_ORGANISMS))
    for (cell,) in _slices(lazy, batch_size):
        yield _cell_names(cell)
