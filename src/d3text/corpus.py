"""Reading the raw document corpus.

The precompute commands are the only readers of the csv/json corpus, and the
first two each got half of it right, which is worse than either being wrong: the
encodings path stripped the XML tags but turned a missing abstract into the
literal string ``"nan"``; the embeddings path handled the missing abstract but
fed raw JATS markup to the transformer. `document_text` is now the single place
both decisions are made, so the two stages cannot describe the same document
differently again.

`stream_documents` reads a row's **gold entity set** off the same file, which
is why the split frames' entity columns are parsed here rather than borrowed
from `brenda_references.preprocess_labels`: that function is in the trunk, and
a labelling command that reached it would pay the whole BRENDA stack to read
four columns of a csv it is already streaming. What it does is small — parse
the cell, prefix each numeric ID with its type's tag — and the tags come off
the schema, so the two spellings cannot drift.

Deliberately a leaf: importing `d3text.data` drags in the whole BRENDA stack
(`brenda_references` -> `d3types` -> `lpsn_interface`, and their database and
API dependencies) to read csv and json rows, which need none of it. The
precompute commands are the only d3text commands that do not already pay that
import cost; reading the corpus must not be what makes them.
"""

import ast
import dataclasses
import pathlib
from collections.abc import Iterator, Mapping
from typing import Any

import polars as pl
import xmlparser

from d3text.schema import BRENDA_SCHEMA, Schema

# The corpus disagrees with itself about the type of a pubmed id: the csv
# splits store it as an integer, the PMC ndjson dump as a string. Both readers
# below preserve whatever they were given; callers stringify it.
PubmedId = int | str

_SEPARATOR = "\n"

# The one entity type whose names live nowhere but this column; see
# `CorpusDocument` and `surface_forms.other_organism_forms`.
_OTHER_ORGANISMS = "other_organisms"


def _present(value: str | float | None) -> str:
    """The cell's text, or ``""`` where the cell is empty.

    A missing cell is `None` from polars and ``float("nan")`` from pandas, so
    anything that is not a `str` is a missing value. The obvious spelling —
    ``str(value) or ""`` — is the bug this replaces: ``str(nan)`` is ``"nan"``,
    a *truthy* string, so the fallback never fired and the word "nan" was
    tokenized into the document.
    """
    return value if isinstance(value, str) else ""


def document_text(
    abstract: str | float | None, fulltext: str | float | None
) -> str:
    """One document's text: its abstract, then its body, with XML tags removed.

    Both halves arrive as JATS markup, and the tags are not part of the
    language the model is meant to read.

    Whitespace is not content. A body that is markup wrapping nothing but
    newlines strips to a *truthy* string of indentation, so every caller's
    ``if not text`` check waves it through and the tokenizer returns a window
    holding ``[CLS]`` and ``[SEP]`` and no token of the document at all. Same
    trap as ``str(nan)`` above, and answered in the same place.
    """
    parts = [part for part in (_present(abstract), _present(fulltext)) if part]
    text = xmlparser.remove_tags(_SEPARATOR.join(parts))
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


def _slices(
    lazy: pl.LazyFrame, total: int, batch_size: int
) -> Iterator[tuple[Any, ...]]:
    """`lazy`'s rows, collected `batch_size` at a time.

    Lazy on purpose: the corpus is ~1 GB of json, and every command consumes it
    one document at a time, so reading it eagerly buys nothing and costs the
    whole file in resident memory.
    """
    for start in range(0, total, batch_size):
        yield from lazy.slice(start, batch_size).collect().iter_rows()


def stream_rows(
    path: pathlib.Path, batch_size: int
) -> tuple[int, Iterator[tuple[PubmedId, str]]]:
    """The corpus's row count, and its ``(pubmed_id, text)`` pairs in slices."""
    lazy = _scan(path).select(
        pl.col("pubmed_id"), pl.col("abstract"), pl.col("fulltext")
    )
    total: int = lazy.select(pl.len()).collect().item()

    def rows() -> Iterator[tuple[PubmedId, str]]:
        for row in _slices(lazy, total, batch_size):
            pubmed_id, abstract, fulltext = row
            yield pubmed_id, document_text(abstract, fulltext)

    return total, rows()


@dataclasses.dataclass(frozen=True, slots=True)
class CorpusDocument:
    """One corpus row: its text, and what it was annotated with.

    `other_organisms` is carried separately from `entity_ids` because it is the
    one namespace whose *names* exist nowhere else. The BRENDA dump has no
    other-organisms table — each document simply spells the names inline — so a
    surface-form index over that namespace can only be built by pooling this
    column across the whole corpus.
    """

    pubmed_id: PubmedId
    text: str
    entity_ids: frozenset[str]
    other_organisms: Mapping[str, str]


def _entity_columns(
    lazy: pl.LazyFrame, schema: Schema
) -> tuple[tuple[str, str], ...]:
    """The schema's entity columns that `lazy` actually carries.

    The PMC noise dump has none of them — it is unannotated text — and a
    document with no gold entities is exactly what that means, so a missing
    column contributes an empty set rather than raising.
    """
    present = set(lazy.collect_schema().names())
    return tuple(
        (entity_type.name, entity_type.prefix)
        for entity_type in schema.entity_types
        if entity_type.name in present
    )


def _cell(value: object) -> list[Any] | dict[str, Any] | None:
    """A split frame's entity cell, parsed.

    The splits store Python `repr`s, not JSON — ``{'2785': 'Jaculus
    orientalis'}`` — which is what `brenda_references` reads them back with.
    The element type is not knowable statically: a list holds numeric IDs, a
    mapping holds IDs against names, and the columns disagree about which.

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

    Iterating covers both shapes the columns take: ``enzymes`` and ``strains``
    are lists of numeric IDs, ``bacteria`` and ``other_organisms`` mappings
    from an ID to the name this document gave it, and iterating a mapping
    yields its keys.
    """
    parsed = _cell(value)
    if parsed is None:
        return set()
    return {f"{prefix}{int(identifier)}" for identifier in parsed}


def _cell_names(value: object) -> dict[str, str]:
    """An entity cell's ID -> name mapping, or nothing if it carries no names."""
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
) -> tuple[int, Iterator[CorpusDocument]]:
    """The corpus's row count, and its rows with their gold entity sets.

    `stream_rows` without the annotations, which the two encoding commands do
    not need and would pay four `literal_eval`s per row for.
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

    def documents() -> Iterator[CorpusDocument]:
        for row in _slices(lazy, total, batch_size):
            pubmed_id, abstract, fulltext = row[:3]
            cells = dict(zip((name for name, _ in columns), row[3:]))
            entity_ids = frozenset(
                identifier
                for name, prefix in columns
                for identifier in _cell_ids(cells[name], prefix)
            )
            yield CorpusDocument(
                pubmed_id=pubmed_id,
                text=document_text(abstract, fulltext),
                entity_ids=entity_ids,
                other_organisms=_cell_names(cells.get(_OTHER_ORGANISMS)),
            )

    return total, documents()


def other_organism_names(
    path: pathlib.Path, batch_size: int
) -> Iterator[Mapping[str, str]]:
    """Each row's inline other-organism names, and nothing else.

    A separate pass because building the surface-form index needs these before
    any document can be labelled, and `stream_documents` would strip a
    gigabyte of JATS markup to hand back a column that is already in the file.
    A file without the column yields nothing.
    """
    lazy = _scan(path)
    if _OTHER_ORGANISMS not in lazy.collect_schema().names():
        return
    lazy = lazy.select(pl.col(_OTHER_ORGANISMS))
    total: int = lazy.select(pl.len()).collect().item()
    for (cell,) in _slices(lazy, total, batch_size):
        yield _cell_names(cell)
