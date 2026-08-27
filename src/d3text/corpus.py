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
    """The cell's text, or ``""`` where the cell is empty.

    A missing cell is `None` from polars and ``float("nan")`` from pandas, so
    anything that is not a `str` is a missing value. The obvious spelling —
    ``str(value) or ""`` — is the bug this replaces: ``str(nan)`` is ``"nan"``,
    a *truthy* string, so the fallback never fired and the word "nan" was
    tokenized into the document.
    """
    return value if isinstance(value, str) else ""


def _remove_tags(markup: str) -> str:
    """``xmlparser.remove_tags``, exempted from nltk's ReDoS guard.

    nltk 3.10 runs every tokenizer pattern under a *wall-clock* timeout
    (``nltk.redos``, five seconds by default, read off the module global at
    match time). ``remove_tags``' pattern is `xmlparser`'s own hardcoded
    constant and strips linearly — no input reaches the bound by matching —
    so a guard that fires here is timing the host, and a few seconds of
    write-back stall during an 80 GiB precompute pass is enough to end a
    multi-hour run on a match costing five milliseconds of CPU.

    ``timeout=None`` is nltk's documented exemption for a trusted pattern.
    Granted per call and restored on the way out: importing this module
    changes nothing, and every caller-supplied pattern elsewhere — the
    tagger, the chunk rules, `tgrep`, which are what the five seconds exist
    for — keeps its guard. Assigning the global at import is what got
    0062e89 reverted (23f1503).

    The lock is for the restore, not the match: two overlapping calls would
    interleave their save/restore and could leave the exemption behind for
    the whole process. No caller strips from more than one thread today, so
    the serialization costs nothing.
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

    Both halves arrive as JATS markup, and the tags are not part of the
    language the model is meant to read.

    Whitespace is not content. A body that is markup wrapping nothing but
    newlines strips to a *truthy* string of indentation, so every caller's
    ``if not text`` check waves it through and the tokenizer returns a window
    holding ``[CLS]`` and ``[SEP]`` and no token of the document at all. Same
    trap as ``str(nan)`` above, and answered in the same place.
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

    Lazy on purpose: the corpus is ~1 GB of json, and every command consumes
    it one document at a time, so reading it eagerly buys nothing and costs
    the whole file in resident memory. `collect_batches` is what keeps it
    lazy without also re-scanning: unlike `lazy.slice(start,
    batch_size).collect()`, which parses the file from the top for every
    batch it produces (CSV and NDJSON have no random access, so a scan
    cannot seek to `start`), the streaming engine here parses the source
    once and hands back one chunk of rows at a time.
    """
    for chunk in lazy.collect_batches(chunk_size=batch_size):
        yield from chunk.iter_rows()


class CorpusStream[RowT](Iterator[RowT]):
    """A corpus iterator that counts the rows it dropped.

    The streaming functions below return a row count and then an iterator
    that may drop a row (one whose tag stripping the ReDoS guard abandoned),
    so the count and the stream can disagree. `dropped` is the difference,
    readable at any point — without it the stream silently shrinks and the
    only signal is one warning per drop in the middle of a multi-hour pass's
    log.
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
    """The row's text, or ``None`` once the row is tallied as dropped.

    One row must not cost the pass: both precompute commands read the whole
    corpus in a single multi-hour run, and a document that cannot be stripped
    is one the consumer already knows how to be without — absent, which is
    what a document the corpus never had looks like too.

    The guard's exception is the *builtin* :class:`TimeoutError` — nltk
    subclasses nothing — and `TimeoutError` is an `OSError`. The catch is
    therefore kept around the one call, which does no I/O, so a future I/O
    timeout raised anywhere else in the stream still ends it loudly instead
    of shrinking it silently.
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
    """The corpus's row count, and its ``(pubmed_id, text)`` pairs in slices.

    The count is the file's; the stream may fall short of it by the rows it
    dropped, which it counts (see `CorpusStream`).
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
) -> tuple[int, CorpusStream[CorpusDocument]]:
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

    A separate pass because building the surface-form index needs these before
    any document can be labelled, and `stream_documents` would strip a
    gigabyte of JATS markup to hand back a column that is already in the file.
    A file without the column yields nothing.
    """
    lazy = _scan(path)
    if _OTHER_ORGANISMS not in lazy.collect_schema().names():
        return
    lazy = lazy.select(pl.col(_OTHER_ORGANISMS))
    for (cell,) in _slices(lazy, batch_size):
        yield _cell_names(cell)
