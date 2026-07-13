"""Reading the raw document corpus.

The two precompute commands are the only readers of the csv/json corpus, and
they each got half of it right, which is worse than either being wrong: the
encodings path stripped the XML tags but turned a missing abstract into the
literal string ``"nan"``; the embeddings path handled the missing abstract but
fed raw JATS markup to the transformer. `document_text` is now the single place
both decisions are made, so the two stages cannot describe the same document
differently again.

Deliberately a leaf: importing `d3text.data` would pull in `brenda_references`
-> `lpsn_interface`, which attaches a log handler to a *relative* path while
being imported. The precompute commands are the only d3text commands that do
not already pay that cost, and so the only ones that still run in a read-only
working directory.
"""

import pathlib
from collections.abc import Iterator

import polars as pl
import xmlparser

# The corpus disagrees with itself about the type of a pubmed id: the csv
# splits store it as an integer, the PMC ndjson dump as a string. Both readers
# below preserve whatever they were given; callers stringify it.
PubmedId = int | str

_SEPARATOR = "\n"


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
    """
    parts = [part for part in (_present(abstract), _present(fulltext)) if part]
    return xmlparser.remove_tags(_SEPARATOR.join(parts))


def _scan(path: pathlib.Path) -> pl.LazyFrame:
    if path.suffix == ".csv":
        return pl.scan_csv(path)
    if path.suffix == ".json":
        # Line-delimited; the PMC dump calls the body what the csv splits call
        # the fulltext.
        return pl.scan_ndjson(path).rename({"body": "fulltext"})

    msg = f"{path} has an unrecognized file format."
    raise ValueError(msg)


def stream_rows(
    path: pathlib.Path, batch_size: int
) -> tuple[int, Iterator[tuple[PubmedId, str]]]:
    """The corpus's row count, and its ``(pubmed_id, text)`` pairs in slices.

    Lazy on purpose: the corpus is ~1 GB of json, and both commands consume it
    one document at a time, so reading it eagerly buys nothing and costs the
    whole file in resident memory.
    """
    lazy = _scan(path).select(
        pl.col("pubmed_id"), pl.col("abstract"), pl.col("fulltext")
    )
    total: int = lazy.select(pl.len()).collect().item()

    def rows() -> Iterator[tuple[PubmedId, str]]:
        for start in range(0, total, batch_size):
            frame = lazy.slice(start, batch_size).collect()
            for pubmed_id, abstract, fulltext in frame.iter_rows():
                yield pubmed_id, document_text(abstract, fulltext)

    return total, rows()
