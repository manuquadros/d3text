"""Reading the raw document corpus.

The two precompute commands are the only readers of the csv/json corpus, and
they each got half of it right, which is worse than either being wrong: the
encodings path stripped the XML tags but turned a missing abstract into the
literal string ``"nan"``; the embeddings path handled the missing abstract but
fed raw JATS markup to the transformer. `document_text` is now the single place
both decisions are made, so the two stages cannot describe the same document
differently again.

Deliberately a leaf: importing `d3text.data` drags in the whole BRENDA stack
(`brenda_references` -> `d3types` -> `lpsn_interface`, and their database and
API dependencies) to read csv and json rows, which need none of it. The
precompute commands are the only d3text commands that do not already pay that
import cost; reading the corpus must not be what makes them.
"""

import logging
import pathlib
from collections.abc import Iterator

import nltk.redos
import polars as pl
import xmlparser

logger = logging.getLogger(__name__)

# nltk 3.10 routes every tokenizer pattern through a wall-clock ReDoS guard,
# five seconds by default. `remove_tags` matches `<\w[^<>]*>|</[^<>]*>` — two
# alternatives over a character class that excludes its own delimiters, so it
# has no nested quantifier to backtrack through and cannot be made pathological
# by its input. Five seconds is calibrated for a *hostile* pattern; ours is
# ours, and the corpus is on disk beside it. What the guard actually measures
# here is how slow the machine is: an 88 KB article strips in 0.005 s on one
# box and tripped the limit on another, 70 minutes into an embedding pass.
#
# Raised rather than removed. A bound that a real document cannot reach still
# ends a genuinely runaway match, and losing the protection process-wide is not
# this module's decision to make for whoever imports it.
_TAG_STRIPPING_TIMEOUT = 300.0
nltk.redos.DEFAULT_TIMEOUT = _TAG_STRIPPING_TIMEOUT

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
                try:
                    text = document_text(abstract, fulltext)
                except TimeoutError:
                    # One row must not cost the pass. Both commands run for
                    # hours over the whole corpus, and a document that cannot
                    # be stripped is a document the consumer already knows how
                    # to be without: it is simply absent from the output, which
                    # is what a document the corpus never had looks like too.
                    logger.warning(
                        "stripping the markup of document %s exceeded %.0fs; "
                        "dropping it from %s",
                        pubmed_id,
                        _TAG_STRIPPING_TIMEOUT,
                        path,
                    )
                    continue
                yield pubmed_id, text

    return total, rows()
