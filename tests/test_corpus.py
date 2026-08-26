"""Reading the corpus: what text actually reaches the tokenizer.

Both precompute commands turn a corpus row into one string, and each used to do
it its own way. The tests below pin the two decisions they disagreed about,
because neither disagreement was visible from the command's output — the run
succeeded, the artifact looked right, and the model quietly read something else.

A missing abstract is `float("nan")` out of pandas and `None` out of polars,
never an empty string, and `str(nan)` is the *truthy* string ``"nan"`` — so the
idiom ``str(row.abstract) or ""`` prepended the word "nan" to every document
that had no abstract (36 of the 1210 in the test split). And both halves of a
document arrive as JATS markup, which one path stripped and the other fed to the
transformer as-is.
"""

import pathlib
import subprocess
import sys

import polars as pl
import pytest

from d3text import corpus

_MISSING = [
    pytest.param(None, id="polars-null"),
    pytest.param(float("nan"), id="pandas-nan"),
]


@pytest.mark.parametrize("missing", _MISSING)
def test_a_missing_abstract_contributes_no_text(missing):
    """The bug this replaces: the document began with the literal word "nan"."""
    text = corpus.document_text(missing, "the body")

    assert text == "the body"
    assert "nan" not in text


@pytest.mark.parametrize("missing", _MISSING)
def test_a_missing_fulltext_contributes_no_text(missing):
    text = corpus.document_text("the abstract", missing)

    assert text == "the abstract"
    assert "nan" not in text


@pytest.mark.parametrize("missing", _MISSING)
def test_a_document_with_neither_half_is_empty(missing):
    """Must be falsy: it is how the commands detect and report an empty
    document, and under `str(nan)` that check could never fire."""
    assert corpus.document_text(missing, missing) == ""


def test_both_halves_are_separated():
    """Without a separator the abstract's last word and the body's first word
    are tokenized as one."""
    assert corpus.document_text("abstract", "body") == "abstract\nbody"


def test_xml_tags_are_stripped_from_both_halves():
    """The corpus is JATS markup. The embeddings path skipped this and fed
    `<jats:body xmlns:jats="...">` to the transformer as if it were prose."""
    text = corpus.document_text(
        '<jats:abstract xmlns:jats="https://example.org"><jats:p>Purpose'
        "</jats:p></jats:abstract>",
        "<jats:body><jats:sec>Method</jats:sec></jats:body>",
    )

    assert text == "Purpose\nMethod"
    assert "<" not in text
    assert "jats" not in text


def write_csv(path: pathlib.Path, rows: str) -> pathlib.Path:
    """The csv split layout: a leading unnamed index column, never selected."""
    path.write_text(f",pubmed_id,abstract,fulltext\n{rows}")
    return path


def test_stream_rows_reads_the_csv_splits(tmp_path):
    path = write_csv(
        tmp_path / "split.csv",
        "0,10,<p>first abstract</p>,<p>first body</p>\n"
        "1,20,,<p>second body</p>\n",
    )

    total, rows = corpus.stream_rows(path, batch_size=10)

    assert total == 2
    assert list(rows) == [
        (10, "first abstract\nfirst body"),
        (20, "second body"),
    ]


def test_stream_rows_reads_the_pmc_ndjson_dump(tmp_path):
    """The dump calls the body `body`, and stores the pubmed id as a string
    where the csv splits store an integer. Both are carried through as-is; the
    callers stringify."""
    path = tmp_path / "dump.json"
    path.write_text(
        '{"pubmed_id": "30", "abstract": "<p>abs</p>", "body": "<p>body</p>"}\n'
    )

    total, rows = corpus.stream_rows(path, batch_size=10)

    assert total == 1
    assert list(rows) == [("30", "abs\nbody")]


def test_stream_rows_yields_every_row_across_slice_boundaries(tmp_path):
    """The corpus is read in slices to keep memory flat, so a document must not
    be dropped or repeated where one slice ends and the next begins."""
    path = write_csv(
        tmp_path / "many.csv",
        "".join(f"{i},{i},abstract {i},body {i}\n" for i in range(7)),
    )

    total, rows = corpus.stream_rows(path, batch_size=3)
    streamed = list(rows)

    assert total == 7
    assert [pubmed_id for pubmed_id, _ in streamed] == list(range(7))
    assert streamed[6] == (6, "abstract 6\nbody 6")


def test_stream_rows_carries_the_missing_abstract_through_to_the_text(tmp_path):
    """The end-to-end version of the unit test above: an empty csv cell must
    reach the tokenizer as nothing at all, not as the word "nan"."""
    path = write_csv(tmp_path / "gap.csv", "0,40,,only a body\n")

    _, rows = corpus.stream_rows(path, batch_size=10)

    assert list(rows) == [(40, "only a body")]


def test_an_unrecognized_file_format_is_rejected(tmp_path):
    path = tmp_path / "corpus.parquet"
    path.touch()

    with pytest.raises(ValueError, match="unrecognized file format"):
        corpus.stream_rows(path, batch_size=10)


def test_the_corpus_reader_does_not_import_the_data_layer(tmp_path):
    """`d3text.data` drags in the whole BRENDA stack — `brenda_references`,
    `d3types`, `lpsn_interface` and their database and API dependencies — to
    read csv and json rows, which need none of it.

    The two precompute commands are the only d3text commands that do not
    already pay that import cost. Reading the corpus must not be what makes
    them.

    Checked in a subprocess: the suite as a whole imports `d3text.data`, so an
    in-process check would pass no matter what this module pulls in.
    """
    probe = (
        "import sys; import d3text.corpus; "
        "print(any(m.startswith(('d3text.data', 'brenda_references', "
        "'lpsn_interface')) for m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=True,
    )

    assert result.stdout.strip().endswith("False"), (
        "d3text.corpus pulled in the BRENDA data layer"
    )


def test_the_corpus_is_not_read_eagerly(tmp_path, monkeypatch):
    """`stream_rows` must scan, not read: the json dump is ~1 GB and both
    commands consume it a document at a time."""
    monkeypatch.setattr(
        pl,
        "read_csv",
        lambda *_args, **_kwargs: pytest.fail("the corpus was read eagerly"),
    )
    path = write_csv(tmp_path / "lazy.csv", "0,50,abstract,body\n")

    _, rows = corpus.stream_rows(path, batch_size=10)

    assert list(rows) == [(50, "abstract\nbody")]


_BLANK = [
    pytest.param("   \n  \n", id="indentation"),
    pytest.param(
        '<jats:body xmlns:jats="https://example.org">\n  <jats:sec>\n  '
        "</jats:sec>\n</jats:body>",
        id="markup-around-nothing",
    ),
]


@pytest.mark.parametrize("blank", _BLANK)
def test_a_whitespace_only_document_is_empty(blank):
    """Markup wrapping nothing strips to indentation, which is *truthy*.

    The commands detect an empty document with `if not text`, so a string of
    newlines passed the check and was tokenized into a window holding `[CLS]`
    and `[SEP]` and no token of the document. That document then reaches the
    pooling as a zero-length token dimension, where the four poolings return
    `-inf`, `NaN`, an `IndexError` and a `ValueError` respectively.
    """
    assert corpus.document_text(None, blank) == ""
    assert corpus.document_text(blank, None) == ""
    assert corpus.document_text(blank, blank) == ""


@pytest.mark.parametrize("blank", _BLANK)
def test_a_whitespace_only_half_does_not_empty_the_document(blank):
    """Only a document with no text at all is empty: the other half stands."""
    assert "Purpose" in corpus.document_text("<p>Purpose</p>", blank)
    assert "Purpose" in corpus.document_text(blank, "<p>Purpose</p>")


def test_a_whitespace_only_row_reaches_the_command_as_empty(tmp_path):
    """End to end: `precompute-encodings` warns on `not text`, so the reader is
    what decides whether a content-free row is reported or silently encoded."""
    path = write_csv(tmp_path / "blank.csv", '0,60,,"<p>  </p>"\n')

    _, rows = corpus.stream_rows(path, batch_size=10)

    assert list(rows) == [(60, "")]
