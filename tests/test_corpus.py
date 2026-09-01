"""Reading the corpus: what text actually reaches the tokenizer.

Both precompute commands turn a row into one string and each used to do it its
own way, neither disagreement visible from the output. A missing abstract is
`nan` or `None`, never `""`, and `str(nan)` is the *truthy* `"nan"`, so `str(
row.abstract) or ""` prepended the word "nan" to 36 of the test split's 1210
documents; and both halves arrive as JATS markup, which one path stripped and
the other fed to the transformer as-is.
"""

import pathlib
import subprocess
import sys

import nltk.redos
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


def write_split_csv(path: pathlib.Path, rows: list[dict[str, object]]):
    """A split csv with its four entity columns.

    Written through polars rather than by hand because the cells are Python
    `repr`s full of commas and quotes, and the quoting is not what is under
    test.
    """
    pl.DataFrame(
        rows,
        schema={
            "pubmed_id": pl.Int64,
            "abstract": pl.Utf8,
            "fulltext": pl.Utf8,
            "enzymes": pl.Utf8,
            "bacteria": pl.Utf8,
            "strains": pl.Utf8,
            "other_organisms": pl.Utf8,
        },
    ).write_csv(path)
    return path


_ANNOTATED = {
    "pubmed_id": 12964952,
    "abstract": "an abstract",
    "fulltext": "a body",
    "enzymes": "[34496]",
    "bacteria": "{}",
    "strains": "[]",
    "other_organisms": "{'2785': 'Jaculus orientalis'}",
}


def test_stream_documents_prefixes_every_gold_id_with_its_type(tmp_path):
    """The gold set has to be spelled the way the surface-form index is.

    `preprocess_labels` builds the same strings but is in the trunk: a
    labelling command that called it would pay the whole BRENDA stack to read
    four columns of a csv it is already streaming.
    """
    path = write_split_csv(tmp_path / "split.csv", [_ANNOTATED])

    total, documents = corpus.stream_documents(path, batch_size=10)
    document = next(iter(documents))

    assert total == 1
    assert document.pubmed_id == 12964952
    assert document.entity_ids == {"enz34496", "oth2785"}


def test_stream_documents_reads_both_column_shapes(tmp_path):
    """`enzymes` and `strains` are lists of IDs; `bacteria` and
    `other_organisms` are mappings from an ID to the name this document gave
    it. Only the keys are IDs, and iterating covers both."""
    path = write_split_csv(
        tmp_path / "split.csv",
        [
            _ANNOTATED
            | {
                "bacteria": "{'42': 'Escherichia coli'}",
                "strains": "[7, 8]",
            }
        ],
    )

    _, documents = corpus.stream_documents(path, batch_size=10)

    assert next(iter(documents)).entity_ids == {
        "enz34496",
        "bac42",
        "str7",
        "str8",
        "oth2785",
    }


def test_stream_documents_carries_the_text_the_encodings_were_built_from(
    tmp_path,
):
    path = write_split_csv(
        tmp_path / "split.csv",
        [_ANNOTATED | {"abstract": "<p>abs</p>", "fulltext": "<p>body</p>"}],
    )

    _, documents = corpus.stream_documents(path, batch_size=10)

    assert next(iter(documents)).text == "abs\nbody"


def test_stream_documents_reads_an_unannotated_dump_as_gold_nothing(tmp_path):
    """The PMC noise dump carries no entity columns at all, and a document with
    no gold entities is exactly what that means."""
    path = tmp_path / "dump.json"
    path.write_text(
        '{"pubmed_id": "30", "abstract": "<p>abs</p>", "body": "<p>body</p>"}\n'
    )

    total, documents = corpus.stream_documents(path, batch_size=10)
    document = next(iter(documents))

    assert total == 1
    assert document.entity_ids == frozenset()
    assert document.other_organisms == {}


def test_other_organism_names_pools_what_no_table_holds(tmp_path):
    """BRENDA's dump has no other-organisms table; the names exist only here.

    Pooled across the corpus on purpose: a document mentioning an organism it
    was not annotated with is the case the abstain target exists for, and that
    mention is only recognizable from another document's naming of it.
    """
    path = write_split_csv(
        tmp_path / "split.csv",
        [
            _ANNOTATED,
            _ANNOTATED
            | {
                "pubmed_id": 2,
                "other_organisms": "{'99': 'Mus musculus'}",
            },
        ],
    )

    pooled = list(corpus.other_organism_names(path, batch_size=1))

    assert pooled == [{"2785": "Jaculus orientalis"}, {"99": "Mus musculus"}]


def test_other_organism_names_yields_nothing_without_the_column(tmp_path):
    path = write_csv(tmp_path / "plain.csv", "0,10,abstract,body\n")

    assert list(corpus.other_organism_names(path, batch_size=10)) == []


def test_the_corpus_reader_does_not_import_the_data_layer(tmp_path):
    """Reading csv and json rows must not cost the whole BRENDA stack.

    The two precompute commands are the only d3text commands that do not
    already pay that import cost. Checked in a subprocess: the suite as a whole
    imports `d3text.data`, so an in-process check would pass no matter what.
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

    assert result.stdout.strip().endswith(
        "False"
    ), "d3text.corpus pulled in the BRENDA data layer"


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


def test_stream_rows_parses_the_file_only_once(tmp_path, monkeypatch):
    """Reading the corpus in slices must not cost one scan per slice.

    A `.slice(start, n).collect()` per batch re-parses the file from the top
    every time, since CSV and NDJSON have no random access.
    """
    calls = []
    original_collect = pl.LazyFrame.collect
    monkeypatch.setattr(
        pl.LazyFrame,
        "collect",
        lambda self, *args, **kwargs: (
            calls.append(1) or original_collect(self, *args, **kwargs)
        ),
    )
    path = write_csv(
        tmp_path / "many.csv",
        "".join(f"{i},{i},abstract {i},body {i}\n" for i in range(7)),
    )

    total, rows = corpus.stream_rows(path, batch_size=3)
    list(rows)

    assert total == 7
    assert len(calls) == 1


_BLANK = [
    pytest.param("   \n  \n", id="indentation"),
    pytest.param(
        '<jats:body xmlns:jats="https://example.org">\n  <jats:sec>\n  '
        "</jats:sec>\n</jats:body>",
        id="markup-around-nothing",
    ),
]


def collect_warnings(monkeypatch) -> list[str]:
    """The module logger's warnings, rendered.

    Captured off `corpus.logger` directly rather than through `caplog`: any
    test that runs a command's `main()` leaves `propagate = False` on the
    `d3text` logger, after which the assertion would pass or fail with the test
    order.
    """
    warnings: list[str] = []
    monkeypatch.setattr(
        corpus.logger, "warning", lambda msg, *args: warnings.append(msg % args)
    )
    return warnings


def test_tag_stripping_outlives_the_redos_budget_on_a_stalled_host(
    monkeypatch,
):
    """A ReDoS budget that runs out here is measuring the host, not the input.

    `remove_tags`' pattern is a hardcoded constant and strips linearly, yet
    write-back stalls during an 80 GiB pass exhausted five seconds on a match
    costing five milliseconds of CPU. A near-zero budget simulates the stall.
    """
    monkeypatch.setattr(nltk.redos, "DEFAULT_TIMEOUT", 1e-9)
    markup = ("<p>" + "word " * 2000 + "</p>") * 50

    text = corpus.document_text(None, markup)

    assert "word" in text
    assert "<" not in text


def test_the_redos_exemption_does_not_outlive_the_call(monkeypatch):
    """The guard exists for nltk's caller-supplied-pattern sinks — the tagger,
    the chunk rules, tgrep. Exempting the one trusted pattern must not lower
    their bound: leaving the global raised at import is what 23f1503
    reverted."""
    monkeypatch.setattr(nltk.redos, "DEFAULT_TIMEOUT", 7.5)

    corpus.document_text("<p>abs</p>", "<p>body</p>")

    assert nltk.redos.DEFAULT_TIMEOUT == 7.5


def test_the_redos_exemption_is_restored_when_stripping_raises(monkeypatch):
    monkeypatch.setattr(nltk.redos, "DEFAULT_TIMEOUT", 7.5)
    monkeypatch.setattr(
        corpus.xmlparser,
        "remove_tags",
        lambda _markup: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(RuntimeError):
        corpus.document_text("<p>abs</p>", None)

    assert nltk.redos.DEFAULT_TIMEOUT == 7.5


def timing_out_on(marker: str):
    """A `document_text` whose guard fires on the rows carrying `marker`."""
    real = corpus.document_text

    def strip(abstract, fulltext):
        if marker in str(abstract):
            raise TimeoutError(
                "regular-expression match exceeded its 5.0s time limit"
            )
        return real(abstract, fulltext)

    return strip


def test_a_row_the_guard_still_kills_is_dropped_and_tallied(
    tmp_path, monkeypatch
):
    """Both precompute commands read the whole corpus in one multi-hour pass.
    A row whose stripping the guard abandons has to cost that row — dropped,
    named, and counted where the consumer can read it — not every row after
    it. The count matters because `stream_rows` returns a total it then does
    not yield: without it the stream silently shrinks."""
    warnings = collect_warnings(monkeypatch)
    monkeypatch.setattr(corpus, "document_text", timing_out_on("pathological"))
    path = write_csv(
        tmp_path / "split.csv",
        "0,10,<p>first</p>,<p>body</p>\n"
        "1,20,<p>pathological</p>,<p>body</p>\n"
        "2,30,<p>third</p>,<p>body</p>\n",
    )

    total, rows = corpus.stream_rows(path, batch_size=10)
    collected = list(rows)

    assert total == 3
    assert [pubmed_id for pubmed_id, _ in collected] == [10, 30]
    assert rows.dropped == 1
    assert any("20" in warning for warning in warnings)
    assert any("1 of 3" in warning for warning in warnings)


def test_a_document_the_guard_still_kills_is_dropped_and_tallied(
    tmp_path, monkeypatch
):
    warnings = collect_warnings(monkeypatch)
    monkeypatch.setattr(corpus, "document_text", timing_out_on("pathological"))
    path = write_split_csv(
        tmp_path / "split.csv",
        [
            _ANNOTATED,
            _ANNOTATED | {"pubmed_id": 2, "abstract": "<p>pathological</p>"},
        ],
    )

    total, documents = corpus.stream_documents(path, batch_size=10)
    collected = list(documents)

    assert total == 2
    assert [document.pubmed_id for document in collected] == [12964952]
    assert documents.dropped == 1
    assert any("1 of 2" in warning for warning in warnings)


def test_only_the_guards_timeout_is_dropped(tmp_path, monkeypatch):
    """`TimeoutError` is the one exception the drop may swallow, and only
    around the stripping call: anything else raising mid-stream is a bug the
    pass must surface, not a row to be quietly short."""
    monkeypatch.setattr(
        corpus,
        "document_text",
        lambda _abstract, _fulltext: (_ for _ in ()).throw(
            RuntimeError("boom")
        ),
    )
    path = write_csv(tmp_path / "split.csv", "0,10,<p>a</p>,<p>b</p>\n")

    _, rows = corpus.stream_rows(path, batch_size=10)

    with pytest.raises(RuntimeError):
        list(rows)


@pytest.mark.parametrize("blank", _BLANK)
def test_a_whitespace_only_document_is_empty(blank):
    """Markup wrapping nothing strips to indentation, which is *truthy*.

    So `if not text` waved it through and it was tokenized into a window
    holding `[CLS]` and `[SEP]` alone — which reaches the pooling as a
    zero-length token dimension, where the four modes return `-inf`, `NaN`, an
    `IndexError` and a `ValueError` respectively.
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
