"""Collecting a candidate pool: the three traps that corrupt it silently.

`xmlparser.parse_jats_article` once resolved `//front` from the *document*
root rather than from the element it was handed, so an efetch article set
parsed in place yielded the first article once per member — no error, no
warning, a sample of four distinct papers repeated seventy-five times. The
pin has moved past that, so the collapse is staged here rather than driven,
and the guard has to keep refusing its shape. esearch refuses a
`retstart` above 9,998, so an unsliced query can only return its newest page.
Neither is reachable from the network here, and neither has to be.
"""

import importlib.util
import pathlib
import random

import polars as pl
import pytest
import xmlparser
from lxml import etree

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts/collect_microbiology_sample.py"
)


def _load_collector():
    """The script as a module, without putting `scripts/` on the path."""
    spec = importlib.util.spec_from_file_location(_SCRIPT.stem, _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collector = _load_collector()


def _article(pmc_id: str, pubmed_id: str, title: str, body: str) -> str:
    return (
        "<article>"
        "<front><article-meta>"
        f'<article-id pub-id-type="pmc">{pmc_id}</article-id>'
        f'<article-id pub-id-type="pmid">{pubmed_id}</article-id>'
        f"<title-group><article-title>{title}</article-title></title-group>"
        "</article-meta></front>"
        f"<body><p>{body}</p></body>"
        "</article>"
    )


_SET = (
    "<pmc-articleset>"
    + _article("9696906", "36431429", "Norepinephrine Effects", "strains")
    + _article("9696907", "36431430", "Women Empowerment Index", "livestock")
    + "</pmc-articleset>"
).encode("utf8")


def test_each_article_of_a_set_parses_as_itself() -> None:
    """The re-rooting invariant. Parsed where they sit, both articles come
    back as the first one, and the pool becomes copies of a handful of
    papers."""
    parsed = list(collector.articles(_SET))

    assert [article.meta.title for _, article in parsed] == [
        "Norepinephrine Effects",
        "Women Empowerment Index",
    ]


def test_each_article_keeps_its_own_identifiers() -> None:
    rows = [
        collector.record(element, parse)
        for element, parse in collector.articles(_SET)
    ]

    assert [row["pubmed_id"] for row in rows] == ["36431429", "36431430"]
    assert [row["pmc_id"] for row in rows] == ["9696906", "9696907"]


def test_an_article_without_a_body_is_not_a_row() -> None:
    """A row with no text screens as a negative on no evidence at all."""
    empty = (
        "<pmc-articleset>"
        "<article><front><article-meta>"
        '<article-id pub-id-type="pmid">1</article-id>'
        "</article-meta></front><body><p>  </p></body></article>"
        "</pmc-articleset>"
    ).encode("utf8")

    element, parse = next(iter(collector.articles(empty)))

    assert collector.record(element, parse) is None


def _collapsed_rows() -> list[dict]:
    """The batch the re-rooting regression produced: every row's ids from its
    own element, its text from the first article's parse."""
    root = etree.fromstring(_SET, collector._PARSER)
    articles = list(root.iter("{*}article"))
    first = xmlparser.parse_jats_article(articles[0])
    return [collector.record(article, first) for article in articles]


def test_a_collapsed_batch_is_refused() -> None:
    """The guard has to fire on the failure it names, and the identifiers
    cannot tell it: `article_id` reads a relative xpath, so every row of a
    collapsed batch still carries its own pmid while the text is one
    article's repeated."""
    rows = _collapsed_rows()

    assert [row["pubmed_id"] for row in rows] == ["36431429", "36431430"]
    assert len({row["title"] for row in rows}) == 1

    with pytest.raises(RuntimeError, match="distinct documents"):
        collector.reject_collapse(rows)


def test_a_correctly_parsed_batch_passes() -> None:
    rows = [
        collector.record(element, parse)
        for element, parse in collector.articles(_SET)
    ]

    collector.reject_collapse(rows)


def test_no_offset_exceeds_what_esearch_accepts() -> None:
    """`retstart` above 9,998 is refused outright, WebEnv or no WebEnv, so a
    226k-hit query cannot be deep-paged and every offset must land inside the
    reachable window."""
    offsets = collector.page_offsets(226_152, 40, random.Random(0))

    assert offsets == sorted(offsets)
    assert max(offsets) <= collector.RETSTART_MAX


def test_a_query_smaller_than_one_page_is_read_from_the_top() -> None:
    assert collector.page_offsets(10, 4, random.Random(0)) == [0]


def test_the_exclusion_reads_the_ids_the_splits_carry(tmp_path) -> None:
    """The pool contaminates every split it is added to if a document BRENDA
    already curates survives into it."""
    path = tmp_path / "split.csv"
    pl.DataFrame(
        {"pmc_id": ["PMC123", "456", ""], "pubmed_id": ["9", "10", "11"]},
        schema={"pmc_id": pl.Utf8, "pubmed_id": pl.Utf8},
    ).write_csv(path)

    assert collector.column_ids([path], "pmc_id") == {"123", "456"}
    assert collector.column_ids([path], "pubmed_id") == {"9", "10", "11"}


def test_the_year_range_covers_both_ends() -> None:
    assert list(collector.year_range("2005-2007")) == [2005, 2006, 2007]
