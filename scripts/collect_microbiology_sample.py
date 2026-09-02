#!/usr/bin/env python
"""Fetch a microbiology candidate pool from PMC, in the noise pool's shape.

The candidates a hard-negative pool is screened from: PMC Open Access articles
carrying a bacteriology MeSH heading, minus every article BRENDA already
curates. Nothing here decides whether a document is a negative —
`screen_enzyme_negatives.py` does that, over the file this writes::

    python scripts/collect_microbiology_sample.py sample.json \\
        --exclude brenda_references/src/brenda_references/data/*_data.csv \\
        --years 2005-2024 --candidates 5000

Each record is one line of JSON keyed as the existing noise pool is —
`pubmed_id`, `pmc_id`, `abstract`, `body` — so `d3text.corpus` reads the result
with no conversion, plus `journal`, `title` and `year` for characterising the
survivors. The output is data, not source: keep it out of the repository, as
the other corpora are.

**This is a small job, not a crawl.** The pool it replaces holds a thousand
documents, and at the screen's measured yield roughly five thousand candidates
cover that, out of a candidate universe of a quarter of a million — hence
`--candidates` rather than a page count. Preprints are kept on purpose:
they are noise-pool documents like any other, and the cleanest hard negative
found by hand was one.

**The sample is drawn per publication year, and that is not cosmetic.**
`retstart` is refused above 9,998 even with `usehistory=y`, so no query's
results can be paged past ten thousand; a single query for 226k hits can only
ever return its newest page, which would make every survivor's date a property
of the fetch. Slicing by year bounds each query near the cap and puts the
sample's year spread in the command line.

NCBI allows three requests a second without a key and ten with one; set
`NCBI_API_KEY` and the delay drops accordingly.
"""

import argparse
import json
import os
import pathlib
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator, Sequence
from typing import Any

import polars as pl
import xmlparser
from lxml import etree

from d3text import corpus, logs

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

TERM = '"Bacteria"[MeSH Terms] AND "open access"[filter]'
"""The prefilter, as a term rather than a constant choice.

`"Bacterial Physiological Phenomena"[MeSH Terms]` is the other heading the
candidate population could be drawn from; both are topic-level indexing and
neither decides anything.
"""

RETSTART_MAX = 9998
"""Highest `retstart` esearch accepts, WebEnv or no WebEnv."""

PAGE = 50
"""Ids per esearch page."""

BATCH = 20
"""Articles per efetch request."""

API_KEY_VARIABLE = "NCBI_API_KEY"

RETRIES = 4
BACKOFF = 2.0
TIMEOUT = 90

_PARSER = etree.XMLParser(recover=True, resolve_entities=False, huge_tree=True)


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="collect-microbiology-sample",
        description=(
            "Fetch PMC Open Access microbiology articles as a candidate pool "
            "for the enzyme-negative screen."
        ),
    )
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument(
        "--term", default=TERM, help=f"esearch term (default {TERM!r})"
    )
    parser.add_argument(
        "--years",
        type=year_range,
        default=year_range("2005-2024"),
        help="publication years to slice the query by, as FIRST-LAST",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=5000,
        help="candidate articles to fetch, spread evenly over the years",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        type=pathlib.Path,
        default=[],
        help="split CSVs whose documents BRENDA already curates",
    )
    parser.add_argument("--seed", type=int, default=20260902)
    return parser.parse_args()


def year_range(text: str) -> range:
    """`"2005-2024"` as the years it names, both ends included.

    :param text: the range as written on the command line.
    :return: the years to slice the query by.
    :raises ValueError: if it is not two years in order.
    """
    first, _, last = text.partition("-")
    years = range(int(first), int(last) + 1)
    if not years:
        raise ValueError(f"{text!r} names no year")
    return years


def column_ids(paths: Sequence[pathlib.Path], column: str) -> set[str]:
    """Every id in one column of the splits, without its `PMC` prefix.

    Scanned with `infer_schema_length=0` so the 560 MB training split is read
    as text and only the one column is materialised.

    :param paths: the split CSVs to read.
    :param column: the id column, `pmc_id` or `pubmed_id`.
    :return: the ids, as bare digits.
    """
    ids: set[str] = set()
    for path in paths:
        values = (
            pl.scan_csv(path, infer_schema_length=0)
            .select(pl.col(column))
            .collect()
            .get_column(column)
            .to_list()
        )
        ids |= {
            str(value).strip().removeprefix("PMC") for value in values if value
        }
    return ids - {""}


def _url(endpoint: str, **params: str | int | None) -> str:
    query = {key: value for key, value in params.items() if value is not None}
    return f"{EUTILS}/{endpoint}?{urllib.parse.urlencode(query)}"


def _get(url: str, delay: float) -> bytes:
    """One eutils response, retried on the transient failures.

    :param url: the request.
    :param delay: seconds to wait afterwards, for NCBI's rate limit.
    :return: the response body.
    :raises RuntimeError: once the retries are spent.
    """
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
                body: bytes = response.read()
            time.sleep(delay)
            return body
        except (urllib.error.URLError, TimeoutError) as failure:
            print(f"  retry {attempt + 1}: {failure}", flush=True)
            time.sleep(BACKOFF * (attempt + 1))
    raise RuntimeError(f"{url} failed {RETRIES} times")


def esearch(
    term: str, retstart: int, retmax: int, api_key: str | None, delay: float
) -> tuple[int, list[str]]:
    """One page of a PMC search.

    :param term: the query.
    :param retstart: the offset into the result set.
    :param retmax: ids to return.
    :param api_key: an NCBI key, or None.
    :param delay: seconds to wait after the request.
    :return: the total hit count and the page's PMC ids.
    """
    payload = _get(
        _url(
            "esearch.fcgi",
            db="pmc",
            term=term,
            retmode="json",
            retstart=retstart,
            retmax=retmax,
            api_key=api_key,
        ),
        delay,
    )
    # esearch's JSON carries raw control characters, which the strict decoder
    # refuses outright rather than escaping.
    found = json.loads(payload.decode("utf8", "replace"), strict=False)
    result = found.get("esearchresult", {})
    return int(result.get("count", 0)), list(result.get("idlist", []))


def page_offsets(
    hits: int, pages: int, rng: random.Random, page: int = PAGE
) -> list[int]:
    """Random `retstart` offsets into one query's reachable results.

    :param hits: the query's total hit count.
    :param pages: how many pages to draw.
    :param rng: the seeded source of the draw.
    :param page: ids per page.
    :return: the offsets, ascending.
    """
    reachable = min(hits, RETSTART_MAX + 1)
    last = max(reachable - page, 0)
    if last == 0:
        return [0]
    return sorted(rng.sample(range(last + 1), min(pages, last + 1)))


def year_candidates(
    term: str,
    year: int,
    wanted: int,
    excluded: set[str],
    rng: random.Random,
    api_key: str | None,
    delay: float,
) -> list[str]:
    """PMC ids drawn from one publication year of the query.

    :param term: the query, before the year is added to it.
    :param year: the publication year to slice by.
    :param wanted: ids to keep.
    :param excluded: PMC ids BRENDA already curates.
    :param rng: the seeded source of the draw.
    :param api_key: an NCBI key, or None.
    :param delay: seconds to wait between requests.
    :return: the drawn ids, in draw order.
    """
    sliced = f"{term} AND {year}[pdat]"
    hits, _ = esearch(sliced, 0, 0, api_key, delay)
    if not hits:
        return []

    pages = -(-wanted // PAGE)
    drawn: dict[str, None] = {}
    for offset in page_offsets(hits, pages, rng):
        _, ids = esearch(sliced, offset, PAGE, api_key, delay)
        drawn.update(
            dict.fromkeys(
                identifier for identifier in ids if identifier not in excluded
            )
        )
    print(f"{year}: {hits} hits, {len(drawn)} candidates", flush=True)
    return list(drawn)[:wanted]


def articles(raw: bytes) -> Iterator[tuple[Any, xmlparser.ParsedArticle]]:
    """Each article of an efetch article set, parsed as its own tree.

    Re-rooted rather than parsed where it sits: `parse_jats_article` resolves
    `//front` from the *document* root rather than from the element it was
    handed, so every article of a set otherwise parses as the set's first one,
    with no error and no warning.

    :param raw: an efetch `pmc-articleset` response.
    :return: each article's own element tree, and its parse.
    """
    root = etree.fromstring(raw, _PARSER)
    for article in root.iter("{*}article"):
        solo = etree.fromstring(etree.tostring(article), _PARSER)
        yield solo, xmlparser.parse_jats_article(solo)


def article_id(article: Any, kind: str) -> str | None:
    """The article's own identifier of one kind, as the JATS front holds it.

    :param article: an article element rooting its own tree.
    :param kind: the `pub-id-type`, `pmc` or `pmid`.
    :return: the identifier, or None if the article carries none.
    """
    found = article.xpath(
        ".//*[local-name()='article-id'][@pub-id-type=$kind]/text()", kind=kind
    )
    return str(found[0]).strip() if found else None


def record(
    article: Any, parsed: xmlparser.ParsedArticle
) -> dict[str, Any] | None:
    """One article as a noise-pool row, or None if it cannot be one.

    Emptiness is judged on `corpus.document_text`, the same function the
    screen will read the row with: a JATS body wrapping nothing still weighs a
    few hundred bytes of markup, and a row of it screens as a negative on no
    evidence at all.

    :param article: the article element rooting its own tree.
    :param parsed: its parse.
    :return: the row, or None where the article has no text or no pubmed id.
    """
    pubmed_id = article_id(article, "pmid")
    if not pubmed_id or not corpus.document_text(parsed.abstract, parsed.body):
        return None
    meta = parsed.meta
    return {
        "pubmed_id": pubmed_id,
        "pmc_id": article_id(article, "pmc"),
        "journal": meta.journal if meta else "",
        "title": meta.title if meta else "",
        "year": meta.year if meta else 0,
        "abstract": parsed.abstract or "",
        "body": parsed.body or "",
    }


def reject_collapse(rows: Sequence[dict[str, Any]]) -> None:
    """Refuse a batch whose articles did not parse as distinct documents.

    Keyed on the parsed text and not on the identifiers, which is the whole
    point: `article_id` reads a relative xpath, so a set parsed against its
    own root still hands every row its own pmid while title, abstract and
    body have all collapsed onto the first article's. Distinct ids over
    identical text is exactly the shape of that regression.

    :param rows: one efetch batch's rows.
    :raises RuntimeError: if two rows carry the same title, abstract and
        body.
    """
    texts = [(row["title"], row["abstract"], row["body"]) for row in rows]
    distinct = len(set(texts))
    if distinct != len(texts):
        raise RuntimeError(
            f"an article set of {len(texts)} yielded {distinct} distinct "
            "documents; either the articles were parsed against the set's "
            "root rather than their own, or PMC served one document twice"
        )


def collect(
    candidates: Sequence[str], api_key: str | None, delay: float
) -> Iterator[dict[str, Any]]:
    """Fetch and parse the candidates, in batches.

    :param candidates: PMC ids to fetch.
    :param api_key: an NCBI key, or None.
    :param delay: seconds to wait between requests.
    :return: one row per article that carries text and a pubmed id.
    """
    for start in range(0, len(candidates), BATCH):
        batch = candidates[start : start + BATCH]
        raw = _get(
            _url(
                "efetch.fcgi",
                db="pmc",
                id=",".join(batch),
                retmode="xml",
                api_key=api_key,
            ),
            delay,
        )
        rows = [
            row
            for article, parsed in articles(raw)
            if (row := record(article, parsed)) is not None
        ]
        reject_collapse(rows)
        print(
            f"  fetched {start + len(batch)} of {len(candidates)}, "
            f"kept {len(rows)} of this batch",
            flush=True,
        )
        yield from rows


def main() -> None:
    logs.configure()
    args = read_args()

    api_key = os.environ.get(API_KEY_VARIABLE) or None
    delay = 0.15 if api_key else 0.4
    rng = random.Random(args.seed)

    excluded_pmc = column_ids(args.exclude, "pmc_id")
    excluded_pubmed = column_ids(args.exclude, "pubmed_id")
    print(
        f"excluding {len(excluded_pmc)} BRENDA pmc ids and "
        f"{len(excluded_pubmed)} pubmed ids"
    )

    per_year = -(-args.candidates // len(args.years))
    candidates: dict[str, None] = {}
    for year in args.years:
        candidates.update(
            dict.fromkeys(
                year_candidates(
                    args.term,
                    year,
                    per_year,
                    excluded_pmc,
                    rng,
                    api_key,
                    delay,
                )
            )
        )

    written: set[str] = set()
    with args.output.open("w", encoding="utf8") as out:
        for row in collect(list(candidates), api_key, delay):
            if row["pubmed_id"] in excluded_pubmed or (
                row["pubmed_id"] in written
            ):
                continue
            written.add(row["pubmed_id"])
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        f"{len(candidates)} candidates over {len(args.years)} years, "
        f"{len(written)} written to {args.output}"
    )


if __name__ == "__main__":
    main()
