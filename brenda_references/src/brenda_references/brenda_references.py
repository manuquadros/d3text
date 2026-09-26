"""Build a database of article references from BRENDA.

Each reference is linked to the enzymes it is associated with on BRENDA and to
the organisms the article references as expressing each one. `sync_doc_db` is
the entry point.
"""

import argparse
import ast
import asyncio
import itertools
import logging
from collections.abc import Iterable, Iterator, Mapping
from functools import cache, partial
from pprint import pformat

import numpy as np
import pandas as pd
import xmlparser
from aiotinydb import AIOTinyDB
from aiotinydb.storage import AIOJSONStorage
from apiadapters.ncbi import AsyncNCBIAdapter
from apiadapters.straininfo import AsyncStrainInfoAdapter
from apiadapters.straininfo import Strain as StrainInfoStrain
from d3types import EC, Bacteria, Document
from lpsn_interface import lpsn_synonyms
from tinydb.table import Document as TDBDocument
from tqdm import tqdm

from brenda_references import db
from brenda_references.utils import CachingMiddleware

from .data_paths import documents_path, noise_pool_path, split_path

logger = logging.getLogger(__name__)

# The permutation of the noise pool has to be identical in every process, not
# merely random: `train` and `evaluate` each build the splits in a process of
# their own, and they must agree on which articles are noise for which split.
NOISE_SEED = 20250818

# A second, independent noise seed for the enzyme-negative pool: it is
# permuted on its own, so its block assignment does not move if the
# psycholinguistics pool's size ever changes, or vice versa.
ENZYME_NOISE_SEED = 20260916

# Split name -> the [first, last) fraction of a permuted pool it draws from.
# Disjoint by construction, which is what keeps a noise article out of both
# training and test. Shared by every noise pool: the fractions are a policy
# about how much of *a* pool each split gets, not a property of one pool.
NOISE_BLOCKS = {
    "training": (0.0, 0.7),
    "validation": (0.7, 0.85),
    "test": (0.85, 1.0),
}


def stderr_logger(level: int = logging.DEBUG) -> logging.Logger:
    """Create a simple stderr logger for debugging purposes.

    A logger of its own, not the module's `logger` — that one's ancestor
    gets routed to a console handler by `d3text.logs.configure()`, and
    sharing it here would print every line through both handlers once an
    entry point has called that.
    """
    ologger = logging.getLogger(f"{__name__}.debug")
    ologger.setLevel(level)
    ologger.propagate = False

    if not ologger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s, %(module)s.%(funcName)s, %(levelname)s, %(message)s",
                datefmt="%H:%M:%S",
            ),
        )
        ologger.addHandler(handler)

    return ologger


# The temporary column `preprocess_relations` uses to hand its per-row,
# per-predicate dropped-pair counts back to `preprocess_labels`, which sums
# them across the `df.apply` call, reports each predicate's split total,
# then removes the column.
_DROPPED_PAIRS_COL = "_dropped_pairs"

# Why `preprocess_relations` drops a pair of that predicate, keyed the same
# way as the `dropped` dict it builds — read out by `preprocess_labels`
# when it logs each predicate's split total.
_DROP_REASONS: Mapping[str, str] = {
    "HasSpecies": (
        "whose subject is not a row strain or whose object is in no "
        "organism column"
    ),
    "HasEnzyme": "whose subject is in no bacteria, strains or "
    "other_organisms column",
}


def preprocess_relations(row: pd.Series) -> pd.Series:
    """Transform the relations column into `(subject, object) -> label` dicts.

    A `HasSpecies` pair is kept only when its strain subject is in
    `row["strains"]` and its object is in `row["bacteria"]` or
    `row["other_organisms"]` — the same columns `row["entities"]` was built
    from, so a kept pair's key always names two of that row's entities. A
    `HasEnzyme` pair is kept only when its subject is in `row["bacteria"]`,
    `row["strains"]` or `row["other_organisms"]`. A pair failing its
    predicate's check is dropped and counted, by predicate, in
    `row[_DROPPED_PAIRS_COL]`.

    :param row: one document's row, with `entities`, `bacteria`, `strains`
        and `other_organisms` already normalised by `preprocess_labels`.
    :return: `row` with `relations` replaced by the pair->label dict and
        `_DROPPED_PAIRS_COL` set to a `{predicate: dropped-pair count}`
        dict for this row.
    """

    def canonical(first: str, second: str) -> tuple[str, str]:
        """The one order every key in `pairs` is spelled in.

        Both the typed keys and the `none` fill go through here, so the
        fill's membership test compares a pair against the spelling a typed
        key of that pair would have.
        """
        low, high = sorted((first, second))
        return low, high

    def get_key(
        entities: tuple[int, int], prefixes: tuple[str, str]
    ) -> tuple[str, str]:
        return canonical(
            f"{prefixes[0]}{entities[0]}", f"{prefixes[1]}{entities[1]}"
        )

    relations = ast.literal_eval(row["relations"])
    pairs = {}
    dropped: dict[str, int] = {"HasSpecies": 0, "HasEnzyme": 0}

    for pair in relations.get("HasSpecies", []):
        if pair["subject"] not in row["strains"]:
            dropped["HasSpecies"] += 1
            continue
        for enttype in ("bacteria", "other_organisms"):
            if pair["object"] in row[enttype]:
                key = get_key(
                    entities=(pair["subject"], pair["object"]),
                    prefixes=("str", enttype[:3]),
                )
                pairs[key] = np.array([0, 1, 0], dtype=np.float16)
                break
        else:
            dropped["HasSpecies"] += 1

    for pair in relations.get("HasEnzyme", []):
        for enttype in (
            "bacteria",
            "strains",
            "other_organisms",
        ):
            if pair["subject"] in row[enttype]:
                key = get_key(
                    entities=(pair["subject"], pair["object"]),
                    prefixes=(enttype[:3], "enz"),
                )
                pairs[key] = np.array([1, 0, 0], dtype=np.float16)
                break
        else:
            dropped["HasEnzyme"] += 1

    for entity_pair in itertools.combinations(row["entities"], r=2):
        key = canonical(*entity_pair)
        if key not in pairs:
            pairs[key] = np.array([0, 0, 1], dtype=np.float16)

    row.loc["relations"] = [pairs]
    row.loc[_DROPPED_PAIRS_COL] = dropped
    return row


def preprocess_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the entity labels on `df` for model training.

    :param df: a split frame with gold columns still Python-literal
        strings, as read from CSV.
    :return: `df` with the entity columns parsed, an `entities` column
        added, and `relations` replaced by `preprocess_relations`'
        pair->label dicts.
    """
    df["bacteria"] = (
        df["bacteria"]
        .apply(ast.literal_eval)
        .apply(lambda bacdic: [int(bacid) for bacid in bacdic])
    )
    df["other_organisms"] = (
        df["other_organisms"]
        .apply(ast.literal_eval)
        .apply(lambda otherdic: [int(otherid) for otherid in otherdic])
    )
    for col in ("strains", "enzymes"):
        df[col] = df[col].apply(ast.literal_eval)

    def merge_entcols(row: pd.Series) -> list[str]:
        ents: list[str] = [
            entcol[:3] + str(ent)
            for entcol in ("bacteria", "enzymes", "strains", "other_organisms")
            for ent in row[entcol]
        ]
        return ents

    df["entities"] = df.apply(merge_entcols, axis=1)

    processed = df.apply(preprocess_relations, axis=1)
    dropped_counts = processed.pop(_DROPPED_PAIRS_COL)
    totals: dict[str, int] = {}
    for row_counts in dropped_counts:
        for predicate, count in row_counts.items():
            totals[predicate] = totals.get(predicate, 0) + count
    for predicate, total in totals.items():
        if total:
            logger.warning(
                "dropped %d %s pair(s) %s",
                total,
                predicate,
                _DROP_REASONS[predicate],
            )
    return processed


def merge_duplicate_documents(df: pd.DataFrame) -> pd.DataFrame:
    """Union rows that repeat a `pubmed_id` into one row per document.

    BRENDA curates one reference per enzyme a paper documents, so a paper
    naming several enzymes lands in the export as several rows sharing one
    `pubmed_id`, the same `pmc_id`, abstract and full text, each carrying
    only part of the paper's gold set. Left unmerged, a consumer keyed by
    `pubmed_id` (the token-label store) silently keeps whichever row it
    read last and drops every other row's labels.

    :param df: a split as read from CSV, gold columns still Python-literal
        strings.
    :return: one row per `pubmed_id`; `enzymes`, `strains`, `entity_spans`,
        `bacteria`, `other_organisms` and `relations` unioned across the
        group, every other column taken from the group's row with a
        non-null `path`, or its first row if none has one.
    """
    if not df["pubmed_id"].duplicated().any():
        return df

    def union_list(values: pd.Series) -> str:
        merged: list = []
        for value in values:
            merged.extend(
                item for item in ast.literal_eval(value) if item not in merged
            )
        return repr(merged)

    def union_dict(values: pd.Series) -> str:
        merged: dict = {}
        for value in values:
            merged.update(ast.literal_eval(value))
        return repr(merged)

    def union_relations(values: pd.Series) -> str:
        merged: dict[str, list] = {}
        for value in values:
            for predicate, pairs in ast.literal_eval(value).items():
                bucket = merged.setdefault(predicate, [])
                bucket.extend(pair for pair in pairs if pair not in bucket)
        return repr(merged)

    def merge_group(group: pd.DataFrame) -> pd.Series:
        with_path = group[group["path"].notna()]
        primary = (
            with_path.iloc[0] if not with_path.empty else group.iloc[0]
        ).copy()
        primary["enzymes"] = union_list(group["enzymes"])
        primary["strains"] = union_list(group["strains"])
        primary["entity_spans"] = union_list(group["entity_spans"])
        primary["bacteria"] = union_dict(group["bacteria"])
        primary["other_organisms"] = union_dict(group["other_organisms"])
        primary["relations"] = union_relations(group["relations"])
        return primary

    rows = [
        merge_group(group) for _, group in df.groupby("pubmed_id", sort=False)
    ]
    return pd.DataFrame(rows).reset_index(drop=True)


def load_split(
    split: str, noise: int = 0, enzyme_noise: int = 0, limit: int = 0
) -> pd.DataFrame:
    """Load dataset split.

    :param split: the split name (`training`, `validation` or `test`).
    :param noise: how many psycholinguistics noise documents to append.
    :param enzyme_noise: how many enzyme-negative noise documents to append,
        on top of `noise` — a second, independent pool, not drawn from the
        same budget.
    :param limit: keep only the first `limit` documents that carry text;
        `noise` and `enzyme_noise` are scaled by the same fraction of the
        split that survives, so a truncated split holds the proportion of
        synthetic documents a whole one holds. 0 or unset keeps all.
    :return: the split, with noise appended and every row's `source` column
        naming which configured corpus file it came from (`split`,
        `psycholinguistics` or `enzyme_negative`) — what a consumer checking
        an encodings store for a whole missing source keys on.
    :raises ValueError: if `limit` is negative — a negative row count drops
        rows off the end of the split rather than refusing the call, which
        would otherwise size a training run's entity vocabulary from
        something far from the argument that caused it.
    """
    if limit < 0:
        msg = f"limit must be non-negative; got {limit}."
        raise ValueError(msg)

    path = split_path(split)
    split_data = merge_duplicate_documents(
        pd.read_csv(path, index_col=0)
    ).dropna(subset=["abstract", "fulltext"])

    # Dropping the textless rows before truncating is what makes `limit` the
    # number of documents actually trained on, and what makes the fraction
    # below exact: a row with no text is in neither the whole run nor the
    # truncated one, so it belongs in neither side of the ratio.
    if limit:
        usable = len(split_data)
        fraction = min(1.0, limit / usable) if usable else 0.0
        noise = round(noise * fraction)
        enzyme_noise = round(enzyme_noise * fraction)
        split_data = split_data.head(limit).copy()

    split_data = preprocess_labels(split_data).assign(source=split)

    # Tagged before the concat, not after: once it has run, a synthetic
    # noise row is a `pubmed_id` indistinguishable from the corpus's own,
    # and a reader checking an encodings store for a whole missing source
    # (`BrendaDataset`) needs to tell them apart at any split size,
    # including a `--limit` subset that shrinks a pool down to a handful of
    # rows.
    noise_data = noise_documents(split, noise).assign(
        source="psycholinguistics"
    )
    enzyme_noise_data = enzyme_negative_documents(split, enzyme_noise).assign(
        source="enzyme_negative"
    )

    # Counted off these three frames, not the concatenated result's `source`
    # value_counts: a pool `--limit` scales down to zero rows is still a
    # frame here, so it logs as 0 instead of leaving no trace at all.
    logger.info(
        "split=%s real=%d psycholinguistics=%d enzyme_negative=%d",
        split,
        len(split_data),
        len(noise_data),
        len(enzyme_noise_data),
    )

    return pd.concat(
        (split_data, noise_data, enzyme_noise_data),
        axis=0,
        ignore_index=True,
    )


CONTAMINATED_PMC_IDS = frozenset(
    {
        5152542,
        5402676,
        5502579,
        5771508,
        7255050,
        7335431,
        7451337,
        7451372,
        7453160,
        7540002,
        7879075,
        7932037,
        8317543,
        8317549,
        8608229,
        8702868,
        8835384,
        8835387,
        8835388,
        9135004,
        9159671,
        9490936,
        9887656,
        10039723,
        10171844,
        10226771,
        10597387,
        10640881,
        10640887,
        10713020,
        11144449,
        11332681,
        11332686,
        11585907,
        11790113,
        11815351,
    }
)
"""`pmc_id`s of noise-pool rows that name an enzyme.

The pool is meant to be enzyme-free by construction, but rescreening it with
the same surface-form index and descriptive-match reading the labelling
pipeline itself matches against turns up genuine and near-genuine enzyme
mentions: liver-panel and oxidative-stress transaminases, ACE2 in COVID
papers, and a handful of same-spelling collisions a dictionary match cannot
tell from a real name without reading the sentence (a cited author surnamed
after an enzyme, a psychometric "factor I", a coagulation "complex, I").
Excluded rather than left in, because the pool's whole purpose is to
guarantee an enzyme-free negative for every row it hands out.
"""


@cache
def psycholinguistics_data() -> pd.DataFrame:
    """The whole noise pool, permuted once under a fixed seed.

    Returns the frame rather than an iterator, and seeds the permutation,
    because `@cache` memoizes whatever this hands back and two callers must see
    the same pool: an iterator is *consumed*, so a sweep's later trials drew no
    noise at all, and an unseeded permutation differs per process, so
    `evaluate` scored the model on noise `train` had trained on.

    Dropping `CONTAMINATED_PMC_IDS` before the permutation changes the pool's
    size, so which article each `NOISE_BLOCKS` fraction draws into training,
    validation or test also changes.

    :return: the permuted pool, contaminated rows excluded.
    """
    path = noise_pool_path("psycholinguistics")
    psyling = pd.read_json(path, lines=True).rename(
        columns={"body": "fulltext"}
    )
    psyling = psyling[
        ~psyling["pmc_id"].isin(CONTAMINATED_PMC_IDS)
    ].reset_index(drop=True)
    psyling["abstract"] = psyling["abstract"].apply(xmlparser.remove_tags)
    for col in (
        "bacteria",
        "enzymes",
        "strains",
        "other_organisms",
        "entities",
        "relations",
    ):
        psyling[col] = [[]] * len(psyling)
    return psyling.sample(
        n=len(psyling), replace=False, random_state=NOISE_SEED
    ).reset_index(drop=True)


@cache
def enzyme_negative_data() -> pd.DataFrame:
    """The enzyme-negative noise pool, permuted once under its own seed.

    Every row survives `negative_screen.LITERAL`'s screen, so it names no
    enzyme under the same surface-form index the positives are labelled
    with; `enzymes` is a true negative here, not a free one. `bacteria`
    and `strains` are blanked the same as
    `psycholinguistics_data`'s, but for a different reason: these documents
    are real microbiology articles that were never curated for those
    entities, so a blank column is an *unknown*, not a verified absence.
    A run drawing on this pool wants `class_negative_abstention` on for
    `bacteria` and `strains`, so the loss abstains where the dictionary
    still finds a mention, while the genuine `enzymes` negative is kept at
    full weight.

    :return: the permuted pool.
    """
    path = noise_pool_path("enzyme_negative")
    pool = pd.read_json(path, lines=True).rename(columns={"body": "fulltext"})
    pool["abstract"] = pool["abstract"].apply(xmlparser.remove_tags)
    for col in (
        "bacteria",
        "enzymes",
        "strains",
        "other_organisms",
        "entities",
        "relations",
    ):
        pool[col] = [[]] * len(pool)
    return pool.sample(
        n=len(pool), replace=False, random_state=ENZYME_NOISE_SEED
    ).reset_index(drop=True)


def _pool_block(
    pool: pd.DataFrame,
    split: str,
    noise: int,
    blocks: Mapping[str, tuple[float, float]] = NOISE_BLOCKS,
) -> pd.DataFrame:
    """The first `noise` rows of `split`'s own block of an already-permuted pool.

    Each split draws from a disjoint block, so no article can be trained on and
    then evaluated on. The bounds are fixed fractions of the pool rather than a
    running offset, which would slide one split's block into another's the
    moment a caller changed how much noise it wanted.

    :param pool: an already-permuted noise pool, as `psycholinguistics_data`
        or `enzyme_negative_data` returns.
    :param split: the split to draw for.
    :param noise: how many rows to draw.
    :param blocks: split name -> the `[first, last)` fraction of `pool` it
        draws from.
    :return: the rows.
    :raises ValueError: if `split` has no block, or its block is smaller than
        `noise` — running short must fail rather than quietly return fewer.
    """
    if noise <= 0:
        return pd.DataFrame()

    if split not in blocks:
        msg = f"{split!r} has no noise block; expected one of {sorted(blocks)}"
        raise ValueError(msg)

    first_fraction, last_fraction = blocks[split]
    start = int(first_fraction * len(pool))
    end = int(last_fraction * len(pool))

    if end - start < noise:
        msg = (
            f"{split!r}'s noise block holds {end - start} rows, fewer "
            f"than the {noise} requested"
        )
        raise ValueError(msg)

    return pool.iloc[start : start + noise]


def noise_documents(split: str, noise: int) -> pd.DataFrame:
    """The first `noise` articles of `split`'s own block of the noise pool.

    :param split: the split to draw for.
    :param noise: how many articles to draw.
    :return: the articles.
    :raises ValueError: if `split` has no block, or its block is smaller than
        `noise`.
    """
    if noise <= 0:
        return pd.DataFrame()
    return _pool_block(psycholinguistics_data(), split, noise)


def enzyme_negative_documents(split: str, noise: int) -> pd.DataFrame:
    """The first `noise` articles of `split`'s own block of the enzyme pool.

    :param split: the split to draw for.
    :param noise: how many articles to draw.
    :return: the articles.
    :raises ValueError: if `split` has no block, or its block is smaller than
        `noise`.
    """
    if noise <= 0:
        return pd.DataFrame()
    return _pool_block(enzyme_negative_data(), split, noise)


def validation_data(
    noise: int = 0, enzyme_noise: int = 0, limit: int = 0
) -> pd.DataFrame:
    """Load validation data."""
    val = load_split(
        "validation", noise=noise, enzyme_noise=enzyme_noise, limit=limit
    )
    return val[
        ~(val["bacteria"].astype("bool") & ~val["strains"].astype("bool"))
    ]


def training_data(
    noise: int = 0, enzyme_noise: int = 0, limit: int = 0
) -> pd.DataFrame:
    """Load training data."""
    train = load_split(
        "training", noise=noise, enzyme_noise=enzyme_noise, limit=limit
    )
    return train[
        ~(train["bacteria"].astype("bool") & ~train["strains"].astype("bool"))
    ]


def test_data(  # noqa: PT028
    noise: int = 0, enzyme_noise: int = 0, limit: int = 0
) -> pd.DataFrame:
    """Load test data."""
    test = load_split(
        "test", noise=noise, enzyme_noise=enzyme_noise, limit=limit
    )
    return test[
        ~(test["bacteria"].astype("bool") & ~test["strains"].astype("bool"))
    ]


async def add_abstracts(
    docs: Iterable[Document],
    adapter: AsyncNCBIAdapter,
) -> list[Document]:
    """Add abstracts to the documents in `docs` where they are available.

    :param docs: the documents to augment.
    :param adapter: the API adapter connecting to NCBI.
    :return: the same documents in the same order, abstracts added where found.
    """
    # Ensure that we have an indexable sequence
    docs = list(docs)

    targets = {
        doc.pubmed_id: ix
        for ix, doc in enumerate(docs)
        if doc.pubmed_id and not getattr(doc, "abstract", None)
    }

    if not targets:
        return docs

    abstracts = await adapter.fetch_ncbi_abstracts(targets.keys())

    for pubmed_id, abstract in abstracts.items():
        index = targets.get(pubmed_id)
        try:
            docs[index] = docs[index].model_copy(update={"abstract": abstract})
        except TypeError:
            logger = stderr_logger()
            logger.debug(pformat(targets))
            logger.debug(index)
            logger.debug(pubmed_id)
            for doc in docs:
                if doc.pubmed_id.strip() == pubmed_id.strip():
                    print(doc)
            raise

    return docs


async def expand_doc(ncbi: AsyncNCBIAdapter, doc: Document) -> Document:
    """Check if we can find a PMCID and a DOI for the article."""
    if not doc.pubmed_id:
        return doc

    try:
        article_ids = await ncbi.article_ids(doc.pubmed_id)
    except KeyError:
        pmc_id = doi = None
        pmc_open = False
    else:
        pmc_id = article_ids.get("pmc")
        doi = article_ids.get("doi")

        if isinstance(pmc_id, str):
            pmc_id = pmc_id.replace("PMC", "")

        pmc_open = await ncbi.is_pmc_open(pmc_id)

    return doc.model_copy(
        update={
            "doi": doi,
            "pmc_id": pmc_id,
            "pmc_open": pmc_open,
        },
    )


class UnknownDocumentError(Exception):
    def __init__(self, reference_id: str) -> None:
        """Custom exception for unknown reference ids"""
        super().__init__(
            f"{reference_id} was not found in the document database"
        )


def get_document(docdb: AIOTinyDB, reference: db._Reference) -> Document:
    """Retrieve document from the JSON database by reference_id."""
    doc = docdb.table("documents").get(doc_id=reference.reference_id)

    if doc is None:
        raise UnknownDocumentError(reference.reference_id)

    return Document.model_validate(doc)


async def add_document(
    docdb: AIOTinyDB,
    ncbi: AsyncNCBIAdapter,
    reference: db._Reference,
) -> None:
    """Add document metadata to the JSON database, retrieving from NCBI.

    :param docdb: the JSON database.
    :param ncbi: the API adapter connecting to NCBI.
    :param reference: the initial metadata retrieved from BRENDA.
    """
    doc = await expand_doc(
        ncbi, Document.model_validate(reference.model_dump())
    )
    docdb.table("documents").insert(
        TDBDocument(doc.model_dump(), doc_id=reference.reference_id),
    )


# How many documents may be expanded (fetched from NCBI) at once. Since
# `sync_doc_db`'s workers pull from a single streamed BRENDA cursor, this is
# also how far reference retrieval is allowed to run ahead of processing —
# not a request-rate cap: NCBI's own request rate is already bounded inside
# AsyncAPIAdapter.
_MAX_CONCURRENT_DOCUMENTS = 8


async def _sync_doc_db_worker(
    docdb: AIOTinyDB,
    ncbi: AsyncNCBIAdapter,
    references: Iterator[db._Reference],
    progress_bar: tqdm,
) -> None:
    """Add each reference from the shared `references` iterator unless
    already stored.

    One of `_MAX_CONCURRENT_DOCUMENTS` workers draining the same iterator;
    `next()` on a plain iterator is synchronous, so calling it from several
    coroutines on one event loop never races.

    :param docdb: the JSON database.
    :param ncbi: the API adapter connecting to NCBI.
    :param references: the shared, single-pass iterator over BRENDA
        references, drained cooperatively by every worker.
    :param progress_bar: updated once per reference this worker processes.
    """
    for reference in references:
        if not docdb.table("documents").contains(doc_id=reference.reference_id):
            await add_document(docdb, ncbi, reference)
        progress_bar.update(1)


def store_enzyme_synonyms(
    docdb: AIOTinyDB,
    enzyme: EC,
    synonyms: Iterable[str],
) -> None:
    """Store enzyme data in the JSON database.

    :param docdb: the JSON database.
    :param enzyme: the EC model describing the enzyme.
    :param synonyms: its synonyms as retrieved from BRENDA.
    """
    enzyme = enzyme.model_copy(update={"synonyms": frozenset(synonyms)})
    docdb.table("enzymes").upsert(
        TDBDocument(enzyme.model_dump(exclude="id"), doc_id=enzyme.id),
    )


def store_strains(
    docdb: AIOTinyDB, strains: Mapping[int, StrainInfoStrain]
) -> None:
    """Write resolved strains to the doc db, keyed by their BRENDA id.

    Bound to `docdb` with `functools.partial` and passed as the `sink`
    `AsyncStrainInfoAdapter` calls on buffer flush. `strains` is keyed by
    BRENDA strain id, not by each `Strain`'s own `id` field (StrainInfo's
    id, `None` when unresolved); the full `model_dump()`, `id` included, is
    stored so `fix_missing_strains.py` can select unresolved rows on it.

    :param docdb: the JSON database.
    :param strains: resolved strains keyed by BRENDA strain id.
    """
    for strain_id, strain in strains.items():
        docdb.table("strains").upsert(
            TDBDocument(strain.model_dump(), doc_id=strain_id),
        )


def store_bacteria(docdb: AIOTinyDB, bacteria: Iterable[Bacteria]) -> None:
    """Retrieve bacterial synonyms from LPSN and add them to the doc db.

    :param docdb: The JSON database
    :param bacteria: Set of Bacteria models to be completed with synonyms
    """
    # TODO: batch the items instead of updating one by one
    for bac in bacteria:
        newbac = bac.model_copy(update={"synonyms": lpsn_synonyms(bac.lpsn_id)})
        docdb.table("bacteria").upsert(
            TDBDocument(newbac.model_dump(exclude="id"), doc_id=newbac.id),
        )


async def sync_doc_db() -> None:
    """Process BRENDA's references into the JSON document database.

    For each reference, stores the entities linked to it in BRENDA and the
    relations between them. No check is made for information changed on BRENDA
    since the last visit, only for references newly added.
    """
    async with (
        AIOTinyDB(
            documents_path(),
            storage=CachingMiddleware(AIOJSONStorage),
        ) as docdb,
        AsyncNCBIAdapter() as ncbi,
        AsyncStrainInfoAdapter(
            sink=partial(store_strains, docdb)
        ) as straininfo,
        db.BRENDA() as brenda,
    ):
        print("Retrieving literature references.")
        with tqdm(total=brenda.count_references()) as progress_bar:
            references = iter(brenda.references())
            async with asyncio.TaskGroup() as task_group:
                for _ in range(_MAX_CONCURRENT_DOCUMENTS):
                    task_group.create_task(
                        _sync_doc_db_worker(
                            docdb, ncbi, references, progress_bar
                        )
                    )

        print("Retrieving enzyme-organism relations from BRENDA.")

        # Collect all organism/enzyme relations for each document
        for doc in tqdm(docdb.table("documents")):
            relations = brenda.enzyme_relations(doc.doc_id)

            for enzyme in relations["enzymes"]:
                if not docdb.table("enzymes").contains(doc_id=enzyme.id):
                    synonyms = brenda.ec_synonyms(enzyme.id)
                    store_enzyme_synonyms(docdb, enzyme, synonyms)

            await straininfo.store_strains(
                [
                    strain
                    for strain in relations["strains"]
                    if not docdb.table("strains").contains(doc_id=strain.id)
                ],
            )
            store_bacteria(docdb, relations["bacteria"])

            document = Document.model_validate(doc).copy(
                update={
                    "relations": relations["triples"],
                    "enzymes": frozenset(
                        enzyme.id for enzyme in relations["enzymes"]
                    ),
                    "bacteria": {
                        bac.id: bac.organism for bac in relations["bacteria"]
                    },
                    "strains": [strain.id for strain in relations["strains"]],
                    "other_organisms": {
                        org.id: org.organism
                        for org in relations["other_organisms"]
                    },
                },
            )

            docdb.table("documents").update(
                document.model_dump(), doc_ids=[doc.doc_id]
            )


def main(argv: list[str] | None = None) -> None:
    """Synchronous entry point for `sync_doc_db`, which is a coroutine.

    :param argv: `None` reads `sys.argv` as an installed console script should;
        a test passes `[]` to call this in-process without inheriting pytest's
        own arguments.
    """
    argparse.ArgumentParser(description=sync_doc_db.__doc__).parse_args(argv)
    asyncio.run(sync_doc_db())
