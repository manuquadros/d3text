"""Brenda References

This module provides functions to build a database of article references from
the BRENDA database. Each article reference is linked to the enzymes it is
associated with on BRENDA as well as with the organisms that are referenced
by the article as expressing each particular enzyme.

The main function is sync_doc_db, which will fetch references from BRENDA and
update the JSON database it founds references that are not already stored in
the latter.
"""

import ast
import itertools
import logging
from collections.abc import Iterable
from functools import cache
from importlib import resources
from pprint import pformat
from typing import Any

import numpy as np
import pandas as pd
import xmlparser
from aiotinydb import AIOTinyDB
from aiotinydb.storage import AIOJSONStorage
from apiadapters.ncbi import AsyncNCBIAdapter
from apiadapters.straininfo import AsyncStrainInfoAdapter
from d3types import EC, Bacteria, Document
from lpsn_interface import lpsn_synonyms
from tinydb.table import Document as TDBDocument
from tqdm import tqdm

from brenda_references import db
from brenda_references.utils import CachingMiddleware

from .config import config

DATA_DIR = resources.files("brenda_references") / "data"


def stderr_logger(level: int = logging.DEBUG) -> logging.Logger:
    """Create a simple stderr logger for debugging purposes."""
    ologger = logging.getLogger(__name__)
    ologger.setLevel(level)

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s, %(module)s.%(funcName)s, %(levelname)s, %(message)s",
            datefmt="%H:%M:%S",
        ),
    )
    ologger.addHandler(handler)

    return ologger


def preprocess_relations(row: pd.Series) -> pd.Series:
    """Transform the relations columns.

    Relations are coded like this on the relations column:

    {'HasEnzyme': [{'subject': 2681, 'object': 26836},
    {'subject': 5301, 'object': 26836},
    {'subject': 6140, 'object': 26836}]}

    :return:
        In this example, [{
            ("oth2681", "enz26836"): "HasEnzyme",
            ("oth5301", "enz26836"): "HasEnzyme",
            ("oth6140", "enz26836"): "HasEnzyme",
        }]
    """

    def get_key(
        entities: tuple[int, int], prefixes: tuple[str, str]
    ) -> tuple[str, str]:
        return tuple(
            sorted(
                (f"{prefixes[0]}{entities[0]}", f"{prefixes[1]}{entities[1]}")
            )
        )

    relations = ast.literal_eval(row["relations"])
    pairs = {}

    for pair in relations.get("HasSpecies", []):
        key = get_key(
            entities=(pair["subject"], pair["object"]),
            prefixes=("str", "bac"),
        )
        pairs[key] = np.array([0, 1, 0], dtype=np.float16)

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

    for entity_pair in itertools.combinations(row["entities"], r=2):
        if entity_pair not in pairs:
            pairs[entity_pair] = np.array([0, 0, 1], dtype=np.float16)

    row.loc["relations"] = [pairs]
    return row


def preprocess_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the entity labels on `df` for model training"""
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

    return df.apply(preprocess_relations, axis=1)


def load_split(split: str, noise: int = 0, limit: int = 0) -> pd.DataFrame:
    """Load dataset split."""
    path = DATA_DIR / f"{split}_data.csv"
    split_data = pd.read_csv(path, index_col=0)

    if limit:
        split_data = split_data.truncate(after=limit - 1)

    split_data = preprocess_labels(
        split_data.dropna(subset=["abstract", "fulltext"])
    )

    noise_data = pd.DataFrame(itertools.islice(psycholinguistics_data(), noise))
    return pd.concat((split_data, noise_data), axis=0, ignore_index=True)


@cache
def psycholinguistics_data() -> Iterable[tuple[Any, ...]]:
    """Load psycholinguistics articles for noise."""
    path = DATA_DIR / "pmc_linguistics_articles.json"
    psyling = pd.read_json(path, lines=True).rename(
        columns={"body": "fulltext"}
    )
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
    return psyling.sample(n=len(psyling), replace=False).itertuples(index=False)


def validation_data(noise: int = 0, limit: int = 0) -> pd.DataFrame:
    """Load validation data."""
    val = load_split("validation", noise=noise, limit=limit)
    return val[
        ~(val["bacteria"].astype("bool") & ~val["strains"].astype("bool"))
    ]


def training_data(noise: int = 0, limit: int = 0) -> pd.DataFrame:
    """Load training data."""
    train = load_split("training", noise=noise, limit=limit)
    return train[
        ~(train["bacteria"].astype("bool") & ~train["strains"].astype("bool"))
    ]


def test_data(noise: int = 0, limit: int = 0) -> pd.DataFrame:  # noqa: PT028
    """Load test data."""
    test = load_split("test", noise=noise, limit=limit)
    return test[
        ~(test["bacteria"].astype("bool") & ~test["strains"].astype("bool"))
    ]


async def add_abstracts(
    docs: Iterable[Document],
    adapter: AsyncNCBIAdapter,
) -> list[Document]:
    """Add abstracts to the documents in `docs` when they are available.

    :param docs: Document models to be augmented with a retrieved abstract
    :param adapter: The API adapter connecting to NCBI

    :return: The documents in `docs` are returned in the same order, but with
             abstracts added to then, when are available
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

    :param docdb: The JSON database
    :param ncbi: The API adapter connecting to NCBI
    :param reference: SQLModel containing the initial metadata retrieved
        from BRENDA.

    :return: Document model containing all the metadata retrieved.
    """
    doc = await expand_doc(
        ncbi, Document.model_validate(reference.model_dump())
    )
    docdb.table("documents").insert(
        TDBDocument(doc.model_dump(), doc_id=reference.reference_id),
    )


def store_enzyme_synonyms(
    docdb: AIOTinyDB,
    enzyme: EC,
    synonyms: Iterable[str],
) -> None:
    """Store enzyme data in the JSON database.

    :param docdb: The JSON database
    :param enzyme: EC model linked describing an enzyme
    :param synonyms: set of synonyms for that EC Class retrieved from BRENDA
    """
    enzyme = enzyme.model_copy(update={"synonyms": frozenset(synonyms)})
    docdb.table("enzymes").upsert(
        TDBDocument(enzyme.model_dump(exclude="id"), doc_id=enzyme.id),
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
    """Ensure that references in BRENDA are processed into the Doc database.

    For each reference, store into the JSON database the entities that are
    linked to it in BRENDA, as well as the relations between these entities
    that are annotated in the database.

    At this point, we are not performing any checks as to whether information
    on BRENDA has changed since the last time we visited it, except as to
    whether new references were added to it.
    """
    async with (
        AIOTinyDB(
            config["documents"],
            storage=CachingMiddleware(AIOJSONStorage),
        ) as docdb,
        AsyncNCBIAdapter() as ncbi,
        AsyncStrainInfoAdapter() as straininfo,
        db.BRENDA() as brenda,
    ):
        straininfo.storage = docdb

        print("Retrieving literature references.")
        # TODO: Improve concurrency here. Use async tasks to speed it up
        with tqdm(total=brenda.count_references()) as progress_bar:
            for reference in brenda.references():
                if not docdb.table("documents").contains(
                    doc_id=reference.reference_id
                ):
                    await add_document(docdb, ncbi, reference)
                progress_bar.update(1)

        print("Retrieving enzyme-organism relations from BRENDA.")

        # Collect all organism/enzyme relations for each document
        for doc in tqdm(docdb.table("documents")):
            relations = brenda.enzyme_relations(doc.doc_id)

            for enzyme in relations["enzymes"]:
                if not docdb.table("enzymes").contains(doc_id=enzyme.id):
                    synonyms = brenda.ec_synonyms(enzyme.id)
                    store_enzyme_synonyms(docdb, enzyme, synonyms)

            straininfo.store_strains(
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
