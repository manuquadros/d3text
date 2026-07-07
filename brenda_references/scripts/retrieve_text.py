"""Retrieve PubMed abstracts and full text content when available.

Feed the document database with abstracts  and full text retrieved from PubMed.
"""

import asyncio
import itertools
from collections.abc import Iterator, MutableMapping

from aiotinydb import AIOTinyDB
from aiotinydb.storage import AIOJSONStorage
from brenda_types import Document
from brenda_references.config import config
from apiadapters.ncbi import AsyncNCBIAdapter
from tinydb import where
from tqdm import tqdm
from utils import AsyncAPIAdapter, CachingMiddleware


async def retrieve(
    field: str, docs: dict[str, Document], api: AsyncAPIAdapter
) -> dict[str, Document]:
    """Retrieve data for the given `field`, for each doc in `docs`.

    :param field: field of the document model to be retrieved
    :param docs: Document instances to be updated, keyed by document id.

    :return: dictionary containing only models that were updated,
        keyed by document id.
    """
    match field:
        case "abstract":
            ncbi_id = "pubmed_id"
            fetch_func = api.fetch_ncbi_abstracts
        case "fulltext":
            ncbi_id = "pmc_id"
            fetch_func = api.fetch_fulltext_articles

    ids_to_retrieve = (
        getattr(doc, ncbi_id) for doc in docs.values() if hasattr(doc, ncbi_id)
    )
    retrieved = await fetch_func(ids_to_retrieve)

    updated_docs: dict[str, Document] = {}

    for doc_id, doc in docs.items():
        if getattr(doc, ncbi_id) in retrieved:
            updated_docs[doc_id] = doc.model_copy(
                update={field: retrieved.get(getattr(doc, ncbi_id))}
            )

    return updated_docs


async def store_in_db(items: dict[str, Document], docdb: AIOTinyDB):
    """Store `items` in `docdb`."""
    for key in items:
        docdb.table("documents").update(
            items[key].model_dump(),
            doc_ids=[key],
        )


async def run() -> None:  # noqa: D103
    async with (
        AIOTinyDB(
            config["documents"],
            storage=CachingMiddleware(AIOJSONStorage),
        ) as docdb,
    ):
        docs = docdb.table("documents")
        missing_abstracts = docs.search(
            where("pubmed_id").exists()
            & (
                (~where("abstract").exists())
                | (where("abstract") == None)
                | (where("abstract") == "")
            )
        )
        missing_fulltext = docs.search(
            where("pmc_id").exists()
            & (where("pmc_open") == True)
            & ((~where("fulltext").exists()) | (where("fulltext") == ""))
        )

        async with AsyncNCBIAdapter() as ncbi:
            print("Retrieving full text:")
            for batch in itertools.batched(tqdm(missing_fulltext), n=250):
                docs = {
                    doc.doc_id: Document.model_validate(doc) for doc in batch
                }
                updates = await retrieve(field="fulltext", docs=docs, api=ncbi)
                await store_in_db(items=updates, docdb=docdb)

            print("Retrieving abstracts:")
            for batch in itertools.batched(tqdm(missing_abstracts), n=250):
                docs = {
                    doc.doc_id: Document.model_validate(doc) for doc in batch
                }
                updates = await retrieve(field="abstract", docs=docs, api=ncbi)
                await store_in_db(items=updates, docdb=docdb)


def main() -> None:
    asyncio.run(run())
