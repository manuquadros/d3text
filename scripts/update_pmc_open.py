"""Script for verifying the status of documents marked as not pmc_open."""

import asyncio

from aiotinydb import AIOTinyDB
from tinydb.middlewares import CachingMiddleware
from tinydb.storages import JSONStorage
from tqdm import tqdm

from brenda_references.config import config
from apiadapters.ncbi import AsyncNCBIAdapter


async def run() -> None:
    async with (
        AIOTinyDB(
            config["documents"], storage=CachingMiddleware(JSONStorage)
        ) as docdb,
        AsyncNCBIAdapter() as ncbi,
    ):
        for doc in tqdm(docdb.table("documents")):
            if doc["pmc_id"] and not doc["pmc_open"]:
                is_open = await ncbi.is_pmc_open(doc["pmc_id"])
                docdb.table("documents").update(
                    {"pmc_open": is_open},
                    doc_ids=[doc.doc_id],
                )


def main() -> None:
    asyncio.run(run())
