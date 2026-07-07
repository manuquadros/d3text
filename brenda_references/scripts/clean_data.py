import re
from typing import Iterator

from brenda_references.config import config
from taxonomy import ncbitax
from tinydb import TinyDB, where
from tinydb.middlewares import CachingMiddleware
from tinydb.storages import JSONStorage
from tinydb.table import Document, Table
from tqdm import tqdm


def strain_queries(strain: Document) -> Iterator[str]:
    try:
        yield strain["taxon"]["name"]
    except TypeError:
        for name in strain["designations"]:
            yield name


def has_bacterial_markers(name: str) -> bool:
    antibiotic_resistance = re.compile(
        r"[^a-zA-Z](Rifr|Kanr|Camr|Spec|Amp)[^a-zA-Z]"
    )
    plasmid = re.compile(r"[^a-zA-Z](pBR|pUC|pET|pXL)[^a-zA-Z]")

    return (
        re.search(antibiotic_resistance, name) is not None
        or re.search(plasmid, name) is not None
    )


def is_bacteria(name: str) -> bool:
    return ncbitax.is_bacteria(name) or has_bacterial_markers(name)


def main() -> None:
    with TinyDB(
        config["documents"], storage=CachingMiddleware(JSONStorage)
    ) as docdb:
        documents = docdb.table("documents")
        strains = docdb.table("strains")
        non_bacterial_strains = docdb.table("non_bacterial_strains")

        strains_to_remove = []
        for strain in tqdm(strains):
            if strain["taxon"] and strain["taxon"]["lpsn"]:
                continue
            if not any(is_bacteria(name) for name in strain_queries(strain)):
                non_bacterial_strains.insert(strain)
                strains_to_remove.append(strain.doc_id)

        strains.remove(doc_ids=strains_to_remove)
        print(
            f"Moved {len(strains_to_remove)} strain entries to the"
            '"non_bacterial_strains" table.'
        )

        for doc in tqdm(documents):
            docstrains: list[int] | None = doc.get("strains")
            if docstrains is not None:
                keep = []


if __name__ == "__main__":
    main()
