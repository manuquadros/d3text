"""Generate a dataset of entity names from the document database.

Examples:
    {"term": "Tyrosine translase", "class": "d3o:Enzyme", "entid": "3494"}
    {"term": "Tyrosyl-tRNA ligase", "class": "d3o:Enzyme", "entid": "3494"}
    {"term": "Escherichia coli", "class": "d3o:Bacteria", "entid": "2026"}
    {"term": "ATCC 35896", "class": "d3o:Strain", "entid": 16526}
"""

import argparse
from pathlib import Path
from typing import Any

import orjson
from tinydb import TinyDB
from tinydb.middlewares import CachingMiddleware
from tinydb.storages import JSONStorage

from brenda_references.config import config


def get_terms(entity: dict[str, Any], table_name: str) -> tuple[str]:
    """Find the designations of `entity` depending on `table_name`."""
    match table_name:
        case "enzymes":
            return (entity["recommended_name"], *entity["synonyms"])
        case "bacteria":
            return (entity["organism"], *entity["synonyms"])
        case "strains":
            return (
                *entity["designations"],
                *(c["strain_number"] for c in entity["cultures"]),
            )
        case _:
            return ()


def main() -> None:  # noqa: D103
    args = argparse.ArgumentParser(
        prog="generate_entity_names_dataset.py",
        description="Generate a dataset of entity names from the document database",
    )
    args.add_argument("output_file")

    with (
        TinyDB(config["documents"], storage=CachingMiddleware(JSONStorage)) as docdb,
        Path(args.parse_args().output_file).open("wb") as output_file,
    ):

        def dump_table(table_name: str, label: str):
            for entity in docdb.table(table_name):
                output_file.writelines(
                    orjson.dumps(
                        {"term": term, "class": label, "entid": entity.doc_id},
                        option=orjson.OPT_APPEND_NEWLINE,
                    )
                    for term in get_terms(entity, table_name)
                )

        dump_table("enzymes", "d3o:Enzyme")
        dump_table("bacteria", "d3o:Bacteria")
        dump_table("strains", "d3o:Strain")
