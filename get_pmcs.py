import json
import logging
import os
import urllib
import xml.etree.ElementTree as ET
from collections.abc import Iterable

import pandas as pd
import xmltodict
from Bio import Entrez
from tqdm import tqdm

import config

Entrez.email = config.entrez_email

references = pd.read_csv(config.references_file)

logging.basicConfig(
    filename="debug.log",
    encoding="utf-8",
    level=logging.DEBUG,
    format=" %(name)s :: %(levelname)s :: %(message)s",
)
logger = logging.getLogger("brenda")

try:
    with open(config.esummaries, "r") as summaries:
        esummaries = json.load(summaries)
except FileNotFoundError:
    esummaries = {}


def format_fields(fields: list[dict] | dict) -> dict:
    """
    Recursively merge all fields into a single dictionary.

    The function eliminates repeated '@Name' keys and '@Type' annotations.
    """
    if isinstance(fields, dict):
        return {fields["@Name"]: fields.get("#text", "")}
    else:
        return {
            field["@Name"]: format_fields(field["Item"])
            if "Item" in field
            else field.get("#text", "")
            for field in fields
        }


def parse_records(records: ET.Element) -> dict:
    """Collect all the article summaries, using their PubMed IDs as dictionary keys."""
    records = xmltodict.parse(ET.tostring(records, encoding="utf-8"))

    try:
        if "ERROR" in records["eSummaryResult"]:
            logger.info(records["eSummaryResult"]["ERROR"])
        records = records["eSummaryResult"]["DocSum"]
    except KeyError:
        return {}
    else:
        if isinstance(records, list):
            return {rec["Id"]: format_fields(rec["Item"]) for rec in records}
        else:
            return {records["Id"]: format_fields(records["Item"])}


def get_pmc(pubmed_id: str) -> str:
    """
    Retrieve PMC ID for a given PubMed ID.

    If the PubMed ID is not found in `config.esummaries`, the function will try to retrieve
    the article summary from Entrez, as long as `retrieve` is set to `True`.
    """
    if pubmed_id != "-":
        try:
            pmc = esummaries[pubmed_id]["ArticleIds"]["pmc"]
            return pmc.replace("PMC", "")
        except KeyError:
            logger.info(f"{pubmed_id} does not have a corresponding PMC")
    return ""


def check_ids(references: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure that we have all the PubMed IDs that we want.

    For IDs that are not already in `config.esummaries`, the function will retrieve the
    summaries from Entrez and update `config.esummaries`.
    """

    ids_to_retrieve = set(
        [
            pid
            for pid in references["pubmed_id"]
            if pid not in esummaries and pid != "-"
        ]
    )

    if ids_to_retrieve:
        print(f"Retrieving {len(ids_to_retrieve)} from Entrez.")

        with Entrez.esummary(
            db="pubmed", id=ids_to_retrieve
        ) as summariesHandle:
            records = parse_records(ET.parse(summariesHandle).getroot())
            if not records:
                print(
                    "The IDs were not retrievable. They will be removed and save in "
                    "unavailable_pubmed_ids.csv"
                )
                unavailable = references.query("pubmed_id in @ids_to_retrieve")
                unavailable.to_csv("unavailable_pubmed_ids.csv", index=False)

                return references.query("pubmed_id not in @ids_to_retrieve")

        with open(config.esummaries, "w") as esummaries_file:
            esummaries.update(records)
            json.dump(esummaries, esummaries_file, indent=1)
    else:
        print("No IDs to retrieve from Entrez.")

    return references


if __name__ == "__main__":
    print("Checking how many ids have to be retrieved from Entrez...")
    references = check_ids(references)

    tqdm.pandas(desc="Progress")
    references["pmc"] = references["pubmed_id"].progress_apply(get_pmc)
    references.to_csv(config.references_file, index=False)

    print(
        f"{len([v for v in esummaries.values() if 'pmc' in v['ArticleIds']])}"
        " articles with PMC IDs."
    )
