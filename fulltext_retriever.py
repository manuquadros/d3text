#!/usr/bin/env python3

import json
import logging
import os
import random
import time
import urllib
from collections.abc import Iterable

import pandas as pd
import requests
import retrying
from tqdm import tqdm

import config


def retry_if_too_many_requests(exception):
    print(
        "HTTP Error 429: Too Many Requests... We are retrying in a few seconds."
    )
    return isinstance(exception, urllib.error.HTTPError)


@retrying.retry(
    retry_on_exception=retry_if_too_many_requests,
    wait_random_min=30000,
    wait_random_max=300000,
)
def get_fulltext(pmc: str) -> str:
    """Retrieve the full-text of the article under the `pmc` identifier in XML."""

    url = (
        "https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi?verb=GetRecord&"
        f"identifier=oai:pubmedcentral.nih.gov:{pmc}&metadataPrefix=pmc"
    )
    request = requests.get(url, headers={"Accept-Encoding": "gzip, deflate"})
    request.encoding = "utf-8"

    return request.text


def nice_retriever(pmcs: Iterable) -> None:
    """
    Respectfully retrieves the full-text for the articles in `pmcs`.

    :param Iterable pmcs: List of PMC IDs to retrieve

    :rtype: None
    """

    counter = 0

    for pmc in tqdm(pmcs):
        filename = os.path.join(config.literature_folder, pmc + "_fulltext.xml")

        if not os.path.exists(filename):
            fulltext = get_fulltext(pmc)
            if '<error code="cannotDisseminateFormat">' not in fulltext:
                counter += 1
            with open(filename, "w", encoding="utf-8") as output_file:
                output_file.write(fulltext)
            time.sleep(random.randrange(30, 60))

    print(f"{counter} new article{'s' if counter != 1 else ''} retrieved.")


if __name__ == "__main__":
    references = pd.read_csv(config.references_file, dtype="str")
    references = references.dropna(subset="pmc").drop_duplicates(subset="pmc")
    print("References file loaded")
    nice_retriever(references["pmc"])
