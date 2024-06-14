#!/usr/bin/env python3

import logging
import json
import os
import requests
import retrying
import time

from tqdm import tqdm

import config


def retry_if_too_many_requests(exception):
    print("HTTP Error 429: Too Many Requests... We are retrying in a few seconds.")
    return isinstance(exception, urllib.error.HTTPError)


@retrying.retry(retry_on_exception=retry_if_too_many_requests,
                wait_random_min=2000,
                wait_random_max=4000)
def get_fulltext(pmc: str) -> str:
    """ Retrieve the full-text of the article under the `pmc` identifier in XML. """

    url = ('https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi?verb=GetRecord&'
           f'identifier=oai:pubmedcentral.nih.gov:{pmc}&metadataPrefix=pmc')
    request = requests.get(url, headers={'Accept-Encoding': 'gzip, deflate'})
    request.encoding = 'utf-8'

    return request.text


def nice_retriever(pmcs: list[str], delay: int = 60) -> None:
    """
    Respectfully retrieves the full-text for the articles in `pmcs`.

    :param list[str] pmcs: List of PMC IDs to retrieve
    :param int delay: Seconds to wait between each request

    :rtype: None
    """

    counter = 0
    
    for pmc in tqdm(pmcs):
        filename = os.path.join(config.literature_folder, pmc + '_fulltext.xml')

        if not os.path.exists(filename):
            fulltext = get_fulltext(pmc)
            if '<error code="cannotDisseminateFormat">' not in fulltext:
                counter += 1
            with open(filename, 'w', encoding='utf-8') as output_file:
                output_file.write(fulltext)
            time.sleep(delay)

    print(f"{counter} new article{'s' if counter != 1 else ''} retrieved.")
