#!/usr/bin/env python3

import logging
import json
import requests
import retrying

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


