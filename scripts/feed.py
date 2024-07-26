#!/usr/bin/env python

import xml.etree.ElementTree as ET
from argparse import ArgumentParser

from lxml import etree
from pydantic import BaseModel, PositiveInt
from sqlmodel import Session

from backend.db import Text, TextChunk, engine


def get_chunks(xml: str, len_threshold: int = 300) -> list[TextChunk]:
    title_path = (
        "//*[name()='article-meta']/*[name()='title-group']"
        "/*[name()='article-title']"
    )
    abstract_path = "//*[name()='abstract'][not(@abstract-type = 'toc')]"
    body_path = "//*[name()='body']//*[name()='p' or name()='title']"

    tree = etree.fromstring(xml)
    paragraphs = tree.xpath(title_path + "|" + abstract_path + "|" + body_path)

    chunks = []
    size = 0
    start = 0
    stop = 0
    for p in paragraphs:
        size += len(
            etree.tostring(p, encoding="unicode", method="text").split()
        )
        stop += 1
        if 0 < len_threshold < size:
            chunks.append(TextChunk(start=start, stop=stop))
            start = stop
            size = 0

    if not chunks or size > 0:
        chunks.append(TextChunk(start=start, stop=stop))

    return chunks


def parse_file(file: str) -> ET.ElementTree:
    try:
        tree = etree.parse(file)
    except etree.XMLSyntaxError:
        print(f"{file} could not be parsed")
    return tree


def get_pmid(tree: ET.ElementTree) -> int:
    pmid_path = "//*[name()='article-id'][@pub-id-type='pmid'][1]"

    try:
        return int(tree.xpath(pmid_path)[0].text)
    except IndexError:
        raise ValueError("Could not find a PMID.")


def get_doi(tree: ET.ElementTree) -> str:
    doi_path = "//*[name()='article-id'][@pub-id-type='doi'][1]"
    try:
        return tree.xpath(doi)[0].text
    except IndexError:
        raise ValueError("Could not find a DOI.")


def store(file: str) -> None:
    tree = parse_file(file)
    content = etree.tostring(
        tree, encoding="unicode", method="c14n2", strip_text=True
    )
    chunks = get_chunks(content)

    try:
        text = TextBase(
            pmid=get_pmid(tree),
            doi=get_doi(tree),
            content=content,
        )
    except ValueError as error:
        error.add_note(f"File: {file}")
        raise

    with Session(engine) as session:
        session.add(text)
        session.commit()
        session.refresh(text)
        for chunk in chunks:
            chunk.source = text.id
            session.add(chunk)
        session.commit()


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("filenames", nargs="+")
    args = parser.parse_args()
    for file in args.filenames:
        p = store(file)
