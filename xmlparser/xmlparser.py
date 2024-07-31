from datamodel import TextChunk
from lxml.etree import (XMLSyntaxError, XPathEvaluator, _Element, _ElementTree,
                        fromstring, parse, tostring)


def parse_file(file: str) -> _ElementTree:
    try:
        tree = parse(file)
    except XMLSyntaxError:
        print(f"{file} could not be parsed")
    return tree


def tree_as_string(tree: _ElementTree | _Element) -> str:
    return tostring(tree, method="c14n2")


def get_text(tree: _ElementTree) -> Text:
    pmid = get_pmid(tree)
    doi = get_doi(tree)

    if not doi:
        print(f"Couldn't find a DOI for PMID: {pmid}.")

    return Text(
        pmid=get_pmid(tree), doi=get_doi(tree), content=tree_as_string(tree)
    )


def get_pmid(tree: _ElementTree) -> int:
    pmid_path = "//*[name()='article-id'][@pub-id-type='pmid'][1]"

    try:
        return int(tree.xpath(pmid_path)[0].text)
    except IndexError:
        raise ValueError("Could not find a PMID.")


def get_doi(tree: _ElementTree) -> str:
    doi_path = "//*[name()='article-id'][@pub-id-type='doi'][1]"
    try:
        return tree.xpath(doi_path)[0].text
    except IndexError:
        return ""


def get_segments(tree: _ElementTree) -> list[_Element]:
    pathfinder = XPathEvaluator(tree)
    main_title = (
        "//*[name()='article-meta']/*[name()='title-group']"
        "/*[name()='article-title']"
    )
    abstract = "//*[name()='abstract'][not(@abstract-type = 'toc')]"
    body = "//*[name()='body']//*[name()='p' or name()='title']"

    segments: list[_Element] = (
        pathfinder(main_title) + pathfinder(abstract) + pathfinder(body)
    )

    return segments


def get_chunk(tree: _ElementTree, start: int, end: int) -> str:
    if isinstance(tree, str | bytes):
        tree = fromstring(tree)
    segs = get_segments(tree)[start:end]
    return "\n".join(tostring(seg, encoding="unicode") for seg in segs)


def get_chunks(tree: _ElementTree, len_threshold: int = 300) -> list[TextChunk]:
    segments = get_segments(tree)

    chunks = []
    size = 0
    start = 0
    stop = 0
    for p in segments:
        size += len(tostring(p, encoding="unicode", method="text").split())
        stop += 1
        if 0 < len_threshold < size:
            chunks.append(TextChunk(start=start, stop=stop))
            start = stop
            size = 0

    if not chunks or size > 0:
        chunks.append(TextChunk(start=start, stop=stop))

    return chunks
