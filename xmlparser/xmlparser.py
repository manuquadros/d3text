import os
import re
from collections.abc import Iterable, Iterator
from typing import NamedTuple

from datamodel import Text, TextChunk
from lxml.etree import (XSLT, XMLSyntaxError, XPathEvaluator, _Element,
                        _ElementTree, fromstring, parse, tostring)
from nltk import RegexpTokenizer

xml_char_tokenizer = RegexpTokenizer(r"<[\w/][^<>]*/?>|.")
tag_pattern = r"<[\w/][^<>]*>"
tag_tokenizer = RegexpTokenizer(tag_pattern)
text_tokenizer = RegexpTokenizer(tag_pattern, gaps=True)


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


def get_metadata(tree: _ElementTree) -> str:
    pathfinder = XPathEvaluator(tree)
    metadata = pathfinder("//*[name()='journal-meta' or name()='article-meta']")
    return "\n".join(
        tostring(block, encoding="unicode").strip() for block in metadata
    )


def get_chunk(tree: _ElementTree, start: int, end: int) -> str:
    if isinstance(tree, str | bytes):
        tree = fromstring(tree)
    segs = get_segments(tree)[start:end]
    return (
        "<chunk>\n"
        + get_metadata(tree)
        + "\n<chunk-body>\n"
        + "\n".join(tostring(seg, encoding="unicode").strip() for seg in segs)
        + "\n</chunk-body>\n</chunk>"
    )


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


def transform_article(article_xml: str) -> str:
    try:
        dom = fromstring(article_xml)
    except XMLSyntaxError as e:
        e.add_note(article_xml)
        raise

    # Do not load the xsl every time in production.
    xslt_transform = XSLT(
        parse(os.path.join(os.path.dirname(__file__), "pubmed.xsl"))
    )

    newdom = xslt_transform(dom)

    return tostring(newdom, pretty_print=True, encoding="unicode")


class Tag(NamedTuple):
    tag: str
    start: int


def get_tags(xml: str) -> Iterator[Tag]:
    """Return tags in `xml`, merging adjacent tags"""

    tags = iter(tag_tokenizer.tokenize(xml))
    offsets = tag_tokenizer.span_tokenize(xml)
    current_tag = next(tags)
    current_offset = next(offsets)

    try:
        while True:
            next_tag = next(tags)
            next_offset = next(offsets)
            if current_offset[1] == next_offset[0]:
                current_tag += next_tag
                current_offset = (current_offset[0], next_offset[1])
            else:
                yield Tag(tag=current_tag, start=current_offset[0])
                current_tag = next_tag
                current_offset = next_offset
    except StopIteration:
        pass
    finally:
        yield Tag(tag=current_tag, start=current_offset[0])


def remove_tags(xml: str) -> str:
    return "".join(text_tokenizer.tokenize(xml))


def tokenize_xml(xml: str) -> str:
    return xml_char_tokenizer.tokenize(xml)


def insert_tags(tags: Iterable[Tag], text: str) -> str:
    """Insert `tags` into their original positions in `text`"""
    result = ""
    counter = 0
    tags = list(tags)
    open_spans: list[str] = []

    print("\n", tags, "\n")

    for char in tokenize_xml(text):
        try:
            currtag = tags[0]
        except IndexError:
            currtag = Tag(tag="", start=-1)

        if char == ("</span>"):
            open_spans = open_spans[:-1]

        if currtag.start == counter:
            tags = tags[1:]
            if currtag.tag.startswith("</") and char.startswith("</"):
                result += char + currtag.tag
                counter += len(currtag.tag)
            else:
                counter += len(currtag.tag)
                if currtag.tag.startswith("</"):
                    result += (
                        len(open_spans) * "</span>"
                        + currtag.tag
                        + "".join(open_spans)
                    )
                    open_spans = open_spans[:-1]
                else:
                    result += currtag.tag + char
        else:
            result += char

        if len(char) == 1:
            counter += 1

        if char.startswith("<span"):
            open_spans.append(char)

    for tag in tags:
        result += tag.tag

    return result


def non_tag_chars(text: str) -> Iterator[str]:
    """Iterate over `text` returning every character plus any XML/HTML tag
    that immediately precedes it.
    """

    filler = ""

    for split in re.split(r"(" + tag_pattern + r")", text):
        if re.match(tag_pattern, split):
            filler += split
        else:
            for char in split:
                yield filler + char
                filler = ""

    if filler:
        yield filler


def split_metadata_body(xml: str) -> tuple[str, str]:
    splits = re.split(r"(</article-meta>)", xml)
    try:
        metadata = splits[0] + splits[1]
        body = splits[2]
        return metadata.strip(), body.strip()
    except IndexError:
        raise RuntimeError("Your XML does not have the expected format")
