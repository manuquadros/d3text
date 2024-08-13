import itertools
import os
import re
from collections.abc import Iterator
from typing import NamedTuple

from datamodel import Text, TextChunk
from lxml.etree import (XSLT, Element, XMLSyntaxError, XPathEvaluator,
                        _Element, _ElementTree, fromstring, iterwalk, parse,
                        tostring)
from nltk import RegexpTokenizer

xml_char_tokenizer = RegexpTokenizer(r"<[\w/][^<>]*/?>|.")
open_tag = r"<\w[^<>]*>"
closed_tag = r"</[^<>]*>"
tag_pattern = open_tag + "|" + closed_tag
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


def remove_tags(xml: str) -> str:
    return "".join(text_tokenizer.tokenize(xml))


def tokenize_xml(xml: str) -> str:
    return xml_char_tokenizer.tokenize(xml)


def reinsert_tags(text: str, xml: _Element | _ElementTree | str) -> str:
    if isinstance(xml, str):
        xml = fromstring(xml)

    text = chars(text)
    open_spans: list[str] = []
    original_elements = tuple(xml.iter())

    for event, elem in iterwalk(xml, events=("start", "end")):
        if elem in original_elements:
            if event == "start" and elem.text is not None:
                segment = "".join(itertools.islice(text, len(elem.text)))
                elem, open_spans = annotate_text(
                    elem, segment, open_spans, "text"
                )
            elif event == "end" and elem.tail is not None:
                segment = "".join(itertools.islice(text, len(elem.tail)))
                elem, open_spans = annotate_text(
                    elem, segment, open_spans, "tail"
                )

    return tostring(xml, method="xml", encoding="unicode")


def annotate_text(
    elem: _Element, text: str, open_spans: list[str], position: str
) -> tuple[_Element, list[str]]:
    new_spans: list[str] = []
    in_tag: bool
    splits = re.findall(rf"{tag_pattern}|[^<>]+", text)

    # Reset `elem`'s text or tail. Those will be decided here.
    if position == "text":
        elem.text = ""
    else:
        elem.tail = ""

    for split in splits:
        if split.startswith("<span"):
            new = Element("span", **attribs(split))
            elem.append(new)
            new_spans.append(split)
            in_tag = True
        elif split.startswith("</span"):
            in_tag = False
            if new_spans:
                new_spans.pop()
            else:
                open_spans.pop()
        elif split.startswith("</") and split.endswith(">"):
            in_tag = False
        elif split.startswith("<") and split.endswith(">"):
            new = Element(tag(split), **attribs(split))
            elem.append(new)
            in_tag = True
        else:
            try:
                if in_tag:
                    new.text = split
                else:
                    new.tail = split
            except NameError:
                if position == "text":
                    elem.text = split
                else:
                    elem.tail = split

    return elem, open_spans + new_spans


def tag(string: str) -> str:
    _match = re.match(r"<([\w_:][\w\d_:\.-]*)", string)
    if _match:
        return string[_match.start() : _match.end()]
    else:
        return ""


def attribs(string: str) -> dict[str, str]:
    attribute = r"([^ \"'>\/=\x00-\x1f\x7f-\x9f]+)"
    value = rf"[\"\']{attribute}[\"\']"
    return dict(re.findall(rf"{attribute}={value}", string))


def chars(text: str) -> Iterator[str]:
    """Iterate over `text` returning every character plus any XML/HTML tag
    that immediately precedes it.

    If a character is followed by a </span>, return it along with the character
    as well.
    """

    tag_char = rf"({open_tag})|({closed_tag})|(.)"
    current: list[str] = []

    for split in re.findall(tag_char, text):
        last_aint_tag = current and re.match(tag_pattern, current[-1]) is None
        match split:
            case ("", s, ""):
                current.append(s)
                if last_aint_tag:
                    yield "".join(current)
                    current = []
            case (s, "", "") | ("", "", s):
                if last_aint_tag:
                    yield "".join(current)
                    current = []
                current.append(s)

    if current:
        yield "".join(current)


def split_metadata_body(xml: str) -> tuple[str, str]:
    xml_sans_chunk_tag = re.sub(r"</?chunk>", "", xml)
    splits = re.split(r"(</article-meta>)", xml_sans_chunk_tag)
    try:
        metadata = splits[0] + splits[1]
        body = splits[2]
        return metadata.strip(), body.strip()
    except IndexError:
        raise RuntimeError("Your XML does not have the expected format")
