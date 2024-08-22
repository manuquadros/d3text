import itertools
import os
import re
from collections.abc import Iterator
from copy import deepcopy
from typing import NamedTuple, Optional

from datamodel import Text, TextChunk
from lxml.etree import (XSLT, Element, XMLSyntaxError, XPathEvaluator,
                        _Element, _ElementTree, fromstring, iterwalk, parse,
                        tostring)
from nltk import RegexpTokenizer

from utils import safe_concat

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


def get_chunk(
    tree: _ElementTree, start: Optional[int] = None, end: Optional[int] = None
) -> str:
    if isinstance(tree, str | bytes):
        tree = fromstring(tree)

    if start is not None and end is not None:
        segs = get_segments(tree)[start:end]
    else:
        segs = get_segments(tree)

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
    start = 0
    stop = 0
    cursum = 0

    for seg in segments:
        length = len(tostring(seg, encoding="unicode", method="text").split())
        stop += 1

        if re.match(r"{.*}abstract", seg.tag):
            chunks.append(TextChunk(start=start, stop=stop))
            start = stop
            cursum = 0
        elif abs(cursum + length - len_threshold) > abs(cursum - len_threshold):
            chunks.append(TextChunk(start=start, stop=stop - 1))
            start = stop - 1
            cursum = length
        else:
            cursum += length

    match cursum:
        case 0:
            pass
        case n if n >= 150:
            chunks.append(TextChunk(start=start, stop=len(segments)))
        case _:
            chunks[-1].stop = len(segments)

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
    open_spans: list[_Element] = []
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

    xml = merge_children(promote_spans(xml))

    return tostring(xml, method="html", encoding="unicode")


def annotate_text(
    elem: _Element, text: str, open_spans: list[str], position: str
) -> tuple[_Element, list[_Element]]:
    new_spans: list[_Element] = []
    splits = re.findall(rf"{tag_pattern}|[^<>]+", text)

    # Reset `elem`'s text or tail. Those will be decided here.
    if position == "text":
        elem.text = ""
    else:
        elem.tail = ""

    context = elem

    for open_span in open_spans:
        subspan = deepcopy(open_span)
        if position == "text":
            context.insert(0, subspan)
        else:
            context.addnext(subspan)
        context = subspan
        position = "text"

    for split in splits:
        if split.startswith("<span"):
            new = Element("span", **attribs(split))
            if position == "text":
                context.append(new)
            else:
                context.getparent().append(new)
            context = new
            new_spans.append(deepcopy(new))
            position = "text"

        elif split == "</span>":
            position = "tail"
            if new_spans:
                new_spans.pop()
            else:
                open_spans.pop()

        elif split.startswith("<div"):
            div = Element("div", **attribs(split))
            for child in elem.getchildren():
                div.append(child)
            elem.append(div)
            position = "text"

        elif split == "</div>":
            pass

        else:
            if position == "text":
                context.text = safe_concat(context.text, split)
            else:
                context.tail = safe_concat(context.tail, split)

    return context, open_spans + new_spans


def promote_spans(tree: _Element | _ElementTree) -> _Element | _ElementTree:
    for node in tree.iter():
        if node.tag == "span":
            promote_span(node)

    return tree


def promote_span(span: _Element) -> None:
    parent = span.getparent()
    while (
        parent is not None
        and len(parent) == 1
        and not parent.text
        and not parent.tail
    ):
        newspan = deepcopy(span)
        parent.remove(span)

        newspan.tail, parent.tail = parent.tail, None
        parent.text, newspan.text = newspan.text, None
        for child in newspan.getchildren():
            parent.append(child)

        parent.getparent().replace(parent, newspan)
        newspan.append(parent)


def merge_children(tree: _Element | _ElementTree) -> _Element | _ElementTree:
    for node in tree.iter():
        for cursor in range(len(node) - 1, 0, -1):
            current = node[cursor]
            preceding = node[cursor - 1]
            if (
                current.tag == preceding.tag
                and current.attrib == preceding.attrib
                and not preceding.tail
            ):
                node.replace(preceding, merge_nodes(preceding, current))
                node.remove(current)

    return tree


def merge_nodes(left: _Element, right: _Element) -> _Element:
    new = Element(left.tag, left.attrib)

    for child in left:
        new.append(child)
    new.text = left.text

    try:
        new[-1].tail = right.text
    except IndexError:
        new.text = safe_concat(new.text, right.text)

    for child in right:
        new.append(child)

    new.tail = right.tail

    return new


def attribs(string: str) -> dict[str, str]:
    invalid_chars = r"\"'>\/=\x00-\x1f\x7f-\x9f"
    attribute = rf"([^ {invalid_chars}]+)"
    value = rf"[\"\']([^{invalid_chars}]+)[\"\']"
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
