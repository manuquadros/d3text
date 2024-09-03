import itertools
import os
import re
from collections.abc import Iterator
from copy import deepcopy
from typing import NamedTuple

from datamodel import Text, TextChunk
from lxml.etree import (XSLT, Element, QName, XMLSyntaxError, XPathEvaluator,
                        _Comment, _Element, _ElementTree,
                        _ProcessingInstruction, cleanup_namespaces, fromstring,
                        iterwalk, parse, tostring)
from nltk import RegexpTokenizer

from utils import concat, safe_concat

xml_char_tokenizer = RegexpTokenizer(r"<[\w/][^<>]*/?>|.")
open_tag = r"<\w[^<>]*>"
closed_tag = r"</[^<>]*>"

tag_pattern = open_tag + "|" + closed_tag
tag_tokenizer = RegexpTokenizer(tag_pattern)
text_tokenizer = RegexpTokenizer(tag_pattern, gaps=True)


def parse_file(file: str) -> _ElementTree:
    try:
        tree: _ElementTree = parse(file)
        return tree
    except XMLSyntaxError:
        print(f"{file} could not be parsed")


def tree_as_string(tree: _ElementTree | _Element) -> str:
    return tostring(tree, method="c14n2")


def get_text(tree: _ElementTree) -> Text:
    pmid = get_pmid(tree)
    doi = get_doi(tree)
    meta = get_metadata(tree)

    return Text(pmid=pmid, doi=doi, meta=meta)


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
    tree = transform_tree(tree)
    pathfinder: XPathEvaluator = XPathEvaluator(tree)
    abstract = "//*[@class='abstract']"

    non_metadata = "//*[@class = 'article-body']//"
    segtags = (
        "*[contains('ph2h3h4h5h6', name()) or name()='table-wrap' or name()='fig']"
    )
    body = non_metadata + segtags

    segments = pathfinder(abstract) + pathfinder(body)

    return segments


def get_metadata(tree: _ElementTree) -> str:
    pathfinder = XPathEvaluator(tree)
    metadata = pathfinder("//*[name()='journal-meta' or name()='article-meta']")
    return "\n".join(tostring(block, encoding="unicode").strip() for block in metadata)


def clean_namespaces(elem: _Element) -> _Element:
    for subelem in elem.getiterator():
        if not (
            isinstance(subelem, _Comment)
            or isinstance(subelem, _ProcessingInstruction)
        ):
            try:
                subelem.tag = QName(subelem).localname
            except ValueError as e:
                e.add_note(tostring(subelem).decode())
                raise

    cleanup_namespaces(elem)

    return elem


def segment_to_string(segment: _Element) -> str:
    segment = clean_namespaces(segment)

    return tostring(segment, method="xml", encoding="unicode")


def build_chunk(content: str, pos: int):
    return TextChunk(content=f"<chunk-body>{content}</chunk-body>", pos=pos)


def get_chunks(
    tree: _ElementTree, minlen: int = 4000, maxlen: int = 6000
) -> Iterator[TextChunk]:
    segments: Iterator[_Element] = iter(get_segments(tree))

    pos = itertools.count()
    yield build_chunk(content=segment_to_string(next(segments)), pos=next(pos))

    content = ""
    content_buffer = ""

    for seg in segments:
        segstring = segment_to_string(seg)

        if (
            seg.tag[-2:] in ("h1", "h2", "h3", "h4", "h5", "h6")
            and len(content) > minlen
        ):
            content_buffer = concat(content_buffer, segstring)
        elif not content_buffer and len(content) + len(segstring) <= maxlen:
            content = concat(content, segstring)
        else:
            content_buffer = concat(content_buffer, segstring)

        if len(content_buffer) >= minlen:
            yield build_chunk(content=content, pos=next(pos))
            content, content_buffer = content_buffer, ""

    content = concat(content, content_buffer)

    if content:
        yield build_chunk(content=content, pos=next(pos))


def transform_tree(tree: _ElementTree) -> _ElementTree:
    xslt_transform = XSLT(parse(os.path.join(os.path.dirname(__file__), "pubmed.xsl")))

    return xslt_transform(tree)


def transform_article(article_xml: str | bytes) -> str:
    if isinstance(article_xml, str):
        article_xml = article_xml.encode()

    try:
        tree = fromstring(article_xml)
    except XMLSyntaxError as e:
        e.add_note(article_xml)
        raise
    finally:
        return tostring(transform_tree(tree), pretty_print=True, encoding="unicode")


class Tag(NamedTuple):
    tag: str
    start: int


def remove_tags(xml: str) -> str:
    return "".join(text_tokenizer.tokenize(xml))


def tokenize_xml(xml: str) -> str:
    return xml_char_tokenizer.tokenize(xml)


def reinsert_tags(text: str, xml: _Element | _ElementTree | str) -> str:
    if isinstance(xml, str):
        xml = fromstring(xml).getroottree()

    textit = chars(text)

    open_spans: list[_Element] = []
    original_elements = tuple(xml.iter())

    for event, elem in iterwalk(xml, events=("start", "end")):
        # check if elem wasn't added in a previous annotation step
        if elem in original_elements:
            elem = clean_namespaces(elem)
            root = True if elem == xml.getroot() else False
            if event == "start" and elem.text is not None:
                segment = "".join(itertools.islice(textit, len(elem.text)))
                elem, open_spans = annotate_text(
                    elem, segment, open_spans, "text"
                )
                if root:
                    xml._setroot(elem)
            elif event == "end" and elem.tail is not None:
                segment = "".join(itertools.islice(textit, len(elem.tail)))
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

    context = elem

    # Reset `elem`'s text or tail. Those will be decided here.
    if position == "text":
        elem.text = ""
    else:
        elem.tail = ""

    for open_span in open_spans:
        subspan = deepcopy(open_span)
        if position == "text":
            context.insert(0, subspan)
        else:
            context.addnext(subspan)
        context = subspan
        position = "text"

    for split in splits:
        if split.startswith("<div") and split == splits[0]:
            div = Element("div", **attribs(split))

            root = elem
            while root.getparent() is not None:
                root = root.getparent()

            if root.tag == "chunk-body":
                root.append(div)
                for child in root:
                    if child != div:
                        div.append(child)
            elif root == elem:
                div.append(root)

            elem = div

        elif split.startswith("<span"):
            new = Element("span", **attribs(split))
            if position == "text":
                context.insert(0, new)
            else:
                context.addnext(new)
            context = new
            new_spans.append(deepcopy(new))
            position = "text"

        elif split == "</span>":
            position = "tail"
            if new_spans:
                new_spans.pop()
            else:
                open_spans.pop()

        elif split == "</div>":
            pass

        else:
            if position == "text":
                context.text = safe_concat(context.text, split)
            else:
                context.tail = safe_concat(context.tail, split)

    return elem, open_spans + new_spans


def promote_spans(tree: _ElementTree) -> _Element | _ElementTree:
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

        try:
            parent.getparent().replace(parent, newspan)
        except AttributeError:
            pass
        finally:
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
    invalid_chars = r"\"'<>=\x00-\x1f\x7f-\x9f"
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
