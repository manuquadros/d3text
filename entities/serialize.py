import re
from collections.abc import Iterable, Sequence

from lxml.etree import fromstring, tostring

from utils import Token, merge_off_tokens


def serialize_triples(tokens: Iterable[Token]) -> str:
    output = '<div prefix="d3o: https://purl.dsmz.de/schema/">'

    tokens = merge_off_tokens(tokens)

    entity_counter = 0
    last_entity_type = ""
    gap = 0
    offset = 0

    for token in tokens:
        space = " " * (token.offset[0] - offset)
        if token.prediction in ("#", "O"):
            output += space + token.string
            gap += 1
        elif token.prediction.startswith("B-"):
            gap = 0
            entity_counter += 1
            output += space + entity_string(token=token, ent_id=entity_counter)
            last_entity_type = token.prediction[2:]
        elif output.endswith("</span>"):
            output = output[:-7]
            output += f"{space}{token.string}</span>"
        else:
            # Here there is a discontinuity. Just use the last id if the type matches
            # and if the last entity is not too far.
            if token.prediction[2:] == last_entity_type and gap <= 3:
                output += space + entity_string(
                    token=token, ent_id=entity_counter
                )
            else:
                last_entity_type = token.prediction[2:]
                entity_counter += 1
                output += space + entity_string(
                    token=token, ent_id=entity_counter
                )
            gap = 0

        offset = token.offset[1]

    output += "</div>"

    return merge_resources(output)


def entity_string(token: Token, ent_id: str | int) -> str:
    if isinstance(ent_id, int):
        ent_id = f"#T{ent_id}"

    label = re.sub(r"[BI]-", "", token.prediction)

    return (
        f'<span class="entity" resource="{ent_id}" '
        f'typeof="d3o:{label}">'
        f"{token.string}</span>"
    )


def merge_resources(xml: str) -> str:
    tree = fromstring(xml)
    entity_map: dict[tuple[str, str], str] = {}

    for elem in tree:
        if elem.tag == "span":
            typeof = elem.attrib.get("typeof")

            if typeof == "d3o:Bacteria":
                reduced_name = re.sub(r"(\w)\w+\.? (\w+)", r"\1. \2", elem.text)
            else:
                reduced_name = ""

            if (elem.text, typeof) in entity_map:
                elem.attrib["resource"] = entity_map[(elem.text, typeof)]
            elif (reduced_name, typeof) in entity_map:
                elem.attrib["resource"] = entity_map[(reduced_name, typeof)]
            else:
                entity_map[(elem.text, typeof)] = elem.attrib["resource"]
                if reduced_name:
                    entity_map[(reduced_name, typeof)] = elem.attrib["resource"]

    return tostring(tree, method="html", encoding="unicode")
