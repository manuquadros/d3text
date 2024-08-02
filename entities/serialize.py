import itertools

import rdflib
from rdflib.namespace import SDO

from entities.utils import Token


def merge_off_tokens(tokens: list[Token]) -> list[Token]:
    """
    Merge the BPE tokens in `tokens` and combine the tags accordingly.

    The function will remove [CLS], [SEP] and [PAD] tokens.
    """
    merged_tokens: list[Token] = []

    for token in tokens:
        if token.string not in ("[SEP]", "[CLS]"):
            if token.string.startswith("##"):
                merged_tokens[-1] = token_merge(merged_tokens[-1], token)
            else:
                merged_tokens.append(token)

    return merged_tokens


def token_merge(a: Token, b: Token) -> Token:
    text = a.string + b.string[2:]
    offset = (
        (a.offset[0], b.offset[1])
        if a.offset is not None and b.offset is not None
        else None
    )
    return Token(text, offset, a.prediction, a.gold_label)


def serialize_triples(tokens: list[Token], source: str) -> str:
    output = (
        '<div prefix="ncbitaxon: http://purl.obolibrary.org/obo/NCBITaxon_>'
    )
    annotated_tokens = filter(
        lambda tk: tk.prediction not in ("#", "O"), merge_off_tokens(tokens)
    )

    entity_tokens = sorted(annotated_tokens, key=lambda ent: ent.offset)
    last_pos = 0
    counter = 1

    for entity in entity_tokens:
        output += source[last_pos : entity.offset[0]]
        output += (
            f'<span resource="#T{counter}" typeof="ncbitaxon:{entity.prediction}">'
            f"{source[entity.offset[0] : entity.offset[1]]}<\span>"
        )
        last_pos = entity.offset[1]
        counter += 1

    output += "</div>"

    """
    # TODO: implement overlapping annotations
    for entity in entiter:
        start = entity.offset[0]
        if start >= last_pos:
            output += source[last_pos : start]
            end = entity.offset[1]

            overlapping = [entity]
            entiter, nextiter = itertools.tee(entiter)
            for nextent in nextiter:
                if nextent.offset[0] < end:
                    overlapping.append(nextent)
                    end = nextent.offset[1]

            last_pos = end
    """

    return output


def get_triples(tokens: list[Token], source: str) -> rdflib.Graph:
    entity_tokens = filter(
        lambda tk: tk.prediction not in ("#", "O"), merge_off_tokens(tokens)
    )
    graph = rdflib.Graph()
    for token in entity_tokens:
        string = source[token.offset[0] : token.offset[1]]
        graph.add(
            (
                rdflib.Literal(string),
                SDO.taxonRank,
                rdflib.Literal(token.prediction),
            )
        )

    return graph
