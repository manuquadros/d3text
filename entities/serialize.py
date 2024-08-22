from collections.abc import Iterable, Sequence

from utils import Token, merge_off_tokens


def serialize_triples(tokens: Sequence[Token], source: str) -> str:
    output = (
        '<div prefix="ncbitaxon: http://purl.obolibrary.org/obo/NCBITaxon_">'
    )

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
            entity_counter += 1
            gap = 0
            output += space + entity_string(token=token, ent_id=entity_counter)
            last_entity_type = token.prediction[2:]
        elif output.endswith("</span>"):
            output = output[:-7]
            output += f"{space}{token.string}</span>"
        else:
            # Here there is a discontinuity. Just use the last id if the type matches
            # and if the last entity is not too far.
            if not (token.prediction[2:] == last_entity_type and gap <= 3):
                last_entity_type = token.prediction[2:]
                gap = 0
                entity_counter += 1
            output += entity_string(token=token, ent_id=entity_counter)

        offset = token.offset[1]

    output += "</div>"

    return output


def entity_string(token: Token, ent_id: int) -> str:
    return (
        f'<span resource="#T{ent_id}" typeof="ncbitaxon:{token.prediction[2:]}">'
        f"{token.string}</span>"
    )
