from collections.abc import Iterable, Sequence

from utils import Token, merge_off_tokens


def serialize_triples(tokens: Sequence[Token]) -> str:
    output = (
        '<div prefix="ncbitaxon: http://purl.obolibrary.org/obo/NCBITaxon_">'
    )

    tokens = merge_off_tokens(tokens)

    entity_counter = 0
    entity_map: dict[tuple[str, str], str] = {}
    last_entity_type = ""
    current_res = ""
    gap = 0
    offset = 0

    for token in tokens:
        space = " " * (token.offset[0] - offset)
        if token.prediction in ("#", "O"):
            output += space + token.string
            gap += 1
        elif token.prediction.startswith("B-"):
            gap = 0
            current_res = entity_map.get((token.string, token.prediction), "")
            if current_res:
                output += space + entity_string(token=token, ent_id=current_res)
            else:
                entity_counter += 1
                entity_map[(token.string, token.prediction)] = (
                    f"#T{entity_counter}"
                )
                output += space + entity_string(
                    token=token, ent_id=entity_counter
                )
            last_entity_type = token.prediction[2:]
        elif output.endswith("</span>"):
            output = output[:-7]
            output += f"{space}{token.string}</span>"
        else:
            # Here there is a discontinuity. Just use the last id if the type matches
            # and if the last entity is not too far.
            if (
                current_res
                and token.prediction[2:] == last_entity_type
                and gap <= 3
            ):
                output += space + entity_string(token=token, ent_id=current_res)
            else:
                last_entity_type = token.prediction[2:]
                entity_counter += 1
                output += space + entity_string(
                    token=token, ent_id=entity_counter
                )
            gap = 0

        offset = token.offset[1]

    output += "</div>"

    return output


def entity_string(token: Token, ent_id: str | int) -> str:
    if isinstance(ent_id, int):
        ent_id = f"#T{ent_id}"
    return (
        f'<span class="entity" resource="{ent_id}" '
        f'typeof="ncbitaxon:{token.prediction[2:]}">'
        f"{token.string}</span>"
    )
