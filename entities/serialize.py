from collections.abc import Iterable, Sequence

from entities.utils import Token


def merge_off_tokens(tokens: Iterable[Token]) -> list[Token]:
    """
    Merge the BPE tokens in `tokens` and combine their labels accordingly.

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


def serialize_triples(tokens: Sequence[Token], source: str) -> str:
    output = (
        '<div prefix="ncbitaxon: http://purl.obolibrary.org/obo/NCBITaxon_">'
    )

    if not isinstance(tokens[0], Token):
        tokens = (Token(*tok) for tok in tokens)

    tokens = merge_off_tokens(tokens)

    entity_counter = 1
    offset = 0

    for token in tokens:
        space = " " * (token.offset[0] - offset)
        if token.prediction in ("#", "O"):
            output += space + token.string
        elif token.prediction.startswith("B-"):
            output += space + entity_string(token=token, ent_id=entity_counter)
            entity_counter += 1
        elif output.endswith("</span>"):
            output = output[:-7]
            output += f"{space}{token.string}</span>"
        else:
            # Here there is a discontinuity. Just use the last id.
            output += entity_string(token=token, ent_id=entity_counter)

        offset = token.offset[1]

    output += "</div>"

    return output


def entity_string(token: Token, ent_id: int) -> str:
    return (
        f'<span resource="#T{ent_id}" typeof="ncbitaxon:{token.prediction[2:]}">'
        f"{token.string}</span>"
    )
