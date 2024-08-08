from entities.utils import Token


def merge_off_tokens(tokens: list[Token]) -> list[Token]:
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


def serialize_triples(tokens: list[Token], source: str) -> str:
    output = (
        '<div prefix="ncbitaxon: http://purl.obolibrary.org/obo/NCBITaxon_>'
    )
    tokens = merge_off_tokens(tokens)

    counter = 1

    for token in tokens:
        if token.prediction in ("#", "O"):
            output += token.string
        else:
            output += (
                f'<span resource="#T{counter}" typeof="ncbitaxon:{token.prediction}">'
                f"{token.string}<\span>"
            )
            counter += 1

    output += "</div>"

    return output
