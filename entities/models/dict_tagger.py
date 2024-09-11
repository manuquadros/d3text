import re
from collections.abc import Iterable, Iterator
from difflib import get_close_matches
from functools import reduce

from utils import Token, repr_sequence, token_merge


class DictTagger:
    def __init__(self, vocab: Iterable[str] | str, label: str) -> None:
        self._label = label

        if isinstance(vocab, str):
            with open(vocab, "r") as f:
                self._vocab = tuple(line.strip() for line in f)
        else:
            self._vocab = tuple(vocab)

        self.prefixes = tuple(
            " ".join(split_term[:ix])
            for split_term in (term.split() for term in vocab)
            for ix in range(1, len(split_term) + 1)
        )

    def tag(self, sequence: Iterable[Token]) -> Iterator[Token]:
        buffer: list[Token] = []

        for token in sequence:
            if token.prediction == "O":
                if re.match(r".\w", token.string) and get_close_matches(
                    repr_sequence(buffer + [token]), self.prefixes, cutoff=0.9
                ):
                    token = token._replace(prediction=self._label)
                    buffer.append(token)
                else:
                    if buffer:
                        yield reduce(token_merge, buffer)
                        buffer = []
                    yield token
            else:
                yield reduce(token_merge, buffer)
                buffer = []
                yield token

        if buffer:
            yield reduce(token_merge, buffer)
