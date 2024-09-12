import re
from collections.abc import Iterable, Iterator
from difflib import SequenceMatcher, get_close_matches
from functools import lru_cache, reduce

from icecream import ic

from utils import Token, debug, debug_iter, repr_sequence, token_merge


class Vocab:
    def __init__(
        self, label: str, vocab: str | Iterable[str], cutoff: float
    ) -> None:
        self.label = label
        self.cutoff = cutoff

        if isinstance(vocab, str):
            with open(vocab, "r") as f:
                vocab = tuple(line.strip() for line in f)

        self._vocab = vocab

        self.prefixes = tuple(
            " ".join(split_term[:ix])
            for split_term in (term.split() for term in self._vocab)
            for ix in range(1, len(split_term) + 1)
        )

    def match(self, tk: Token | Iterable[Token]) -> float:
        if hasattr(tk, "_fields"):
            tk = (tk,)

        ratio = 0
        s = SequenceMatcher()
        s.set_seq2(repr_sequence(tk))

        for seq in self.prefixes:
            s.set_seq1(seq)
            if (
                s.real_quick_ratio() > self.cutoff
                and s.quick_ratio() > self.cutoff
            ):
                r = s.ratio()
                if r == 1:
                    return r
                if r > ratio:
                    ratio = r

        if ratio > self.cutoff:
            return ratio
        else:
            return 0

    def whole_match(self, tk: Iterable[Token]) -> bool:
        seq = repr_sequence(tk)
        matches = get_close_matches(seq, self._vocab, cutoff=self.cutoff)
        if matches:
            return True
        else:
            return False


class DictTagger:
    def __init__(self, vocabs: dict[str, str | list[str]], cutoff=0.93) -> None:
        self._vocabs = tuple(
            Vocab(label, vocab, cutoff) for label, vocab in vocabs.items()
        )

    def find_match(self, token: Token) -> tuple[float, Vocab | None]:
        for vocab in self._vocabs:
            ratio = vocab.match(token)
            if ratio:
                return ratio, vocab

        return 0, None

    def tag(self, sequence: Iterable[Token]) -> Iterator[Token]:
        """Tokens that have not received a specific annotation may get one if
        they match one of the wordlists in self._vocab"""

        buffer: list[Token] = []
        ratio = 0
        current_vocab: Vocab | None

        for token in sequence:
            if token.prediction == "O":
                if buffer:
                    match_ratio = current_vocab.match(buffer + [token])

                    if match_ratio >= ratio:
                        buffer.append(token)
                    else:
                        yield from self.dispatch(buffer, current_vocab)

                        ratio, current_vocab = self.find_match(token)

                        if ratio:
                            buffer = [token]
                        else:
                            yield token
                            buffer = []
                else:
                    ratio, current_vocab = self.find_match(token)
                    if ratio:
                        buffer = [token]
                    else:
                        yield token
            else:
                if buffer:
                    yield from self.dispatch(buffer, current_vocab)
                    buffer, ratio, current_vocab = [], 0, None
                yield token

        if buffer:
            yield from self.dispatch(buffer, current_vocab)

    def dispatch(self, tokens: list[Token], vocab: Vocab) -> Iterator[Token]:
        if vocab.whole_match(tokens):
            yield reduce(
                token_merge,
                (tk._replace(prediction=vocab.label) for tk in tokens),
            )
        else:
            for tk in tokens:
                yield tk
