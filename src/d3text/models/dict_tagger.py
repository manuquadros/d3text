from collections.abc import Iterable, Iterator, Sequence
from functools import reduce
from operator import itemgetter
from itertools import groupby, chain, takewhile

from rapidfuzz import fuzz, process

from d3text.utils import Token, repr_sequence, token_merge


class Vocab:
    def __init__(
        self, label: str, vocab: str | Iterable[str], cutoff: float
    ) -> None:
        self.label = label
        self.cutoff = cutoff

        if isinstance(vocab, str):
            with open(vocab, "r") as f:
                vocab = sorted((line.strip() for line in f), key=len)

        self._vocab = {
            length: tuple(terms) for length, terms in groupby(vocab, len)
        }

    def match(self, tk: Token | tuple[Token, ...]) -> float:
        if hasattr(tk, "_fields"):
            tk = (tk,)

        query = repr_sequence(tk)
        search_space = chain.from_iterable(
            self._vocab[k]
            for k in self._vocab.keys()
            if abs(k - len(query)) <= 2
        )

        try:
            best_match = process.extract(
                query,
                search_space,
                scorer=fuzz.QRatio,
                limit=1,
            )
            _, ratio, _ = best_match[0]
        except IndexError:
            return 0.0

        return ratio if ratio >= self.cutoff else 0.0


class DictTagger:
    def __init__(self, vocabs: dict[str, str | list[str]], cutoff=93) -> None:
        self._vocabs = tuple(
            Vocab(label, vocab, cutoff) for label, vocab in vocabs.items()
        )

    def tag(self, tokens: Sequence[Token]) -> Iterator[Token]:
        """Tokens that have not received a specific annotation may get one if
        they match one of the wordlists in self._vocab"""

        ix = 0
        tokens = tuple(tokens)
        while ix < len(tokens):
            if tokens[ix].prediction == "O":
                window = tuple(
                    takewhile(lambda tk: tk.prediction == "O", tokens[ix:])
                )
                best_match = self._find_best_match(window)
                if best_match:
                    label, score, matched_tokens = best_match
                    merged = reduce(token_merge, matched_tokens)._replace(
                        prediction=label
                    )
                    yield merged
                    ix += len(matched_tokens)
                else:
                    yield tokens[ix]
                    ix += 1
            else:
                yield tokens[ix]
                ix += 1

    def _find_best_match(
        self, tokens: Sequence[Token]
    ) -> tuple[str, float, tuple[Token, ...]] | None:
        def match_vocab(vocab):
            best_score = 0.0
            best_sequence = ()

            for i in range(1, min(len(tokens), 10) + 1):
                score = vocab.match(tokens[:i])
                if score > best_score:
                    best_score = score
                    best_sequence = tokens[:i]

            return vocab.label, best_score, best_sequence

        results = map(match_vocab, self._vocabs)

        best_match = max(results, key=itemgetter(1))
        return best_match if best_match[1] > 0 else None
