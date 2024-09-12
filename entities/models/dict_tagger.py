import re
from collections.abc import Iterable, Iterator, Sequence
from difflib import get_close_matches
from functools import reduce
from itertools import count, takewhile
from operator import itemgetter
from typing import Any

try:
    from cfuzzyset import cFuzzySet as FuzzySet
except ImportError:
    from fuzzyset import FuzzySet

from icecream import ic

from utils import Token, debug, debug_iter, repr_sequence, token_merge

type ScoredSequence = list[tuple[float, tuple[Token, ...]]]


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
        self.vocab_set = FuzzySet()
        for item in self._vocab:
            self.vocab_set.add(item)

    def match(self, tk: Token | tuple[Token, ...]) -> float:
        if hasattr(tk, "_fields"):
            tk = (tk,)

        try:
            matches = self.vocab_set.get(repr_sequence(tk))
            ratio, _ = matches[0]
        except TypeError:
            return 0.0

        if ratio >= self.cutoff:
            return ratio
        else:
            return 0.0


class DictTagger:
    def __init__(self, vocabs: dict[str, str | list[str]], cutoff=0.93) -> None:
        self._vocabs = tuple(
            Vocab(label, vocab, cutoff) for label, vocab in vocabs.items()
        )

    def tag(self, tokens: Sequence[Token]) -> Iterator[Token]:
        """Tokens that have not received a specific annotation may get one if
        they match one of the wordlists in self._vocab"""

        ix = 0
        tokens = tuple(tokens)
        while ix < len(tokens):
            token = tokens[ix]
            if token.prediction == "O":
                window = takewhile(
                    lambda j: j < len(tokens) and tokens[j].prediction == "O",
                    count(ix),
                )

                vocab_candidates: dict[str, dict[str, Any]] = {
                    v.label: {"max": 0.0, "sequence": []} for v in self._vocabs
                }

                for vocab in self._vocabs:
                    current = vocab_candidates[vocab.label]
                    candidates: list[tuple[float, tuple[Token, ...]]] = []
                    for jx in window:
                        score = vocab.match(tokens[ix : jx + 1])
                        if score:
                            current["sequence"].append(
                                (score, tokens[ix : jx + 1])
                            )
                            current["max"] = max(current["max"], score)

                label = max(
                    vocab_candidates.keys(),
                    key=lambda key: vocab_candidates[key]["max"],
                )
                candidates = vocab_candidates[label]["sequence"]
                # sorting by length (from longest to shortest) because `max` is
                # order-sensitive.
                candidates.reverse()

                try:
                    best = max(candidates, key=itemgetter(0))
                except ValueError:
                    yield token
                    ix += 1
                else:
                    # yield from self.dispatch(best[1], vocab)
                    merged = reduce(token_merge, best[1])._replace(
                        prediction=label
                    )
                    yield merged
                    ix += len(best[1])
            else:
                yield token
                ix += 1

    def dispatch(
        self, tokens: Sequence[Token], vocab: Vocab
    ) -> Iterator[Token]:
        if vocab.whole_match(tokens):
            merged = reduce(token_merge, tokens)
            yield merged._replace(prediction=vocab.label)
        else:
            for token in tokens:
                yield token
