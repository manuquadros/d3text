import os
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import reduce
from itertools import chain, takewhile
from typing import cast

from rapidfuzz import fuzz, process

from d3text.utils import Token, repr_sequence, token_merge


@dataclass(frozen=True, slots=True)
class VocabMatch:
    """A dictionary hit: which term fired, and how well it scored.

    `entity_ids` is the slot a linker fills in once the wordlists carry BRENDA
    identifiers; a bare wordlist has none, so it stays empty here. It is a set
    because a surface form is not owned by one entity — `AS-A` names four
    separate enzymes — and because a species nested in a strain designation is
    meant to yield both entities rather than force a choice at match time.
    """

    term: str
    score: float
    entity_ids: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class SpanMatch:
    """A `VocabMatch` together with the token span that produced it."""

    label: str
    tokens: tuple[Token, ...]
    match: VocabMatch


class Vocab:
    def __init__(
        self,
        label: str,
        vocab: str | os.PathLike[str] | Iterable[str],
        cutoff: float,
    ) -> None:
        self.label = label
        self.cutoff = cutoff

        # A str or any os.PathLike names a wordlist file; anything else
        # iterable is the wordlist itself.
        if isinstance(vocab, (str, os.PathLike)):
            with open(vocab, "r") as f:
                vocab = [line.strip() for line in f]

        # Accumulated rather than grouped: consecutive-run grouping would
        # require the caller to hand over a length-sorted iterable, which
        # nothing at the call site says and nothing here could enforce.
        buckets: defaultdict[int, list[str]] = defaultdict(list)
        for term in vocab:
            buckets[len(term)].append(term)

        self._vocab = {
            length: tuple(terms) for length, terms in buckets.items()
        }

    def match(self, tk: Token | tuple[Token, ...]) -> VocabMatch | None:
        """Best wordlist entry for `tk`, or None if nothing reached `cutoff`.

        None rather than a zero score: 0.0 is a score rapidfuzz really
        returns, so a caller cannot tell "no candidate" from "scored 0.0" if
        both come back as a number.

        Only the single best-scoring term comes back, and among equally
        scoring terms which one that is follows rapidfuzz's iteration order.
        """

        # A single Token is itself a NamedTuple, so `_fields` tells it apart
        # from a tuple of Tokens; cast because hasattr does not narrow for mypy.
        tokens = cast(
            "tuple[Token, ...]", (tk,) if hasattr(tk, "_fields") else tk
        )
        query = repr_sequence(tokens)
        search_space = chain.from_iterable(
            self._vocab[k]
            for k in self._vocab.keys()
            if abs(k - len(query)) <= 2
        )

        best_match = process.extract(
            query,
            search_space,
            scorer=fuzz.QRatio,
            limit=1,
        )
        if not best_match:
            return None

        term, ratio, _ = best_match[0]
        if ratio < self.cutoff:
            return None

        return VocabMatch(term=term, score=ratio)


class DictTagger:
    def __init__(
        self,
        # Mapping, not dict: dict is invariant in its value type, so a
        # dict[str, Path] would still be rejected by the widened union.
        vocabs: Mapping[str, str | os.PathLike[str] | Iterable[str]],
        cutoff: float = 93.0,
    ) -> None:
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
                    merged = reduce(token_merge, best_match.tokens)._replace(
                        prediction=best_match.label
                    )
                    yield merged
                    ix += len(best_match.tokens)
                else:
                    yield tokens[ix]
                    ix += 1
            else:
                yield tokens[ix]
                ix += 1

    def _find_best_match(self, tokens: Sequence[Token]) -> SpanMatch | None:
        def match_vocab(vocab: Vocab) -> SpanMatch | None:
            best: SpanMatch | None = None

            for i in range(1, min(len(tokens), 10) + 1):
                match = vocab.match(tuple(tokens[:i]))
                if match is None:
                    continue
                if best is None or match.score > best.match.score:
                    best = SpanMatch(vocab.label, tuple(tokens[:i]), match)

            return best

        candidates = [
            span for span in map(match_vocab, self._vocabs) if span is not None
        ]
        if not candidates:
            return None

        return max(candidates, key=lambda span: span.match.score)
