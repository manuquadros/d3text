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


AMBIGUOUS = "AMBIGUOUS"
"""`Token.prediction` for a span more than one wordlist matched equally well.

Distinct from `"O"`, which says no wordlist matched at all: the two are
different facts, and a consumer that has to exclude ambiguous spans from its
targets can only do so if a match that happened is still recorded as one. The
tied labels are in `Token.candidate_labels`.
"""


@dataclass(frozen=True, slots=True)
class SpanMatch:
    """A token span, and every label whose wordlist matched it best.

    `matches` is keyed by label because two wordlists can score one span
    identically, and nothing here can say which of them is right — a strain
    designation and an enzyme abbreviation are not the same claim about the
    span. Picking one by the order the vocabularies were constructed would
    only make the arbitrary answer reproducible, so every tied label is kept
    and the span is marked ambiguous instead.
    """

    tokens: tuple[Token, ...]
    matches: Mapping[str, VocabMatch]

    @property
    def labels(self) -> frozenset[str]:
        return frozenset(self.matches)


# One vocabulary's best window over a span, before the vocabularies are
# compared with each other.
_Candidate = tuple[tuple[Token, ...], VocabMatch]


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
                    labels = best_match.labels
                    if len(labels) > 1:
                        prediction, tied = AMBIGUOUS, labels
                    else:
                        prediction, tied = next(iter(labels)), frozenset[str]()
                    merged = reduce(token_merge, best_match.tokens)._replace(
                        prediction=prediction, candidate_labels=tied
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
        def match_vocab(vocab: Vocab) -> _Candidate | None:
            best: _Candidate | None = None

            for i in range(1, min(len(tokens), 10) + 1):
                match = vocab.match(tuple(tokens[:i]))
                if match is None:
                    continue
                if best is None or match.score > best[1].score:
                    best = (tuple(tokens[:i]), match)

            return best

        def rank(candidate: _Candidate) -> tuple[float, int]:
            span, match = candidate
            return match.score, -len(span)

        candidates: dict[str, _Candidate] = {}
        for vocab in self._vocabs:
            found = match_vocab(vocab)
            if found is not None:
                candidates[vocab.label] = found

        if not candidates:
            return None

        # Score first and then the shorter span, which is the tie-break
        # match_vocab already applies within one vocabulary. What survives it
        # is the same span scored identically by two wordlists, and nothing
        # but the order the vocabularies were passed in could separate those,
        # so all of them are returned.
        best = max(rank(candidate) for candidate in candidates.values())
        winners = {
            label: candidate
            for label, candidate in candidates.items()
            if rank(candidate) == best
        }
        span, _ = next(iter(winners.values()))

        return SpanMatch(
            tokens=span,
            matches={label: match for label, (_, match) in winners.items()},
        )
