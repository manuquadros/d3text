import math
import os
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import reduce
from itertools import chain, takewhile
from typing import cast

from rapidfuzz import fuzz, process

from d3text.surface_forms import is_symbol_like
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


_PUNCTUATION = re.compile(r"[\W_]")


def _normalize(term: str) -> str:
    """Punctuation to spaces, one character in for one character out.

    `MMP-3` and `MMP 3` are the same enzyme written two ways, and a scorer
    comparing them raw puts them at 80. Punctuation is replaced rather than
    deleted so that the words on either side of it stay separate words; the
    length is left untouched as a side effect, but `Vocab` buckets terms by
    their processed length rather than resting on that.
    """

    return _PUNCTUATION.sub(" ", term)


@dataclass(frozen=True, slots=True)
class _Population:
    """One scoring regime's terms, bucketed by the length that is scored.

    Keying by the *processed* length rather than the raw one is what keeps
    `Vocab`'s cutoff-derived band sound: the band bounds `len(term)` against
    `len(query)` as `QRatio` sees them, so a bucket keyed by a length the
    scorer never sees would prune terms that clear the cutoff.

    `scored` and `surface` are parallel per bucket — same length, same order —
    so the search space stays the lazy chain of tuples rapidfuzz iterates
    fastest, and the surface form is recovered afterwards from the winner's
    position alone. Zipping them into pairs up front costs about 2.5x per
    window on a full wordlist, and `match` runs once per prefix window.
    """

    fold_case: bool
    scored: Mapping[int, tuple[str, ...]]
    surface: Mapping[int, tuple[str, ...]]

    @classmethod
    def build(cls, terms: Iterable[str], fold_case: bool) -> "_Population":
        # Accumulated rather than grouped: consecutive-run grouping would
        # require the caller to hand over a length-sorted iterable, which
        # nothing at the call site says and nothing here could enforce.
        scored: defaultdict[int, list[str]] = defaultdict(list)
        surface: defaultdict[int, list[str]] = defaultdict(list)

        for term in terms:
            key = _normalize(term)
            if fold_case:
                key = key.lower()
            scored[len(key)].append(key)
            surface[len(key)].append(term)

        return cls(
            fold_case=fold_case,
            scored={length: tuple(keys) for length, keys in scored.items()},
            surface={
                length: tuple(entries) for length, entries in surface.items()
            },
        )

    def term_at(self, lengths: Sequence[int], index: int) -> str:
        """The surface form behind `index` into the chained `lengths`."""

        for length in lengths:
            bucket = len(self.scored[length])
            if index < bucket:
                return self.surface[length][index]
            index -= bucket

        raise IndexError(f"{index} is past the end of the search space")


def _length_band_ratios(cutoff: float) -> tuple[float, float] | None:
    """Bounds on `len(term) / len(query)` for a term that can reach `cutoff`.

    `fuzz.QRatio` scores `200 * M / (len(a) + len(b))`, where `M` is the
    length of the longest common subsequence and so is at most
    `min(len(a), len(b))`. A term of length `t` therefore cannot score above
    `200 * min(t, q) / (t + q)` against a query of length `q`, and reaches
    exactly that when one string's characters are a subsequence of the
    other's. Requiring that ceiling to reach `cutoff` gives the inclusive
    band `q * cutoff / (200 - cutoff) <= t <= q * (200 - cutoff) / cutoff`.

    None asks for no pruning at all, which is what a degenerate cutoff gets:
    at or below 0 every term clears it, at or above 200 no term can, and
    neither has a finite band to divide out. Scoring a term that cannot win
    only costs time, so declining to prune is always the safe answer.
    """

    if not 0.0 < cutoff < 200.0:
        return None

    return cutoff / (200.0 - cutoff), (200.0 - cutoff) / cutoff


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

        symbols: list[str] = []
        descriptive: list[str] = []
        for term in vocab:
            (symbols if is_symbol_like(term) else descriptive).append(term)

        # The order is also the cross-half tie-break: `match` keeps the first
        # of two equal scores, and nothing here separates a symbol from a
        # descriptive name that scored the same.
        self._populations = (
            _Population.build(symbols, fold_case=False),
            _Population.build(descriptive, fold_case=True),
        )
        self._lengths = frozenset(
            length
            for population in self._populations
            for length in population.scored
        )

        # The band a cutoff implies is proportional to the query, so what is
        # fixed for the life of a Vocab is the ratio, not the band itself.
        self._length_ratios = _length_band_ratios(cutoff)

    def _candidate_lengths(self, query_length: int) -> Iterable[int]:
        """Bucket keys that could still hold a term reaching `cutoff`.

        The bounds are rounded outwards because the two errors are not
        symmetric: scoring a term that cannot clear the cutoff costs time,
        while skipping one that could is a silent miss no score can explain.
        """

        if self._length_ratios is None:
            return self._lengths

        shortest, longest = self._length_ratios
        low = math.floor(query_length * shortest)
        high = math.ceil(query_length * longest)

        return (length for length in self._lengths if low <= length <= high)

    def match(self, tk: Token | tuple[Token, ...]) -> VocabMatch | None:
        """Best wordlist entry for `tk`, or None if nothing reached `cutoff`.

        None rather than a zero score: 0.0 is a score rapidfuzz really
        returns, so a caller cannot tell "no candidate" from "scored 0.0" if
        both come back as a number.

        The query is punctuation-normalized before scoring, and case-folded
        as well against the descriptive half of the wordlist — `Catalase`
        scores 87.5 against `catalase` raw and so misses at any usable
        cutoff. The symbol half is scored with case intact; see
        `is_symbol_like`.

        Only the single best-scoring term comes back, and among equally
        scoring terms which one that is follows rapidfuzz's iteration order,
        the symbol half first.
        """

        # A single Token is itself a NamedTuple, so `_fields` tells it apart
        # from a tuple of Tokens; cast because hasattr does not narrow for mypy.
        tokens = cast(
            "tuple[Token, ...]", (tk,) if hasattr(tk, "_fields") else tk
        )
        query = _normalize(repr_sequence(tokens))

        best: tuple[str, float] | None = None
        for population in self._populations:
            probe = query.lower() if population.fold_case else query
            lengths = [
                length
                for length in self._candidate_lengths(len(probe))
                if length in population.scored
            ]
            found = process.extract(
                probe,
                chain.from_iterable(
                    population.scored[length] for length in lengths
                ),
                scorer=fuzz.QRatio,
                limit=1,
            )
            if not found:
                continue

            _, ratio, index = found[0]
            if best is None or ratio > best[1]:
                best = population.term_at(lengths, index), ratio

        if best is None or best[1] < self.cutoff:
            return None

        term, ratio = best

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
