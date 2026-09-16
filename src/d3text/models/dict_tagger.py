import math
import os
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import reduce
from itertools import takewhile
from typing import cast

from rapidfuzz import fuzz, process

from d3text.constraints import FuzzyScore
from d3text.schema import Schema
from d3text.surface_forms import is_symbol_like, length_band_ratios
from d3text.utils import Token, repr_sequence, token_merge


@dataclass(frozen=True, slots=True)
class VocabMatch:
    """A dictionary hit: which term fired, and how well it scored."""

    term: str
    score: float


AMBIGUOUS = "AMBIGUOUS"
"""`Token.prediction` for a span more than one wordlist matched equally well.

Distinct from `"O"`, which says no wordlist matched at all: a consumer
excluding ambiguous spans from its targets can only do so if a match that
happened is still recorded as one. The tied labels are in
`Token.candidate_labels`.
"""


@dataclass(frozen=True, slots=True)
class SpanMatch:
    """A token span, and every label whose wordlist matched it best.

    Keyed by label because two wordlists can score one span identically and
    nothing here can say which is right; picking one by construction order
    would only make the arbitrary answer reproducible.
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

_TRIGRAM_Q = 3
"""Q-gram length for `Vocab`'s pre-rapidfuzz blocking index."""


def _trigram_counts(term: str, q: int = _TRIGRAM_Q) -> Counter[str]:
    """`term`'s q-grams, with multiplicity.

    :param term: the (already normalized) string to shingle.
    :param q: the q-gram length.
    :return: each q-gram mapped to how many times it occurs in `term`.
    """
    return Counter(term[i : i + q] for i in range(len(term) - q + 1))


def _indel_distance_bound(
    query_length: int, term_length: int, cutoff: FuzzyScore
) -> int:
    """Greatest InDel distance a term can have from the query and reach `cutoff`.

    `QRatio` is `200 * M / (len_a + len_b)` with `M` an LCS length, so
    reaching `cutoff` bounds how many single-character inserts/deletes can
    separate the two strings.

    :param query_length: the query's length.
    :param term_length: the candidate term's length.
    :param cutoff: the score a match has to reach.
    :return: the bound; can be negative, meaning no term of this length pair
        can reach `cutoff` at all.
    """
    total = query_length + term_length
    return total - 2 * math.ceil(cutoff * total / 200.0)


def _min_shared_trigrams(
    query_length: int, term_length: int, cutoff: FuzzyScore, q: int = _TRIGRAM_Q
) -> int:
    """Fewest q-grams a term of `term_length` must share with the query.

    A sound lower bound, mirroring `length_band_ratios`: each single-character
    insert/delete can destroy at most `q` positional q-grams, so a term below
    this many shared q-grams (counted with multiplicity) cannot reach `cutoff`
    and is safe to skip.

    :param query_length: the query's length.
    :param term_length: the candidate term's length.
    :param cutoff: the score a match has to reach.
    :param q: the q-gram length.
    :return: the minimum shared-q-gram count a survivor must reach; 0 means no
        term can be excluded on this basis.
    """
    indel_distance_max = _indel_distance_bound(
        query_length, term_length, cutoff
    )
    return max(
        0, (min(query_length, term_length) - q + 1) - q * indel_distance_max
    )


def _normalize(term: str) -> str:
    """Punctuation to spaces, one character in for one character out.

    `MMP-3` and `MMP 3` are the same enzyme written two ways. Punctuation is
    replaced rather than deleted so the words on either side stay separate
    words.
    """

    return _PUNCTUATION.sub(" ", term)


@dataclass(frozen=True, slots=True)
class _Population:
    """One scoring regime's terms, bucketed by the length that is scored.

    Keyed by the *processed* length, since the cutoff-derived band bounds the
    lengths `QRatio` sees. `scored` and `surface` are parallel per bucket so
    the search space stays the lazy chain rapidfuzz iterates fastest; zipping
    them into pairs up front costs about 2.5x per window.
    """

    fold_case: bool
    scored: Mapping[int, tuple[str, ...]]
    surface: Mapping[int, tuple[str, ...]]
    _trigram_index_cache: dict[int, dict[str, dict[int, int]]]
    """Bucket length -> q-gram -> {local index in `scored[length]`: count}.

    Built lazily per bucket by `_trigram_index`, not by `build()`: a `Vocab`'s
    length-band pruning means most queries only ever land in a small fraction
    of the vocabulary's buckets, so indexing every bucket up front pays for
    buckets a run may never probe. Empty out of `build()`, populated (and
    memoized) on first probe of each bucket.
    """
    min_upper_count: int
    max_upper_count: int
    """Least/most uppercase letters any one term in `scored` carries.

    For a folded population every key is lowercase, so both are always 0 and
    `admits_case_shape` becomes a no-op — case genuinely cannot rule anything
    out there, only in the unfolded (symbol) population.
    """

    @classmethod
    def build(cls, terms: Iterable[str], fold_case: bool) -> "_Population":
        # Accumulated rather than grouped: consecutive-run grouping would
        # require the caller to hand over a length-sorted iterable, which
        # nothing at the call site says and nothing here could enforce.
        scored: defaultdict[int, list[str]] = defaultdict(list)
        surface: defaultdict[int, list[str]] = defaultdict(list)
        upper_counts: list[int] = []

        for term in terms:
            key = _normalize(term)
            if fold_case:
                key = key.lower()
            scored[len(key)].append(key)
            surface[len(key)].append(term)
            upper_counts.append(sum(1 for c in key if c.isupper()))

        return cls(
            fold_case=fold_case,
            scored={length: tuple(keys) for length, keys in scored.items()},
            surface={
                length: tuple(entries) for length, entries in surface.items()
            },
            _trigram_index_cache={},
            min_upper_count=min(upper_counts, default=0),
            max_upper_count=max(upper_counts, default=0),
        )

    def admits_case_shape(self, query_upper_count: int, max_indel: int) -> bool:
        """Whether some term's uppercase-letter count could still be in reach.

        For any two strings, `|upper(a) - upper(b)| <= edit_distance(a, b)`
        (an insert/delete changes an uppercase count by at most one), so a
        gap wider than `max_indel` rules out every term in this population
        before `Vocab._candidates` normalizes or trigrams the query.

        :param query_upper_count: uppercase letters in the case-appropriate
            query string.
        :param max_indel: the greatest edit distance a term in the current
            candidate length band could have and still reach `cutoff`.
        :return: False only when no term's uppercase count could be in reach.
        """
        if not self.scored:
            return False

        return (
            self.min_upper_count <= query_upper_count + max_indel
            and query_upper_count <= self.max_upper_count + max_indel
        )

    def _trigram_index(self, length: int) -> Mapping[str, Mapping[int, int]]:
        """The q-gram inversion for bucket `length`, building it on first use.

        :param length: the bucket to index.
        :return: q-gram -> {local index in `scored[length]`: count}.
        """
        cached = self._trigram_index_cache.get(length)
        if cached is not None:
            return cached

        by_trigram: defaultdict[str, dict[int, int]] = defaultdict(dict)
        for local_index, key in enumerate(self.scored.get(length, ())):
            for trigram, count in _trigram_counts(key).items():
                by_trigram[trigram][local_index] = count

        index = dict(by_trigram)
        self._trigram_index_cache[length] = index
        return index

    def shared_trigram_counts(
        self, length: int, query_counts: Mapping[str, int]
    ) -> dict[int, int]:
        """Local index -> q-grams shared with the query, for bucket `length`.

        Multiset intersection (`min` of each side's count), which is what the
        `_min_shared_trigrams` bound is derived against. A candidate absent
        from the result shares no q-gram with the query at all.

        :param length: the bucket to probe.
        :param query_counts: the query's q-gram counts, from `_trigram_counts`.
        :return: each candidate's shared-q-gram count, omitting zeros.
        """
        if length not in self.scored:
            return {}
        index = self._trigram_index(length)

        shared: dict[int, int] = {}
        for trigram, query_count in query_counts.items():
            postings = index.get(trigram)
            if postings is None:
                continue
            for local_index, term_count in postings.items():
                shared[local_index] = shared.get(local_index, 0) + min(
                    query_count, term_count
                )
        return shared


class Vocab:
    def __init__(
        self,
        label: str,
        vocab: str | os.PathLike[str] | Iterable[str],
        cutoff: FuzzyScore,
    ) -> None:
        self.label = label
        self.cutoff = cutoff

        # A str or any os.PathLike names a wordlist file; anything else
        # iterable is the wordlist itself.
        if isinstance(vocab, (str, os.PathLike)):
            # A blank line in a line-separated wordlist is a separator, not a
            # term: kept, it enters the vocabulary as "", which no query can
            # ever match but which is still scored once per prefix window
            # whenever the cutoff is degenerate enough to disable the length
            # prune, and which counts as an entry everywhere the index is
            # measured.
            with open(vocab, "r") as f:
                vocab = [term for line in f if (term := line.strip())]

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
        self._length_ratios = length_band_ratios(cutoff)

        # Both bounds below depend only on query_length (plus state fixed for
        # the life of this Vocab), so a query_length that recurs across the
        # document — common, since window length varies with token count, not
        # text length — hits the cache instead of re-deriving the band.
        self._candidate_lengths_cache: dict[int, tuple[int, ...]] = {}
        self._max_indel_cache: dict[int, int | None] = {}

    def _candidate_lengths(self, query_length: int) -> tuple[int, ...]:
        """Bucket keys that could still hold a term reaching `cutoff`.

        The bounds are rounded outwards: scoring a term that cannot clear the
        cutoff costs time, while skipping one that could is a silent miss.
        Memoized per `query_length`, since the answer never changes for a
        given `Vocab`.
        """

        cached = self._candidate_lengths_cache.get(query_length)
        if cached is not None:
            return cached

        if self._length_ratios is None:
            result = tuple(self._lengths)
        else:
            shortest, longest = self._length_ratios
            low = math.floor(query_length * shortest)
            high = math.ceil(query_length * longest)
            result = tuple(
                length for length in self._lengths if low <= length <= high
            )

        self._candidate_lengths_cache[query_length] = result
        return result

    def _max_indel_distance(self, query_length: int) -> int | None:
        """Most generous InDel budget any in-band term could be scored under.

        `_indel_distance_bound` only trends upward with total length — the
        `ceil` inside it makes it non-monotonic step to step — so the band's
        longest length alone is not a safe stand-in for the true max; every
        in-band length is checked. None mirrors `_candidate_lengths`' own
        escape hatch for a degenerate cutoff. Memoized per `query_length`,
        same reasoning as `_candidate_lengths`.

        :param query_length: the query's length.
        :return: the bound, or None when pruning is disabled entirely.
        """
        if query_length in self._max_indel_cache:
            return self._max_indel_cache[query_length]

        if self._length_ratios is None:
            result = None
        else:
            shortest, longest = self._length_ratios
            low = max(0, math.floor(query_length * shortest))
            high = math.ceil(query_length * longest)
            result = max(
                _indel_distance_bound(query_length, term_length, self.cutoff)
                for term_length in range(low, high + 1)
            )

        self._max_indel_cache[query_length] = result
        return result

    def _candidates(
        self, population: _Population, probe: str
    ) -> tuple[list[str], list[tuple[int, int]]]:
        """`population`'s terms still eligible to score `probe`.

        Length-band pruned as `_candidate_lengths` decides, then further
        pruned per bucket by `_min_shared_trigrams` — a term below the shared
        q-gram bound cannot reach `cutoff`, so it is never handed to
        rapidfuzz. Both prunes are score-preserving: a survivor's rapidfuzz
        score is unaffected by having been pre-filtered.

        :param population: the half of the wordlist to search.
        :param probe: the case-appropriate query string.
        :return: the eligible scored strings, and their `(length, local
            index)` so a rapidfuzz hit index can be traced to a surface form.
        """
        query_length = len(probe)
        lengths = [
            length
            for length in self._candidate_lengths(query_length)
            if length in population.scored
        ]

        # The trigram filter shares the length band's escape hatch: a
        # degenerate cutoff disables both rather than only one of them.
        query_counts = (
            _trigram_counts(probe) if self._length_ratios is not None else None
        )

        terms: list[str] = []
        locations: list[tuple[int, int]] = []
        for length in lengths:
            bucket = population.scored[length]
            threshold = (
                0
                if query_counts is None
                else _min_shared_trigrams(query_length, length, self.cutoff)
            )
            indices: Iterable[int]
            if query_counts is None or threshold <= 0:
                indices = range(len(bucket))
            else:
                shared = population.shared_trigram_counts(length, query_counts)
                indices = (
                    local_index
                    for local_index, count in shared.items()
                    if count >= threshold
                )
            for local_index in indices:
                terms.append(bucket[local_index])
                locations.append((length, local_index))

        # rapidfuzz's `limit=1` keeps whichever tied choice it sees first, so
        # a score tie must be broken by something other than the order terms
        # were appended above (bucket order, in turn wordlist order).
        # Sorting by the scored text itself is stated and file-order-free.
        order = sorted(range(len(terms)), key=lambda i: terms[i])
        terms = [terms[i] for i in order]
        locations = [locations[i] for i in order]

        return terms, locations

    def match(self, tk: Token | tuple[Token, ...]) -> VocabMatch | None:
        """Best wordlist entry for `tk`, or None if nothing reached `cutoff`.

        None rather than a zero score, since 0.0 is a score rapidfuzz really
        returns. The query is punctuation-normalized, and case-folded against
        the descriptive half of the wordlist only; the symbol half is scored
        with case intact.

        :param tk: the token or token span to match.
        :return: the single best-scoring term. A tie within one population
            (symbol or descriptive) is broken alphabetically by the scored
            text; a tie across the two populations still favors the symbol
            half, or None.
        """

        # A single Token is itself a NamedTuple, so `_fields` tells it apart
        # from a tuple of Tokens; cast because hasattr does not narrow for mypy.
        tokens = cast(
            "tuple[Token, ...]", (tk,) if hasattr(tk, "_fields") else tk
        )
        query = _normalize(repr_sequence(tokens))
        max_indel = self._max_indel_distance(len(query))

        best: tuple[str, float] | None = None
        for population in self._populations:
            probe = query.lower() if population.fold_case else query
            if max_indel is not None and not population.admits_case_shape(
                sum(1 for c in probe if c.isupper()), max_indel
            ):
                continue
            terms, locations = self._candidates(population, probe)
            found = process.extract(
                probe,
                terms,
                scorer=fuzz.QRatio,
                limit=1,
                score_cutoff=self.cutoff,
            )
            if not found:
                continue

            _, ratio, index = found[0]
            if best is None or ratio > best[1]:
                length, local_index = locations[index]
                best = population.surface[length][local_index], ratio

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
        cutoff: FuzzyScore = 93.0,
    ) -> None:
        self._vocabs = tuple(
            Vocab(label, vocab, cutoff) for label, vocab in vocabs.items()
        )

    @classmethod
    def from_schema(
        cls, schema: Schema, cutoff: FuzzyScore = 93.0
    ) -> "DictTagger":
        """Build a tagger from the entity types that declare a `vocab_path`.

        A type with no wordlist is a detectable class with nothing to match it
        against, so it is skipped rather than replaced by a hard-coded skip
        list.

        :param schema: declares the types and their wordlists.
        :param cutoff: the score a match has to reach.
        :return: the tagger.
        """

        vocabs: dict[str, os.PathLike[str]] = {
            entity_type.name: entity_type.vocab_path
            for entity_type in schema.entity_types
            if entity_type.vocab_path is not None
        }
        return cls(vocabs=vocabs, cutoff=cutoff)

    def tag(self, tokens: Sequence[Token]) -> Iterator[Token]:
        """Annotate the tokens no earlier stage has already labelled.

        :param tokens: the document's tokens.
        :return: the same tokens, wordlist matches applied.
        """

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
