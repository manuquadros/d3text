"""Ported from the deleted src/tests/test_dict_tagger.py (entities layer).

Re-targeted at the live d3text.models.dict_tagger. DictTagger/Vocab are fully
pure — no disk, no model — so these run on CPU with no data or network.
"""

import pathlib
from dataclasses import FrozenInstanceError

import pytest
from rapidfuzz import fuzz

from d3text.models.dict_tagger import DictTagger, Vocab, VocabMatch
from d3text.schema import EntityType, Schema
from d3text.utils import Token, repr_sequence

sample = [
    Token(string="on", offset=(3475, 3477), prediction="O", gold_label=None),
    Token(string="the", offset=(3478, 3481), prediction="O", gold_label=None),
    Token(
        string="production",
        offset=(3482, 3492),
        prediction="O",
        gold_label=None,
    ),
    Token(string="of", offset=(3493, 3495), prediction="O", gold_label=None),
    Token(string="COX", offset=(3496, 3499), prediction="O", gold_label=None),
    Token(string=".", offset=(3499, 3500), prediction="O", gold_label=None),
]


def test_dict_tagger_merges_matching_span() -> None:
    expected = [
        Token(
            string="on", offset=(3475, 3477), prediction="O", gold_label=None
        ),
        Token(
            string="the", offset=(3478, 3481), prediction="O", gold_label=None
        ),
        Token(
            string="production of COX",
            offset=(3482, 3499),
            prediction="process",
            gold_label=None,
        ),
        Token(string=".", offset=(3499, 3500), prediction="O", gold_label=None),
    ]

    dtagger = DictTagger(
        vocabs={"process": ["production of COX", "enzyme activity alteration"]}
    )
    assert list(dtagger.tag(sample)) == expected


def test_dict_tagger_leaves_non_o_tokens_untouched() -> None:
    tokens = [
        Token(
            string="production",
            offset=(3482, 3492),
            prediction="enzyme",
            gold_label=None,
        ),
        Token(
            string="of", offset=(3493, 3495), prediction="O", gold_label=None
        ),
    ]
    # The first token already carries a non-"O" prediction, so it must be
    # yielded unchanged rather than pulled into a dictionary match.
    dtagger = DictTagger(vocabs={"process": ["production of"]})
    assert list(dtagger.tag(tokens)) == tokens


def test_dict_tagger_below_cutoff_no_match() -> None:
    # A vocabulary term that does not resemble the input at all should not
    # trigger a match; every token is passed through unchanged.
    dtagger = DictTagger(
        vocabs={"process": ["completely unrelated phrase"]}, cutoff=93.0
    )
    assert list(dtagger.tag(sample)) == sample


def test_dict_tagger_cutoff_gates_imperfect_matches() -> None:
    # The same match that succeeds at cutoff 93 is rejected at cutoff 100
    # (an exact-similarity requirement the near-match cannot meet).
    near_miss = [
        Token(
            string="production of cox",  # lowercase -> not an exact match
            offset=(0, 17),
            prediction="O",
            gold_label=None,
        )
    ]
    strict = DictTagger(vocabs={"process": ["production of COX"]}, cutoff=100.0)
    assert list(strict.tag(near_miss)) == near_miss


def test_vocab_reads_wordlist_from_a_path_object(
    tmp_path: pathlib.Path,
) -> None:
    # A pathlib.Path must reach open() exactly like a str does: the vocabulary
    # files are addressed as paths everywhere else in the package.
    vocab_file = tmp_path / "enzymes.txt"
    vocab_file.write_text("catalase\ncytochrome c oxidase\n")

    vocab = Vocab("enzyme", vocab_file, 93.0)
    token = Token(
        string="catalase", offset=(0, 8), prediction="O", gold_label=None
    )

    match = vocab.match(token)
    assert match is not None
    assert match.term == "catalase"
    assert match.score == 100.0


def test_vocab_match_reports_which_term_it_matched() -> None:
    # An inexact query over a wordlist with several eligible entries: the
    # matched term is the dictionary entry, not the query, and a linker needs
    # to know which one fired.
    vocab = Vocab("enzyme", ["urease", "catalase"], 93.0)
    token = Token(
        string="catalse", offset=(0, 7), prediction="O", gold_label=None
    )

    match = vocab.match(token)
    assert match is not None
    assert match.term == "catalase"
    assert match.score > 93.0


def test_vocab_match_stays_frozen_and_hashable() -> None:
    match = VocabMatch(term="AS-A", score=100.0)

    # Hashable only while every field is; a plain (mutable) instance
    # would raise here.
    assert len({match, VocabMatch("AS-A", 100.0)}) == 1

    with pytest.raises(FrozenInstanceError):
        match.score = 0.0  # type: ignore[misc]


def test_vocab_match_below_cutoff_is_not_a_match() -> None:
    vocab = Vocab("enzyme", ["urease", "catalase"], 100.0)
    token = Token(
        string="catalse", offset=(0, 7), prediction="O", gold_label=None
    )

    assert vocab.match(token) is None


def test_vocab_separates_empty_search_space_from_a_zero_score() -> None:
    token = Token(
        string="catalase", offset=(0, 8), prediction="O", gold_label=None
    )

    # An empty wordlist offers no candidate, so nothing was scored at all.
    assert Vocab("enzyme", [], 0.0).match(token) is None

    # A candidate sharing no character with the query scores 0.0, which at a
    # cutoff of 0.0 is a match: the score and the "no candidate" answer are
    # different values.
    scored = Vocab("enzyme", ["hippodrom"], 0.0).match(token)
    assert scored is not None
    assert scored.score == 0.0
    assert scored.term == "hippodrom"


def test_vocab_matches_a_term_the_cutoff_admits_but_a_fixed_band_would_not() -> (
    None
):
    # 35 characters against 39: a fixed +-2 window over the length buckets
    # dropped this pair before anything was scored, even though QRatio puts
    # it comfortably over the cutoff.
    entry = "Bacillus subtilis subsp. spizizenii"
    query = "Bacillus subtilis subspecies spizizenii"
    assert abs(len(entry) - len(query)) > 2

    match = Vocab("bacteria", [entry], 90.0).match(
        Token(
            string=query,
            offset=(0, len(query)),
            prediction="O",
            gold_label=None,
        )
    )

    assert match is not None
    assert match.term == entry
    assert match.score > 90.0


def test_vocab_matches_a_term_longer_than_the_query_by_more_than_two() -> None:
    # The other side of the band: 45 characters against 39 tops out at
    # 200 * 39 / 84 = 92.86, which clears a cutoff of 90.
    entry, query = "x" * 45, "x" * 39

    match = Vocab("enzyme", [entry], 90.0).match(
        Token(
            string=query,
            offset=(0, len(query)),
            prediction="O",
            gold_label=None,
        )
    )

    assert match is not None
    assert match.term == entry
    assert match.score == pytest.approx(200 * 39 / 84)


def test_vocab_still_prunes_lengths_the_cutoff_puts_out_of_reach() -> None:
    # Pruning is score-preserving by construction, so `match` answers None
    # whether a hopeless term was skipped or scored and rejected: the band
    # itself is the only place the prune is observable.
    vocab = Vocab("enzyme", ["x" * n for n in (20, 33, 40, 48, 61)], 90.0)
    admitted = set(vocab._candidate_lengths(40))

    # QRatio cannot exceed 200 * min(t, q) / (t + q), which is 66.7 at 20
    # characters and 79.2 at 61 -- neither is worth scoring against a cutoff
    # of 90.
    assert admitted.isdisjoint({20, 61})
    # Inside the band, and the query's own length, must survive.
    assert {33, 40, 48} <= admitted


def test_vocab_length_band_never_drops_a_term_that_could_clear_the_cutoff() -> (
    None
):
    # A run of one repeated character makes the ceiling attainable: the
    # shorter string is a subsequence of the longer, so QRatio hits exactly
    # 200 * min(t, q) / (t + q) and the band must admit every length that
    # reaches the cutoff.
    cutoff = 90.0
    lengths = range(1, 41)
    vocab = Vocab("enzyme", ["x" * n for n in lengths], cutoff)

    for query_length in lengths:
        admitted = set(vocab._candidate_lengths(query_length))
        for term_length in lengths:
            score = fuzz.QRatio("x" * term_length, "x" * query_length)
            if score >= cutoff:
                assert term_length in admitted, (
                    f"{term_length} scores {score} against a query of "
                    f"{query_length} and was pruned anyway"
                )


def test_a_cutoff_no_term_can_fail_prunes_nothing() -> None:
    # cutoff 0 puts a zero in the band's denominator; every term clears it,
    # so the answer is to prune nothing rather than to divide.
    vocab = Vocab("enzyme", ["ox", "hippodrom"], 0.0)

    assert set(vocab._candidate_lengths(8)) == {2, 9}


def test_dict_tagger_accepts_path_valued_vocabs(tmp_path: pathlib.Path) -> None:
    vocab_file = tmp_path / "processes.txt"
    vocab_file.write_text("production of COX\n")

    dtagger = DictTagger(vocabs={"process": vocab_file})

    tagged = list(dtagger.tag(sample))
    assert [tok.string for tok in tagged] == [
        "on",
        "the",
        "production of COX",
        ".",
    ]
    assert [tok.prediction for tok in tagged] == ["O", "O", "process", "O"]


def test_dict_tagger_window_cap_limits_span() -> None:
    # _find_best_match only considers windows up to min(len, 10) tokens, so a
    # 12-token phrase can never be matched as a single span.
    tokens = [Token(f"w{i}", (i * 3, i * 3 + 2), "O", None) for i in range(12)]
    full_phrase = repr_sequence(tokens)
    tagged = list(
        DictTagger(vocabs={"p": [full_phrase]}, cutoff=93.0).tag(tokens)
    )
    assert len(tagged) == 12  # nothing merged
    assert all(tok.prediction == "O" for tok in tagged)


def test_vocab_keeps_entries_whose_lengths_repeat_out_of_order() -> None:
    # Lengths 3, 2, 3: consecutive-run grouping builds one bucket per run and
    # the later run of length 3 replaces the earlier one, so "abc" leaves the
    # search space entirely and no score can ever mention it.
    vocab = Vocab("enzyme", ["abc", "de", "fgh"], 100.0)

    for term in ("abc", "de", "fgh"):
        token = Token(
            string=term,
            offset=(0, len(term)),
            prediction="O",
            gold_label=None,
        )
        match = vocab.match(token)
        assert match is not None, f"{term!r} is missing from the search space"
        assert match.term == term
        assert match.score == 100.0


# `NPP 1` is the one surface form that data/enzymes.txt and data/strains.txt
# actually share, so the tie below is the real one rather than an invented one.
AMBIGUOUS_ENZYMES = ["NPP 1", "catalase"]
AMBIGUOUS_STRAINS = ["NPP 1", "P-24"]


def ambiguous_span() -> list[Token]:
    return [
        Token(string="NPP", offset=(0, 3), prediction="O", gold_label=None),
        Token(string="1", offset=(4, 5), prediction="O", gold_label=None),
    ]


def test_dict_tagger_tie_does_not_depend_on_vocab_order() -> None:
    # Both wordlists score the span 100.0, so under a first-maximum tie-break
    # the winning label is whichever vocabulary was passed first — stable per
    # construction and silently different across constructions.
    forward = list(
        DictTagger(
            vocabs={
                "enzyme": AMBIGUOUS_ENZYMES,
                "strain": AMBIGUOUS_STRAINS,
            }
        ).tag(ambiguous_span())
    )
    reverse = list(
        DictTagger(
            vocabs={
                "strain": AMBIGUOUS_STRAINS,
                "enzyme": AMBIGUOUS_ENZYMES,
            }
        ).tag(ambiguous_span())
    )

    assert forward == reverse


def test_dict_tagger_separates_no_match_from_an_ambiguous_one() -> None:
    tokens = [
        Token(string="in", offset=(0, 2), prediction="O", gold_label=None),
        Token(
            string="catalase", offset=(3, 11), prediction="O", gold_label=None
        ),
        Token(string="NPP", offset=(12, 15), prediction="O", gold_label=None),
        Token(string="1", offset=(16, 17), prediction="O", gold_label=None),
    ]
    tagged = list(
        DictTagger(
            vocabs={
                "enzyme": AMBIGUOUS_ENZYMES,
                "strain": AMBIGUOUS_STRAINS,
            }
        ).tag(tokens)
    )

    unmatched, unique, ambiguous = tagged
    assert (unmatched.prediction, unmatched.candidate_labels) == (
        "O",
        frozenset(),
    )
    assert (unique.prediction, unique.candidate_labels) == (
        "enzyme",
        frozenset(),
    )
    # A match did occur, so it is not "O"; which of the two types it is, is
    # not decided, so it is neither of them either.
    assert ambiguous.string == "NPP 1"
    assert ambiguous.candidate_labels == frozenset({"enzyme", "strain"})
    assert ambiguous.prediction not in ("O", "enzyme", "strain")


def as_token(string: str) -> Token:
    return Token(
        string=string, offset=(0, len(string)), prediction="O", gold_label=None
    )


def as_span(*strings: str) -> tuple[Token, ...]:
    tokens: list[Token] = []
    offset = 0
    for string in strings:
        tokens.append(
            Token(
                string=string,
                offset=(offset, offset + len(string)),
                prediction="O",
                gold_label=None,
            )
        )
        offset += len(string) + 1

    return tuple(tokens)


def test_vocab_matches_a_descriptive_name_written_in_another_case() -> None:
    # Scored raw, "Catalase" against "catalase" is 87.5 and misses at any
    # usable cutoff, so a sentence-initial mention of a wordlist term was
    # invisible to the tagger.
    vocab = Vocab("enzyme", ["catalase"], 93.0)

    for written in ("Catalase", "CATALASE", "catalase"):
        match = vocab.match(as_token(written))
        assert match is not None, f"{written!r} did not match"
        # The surface form the wordlist holds, not the folded key it is
        # indexed under: a linker resolves the entry, not the query.
        assert match.term == "catalase"
        assert match.score == 100.0


def test_vocab_matches_a_symbol_written_with_other_punctuation() -> None:
    # "MMP 3" against "MMP-3" is 80.0 raw. Punctuation is normalized for both
    # halves of the wordlist; only case folding is withheld from symbols.
    match = Vocab("enzyme", ["MMP-3"], 93.0).match(as_span("MMP", "3"))

    assert match is not None
    assert match.term == "MMP-3"
    assert match.score == 100.0


def test_vocab_keeps_symbol_like_forms_case_separated() -> None:
    # `FOR` is formaldehyde ferredoxin oxidoreductase and case is the only
    # feature separating it from the English word, so the fold that rescues
    # "Catalase" must not reach it.
    assert Vocab("enzyme", ["FOR"], 93.0).match(as_token("for")) is None
    assert Vocab("enzyme", ["for"], 93.0).match(as_token("FOR")) is None

    exact = Vocab("enzyme", ["FOR"], 93.0).match(as_token("FOR"))
    assert exact is not None and exact.score == 100.0


def test_vocab_keeps_a_capital_past_the_first_character_case_separated() -> (
    None
):
    # Long enough to be a phrase, but `COX` inside it is a symbol, so the
    # whole form is scored with case intact. An initial capital alone is a
    # sentence or a genus and does not make a form symbol-like.
    vocab = Vocab("process", ["production of COX"], 93.0)

    assert vocab.match(as_span("production", "of", "cox")) is None

    folded = Vocab("bacteria", ["Bacillus subtilis"], 93.0)
    match = folded.match(as_span("bacillus", "subtilis"))
    assert match is not None
    assert match.term == "Bacillus subtilis"


def test_vocab_buckets_are_keyed_by_the_length_the_scorer_sees() -> None:
    # The cutoff-derived band bounds len(term) against len(query) as QRatio
    # sees them, so a bucket keyed by a length the scorer never sees prunes
    # terms that would have cleared the cutoff. "İnulinase" is the case that
    # tells the two keyings apart: it folds to ten characters, not nine.
    terms = [
        "catalase",
        "MMP-3",
        "FOR",
        "İnulinase",
        "Bacillus subtilis subsp. spizizenii",
    ]
    vocab = Vocab("enzyme", terms, 93.0)

    kept: list[str] = []
    for population in vocab._populations:
        for length, keys in population.scored.items():
            assert all(len(key) == length for key in keys)
            assert len(population.surface[length]) == len(keys)
            kept.extend(population.surface[length])

    # Nothing is dropped by the split, and the entry whose folded form is
    # longer than its surface form is indexed under the folded length.
    assert sorted(kept) == sorted(terms)
    assert 10 in vocab._lengths and 9 not in vocab._lengths


def test_length_prune_never_hides_a_match_the_processor_admits() -> None:
    # A cutoff of 0.0 prunes nothing, so it is the unpruned oracle for the
    # same population: whatever it scores at or above the cutoff must survive
    # the band at that cutoff with the same score. The species pair is the
    # length gap the band is there to admit -- 35 characters against 39.
    cutoff = 90.0
    terms = [
        "catalase",
        "cytochrome c oxidase",
        "MMP-3",
        "Bacillus subtilis subsp. spizizenii",
    ]
    queries = (
        as_span("Catalase"),
        as_span("CYTOCHROME", "C", "OXIDASE"),
        as_span("Cytochrome", "c", "oxidase", "activity"),
        as_span("MMP", "3"),
        as_span("Bacillus", "subtilis", "subspecies", "spizizenii"),
        as_span("urease"),
    )

    admitted = 0
    for query in queries:
        oracle = Vocab("enzyme", terms, 0.0).match(query)
        assert oracle is not None
        if oracle.score < cutoff:
            continue

        admitted += 1
        pruned = Vocab("enzyme", terms, cutoff).match(query)
        assert pruned is not None, f"{repr_sequence(query)!r} was pruned away"
        assert pruned.score == oracle.score
        assert pruned.term == oracle.term

    assert admitted >= 4


def test_dict_tagger_from_schema_builds_one_vocab_per_backed_entity_type(
    tmp_path: pathlib.Path,
) -> None:
    # A schema with a mix of backed and unbacked entity types: the factory
    # must build a Vocab for each of the former and silently skip the latter,
    # rather than requiring every entity type to carry a wordlist.
    enzymes_file = tmp_path / "enzymes.txt"
    enzymes_file.write_text("catalase\n")
    bacteria_file = tmp_path / "bacteria.txt"
    bacteria_file.write_text("Bacillus subtilis\n")

    schema = Schema(
        entity_types=(
            EntityType(name="enzymes", prefix="enz", vocab_path=enzymes_file),
            EntityType(name="bacteria", prefix="bac", vocab_path=bacteria_file),
            EntityType(name="other_organisms", prefix="oth"),
        )
    )

    dtagger = DictTagger.from_schema(schema)

    # Exactly the backed types, keyed by entity-type name, and no more: a
    # wrong key or a spurious "other_organisms" vocab would surface here.
    assert {vocab.label for vocab in dtagger._vocabs} == {"enzymes", "bacteria"}

    # The path reached Vocab intact -- a match against the file's one entry
    # is the only way to tell a correctly wired Path from one silently
    # swapped for the wrong file or a bare string that failed to open.
    tagged = list(dtagger.tag(list(as_span("Catalase"))))
    assert [tok.prediction for tok in tagged] == ["enzymes"]


def test_dict_tagger_tags_a_mention_written_in_another_case() -> None:
    tagged = list(
        DictTagger(vocabs={"enzyme": ["catalase"]}).tag(
            list(as_span("Catalase"))
        )
    )

    assert [tok.prediction for tok in tagged] == ["enzyme"]
