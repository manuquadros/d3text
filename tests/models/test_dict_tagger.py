"""Ported from the deleted src/tests/test_dict_tagger.py (entities layer).

Re-targeted at the live d3text.models.dict_tagger. DictTagger/Vocab are fully
pure — no disk, no model — so these run on CPU with no data or network.
"""

import pathlib
from dataclasses import FrozenInstanceError

import pytest

from d3text.models.dict_tagger import DictTagger, Vocab, VocabMatch
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
    assert match.entity_ids == frozenset()


def test_vocab_match_carries_a_set_of_entity_ids() -> None:
    # A surface form is not owned by one entity: `AS-A` is a synonym of four
    # separate enzymes, and a species nested in a strain designation is meant
    # to yield both. An empty set rather than None also lets a consumer
    # iterate the slot without first testing it.
    match = VocabMatch(term="catalase", score=100.0)

    assert match.entity_ids == frozenset()
    assert isinstance(match.entity_ids, frozenset)

    linked = VocabMatch(
        term="AS-A", score=100.0, entity_ids=frozenset({"enz1", "enz2"})
    )
    assert linked.entity_ids == frozenset({"enz1", "enz2"})


def test_vocab_match_stays_frozen_and_hashable() -> None:
    ids = frozenset({"enz1", "enz2"})
    match = VocabMatch(term="AS-A", score=100.0, entity_ids=ids)

    # Hashable only while every field is; a plain set would raise here.
    assert len({match, VocabMatch("AS-A", 100.0, ids)}) == 1

    with pytest.raises(FrozenInstanceError):
        match.entity_ids = frozenset()  # type: ignore[misc]


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

    # Nothing is within the +-2 length window, so no candidate was scored.
    assert Vocab("enzyme", ["ox"], 0.0).match(token) is None

    # A candidate sharing no character with the query scores 0.0, which at a
    # cutoff of 0.0 is a match: the score and the "no candidate" answer are
    # different values.
    scored = Vocab("enzyme", ["hippodrom"], 0.0).match(token)
    assert scored is not None
    assert scored.score == 0.0
    assert scored.term == "hippodrom"


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
