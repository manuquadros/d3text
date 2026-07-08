"""Ported from the deleted src/tests/test_dict_tagger.py (entities layer).

Re-targeted at the live d3text.models.dict_tagger. DictTagger/Vocab are fully
pure — no disk, no model — so these run on CPU with no data or network.
"""

from d3text.models.dict_tagger import DictTagger
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


def test_dict_tagger_window_cap_limits_span() -> None:
    # _find_best_match only considers windows up to min(len, 10) tokens, so a
    # 12-token phrase can never be matched as a single span.
    tokens = [
        Token(f"w{i}", (i * 3, i * 3 + 2), "O", None) for i in range(12)
    ]
    full_phrase = repr_sequence(tokens)
    tagged = list(DictTagger(vocabs={"p": [full_phrase]}, cutoff=93.0).tag(tokens))
    assert len(tagged) == 12  # nothing merged
    assert all(tok.prediction == "O" for tok in tagged)
