from entities.models import DictTagger
from utils import Token

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


def test_dict_tagger():
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
