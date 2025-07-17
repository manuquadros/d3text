import pathlib

import torch
import transformers
from d3text import utils
from d3text.utils import (
    Token,
    merge_off_tokens,
    merge_predictions,
    repr_sequence,
)

og = [
    Token(
        string="Effect",
        offset=(3427, 3433),
        prediction="O",
        gold_label=None,
    ),
    Token(string="of", offset=(3434, 3436), prediction="O", gold_label=None),
    Token(string="the", offset=(3437, 3440), prediction="O", gold_label=None),
    Token(
        string="cholesterol",
        offset=(3441, 3452),
        prediction="O",
        gold_label=None,
    ),
    Token(string="em", offset=(3453, 3455), prediction="O", gold_label=None),
    Token(string="##uIs", offset=(3455, 3458), prediction="O", gold_label=None),
    Token(
        string="##ification",
        offset=(3458, 3467),
        prediction="O",
        gold_label=None,
    ),
    Token(
        string="method",
        offset=(3468, 3474),
        prediction="O",
        gold_label=None,
    ),
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
    Token(string="[SEP]", offset=(0, 0), prediction="O", gold_label=None),
]

merged = [
    Token(
        string="Effect",
        offset=(3427, 3433),
        prediction="O",
        gold_label=None,
    ),
    Token(string="of", offset=(3434, 3436), prediction="O", gold_label=None),
    Token(string="the", offset=(3437, 3440), prediction="O", gold_label=None),
    Token(
        string="cholesterol",
        offset=(3441, 3452),
        prediction="O",
        gold_label=None,
    ),
    Token(
        string="emuIsification",
        offset=(3453, 3467),
        prediction="O",
        gold_label=None,
    ),
    Token(
        string="method",
        offset=(3468, 3474),
        prediction="O",
        gold_label=None,
    ),
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


def test_merge_predictions_does_not_duplicate() -> None:
    assert list(
        merge_predictions(
            preds=[og], sample_mapping=torch.tensor([0]), stride=50
        )
    ) == [og]


def test_merge_off_tokens_does_not_duplicate() -> None:
    assert merge_off_tokens(og) == merged


def test_sequence_is_printed_correctly() -> None:
    assert (
        repr_sequence(merged)
        == "Effect of the cholesterol emuIsification method on the production of COX."
    )


def test_aggregate_embeddings_across_document() -> None:
    fp = pathlib.Path(__file__).parent / "test_abstract.txt"
    with fp.open() as abstract_file:
        abstract = abstract_file.read()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "prajjwal1/bert-mini"
    )
    model = transformers.AutoModel.from_pretrained("prajjwal1/bert-mini")
    tokenized = utils.split_and_tokenize(
        tokenizer=tokenizer, inputs=abstract, stride=20
    )

    text = []
    for idseq, attn_mask in zip(
        tokenized["input_ids"], tokenized["attention_mask"]
    ):
        ids = idseq[attn_mask.bool()][1:-1]
        text.extend(tokenizer.convert_ids_to_tokens(ids))
    print(len(text))
    print(tokenizer.convert_tokens_to_string(text))

    embeddings = model(
        tokenized["input_ids"], tokenized["attention_mask"]
    ).last_hidden_state
    aggregated = utils.aggregate_embeddings(
        embeddings, tokenized["attention_mask"], stride=20
    )
    assert len(aggregated) == 609
