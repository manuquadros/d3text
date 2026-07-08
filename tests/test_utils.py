import pathlib

import pytest
import torch
import transformers
from d3text import utils
from d3text.utils import (
    Token,
    merge_off_tokens,
    merge_predictions,
    repr_sequence,
    token_merge,
)
from d3text.utils.utils import (
    aggregate_embeddings,
    concat,
    entity_counter,
    midhash,
    pad_offsets,
    safe_concat,
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


# --------------------------------------------------------------------------- #
# Pure helpers (no network, no model)                                          #
# --------------------------------------------------------------------------- #
def test_token_merge_adjacent_offsets_no_space() -> None:
    a = Token(string="em", offset=(0, 2), prediction="B-enz", gold_label=None)
    b = Token(string="##uls", offset=(2, 7), prediction="O", gold_label=None)
    merged = token_merge(a, b)
    assert merged.string == "emuls"
    assert merged.offset == (0, 7)
    assert merged.prediction == "B-enz"  # inherited from the first token


def test_token_merge_offset_gap_inserts_single_space() -> None:
    a = Token(string="em", offset=(0, 2), prediction="O", gold_label=None)
    b = Token(string="##uls", offset=(3, 8), prediction="O", gold_label=None)
    merged = token_merge(a, b)
    assert merged.string == "em uls"
    assert merged.offset == (0, 8)


def test_midhash() -> None:
    assert midhash("##ing") == "##"
    assert midhash("cat") == ""
    assert midhash("") == ""  # must not raise IndexError


def test_entity_counter_counts_only_b_tags() -> None:
    counted = entity_counter(["B-enz", "I-enz", "B-bac", "B-enz"])
    assert dict(counted) == {"B-enz": 2, "B-bac": 1}


def test_safe_concat_handles_none() -> None:
    assert safe_concat("a", "b") == "ab"
    assert safe_concat("a", None) == "a"
    assert safe_concat(None, "b") == "b"
    assert safe_concat(None, None) is None


def test_concat_uses_separator_only_between_non_empty() -> None:
    assert concat("a", "b", "-") == "a-b"
    assert concat("", "b", "-") == "b"
    assert concat("a", "") == "a"


@pytest.mark.xfail(
    reason="pad_offsets pads with float32 zeros, so an int input yields a float "
    "tensor that violates its Integer return hint under beartype",
    strict=True,
)
def test_pad_offsets_preserves_integer_dtype_and_pads_with_zeros() -> None:
    out = pad_offsets(torch.tensor([[1, 2], [3, 4]]), length=4)
    assert tuple(out.shape) == (4, 2)
    assert out.dtype in (torch.int32, torch.int64)
    assert out[2:].tolist() == [[0, 0], [0, 0]]


def test_aggregate_embeddings_pure_stride_merge() -> None:
    # Two 6-token sequences (CLS + 4 real tokens + SEP), embedding dim 1 whose
    # value is the token id. With stride=2, the overlap between the two windows
    # must be resolved token-by-token rather than duplicated.
    seq0 = torch.arange(6).reshape(6, 1).float()  # ids 0..5
    seq1 = (torch.arange(6) + 100).reshape(6, 1).float()  # ids 100..105
    embeddings = torch.stack([seq0, seq1])  # [2, 6, 1]
    mask = torch.ones(2, 6, dtype=torch.long)
    out = aggregate_embeddings(embeddings, mask, stride=2)
    # CLS/SEP dropped; overlap token 4 (seq0) and 101 (seq1) resolved away.
    assert out.flatten().tolist() == [1.0, 2.0, 3.0, 102.0, 103.0, 104.0]


@pytest.mark.integration
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
