import pathlib

import pytest
import torch
import transformers
from d3text import utils
from d3text.utils import (
    Token,
    load_fast_tokenizer,
    merge_off_tokens,
    merge_predictions,
    repr_sequence,
    token_merge,
)
from d3text.models.base import load_base_model
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


def test_empty_sequence_is_printed_as_the_empty_string() -> None:
    assert repr_sequence(()) == ""


def test_first_token_gets_no_leading_gap() -> None:
    tokens = [
        Token(string="COX", offset=(3496, 3499), prediction="O"),
        Token(string=".", offset=(3499, 3500), prediction="O"),
    ]
    assert repr_sequence(tokens) == "COX."


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
    """Overlapping windows of a long document aggregate back to exactly one
    embedding per document token.

    Runs against a maintained tiny BERT rather than ``prajjwal1/bert-mini``,
    whose legacy tokenizer no longer instantiates on the pinned
    ``transformers``.
    """
    fp = pathlib.Path(__file__).parent / "test_abstract.txt"
    abstract = fp.read_text()

    base_model = "hf-internal-testing/tiny-random-BertModel"
    tokenizer = load_fast_tokenizer(base_model)
    model = load_base_model(base_model)
    window = model.config.max_position_embeddings
    tokenized = utils.split_and_tokenize(
        tokenizer=tokenizer, inputs=abstract, stride=20, max_length=window
    )

    embeddings = model(
        tokenized["input_ids"], tokenized["attention_mask"]
    ).last_hidden_state
    aggregated = utils.aggregate_embeddings(
        embeddings, tokenized["attention_mask"], stride=20
    )

    # The windows are a strided, overlapping view of one token stream, so the
    # merge must reproduce it exactly: one row per non-special document token.
    expected = len(tokenizer(abstract, add_special_tokens=False)["input_ids"])
    assert len(aggregated) == expected


def test_split_and_tokenize_windows_the_whole_document() -> None:
    """A document longer than the window survives it whole.

    Not marked `integration`, unlike the aggregation test above that asserts
    the same invariant from the other side: that one is deselected by the
    `-m "not integration"` gate, and this failure has to be caught by the gate
    that actually runs before a commit.

    The failure it guards against is silent, which is why the count is
    asserted rather than the call merely exercised. `transformers` 5.16.1
    returns two overflow windows for a 5,989-token document where 5.15.1
    returns thirteen — every fulltext truncated to the first ~1,000 tokens,
    with no error and a perfectly well-formed `BatchEncoding` on the way out.
    Nothing downstream can tell: the encodings, the token labels and the
    training run would all agree with each other about the truncated text.
    """
    tokenizer = load_fast_tokenizer("hf-internal-testing/tiny-random-BertModel")
    text = " ".join(f"token{n} of the sequence," for n in range(600))

    window, stride = 64, 20
    tokenized = utils.split_and_tokenize(
        tokenizer=tokenizer, inputs=text, max_length=window, stride=stride
    )

    length = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    # Each window spends two positions on [CLS]/[SEP] and re-reads `stride`
    # tokens of the one before it, so this many carry new text.
    per_window = window - 2 - stride
    assert len(tokenized["input_ids"]) >= -(-length // per_window)

    # The end of the text has to be inside the last window. The count above
    # would still pass if the windows overlapped more than they should; this
    # is what pins them to the *document* rather than to each other.
    assert int(tokenized["offset_mapping"].max()) == len(text)


@pytest.mark.integration
def test_load_base_model_handles_legacy_config() -> None:
    """`prajjwal1/bert-mini`'s config.json has no `model_type`, so plain
    `AutoModel.from_pretrained` raises `ValueError`; `load_base_model`
    falls back to an explicit BERT config and loads the encoder."""
    with pytest.raises(ValueError):
        transformers.AutoModel.from_pretrained("prajjwal1/bert-mini")

    model = load_base_model("prajjwal1/bert-mini")
    assert model.config.model_type == "bert"
    # 256 == embedding_dims['prajjwal1/bert-mini']
    assert model.config.hidden_size == 256


def test_load_fast_tokenizer_rejects_a_slow_tokenizer(monkeypatch) -> None:
    """A slow tokenizer must be refused where the base model is named.

    `split_and_tokenize` and `embed_document` need `return_overflowing_tokens`
    and `offset_mapping`, which only the fast tokenizers provide, so accepting
    a slow one just defers the failure to the middle of a precompute run.
    """

    class SlowTokenizer:
        pass

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        classmethod(lambda cls, *a, **kw: SlowTokenizer()),
    )

    with pytest.raises(TypeError, match="slow tokenizer"):
        load_fast_tokenizer("some/slow-model")


@pytest.mark.integration
def test_load_fast_tokenizer_returns_a_fast_tokenizer() -> None:
    tokenizer = load_fast_tokenizer("hf-internal-testing/tiny-random-BertModel")
    assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
