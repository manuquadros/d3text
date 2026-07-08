"""Ported from the deleted src/tests/test_predictions.py (entities layer).

Re-targeted at the live d3text.utils.merge_tokens.
"""

from d3text.utils import merge_tokens


def test_merging_without_gold_standard() -> None:
    tokens = (
        "[CLS]",
        "genetic",
        "analysis",
        "of",
        "the",
        "xenobiotic",
        "resistance",
        "-",
        "associated",
        "abc",
        "gene",
        "subfamilies",
        "of",
        "the",
        "lepid",
        "##optera",
        ".",
        "[SEP]",
    )

    tags = (
        "#",
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
        "B-OOS",
        "B-OOS",
        "O",
        "#",
    )

    merged = {
        "tokens": [
            "genetic",
            "analysis",
            "of",
            "the",
            "xenobiotic",
            "resistance",
            "-",
            "associated",
            "abc",
            "gene",
            "subfamilies",
            "of",
            "the",
            "lepidoptera",
            ".",
        ],
        "predicted": [
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "B-OOS",
            "O",
        ],
    }

    assert merge_tokens(tokens, tags) == merged


def test_merge_tokens_omits_gold_key_when_absent() -> None:
    # gold_labels defaults to None -> the result must not carry a gold key.
    result = merge_tokens(("[CLS]", "cat", "[SEP]"), ("#", "O", "#"))
    assert "gold_labels" not in result
