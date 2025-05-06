import utils


def test_merging_without_gold_standard():
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

    assert utils.merge_tokens(tokens, tags) == merged
