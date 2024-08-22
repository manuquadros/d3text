from itertools import starmap

import torch

from utils import Token, merge_predictions


def test_merged_predictions():
    preds = [
        [
            Token(
                string="from",
                offset=(2442, 2446),
                prediction="O",
                gold_label=None,
            ),
            Token(
                string="Mycobacterium",
                offset=(2447, 2460),
                prediction="B-OOS",
                gold_label=None,
            ),
            Token(
                string="sp",
                offset=(2461, 2463),
                prediction="O",
                gold_label=None,
            ),
        ],
        [
            Token(
                string="sp",
                offset=(2461, 2463),
                prediction="O",
                gold_label=None,
            ),
            Token(
                string="5", offset=(2545, 2546), prediction="O", gold_label=None
            ),
            Token(
                string="]", offset=(2546, 2547), prediction="O", gold_label=None
            ),
        ],
        [
            Token(
                string=".", offset=(2547, 2548), prediction="O", gold_label=None
            ),
            Token(
                string="Enzymatic",
                offset=(2549, 2558),
                prediction="O",
                gold_label=None,
            ),
            Token(
                string="properties",
                offset=(2559, 2569),
                prediction="O",
                gold_label=None,
            ),
        ],
    ]
    merged = [
        [
            Token(
                string="from",
                offset=(2442, 2446),
                prediction="O",
                gold_label=None,
            ),
            Token(
                string="Mycobacterium",
                offset=(2447, 2460),
                prediction="B-OOS",
                gold_label=None,
            ),
            Token(
                string="sp",
                offset=(2461, 2463),
                prediction="O",
                gold_label=None,
            ),
            Token(
                string="5", offset=(2545, 2546), prediction="O", gold_label=None
            ),
            Token(
                string="]", offset=(2546, 2547), prediction="O", gold_label=None
            ),
        ],
        [
            Token(
                string=".", offset=(2547, 2548), prediction="O", gold_label=None
            ),
            Token(
                string="Enzymatic",
                offset=(2549, 2558),
                prediction="O",
                gold_label=None,
            ),
            Token(
                string="properties",
                offset=(2559, 2569),
                prediction="O",
                gold_label=None,
            ),
        ],
    ]

    sample_mapping = torch.tensor([0, 0, 1])
    stride = 1

    assert merge_predictions(preds, sample_mapping, stride) == merged
