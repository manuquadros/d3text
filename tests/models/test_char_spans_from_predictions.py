"""Grounding a tagger's aggregated-axis spans in an external document's own
char offsets: the same window merge `aggregate_embeddings` gives the
embeddings, applied to the stored offset mapping instead, with no BRENDA
mention store consulted."""

import numpy
import torch
from d3text.mention_metrics import PredictedMention
from d3text.models.token_supervision import char_spans_from_predictions
from d3text.token_labels import BRENDA_LABELS

ENZYMES = BRENDA_LABELS.by_prefix["enz"]


def test_a_span_is_grounded_through_the_window_merge() -> None:
    """One 12-token window: CLS and SEP drop out of the aggregated axis, so
    aggregated position 2 is the window's token 3, not its token 2."""
    text = "ABCDEFGHIJKL"
    offset_mapping = numpy.zeros((1, 12, 2), dtype=numpy.uint32)
    for token in range(1, 11):
        offset_mapping[0, token] = (token - 1, token)
    attention_mask = numpy.ones((1, 12), dtype=numpy.int64)

    (span,) = char_spans_from_predictions(
        [PredictedMention(start=2, end=5, type_code=ENZYMES)],
        offset_mapping,
        attention_mask,
        text=text,
        document="s800:doc1",
    )

    assert (span.document, span.start, span.end) == ("s800:doc1", 2, 5)
    assert span.surface == "CDE"
    assert span.entity_type == BRENDA_LABELS.type_of(ENZYMES)


def test_two_windows_merge_before_grounding() -> None:
    """A span straddling the seam is grounded from whichever window
    `aggregate_embeddings` actually kept for each side of it."""
    offset_mapping = numpy.zeros((2, 32, 2), dtype=numpy.uint32)
    offset_mapping[0, 1:31] = [(i - 1, i) for i in range(1, 31)]
    offset_mapping[1, 1:31] = [(i + 9, i + 10) for i in range(1, 31)]
    attention_mask = numpy.ones((2, 32), dtype=numpy.int64)
    text = "".join(chr(ord("a") + (i % 26)) for i in range(40))

    (span,) = char_spans_from_predictions(
        [PredictedMention(start=18, end=22, type_code=ENZYMES)],
        torch.as_tensor(offset_mapping),
        torch.as_tensor(attention_mask),
        text=text,
        document="enzymener:s1",
    )

    assert span.document == "enzymener:s1"
    assert span.surface == text[span.start : span.end]
    assert (span.start, span.end) == (18, 22)
