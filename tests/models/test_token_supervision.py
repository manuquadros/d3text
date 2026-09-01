"""The label-store reader: space checked at open, codes carried across the
window merge by the same arithmetic that merges the embeddings."""

import h5py
import numpy
import pytest
import torch
from d3text import token_labels
from d3text.models.token_supervision import (
    TokenLabelReader,
    document_lengths,
    padded_targets,
)
from d3text.token_labels import BRENDA_LABELS, IGNORE_INDEX, DocumentLabels

NO_SPANS = numpy.zeros((0, token_labels.SPAN_COLUMNS), dtype=numpy.int32)
_STAMP = token_labels.IndexStamp(digest="test-index")


def write_store(path, documents, space=BRENDA_LABELS):
    """A label store holding `documents` (pubmed id -> [windows, T] codes)."""
    with h5py.File(path, "w") as store:
        token_labels.write_label_space(store, space, stamp=_STAMP)
        for pubmed_id, codes in documents.items():
            token_labels.store_token_labels(
                store,
                pubmed_id,
                DocumentLabels(
                    codes=numpy.asarray(codes, dtype=numpy.int8),
                    spans=NO_SPANS,
                    text_length=0,
                ),
            )
    return path


def test_a_store_of_another_space_is_refused_at_open(tmp_path) -> None:
    """A permuted space re-means every integer; the reader must not serve it."""
    permuted = token_labels.LabelSpace(
        types=tuple(reversed(BRENDA_LABELS.types)),
        prefixes=tuple(reversed(BRENDA_LABELS.prefixes)),
    )
    path = write_store(tmp_path / "labels.hdf5", {}, space=permuted)

    with pytest.raises(ValueError, match="label space"):
        TokenLabelReader(path)


def test_codes_ride_the_same_window_merge_as_the_embeddings(tmp_path) -> None:
    """Two 32-token windows under the 20-token stride: the first window
    keeps its half of the overlap, the second supplies the rest — element
    for element what `aggregate_embeddings` selects for the embeddings."""
    codes = numpy.zeros((2, 32), dtype=numpy.int8)
    codes[0] = numpy.arange(32)
    codes[1] = 64 + numpy.arange(32)
    reader = TokenLabelReader(
        write_store(tmp_path / "labels.hdf5", {"77": codes})
    )

    aggregated = reader.document_codes("77", numpy.ones((2, 32)))

    assert aggregated is not None
    assert aggregated.dtype == torch.int64
    assert aggregated.tolist() == list(range(1, 21)) + list(range(75, 95))


def test_a_collated_mask_is_flattened_before_the_merge(tmp_path) -> None:
    """The DataLoader path hands the mask as [1, windows, T]; the reader must
    read it as the [windows, T] it masks."""
    codes = numpy.zeros((1, 32), dtype=numpy.int8)
    codes[0, 5] = 2
    reader = TokenLabelReader(
        write_store(tmp_path / "labels.hdf5", {"77": codes})
    )

    aggregated = reader.document_codes("77", torch.ones((1, 1, 32)))

    assert aggregated is not None
    assert aggregated.shape[0] == 30  # 32 minus [CLS] and [SEP]
    assert aggregated[4] == 2


def test_a_document_the_store_lacks_is_none(tmp_path) -> None:
    reader = TokenLabelReader(write_store(tmp_path / "labels.hdf5", {}))

    assert reader.document_codes("404", numpy.ones((1, 32))) is None


def write_store_with_spans(path, spans_by_document, space=BRENDA_LABELS):
    """A label store holding one row of `spans` per document, no codes."""
    with h5py.File(path, "w") as store:
        token_labels.write_label_space(store, space, stamp=_STAMP)
        for pubmed_id, spans in spans_by_document.items():
            token_labels.store_token_labels(
                store,
                pubmed_id,
                DocumentLabels(
                    codes=numpy.zeros((0,), dtype=numpy.int8),
                    spans=numpy.asarray(spans, dtype=numpy.int32).reshape(
                        -1, token_labels.SPAN_COLUMNS
                    ),
                    text_length=0,
                ),
            )
    return path


def test_mentioned_types_reads_the_spans_regardless_of_gold(tmp_path) -> None:
    enzyme = BRENDA_LABELS.code_of("enz1")
    bacterium = BRENDA_LABELS.code_of("bac3")
    reader = TokenLabelReader(
        write_store_with_spans(
            tmp_path / "labels.hdf5",
            {"77": [(0, 8, enzyme, 1), (9, 20, bacterium, 0)]},
        )
    )

    assert reader.mentioned_types("77") == {enzyme, bacterium}


def test_mentioned_types_of_a_document_the_store_lacks_is_none(
    tmp_path,
) -> None:
    reader = TokenLabelReader(
        write_store_with_spans(tmp_path / "labels.hdf5", {})
    )

    assert reader.mentioned_types("404") is None


def test_mentioned_types_min_chars_gates_per_type(tmp_path) -> None:
    enzyme = BRENDA_LABELS.code_of("enz1")
    bacterium = BRENDA_LABELS.code_of("bac3")
    reader = TokenLabelReader(
        write_store_with_spans(
            tmp_path / "labels.hdf5",
            # enzyme: 3 chars; bacterium: 10 chars
            {"77": [(0, 3, enzyme, 1), (10, 20, bacterium, 0)]},
        )
    )

    assert reader.mentioned_types(
        "77", min_chars={enzyme: 8, bacterium: 8}
    ) == {bacterium}


def test_mentioned_types_min_chars_drops_a_short_span(tmp_path) -> None:
    """A short match should not, on its own, carry a type through the gate
    that feeds the class-negative abstention mask."""
    enzyme = BRENDA_LABELS.code_of("enz1")
    bacterium = BRENDA_LABELS.code_of("bac3")
    reader = TokenLabelReader(
        write_store_with_spans(
            tmp_path / "labels.hdf5",
            # enzyme span is 3 chars long, bacterium span is 11
            {"77": [(0, 3, enzyme, 1), (9, 20, bacterium, 0)]},
        )
    )

    assert reader.mentioned_types("77") == {enzyme, bacterium}
    assert reader.mentioned_types("77", min_chars=8) == {bacterium}


def test_mismatched_window_geometry_raises(tmp_path) -> None:
    """Codes stored against other encodings would land on the wrong tokens."""
    reader = TokenLabelReader(
        write_store(
            tmp_path / "labels.hdf5",
            {"77": numpy.zeros((2, 32), dtype=numpy.int8)},
        )
    )

    with pytest.raises(ValueError, match="different encodings"):
        reader.document_codes("77", numpy.ones((3, 32)))


def test_padded_targets_pad_with_the_ignore_index() -> None:
    padded = padded_targets([torch.tensor([1, 2]), torch.tensor([3])], length=4)

    assert padded.tolist() == [
        [1, 2, IGNORE_INDEX, IGNORE_INDEX],
        [3, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX],
    ]


def test_document_lengths_count_unpadded_tokens() -> None:
    mask = torch.tensor([[True, True, False], [True, False, False]])

    assert document_lengths(mask) == [2, 1]
