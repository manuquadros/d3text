"""Pure unit tests for the vocabulary-independent helpers in data/data.py.

None of these touch HDF5 or the BRENDA files (see tests/data/test_dataset.py
for the fixture-backed dataset tests).
"""

import numpy
import pandas as pd
import pytest
import torch

from d3text.data.data import (
    BrendaDataset,
    compute_frequencies,
    index_tensor,
    multi_hot_encode_series,
)


def test_index_tensor_encodes_known_values():
    index = {"enz1": 0, "bac2": 1, "str3": 2}
    out = index_tensor(["enz1", "str3", "zzz"], index)  # "zzz" is unknown
    assert out.tolist() == [1, 0, 1]
    assert out.dtype == torch.uint8


def test_index_tensor_empty_input_is_all_zeros():
    index = {"enz1": 0, "bac2": 1, "str3": 2}
    assert index_tensor([], index).tolist() == [0, 0, 0]


class CountingIndex(dict[str, int]):
    """An entity index that records how often its columns were enumerated."""

    def __init__(self, mapping: dict[str, int]) -> None:
        super().__init__(mapping)
        self.enumerations = 0

    def values(self):
        self.enumerations += 1
        return super().values()


def test_index_tensor_length_follows_max_index():
    # Non-contiguous index: width is max(index)+1, not len(index).
    out = index_tensor(["a"], {"a": 0, "b": 5})
    assert out.shape[0] == 6
    assert out.tolist() == [1, 0, 0, 0, 0, 0]


def test_index_tensor_takes_the_width_it_is_given():
    """A given width is used as-is, and spares the pass over the index."""
    index = CountingIndex({"a": 0, "b": 5})

    out = index_tensor(["a"], index, width=6)

    assert out.tolist() == [1, 0, 0, 0, 0, 0]
    assert index.enumerations == 0


def test_multi_hot_encode_series_derives_the_width_once():
    """The width belongs to the vocabulary, not to a document: derived inside
    the per-row encode it walks thousands of entities per document, and the
    entity index is fixed for the whole split."""
    index = CountingIndex({"enz1": 0, "bac2": 1, "str3": 2})

    encoded = multi_hot_encode_series(
        pd.Series([["enz1"], ["bac2", "str3"], []]), index
    )

    assert index.enumerations == 1
    assert [row.tolist() for row in encoded] == [
        [1, 0, 0],
        [0, 1, 1],
        [0, 0, 0],
    ]


def test_compute_frequencies_means_and_clamps(stub):
    df = pd.DataFrame(
        {"entities": [torch.tensor([1, 0, 1]), torch.tensor([1, 1, 0])]}
    )
    dataset = stub(BrendaDataset, data=df)
    freq = compute_frequencies(dataset, "entities")
    # column means are [1.0, 0.5, 0.5]; the all-ones column is clamped below 1.
    assert freq[0].item() < 1.0
    assert freq[0].item() == pytest.approx(1 - 1e-5)
    assert freq[1].item() == pytest.approx(0.5)
    assert freq[2].item() == pytest.approx(0.5)


def test_compute_frequencies_never_stacks_the_whole_column(stub, monkeypatch):
    """The column is summed a row at a time, so no `[docs, labels]` tensor is
    ever materialised — `torch.stack` must not be reached at all."""

    def no_stack(*args, **kwargs):
        raise AssertionError("compute_frequencies materialised the column")

    rows = [torch.tensor([1, 0, 1]), torch.tensor([1, 1, 0])]
    dataset = stub(BrendaDataset, data=pd.DataFrame({"entities": rows}))

    monkeypatch.setattr(torch, "stack", no_stack)
    freq = compute_frequencies(dataset, "entities")

    assert freq[1].item() == pytest.approx(0.5)


def test_compute_frequencies_does_not_alias_the_frames_first_row(stub):
    """The accumulator starts at a fresh zero row, not at the first row.

    The column has to be float32 for this to bind: `Tensor.float()` returns
    *self* only when no conversion is needed, so an accumulator seeded with it
    adds every document into the frame's own labels. On the uint8 column the
    splits actually carry, the conversion copies and the same bug is invisible.
    """
    rows = [torch.tensor([1.0, 0.0, 1.0]), torch.tensor([1.0, 1.0, 0.0])]
    dataset = stub(BrendaDataset, data=pd.DataFrame({"entities": rows}))

    compute_frequencies(dataset, "entities")

    assert rows[0].tolist() == [1.0, 0.0, 1.0]
    assert rows[1].tolist() == [1.0, 1.0, 0.0]


@pytest.mark.parametrize(
    "rows",
    [
        pytest.param([[1, 0, 1], [1]], id="short-after-wide"),
        pytest.param([[1], [1, 0, 1]], id="wide-after-short"),
    ],
)
def test_compute_frequencies_rejects_a_ragged_column(stub, rows):
    """A short row is *broadcast* by `+=` where `torch.stack` used to raise, so
    `[[1, 0, 1], [1]]` would average to a plausible `[1.0, 0.5, 1.0]` instead
    of failing. The shape check is what keeps that a crash."""
    column = pd.Series(
        [torch.tensor(row, dtype=torch.float32) for row in rows], dtype=object
    )
    dataset = stub(BrendaDataset, data=pd.DataFrame({"entities": column}))

    with pytest.raises(ValueError, match="Ragged"):
        compute_frequencies(dataset, "entities")


def test_compute_frequencies_equals_the_stacked_mean_bitwise(stub):
    """Value identity with the tensor the stacked mean returned. Passes with
    either implementation; it is here so a future rewrite of the reduction
    cannot drift the numbers, which seed a classification head's bias."""
    numpy.random.seed(0)
    column = list(numpy.random.randint(0, 2, size=(97, 311), dtype="uint8"))
    dataset = stub(BrendaDataset, data=pd.DataFrame({"entities": column}))

    expected = (
        torch.stack([torch.tensor(e, dtype=torch.float32) for e in column])
        .mean(dim=0)
        .clamp(min=1e-5, max=1 - 1e-5)
    )

    assert torch.equal(compute_frequencies(dataset, "entities"), expected)


def test_compute_frequencies_rejects_an_empty_column(stub):
    dataset = stub(
        BrendaDataset, data=pd.DataFrame({"entities": pd.Series(dtype=object)})
    )
    with pytest.raises(ValueError, match="empty"):
        compute_frequencies(dataset, "entities")


def test_multi_hot_encode_series():
    index = {"enz1": 0, "bac2": 1, "str3": 2}
    encoded = multi_hot_encode_series(
        pd.Series([["enz1"], ["bac2", "str3"]]), index
    )
    assert encoded.iloc[0].tolist() == [1, 0, 0]
    assert encoded.iloc[1].tolist() == [0, 1, 1]
