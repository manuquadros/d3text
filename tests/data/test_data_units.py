"""Pure unit tests for the vocabulary-independent helpers in data/data.py.

None of these touch HDF5 or the BRENDA files (see tests/data/test_dataset.py
for the fixture-backed dataset tests).
"""

import numpy
import pandas as pd
import pytest
import torch

from d3text.data.data import BrendaDataset, compute_frequencies


def test_compute_frequencies_means_and_clamps(stub):
    df = pd.DataFrame(
        {"classes": [torch.tensor([1, 0, 1]), torch.tensor([1, 1, 0])]}
    )
    dataset = stub(BrendaDataset, data=df)
    freq = compute_frequencies(dataset, "classes")
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
    dataset = stub(BrendaDataset, data=pd.DataFrame({"classes": rows}))

    monkeypatch.setattr(torch, "stack", no_stack)
    freq = compute_frequencies(dataset, "classes")

    assert freq[1].item() == pytest.approx(0.5)


def test_compute_frequencies_does_not_alias_the_frames_first_row(stub):
    """The accumulator starts at a fresh zero row, not at the first row.

    The column has to be float32 for this to bind: `Tensor.float()` returns
    *self* only when no conversion is needed, so an accumulator seeded with it
    adds every document into the frame's own labels. On the uint8 column the
    splits actually carry, the conversion copies and the same bug is invisible.
    """
    rows = [torch.tensor([1.0, 0.0, 1.0]), torch.tensor([1.0, 1.0, 0.0])]
    dataset = stub(BrendaDataset, data=pd.DataFrame({"classes": rows}))

    compute_frequencies(dataset, "classes")

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
    dataset = stub(BrendaDataset, data=pd.DataFrame({"classes": column}))

    with pytest.raises(ValueError, match="Ragged"):
        compute_frequencies(dataset, "classes")


def test_compute_frequencies_equals_the_stacked_mean_bitwise(stub):
    """Value identity with the tensor the stacked mean returned. Passes with
    either implementation; it is here so a future rewrite of the reduction
    cannot drift the numbers, which seed a classification head's bias."""
    numpy.random.seed(0)
    column = list(numpy.random.randint(0, 2, size=(97, 311), dtype="uint8"))
    dataset = stub(BrendaDataset, data=pd.DataFrame({"classes": column}))

    expected = (
        torch.stack([torch.tensor(e, dtype=torch.float32) for e in column])
        .mean(dim=0)
        .clamp(min=1e-5, max=1 - 1e-5)
    )

    assert torch.equal(compute_frequencies(dataset, "classes"), expected)


def test_compute_frequencies_rejects_an_empty_column(stub):
    dataset = stub(
        BrendaDataset, data=pd.DataFrame({"classes": pd.Series(dtype=object)})
    )
    with pytest.raises(ValueError, match="empty"):
        compute_frequencies(dataset, "classes")
