"""Pure unit tests for the vocabulary-independent helpers in data/data.py.

None of these touch HDF5 or the BRENDA files (see tests/data/test_dataset.py
for the fixture-backed dataset tests).
"""

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


def test_index_tensor_length_follows_max_index():
    # Non-contiguous index: width is max(index)+1, not len(index).
    out = index_tensor(["a"], {"a": 0, "b": 5})
    assert out.shape[0] == 6
    assert out.tolist() == [1, 0, 0, 0, 0, 0]


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


def test_multi_hot_encode_series():
    index = {"enz1": 0, "bac2": 1, "str3": 2}
    encoded = multi_hot_encode_series(
        pd.Series([["enz1"], ["bac2", "str3"]]), index
    )
    assert encoded.iloc[0].tolist() == [1, 0, 0]
    assert encoded.iloc[1].tolist() == [0, 1, 1]
