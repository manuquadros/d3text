"""Property-based tests for the embeddings-store codec.

`test_embeddings_store.py` pins the round trip at a handful of hand-picked
tensors. What the codec actually promises is a property over any token/feature
matrix — shape preserved exactly, values surviving up to bf16's rounding — so
this generates the shapes and values. Marked `slow`.
"""

import numpy
import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from d3text.embeddings_store import bytes_to_tensor, tensor_to_bytes

pytestmark = pytest.mark.slow

# 0 is a legal dimension the uint32 shape header must carry, so it is
# drawn rather than assumed away.
_DIM = st.integers(min_value=0, max_value=64)

# Real activations stay far below float32's max; the round trip is claimed
# only over this range -- near the max, bf16 rounding can overflow (see
# `test_a_value_near_the_float32_max_can_overflow_to_infinity`).
_REALISTIC_FLOAT32 = st.floats(
    width=32,
    allow_nan=False,
    allow_infinity=False,
    min_value=-1e10,
    max_value=1e10,
)


@st.composite
def _token_feature_tensor(
    draw: st.DrawFn, values: st.SearchStrategy[float] = _REALISTIC_FLOAT32
) -> torch.Tensor:
    rows = draw(_DIM)
    columns = draw(_DIM)
    entries = draw(
        st.lists(values, min_size=rows * columns, max_size=rows * columns)
    )
    return torch.tensor(entries, dtype=torch.float32).reshape(rows, columns)


@given(tensor=_token_feature_tensor())
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_any_token_feature_matrix_survives_the_round_trip(tensor):
    restored = bytes_to_tensor(tensor_to_bytes(tensor))

    assert restored.shape == tensor.shape
    assert restored.dtype == torch.bfloat16
    # bf16 rounding, not exactness, is the contract -- `atol` covers the
    # region near zero where a relative tolerance alone is meaningless.
    torch.testing.assert_close(restored.float(), tensor, rtol=1e-2, atol=1e-2)


@given(tensor=_token_feature_tensor())
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_a_second_round_trip_is_a_fixed_point(tensor):
    """Once a value has been rounded to bf16, re-encoding it must not move it
    again -- the property `EmbeddingsStore` relies on implicitly, since a
    document can be re-embedded and re-stored (`precompute-embeddings -f`)."""
    once = bytes_to_tensor(tensor_to_bytes(tensor))
    twice = bytes_to_tensor(tensor_to_bytes(once.float()))

    torch.testing.assert_close(twice, once)


# bf16 keeps fp32's exponent but only 7 mantissa bits, so round-to-nearest
# can carry a finite fp32 value near its max past bf16's largest finite value
# and into +/-inf.
_FLOAT32_MAX = float(numpy.finfo(numpy.float32).max)
_NEAR_FLOAT32_MAX_LOWER = float(numpy.float32(3.0e38))
_NEAR_FLOAT32_MAX = st.floats(
    width=32,
    allow_nan=False,
    min_value=_NEAR_FLOAT32_MAX_LOWER,
    max_value=_FLOAT32_MAX,
) | st.floats(
    width=32,
    allow_nan=False,
    min_value=-_FLOAT32_MAX,
    max_value=-_NEAR_FLOAT32_MAX_LOWER,
)


@given(tensor=_token_feature_tensor(values=_NEAR_FLOAT32_MAX))
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_a_value_near_the_float32_max_can_overflow_to_infinity(tensor):
    restored = bytes_to_tensor(tensor_to_bytes(tensor))

    # Not a round-trip assertion: the point is that this region is where the
    # round-trip property stops holding, documented rather than silently
    # left for the store's first NaN-shaped bug report to rediscover.
    still_finite = torch.isfinite(restored)
    close_where_finite = torch.zeros_like(tensor, dtype=torch.bool)
    close_where_finite[still_finite] = torch.isclose(
        restored[still_finite].float(),
        tensor[still_finite],
        rtol=1e-2,
        atol=1e-2,
    )
    assert bool((still_finite == close_where_finite).all())
