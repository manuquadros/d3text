"""Property-based tests for the embeddings-store codec.

`test_embeddings_store.py` pins the round trip at a handful of hand-picked
tensors (a small literal matrix, values past the fp16 range, a non-contiguous
view, ...). What `tensor_to_bytes`/`bytes_to_tensor` actually promise is a
*property* over any token/feature matrix — shape is preserved exactly and
values survive up to bf16's rounding — so this file generates the shapes and
values instead of enumerating them by hand.

Marked `slow`: `@given` draws many examples per test, so each one costs a
handful of seconds rather than milliseconds.
"""

import numpy
import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from d3text.embeddings_store import bytes_to_tensor, tensor_to_bytes

pytestmark = pytest.mark.slow

# `tensor_to_bytes` is annotated `Float[Tensor, "token feature"]` and its
# header packs rows/columns as unsigned 32-bit ints -- 0 rows is a legal
# matrix shape numpy/torch both accept, so it is included rather than assumed
# away.
_DIM = st.integers(min_value=0, max_value=64)

# Real activations out of a frozen transformer stay orders of magnitude below
# float32's max (~3.4e38): the existing pinned test only goes as far as 1e6.
# This range is what the round-trip property below actually claims -- see
# `test_a_value_near_the_float32_max_can_overflow_to_infinity` for what
# happens once a value approaches the exponent's edge.
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


# Range surrounding float32's largest finite magnitude (~3.4028e38): bf16
# keeps fp32's 8-bit exponent but only 7 mantissa bits, so round-to-nearest on
# the cast can round a finite fp32 value up past bf16's largest finite value
# and into +/-inf. `test_a_value_past_the_half_precision_range_is_kept` pins
# one point (1e6) well inside the safe region; Hypothesis found this edge
# unprompted by drawing from the full float32 range.
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
