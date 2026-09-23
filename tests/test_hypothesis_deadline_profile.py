"""Regression test for the deadline profile `tests/conftest.py` loads.

None of the suite's property tests measure timing, so a slow machine
tripping Hypothesis's 200ms default deadline is a flake, not a signal.
"""

import time

from hypothesis import given, settings
from hypothesis import strategies as st


@given(st.just(None))
@settings(max_examples=1)
def test_a_slow_example_does_not_trip_the_default_deadline(_):
    """This decorator leaves `deadline` unset, the same shape every
    `*_hypothesis.py` test in the suite uses, so it only stays green
    because `conftest.py`'s profile makes that mean `None` here rather
    than Hypothesis's built-in 200ms.
    """
    time.sleep(0.25)
