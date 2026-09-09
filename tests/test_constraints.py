"""The value-constrained aliases admit their range and refuse the rest."""

import math

import pytest
import torch
from beartype.door import is_bearable
from beartype.roar import BeartypeCallHintParamViolation

from d3text import constraints
from d3text.embeddings_store import StoreProvenance
from d3text.models.base import focal_cross_entropy
from d3text.models.heads import ClassificationHead

ADMITTED = {
    "Positive": (1, 32, 10**9),
    "NonNegative": (0, 1, 10**9),
    "PositiveReal": (1e-5, 1, 2.0),
    "NonNegativeReal": (0, 0.0, 2.0, 1e9),
    "UnitInterval": (0, 0.0, 0.5, 1, 1.0),
    "FuzzyScore": (0, 80.0, 93.0, 100),
    "EntityId": ("enz26836", "bac42", "str1234", "oth567", "e0"),
}
REFUSED = {
    "Positive": (0, -1, 1.0),
    "NonNegative": (-1, 0.0),
    "PositiveReal": (0, 0.0, -1e-5, math.nan),
    "NonNegativeReal": (-1, -0.1, math.nan),
    "UnitInterval": (-0.1, 1.5, 5.0, math.nan),
    "FuzzyScore": (-1, 100.5, 200.0, math.nan),
    "EntityId": ("UNK", "enz", "26836", "ENZ1", "enz 1", ""),
}


@pytest.mark.parametrize(
    "alias, value",
    [(a, v) for a, values in ADMITTED.items() for v in values],
)
def test_the_alias_admits_its_range(alias, value):
    assert is_bearable(value, getattr(constraints, alias))


@pytest.mark.parametrize(
    "alias, value",
    [(a, v) for a, values in REFUSED.items() for v in values],
)
def test_the_alias_refuses_what_is_outside_it(alias, value):
    assert not is_bearable(value, getattr(constraints, alias))


def test_a_real_alias_admits_an_integer():
    """beartype does not apply the numeric tower to `float`, so the aliases
    spell `int | float` themselves; a caller's `0` and `1` must pass."""
    assert is_bearable(0, constraints.UnitInterval)
    assert is_bearable(1, constraints.UnitInterval)
    assert not is_bearable(0, constraints.PositiveReal)


def test_the_claw_enforces_an_alias_on_a_head_size():
    with pytest.raises(BeartypeCallHintParamViolation):
        ClassificationHead(input_size=0, n_entities=5, n_classes=3)


def test_the_claw_enforces_an_alias_on_a_loss_exponent():
    """A negative gamma gives inf or nan losses rather than an error, which
    is why the exponent's range lives in the signature."""
    preds = torch.zeros(2, 3)
    targets = torch.tensor([0, 1])
    with pytest.raises(BeartypeCallHintParamViolation):
        focal_cross_entropy(preds, targets, gamma=-1.0)


def test_the_claw_enforces_an_alias_on_a_dataclass_field():
    with pytest.raises(BeartypeCallHintParamViolation):
        StoreProvenance(base_model="m", max_length=0, stride=0)
