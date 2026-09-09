"""Value-constrained aliases for the ranges a bare `int` or `float` hides.

Each alias is an `Annotated` type whose metadata beartype checks at call
time and every other reader ignores, so a signature states its range where
the parameter is declared. beartype is optional at runtime, so an alias is a
contract the claw enforces where it is installed, not a guard: a check whose
error message a caller relies on stays a check.
"""

import re
from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from beartype.vale import Is
else:
    try:
        from beartype.vale import Is
    except ModuleNotFoundError:

        class Is:
            """Metadata nobody reads, for an install without beartype."""

            def __class_getitem__(cls, predicate: object) -> object:
                return predicate


Positive = Annotated[int, Is[lambda n: n >= 1]]
"""A count that must be at least one: a size, a batch, a window."""

NonNegative = Annotated[int, Is[lambda n: n >= 0]]
"""A count that may be zero: a stride, a gap, an epoch, a step."""

# beartype reads `float` as float alone, not as the PEP 484 numeric tower,
# so the real-valued aliases admit `int` or a caller's `0` and `1` would be
# violations.
PositiveReal = Annotated[int | float, Is[lambda x: x > 0]]
"""A strictly positive number: a floor that keeps a logarithm finite."""

NonNegativeReal = Annotated[int | float, Is[lambda x: x >= 0]]
"""A number that may be zero: a focusing exponent, a loss weight."""

UnitInterval = Annotated[int | float, Is[lambda x: 0 <= x <= 1]]
"""A probability, threshold or share, inclusive at both ends."""

FuzzyScore = Annotated[int | float, Is[lambda x: 0 <= x <= 100]]
"""A `rapidfuzz` similarity, or the cutoff one is compared against."""

_ENTITY_ID = re.compile(r"[a-z]+[0-9]+")

EntityId = Annotated[str, Is[lambda s: _ENTITY_ID.fullmatch(s) is not None]]
"""A prefixed BRENDA entity ID such as `enz26836`: type prefix, then number.

Not a vocabulary column label, which may be the `UNK` sentinel instead.
"""
