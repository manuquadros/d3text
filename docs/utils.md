::: d3text.utils

## Value constraints

`d3text.constraints` holds the `Annotated` aliases that give a bare `int` or
`float` parameter its range: `Positive`, `NonNegative`, `PositiveReal`,
`NonNegativeReal`, `UnitInterval`, `FuzzyScore`, and `EntityId` for a
prefixed BRENDA ID. beartype checks the metadata at call time under the
package's import hook; every other reader, mypy included, sees the plain
base type. beartype is a development dependency, so the aliases are a
contract the hook enforces where it is installed, not a replacement for a
guard whose error message a caller relies on. The configuration fields keep
their pydantic bounds, which are checked at load time and which beartype
does not read.

::: d3text.constraints
