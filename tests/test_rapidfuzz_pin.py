"""`rapidfuzz`'s upper bound must not silently harden into an exact pin.

`d0e4f14` turned `rapidfuzz>=3.14.1,<4` into `rapidfuzz==3.14.1,<4` as a side
effect of an unrelated python-version change; nothing in that commit's
message mentioned rapidfuzz. An exact pin beside a `<4` upper bound is the
tell that survives such an edit, since nobody writes both deliberately.
"""

import pathlib
import tomllib

from packaging.requirements import Requirement

_PYPROJECT = pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"


def test_rapidfuzz_is_not_exact_pinned() -> None:
    with _PYPROJECT.open("rb") as pyproject:
        config = tomllib.load(pyproject)

    requirement = next(
        Requirement(spec)
        for spec in config["project"]["dependencies"]
        if Requirement(spec).name == "rapidfuzz"
    )

    assert not any(
        specifier.operator == "==" for specifier in requirement.specifier
    )
