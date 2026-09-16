"""The StrainInfo write-back must not join on a generic designation.

`apiadapters.straininfo.StrainInfoAdapterBase.retrieve_strain_models` (not
editable here) pairs a returned record with whichever BRENDA strain shares
its *first* matching designation, with no organism constraint. `A2` is short
enough that StrainInfo can return a record for an unrelated species under
that name, and the unconstrained rule would join it anyway; these tests pin
that `gated_pairing` refuses that join while still admitting a real
culture-collection accession, and that reverting to the unconstrained rule
(passing a shape predicate that admits everything) reproduces the bug.
"""

from apiadapters.straininfo.straininfo import Culture, Strain
from d3types import Strain as BrendaStrain

from scripts.fix_missing_strains import (
    gated_pairing,
    known_designations,
    matching_designations,
)


def _strain(
    designations: frozenset[str], cultures: frozenset[Culture]
) -> Strain:
    """A StrainInfo `Strain`, with both defaulted fields spelled out.

    `designations` and `cultures` carry their pydantic default inside
    `Annotated[...]` metadata rather than as an assigned class attribute, a
    form mypy's PEP 681 support does not read as optional — so every call
    site passes both explicitly rather than tripping a false `call-arg`.
    """
    return Strain(designations=designations, cultures=cultures)


def _culture(strain_number: str) -> Culture:
    """A `Culture`, sidestepping the `id`/`siid` validation-alias mismatch.

    `Culture.siid`'s validation alias is `"id"`, which `Culture(id=...)`
    satisfies at runtime but which mypy's synthesized signature (keyed on
    the field name) rejects; `model_validate` takes the alias dynamically.
    """
    return Culture.model_validate({"id": 1, "strain_number": strain_number})


def test_known_designations_maps_normalized_names_to_brenda_ids() -> None:
    strains = {
        1: BrendaStrain(designations=frozenset({"A2"}), cultures=frozenset()),
        2: BrendaStrain(
            designations=frozenset({"DSM 4252"}), cultures=frozenset()
        ),
    }

    assert known_designations(strains) == {"A2": 1, "DSM 4252": 2}


def test_gated_pairing_refuses_a_generic_designation() -> None:
    """The regression case: `A2` matches, but is not a registry identifier."""
    known_names = {"A2": 1}
    entry = _strain(designations=frozenset({"A2"}), cultures=frozenset())

    assert gated_pairing(entry, known_names) is None


def test_gated_pairing_admits_a_collection_number() -> None:
    known_names = {"DSM 4252": 2}
    entry = _strain(
        designations=frozenset(), cultures=frozenset({_culture("DSM 4252")})
    )

    assert gated_pairing(entry, known_names) == (2, "DSM 4252")


def test_gated_pairing_without_the_shape_gate_reproduces_the_bug() -> None:
    """Pre-fix behavior: an unconstrained predicate joins the generic name.

    This is what `retrieve_strain_models`'s own unconstrained
    `next(filter(lambda w: w in known_names, names))` does, and is exactly
    the join `gated_pairing`'s default `is_collection_number` gate exists to
    refuse.
    """
    known_names = {"A2": 1}
    entry = _strain(designations=frozenset({"A2"}), cultures=frozenset())

    assert gated_pairing(
        entry, known_names, is_collection_number=lambda _name: True
    ) == (1, "A2")


def test_matching_designations_is_empty_for_an_unrelated_record() -> None:
    known_names = {"DSM 4252": 2}
    entry = _strain(designations=frozenset({"F1"}), cultures=frozenset())

    assert matching_designations(entry, known_names) == frozenset()
    assert gated_pairing(entry, known_names) is None
