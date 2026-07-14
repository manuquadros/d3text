"""Declarative description of an extraction target.

A `Schema` names the entity types a corpus annotates and the relations that
may hold between them, so that the dataset adapters, the models and the
taggers can read one description instead of each hard-coding its own copy of
the BRENDA one.

The dataclasses are frozen and validated on construction: an invalid `Schema`
cannot be built, so consumers may trust `class_names`, `prefix_to_type` and
`none_relation_index` without re-checking them.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass

__all__ = ["EntityType", "RelationType", "Schema"]


@dataclass(frozen=True)
class EntityType:
    """One entity class: a column of the class head, a row block of the
    class matrix, and a prefix namespace for entity IDs.

    :param name: The class name, as the corpus column names it (`enzymes`).
    :param prefix: Prepended to a corpus ID to make it unique across types
        (`enz` + `26836` -> `enz26836`).
    :param has_ids: Whether instances of the type are identified by a
        database ID at all. A type without IDs is detected but never linked,
        so it contributes a class column and no entity columns.
    :param vocab_path: Term list for the dictionary tagger, named relative to
        the corpus data directory. `None` for a type with no term list.
    :param abbreviation_fn: Renders an entity name in its abbreviated form
        for serialisation (`Escherichia coli` -> `E. coli`).
    """

    name: str
    prefix: str
    has_ids: bool = True
    vocab_path: str | None = None
    abbreviation_fn: Callable[[str], str] | None = None


@dataclass(frozen=True)
class RelationType:
    """One column of the relation head.

    The argument types are the types the relation is *declared* over; they say
    nothing about which arguments a given corpus row actually pairs.

    :param name: The label, as the corpus names it (`HasEnzyme`).
    :param subject_types: Names of the entity types admissible as subject.
    :param object_types: Idem, as object.
    :param is_none: Marks the label for a candidate pair that stands in no
        relation. Exactly one relation type carries it, and it takes no
        arguments.
    """

    name: str
    subject_types: tuple[str, ...] = ()
    object_types: tuple[str, ...] = ()
    is_none: bool = False


@dataclass(frozen=True)
class Schema:
    """The entity types and relation types of one extraction target.

    Order is meaningful in both tuples: `entity_types` fixes the class head's
    column order (and so the class matrix's columns), `relation_types` the
    relation head's. Neither may be permuted without retraining.
    """

    entity_types: tuple[EntityType, ...]
    relation_types: tuple[RelationType, ...]

    def __post_init__(self) -> None:
        self.validate()

    @property
    def class_names(self) -> tuple[str, ...]:
        """Entity type names in class-head column order, without OOS."""
        return tuple(et.name for et in self.entity_types)

    @property
    def relation_names(self) -> tuple[str, ...]:
        """Relation names in relation-head column order, `none` last."""
        return tuple(rt.name for rt in self.relation_types)

    @property
    def prefix_to_type(self) -> dict[str, EntityType]:
        """Entity ID prefix (`enz`) -> the type it namespaces."""
        return {et.prefix: et for et in self.entity_types}

    @property
    def none_relation_index(self) -> int:
        """Column of the relation head that scores "no relation"."""
        return next(
            ix for ix, rt in enumerate(self.relation_types) if rt.is_none
        )

    def validate(self) -> None:
        """Raise `ValueError` unless the schema upholds every invariant the
        heads rely on. Called on construction."""
        if not self.entity_types:
            raise ValueError("A schema needs at least one entity type")

        _reject_duplicates(
            (et.name for et in self.entity_types), "entity type name"
        )
        _reject_duplicates(
            (et.prefix for et in self.entity_types), "entity ID prefix"
        )
        _reject_duplicates(
            (rt.name for rt in self.relation_types), "relation type name"
        )

        known = set(self.class_names)
        for rt in self.relation_types:
            unknown = (set(rt.subject_types) | set(rt.object_types)) - known
            if unknown:
                raise ValueError(
                    f"Relation {rt.name!r} takes unknown entity types "
                    f"{sorted(unknown)}; the schema declares {sorted(known)}"
                )
            if rt.is_none:
                if rt.subject_types or rt.object_types:
                    raise ValueError(
                        f"The `none` relation {rt.name!r} stands for the "
                        "absence of a relation, so it takes no arguments"
                    )
            elif not (rt.subject_types and rt.object_types):
                raise ValueError(
                    f"Relation {rt.name!r} must declare both a subject and "
                    "an object type"
                )

        none_types = [rt.name for rt in self.relation_types if rt.is_none]
        if len(none_types) != 1:
            raise ValueError(
                "A schema needs exactly one `none` relation type, "
                f"got {none_types}"
            )
        if not self.relation_types[-1].is_none:
            # The heads score `none` in the tail column, and the aligner writes
            # gold the entity head never proposed into that same column.
            raise ValueError(
                f"The `none` relation {none_types[0]!r} must come last, "
                f"got {list(self.relation_names)}"
            )


def _reject_duplicates(names: Iterable[str], what: str) -> None:
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise ValueError(f"Duplicate {what}: {name!r}")
        seen.add(name)
