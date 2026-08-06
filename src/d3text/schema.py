"""Declarative description of the entity and relation types of a dataset.

A `Schema` is meant to be the single place that answers which entity types
exist, which prefix their IDs carry, and which relation types hold between
them. Today those facts are spelled out once per call site and have to be kept
in step by hand: `entity_cols` plus the `col[:3]` prefix slicing in
`data.brenda_dataset`, `list(classes.keys()) + ["OOS"]` in the model
constructors, and the hardcoded `("HasEnzyme", "HasSpecies", "none")` tuple in
`ETEBrendaModel.__init__`.

Nothing consumes this module yet — moving each of those call sites onto it is
SCHEMA-02..05 in `design/tickets.md`.

This is a leaf module: it imports nothing from `d3text`, so `d3text/__init__.py`
can export it without dragging in the BRENDA data layer.
"""

import dataclasses
import pathlib
from collections.abc import Callable


@dataclasses.dataclass(frozen=True)
class EntityType:
    """One entity type: a class label, and how its instances are identified.

    :param name: The class label, e.g. ``"enzymes"``. Doubles as the column
        name the corpus carries the type's mentions under.
    :param prefix: The tag prepended to a numeric database ID to make an entity
        ID unique across types, e.g. ``"enz"`` in ``enz26836``.
    :param has_ids: Whether instances of this type are grounded in database IDs
        at all. A type without IDs is detected as a class but never linked.
    :param vocab_path: Word list backing `DictTagger` for this type, if any.
    :param abbreviation_fn: How to shorten a mention of this type for the
        serialiser (e.g. ``"Escherichia coli"`` -> ``"E. coli"``).
    """

    name: str
    prefix: str
    has_ids: bool = True
    vocab_path: pathlib.Path | None = None
    abbreviation_fn: Callable[[str], str] | None = None


@dataclasses.dataclass(frozen=True)
class RelationType:
    """One relation type, or the null class of the relation classifier.

    :param name: The label, e.g. ``"HasEnzyme"``.
    :param subject_type: `EntityType.name` of the first argument.
    :param object_type: `EntityType.name` of the second argument.
    :param is_none: Marks the null class — the label a candidate pair gets when
        no relation holds. It has no arguments, which is why the argument types
        are optional; `Schema.validate` requires them of every other relation.
    """

    name: str
    subject_type: str | None = None
    object_type: str | None = None
    is_none: bool = False


@dataclasses.dataclass(frozen=True)
class Schema:
    """The entity and relation types a model is built over.

    Frozen and built from tuples, hence hashable: a schema is identity, not
    state — two runs over the same schema must be comparable, and a mutable one
    could drift out of step with a model's already-sized output layers.
    """

    entity_types: tuple[EntityType, ...]
    relation_types: tuple[RelationType, ...] = ()

    def __post_init__(self) -> None:
        self.validate()

    @property
    def class_names(self) -> tuple[str, ...]:
        """Entity-type names, in declaration order.

        This is the order of the class head's target columns. The extra column
        the head scores on top — ``OOS`` — is deliberately absent: it is a
        property of the head, not of the data, and the models append and locate
        it by name themselves.
        """
        return tuple(entity_type.name for entity_type in self.entity_types)

    @property
    def relation_names(self) -> tuple[str, ...]:
        """Relation labels, in declaration order.

        Unlike the entity head's ``UNK`` and the class head's ``OOS``, the null
        relation class *is* part of the schema: it is one of the relation
        head's ordinary softmax columns, and the loss targets index it.
        """
        return tuple(
            relation_type.name for relation_type in self.relation_types
        )

    @property
    def prefix_to_type(self) -> dict[str, EntityType]:
        """Entity-ID prefix -> the type it denotes (``"enz"`` -> enzymes).

        `validate` rejects duplicate prefixes, so no type is shadowed here.
        """
        return {
            entity_type.prefix: entity_type for entity_type in self.entity_types
        }

    @property
    def none_relation_index(self) -> int:
        """Column of the null relation class in the relation head's output.

        Found by the `is_none` flag, not by name or by position: the training
        targets are filled with this index, so a schema that names its null
        class something other than ``"none"``, or declares it first, still has
        to land on the right column.

        :raises ValueError: if the schema declares no relation types.
        """
        for index, relation_type in enumerate(self.relation_types):
            if relation_type.is_none:
                return index
        raise ValueError(
            "schema declares no relation types, so there is no `none` column"
        )

    def validate(self) -> None:
        """Check the schema's internal consistency.

        Called from `__post_init__`, so an invalid `Schema` cannot be built and
        no consumer has to remember to ask; public so that a schema assembled
        elsewhere (parsed from a config, read back from a checkpoint) can be
        re-checked at the boundary.

        :raises ValueError: on an empty or blank-named entity type, duplicate
            entity-type names or prefixes, duplicate relation names, a relation
            naming an unknown entity type, a non-null relation missing an
            argument type, or any number of null relation classes other than
            exactly one (unless no relation types are declared at all).
        """
        if not self.entity_types:
            raise ValueError("a schema must declare at least one entity type")

        for entity_type in self.entity_types:
            if not entity_type.name or not entity_type.prefix:
                raise ValueError(
                    "entity types need a non-empty name and prefix, got "
                    f"name={entity_type.name!r} prefix={entity_type.prefix!r}"
                )

        _reject_duplicates(self.class_names, "entity type names")
        _reject_duplicates(
            tuple(entity_type.prefix for entity_type in self.entity_types),
            "entity ID prefixes",
        )
        _reject_duplicates(self.relation_names, "relation type names")

        known = set(self.class_names)
        for relation_type in self.relation_types:
            arguments = (relation_type.subject_type, relation_type.object_type)
            if not relation_type.is_none and None in arguments:
                raise ValueError(
                    f"relation type {relation_type.name!r} must declare both a "
                    "subject_type and an object_type"
                )
            for argument in arguments:
                if argument is not None and argument not in known:
                    raise ValueError(
                        f"relation type {relation_type.name!r} names unknown "
                        f"entity type {argument!r}; known: {sorted(known)}"
                    )

        # No relation types at all is a classification-only schema
        # (`BrendaClassificationModel`, `NERClassificationModel`), which needs
        # no null class. Declaring some obliges exactly one.
        none_classes = [rt.name for rt in self.relation_types if rt.is_none]
        if self.relation_types and len(none_classes) != 1:
            raise ValueError(
                "a schema with relation types needs exactly one `is_none` "
                f"relation type, got {len(none_classes)}: {none_classes}"
            )


def _reject_duplicates(names: tuple[str, ...], what: str) -> None:
    """:raises ValueError: if `names` repeats a value."""
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate {what}: {duplicates}")
