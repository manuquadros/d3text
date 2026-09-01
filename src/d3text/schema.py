"""Declarative description of the entity and relation types of a dataset.

The single place that answers which entity types exist, which prefix their IDs
carry, and which relation types hold between them. `BRENDA_SCHEMA` lives here
rather than beside its loader because the leaf modules need it and none of them
may import the BRENDA data layer. A leaf itself: it imports nothing from
`d3text`.
"""

import dataclasses
import pathlib
from collections.abc import Callable

# Mirrors d3text.data.data.DATA_DIR without importing it: that module pulls in
# torch, h5py and the rest of the training stack, and this one has to stay a
# leaf so d3text/__init__.py can export it before the BRENDA data layer loads.
DATA_DIR = pathlib.Path(__file__).parent.parent.parent / "data"


@dataclasses.dataclass(frozen=True)
class EntityType:
    """One entity type: a class label, and how its instances are identified.

    :param name: the class label, e.g. `"enzymes"`. Doubles as the column name
        the corpus carries the type's mentions under.
    :param prefix: the tag prepended to a numeric database ID to make an entity
        ID unique across types, e.g. `"enz"` in `enz26836`.
    :param has_ids: whether instances of this type are grounded in database IDs
        at all. A type without IDs is detected as a class but never linked.
    :param vocab_path: word list backing `DictTagger` for this type, if any.
    :param abbreviation_fn: how to shorten a mention of this type for the
        serialiser.
    """

    name: str
    prefix: str
    has_ids: bool = True
    vocab_path: pathlib.Path | None = None
    abbreviation_fn: Callable[[str], str] | None = None


@dataclasses.dataclass(frozen=True)
class RelationType:
    """One relation type, or the null class of the relation classifier.

    :param name: the label, e.g. `"HasEnzyme"`.
    :param subject_types: the entity-type names the first argument may take. A
        tuple because a relation's subject is not always one type — BRENDA's
        `HasEnzyme` holds between an enzyme and whichever of a bacterium, a
        strain or an other-organism names it.
    :param object_type: the entity-type name of the second argument.
    :param is_none: marks the null class, which has no arguments;
        `Schema.validate` requires them of every other relation.
    """

    name: str
    subject_types: tuple[str, ...] = ()
    object_type: str | None = None
    is_none: bool = False


@dataclasses.dataclass(frozen=True)
class Schema:
    """The entity and relation types a model is built over.

    Frozen and built from tuples, hence hashable: a schema is identity, not
    state, and a mutable one could drift out of step with a model's
    already-sized output layers.
    """

    entity_types: tuple[EntityType, ...]
    relation_types: tuple[RelationType, ...] = ()

    def __post_init__(self) -> None:
        self.validate()

    @property
    def class_names(self) -> tuple[str, ...]:
        """Entity-type names, in declaration order.

        This is the order of the class head's target columns. `OOS` is
        deliberately absent: it is a property of the head, not of the data.

        :return: the type names in declaration order.
        """
        return tuple(entity_type.name for entity_type in self.entity_types)

    @property
    def relation_names(self) -> tuple[str, ...]:
        """Relation labels, in declaration order.

        Unlike `UNK` and `OOS`, the null relation class *is* part of the
        schema: it is one of the head's ordinary softmax columns and the
        targets index it.

        :return: the relation labels in declaration order.
        """
        return tuple(
            relation_type.name for relation_type in self.relation_types
        )

    @property
    def prefix_to_type(self) -> dict[str, EntityType]:
        """Entity-ID prefix -> the type it denotes (`"enz"` -> enzymes).

        :return: the mapping; `validate` rejects duplicate prefixes, so no type
            is shadowed.
        """
        return {
            entity_type.prefix: entity_type for entity_type in self.entity_types
        }

    @property
    def none_relation_index(self) -> int:
        """Column of the null relation class in the relation head's output.

        Found by the `is_none` flag rather than by name or position, so a
        schema that names its null class something else still lands on the
        right column.

        :return: the column index.
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

        Called from `__post_init__`, so an invalid schema cannot be built;
        public so one assembled elsewhere can be re-checked at the boundary.

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
            if not relation_type.is_none and (
                not relation_type.subject_types
                or relation_type.object_type is None
            ):
                raise ValueError(
                    f"relation type {relation_type.name!r} must declare both a "
                    "subject_types and an object_type"
                )
            arguments = tuple(relation_type.subject_types) + (
                ()
                if relation_type.object_type is None
                else (relation_type.object_type,)
            )
            for argument in arguments:
                if argument not in known:
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


# Declaration order is the class head's column order, the class matrix's, and
# the token-label codes', so it is not free to change: a checkpoint's class
# logits are positional, and `d3text.token_labels` records this order inside
# every artifact it writes for the same reason. The prefixes are the ones
# `brenda_references.preprocess_labels` stamps onto the numeric BRENDA IDs.
#
# Relation declaration order is the relation head's column order, matching the
# `("HasEnzyme", "HasSpecies", "none")` tuple `ETEBrendaModel` used to hardcode.
# `HasEnzyme`'s subject is a bacterium, a strain or an other-organism —
# `brenda_references.preprocess_relations` accepts any of the three and keys
# the pair by whichever one actually holds the subject ID — hence the tuple of
# `subject_types` rather than one name. `HasSpecies` holds strain -> bacterium.
BRENDA_SCHEMA = Schema(
    entity_types=(
        EntityType(
            name="strains", prefix="str", vocab_path=DATA_DIR / "strains.txt"
        ),
        EntityType(
            name="bacteria", prefix="bac", vocab_path=DATA_DIR / "bacteria.txt"
        ),
        EntityType(name="other_organisms", prefix="oth"),
        EntityType(
            name="enzymes", prefix="enz", vocab_path=DATA_DIR / "enzymes.txt"
        ),
    ),
    relation_types=(
        RelationType(
            name="HasEnzyme",
            subject_types=("bacteria", "strains", "other_organisms"),
            object_type="enzymes",
        ),
        RelationType(
            name="HasSpecies",
            subject_types=("strains",),
            object_type="bacteria",
        ),
        RelationType(name="none", is_none=True),
    ),
)
"""The BRENDA corpus's entity types.

Every name is a column of the split frames as well as a class label, which is
what lets `d3text.corpus` read a document's gold entity set off a row without
being told the column names.
"""
