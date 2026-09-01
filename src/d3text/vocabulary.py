"""The label vocabulary a model's heads were sized to.

Both heads are positional and nothing in a `state_dict` records which ID owns
which column, so a same-width repermutation scores every entity against another
entity's logits and reads as a mediocre model rather than a broken one.
`Vocabulary` is that order made explicit, written into the checkpoint and read
back. Leaf module: torch and `d3text.schema` only.
"""

import collections
import dataclasses
from collections.abc import Mapping, Sequence, Set
from typing import Any

import torch
from jaxtyping import Float
from ordered_set import OrderedSet
from torch import Tensor

from d3text.schema import Schema

# `torch.load` defaults to `weights_only=True`, which admits tensors and plain
# builtins and nothing else, so the payload is lists and dicts rather than a
# pickled `Vocabulary`. Keeping the checkpoint loadable without trusting it is
# worth more than the convenience of pickling the dataclass.
Payload = dict[str, Any]


@dataclasses.dataclass(frozen=True)
class Vocabulary:
    """The entity and class columns a checkpoint's heads were trained on.

    :param entities: entity IDs in entity-head column order. The head's
        trailing `UNK` column is deliberately absent, as `Schema.class_names`
        omits `OOS`.
    :param class_map: class name -> the entity IDs of that class, in class-head
        column order. A class with no groundable instances still holds its key,
        because the class head is sized from this mapping.
    """

    entities: tuple[str, ...]
    class_map: dict[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def from_class_map(cls, class_map: Mapping[str, Set[str]]) -> "Vocabulary":
        """The vocabulary a corpus's class map implies.

        Each type's IDs are sorted before they are laid down, so one training
        split yields one column order in every process: a `set` of strings
        iterates in an order that depends on `PYTHONHASHSEED`.

        :param class_map: class name -> its entity IDs, in the schema's order.
        :return: the vocabulary those columns define.
        """
        ordered = {
            name: tuple(sorted(entity_ids))
            for name, entity_ids in class_map.items()
        }
        return cls(
            entities=tuple(OrderedSet[str]().union(*ordered.values())),
            class_map=ordered,
        )

    @classmethod
    def from_index(
        cls,
        entity_index: Mapping[str, int],
        class_map: Mapping[str, Set[str]],
    ) -> "Vocabulary":
        """The vocabulary a built dataset is carrying.

        `entity_index` is authoritative for the column order, since it is what
        the labels were encoded against; `class_map` contributes only
        membership.

        :param entity_index: entity ID -> its column.
        :param class_map: class name -> its entity IDs.
        :return: the vocabulary those columns define.
        :raises ValueError: if `entity_index` is not a bijection onto
            `range(len(entity_index))`; a caller holding one is already wrong
            and must not have it written to disk.
        """
        columns = sorted(entity_index.values())
        if columns != list(range(len(entity_index))):
            raise ValueError(
                "entity_index must number its entities 0..n-1 exactly once "
                f"each, got {len(entity_index)} entities over columns "
                f"{columns[:8]}{'...' if len(columns) > 8 else ''}"
            )
        return cls(
            entities=tuple(
                sorted(entity_index, key=lambda name: entity_index[name])
            ),
            class_map={
                name: tuple(sorted(entity_ids))
                for name, entity_ids in class_map.items()
            },
        )

    @property
    def entity_index(self) -> dict[str, int]:
        """Entity ID -> the column it owns in the entity head's output.

        :return: the index the labels were encoded against.
        """
        return {
            entity_id: column for column, entity_id in enumerate(self.entities)
        }

    @property
    def class_names(self) -> tuple[str, ...]:
        """Class labels in class-head column order.

        :return: the labels in column order.
        """
        return tuple(self.class_map)

    def class_matrix(self) -> Float[Tensor, "entities classes"]:
        """One-hot rows mapping each entity onto the classes it belongs to.

        Built by walking the classes rather than by inverting them, so an ID
        declared under two types lights both columns.

        :return: an `[entities, classes]` matrix.
        """
        index = self.entity_index
        matrix = torch.zeros(
            len(self.entities), len(self.class_map), dtype=torch.float32
        )
        for column, entity_ids in enumerate(self.class_map.values()):
            for entity_id in entity_ids:
                matrix[index[entity_id], column] = 1.0
        return matrix

    def as_class_map(self) -> dict[str, set[str]]:
        """`class_map` in the `set`-valued shape the model constructors take.

        :return: class name -> its entity IDs.
        """
        return {
            name: set(entity_ids) for name, entity_ids in self.class_map.items()
        }

    def validate(self) -> None:
        """Check the vocabulary's internal consistency.

        Called from `__post_init__`; public so one read back off a checkpoint
        can be re-checked at the boundary.

        :raises ValueError: on a repeated entity ID or class name, or on a
            class naming an entity that owns no column — what a truncated or
            hand-edited payload looks like, which would otherwise surface as a
            `KeyError` deep inside `class_matrix`.
        """
        _reject_duplicates(self.entities, "entity IDs")
        _reject_duplicates(tuple(self.class_map), "class names")

        for name, entity_ids in self.class_map.items():
            _reject_duplicates(entity_ids, f"entity IDs under class {name!r}")

        classified = {
            entity_id
            for entity_ids in self.class_map.values()
            for entity_id in entity_ids
        }
        unknown = sorted(classified - set(self.entities))
        if unknown:
            raise ValueError(
                "class_map names entities that own no column: "
                f"{unknown[:8]}{'...' if len(unknown) > 8 else ''}"
            )

    def check_fits(self, schema: Schema) -> None:
        """Check that a model built under `schema` can wear this vocabulary.

        The class head's targets are built in schema order and its columns in
        vocabulary order, so equal sets in a different order is the dangerous
        case.

        :param schema: the schema the model was built under.
        :raises ValueError: if the class names differ from the schema's in
            content or in order.
        """
        if self.class_names != schema.class_names:
            raise ValueError(
                "the recorded vocabulary's classes do not match the schema's: "
                f"recorded {list(self.class_names)}, "
                f"schema {list(schema.class_names)}"
            )

    def disagreement_with(self, other: "Vocabulary") -> str | None:
        """How `other` differs from this vocabulary, or `None` if it does not.

        A one-line report rather than a bool: resized and repermuted call for
        different responses, and only the first is visible in the shapes.

        :param other: the vocabulary to compare against.
        :return: the difference, or None if there is none.
        """
        same_entities = self.entities == other.entities
        if same_entities and self.class_map == other.class_map:
            return None

        if len(self.entities) != len(other.entities):
            return (
                f"{len(self.entities)} entities recorded against "
                f"{len(other.entities)} derived from the corpus"
            )

        missing = sorted(set(self.entities) - set(other.entities))
        if missing:
            return (
                f"{len(missing)} recorded entities are absent from the "
                f"corpus, starting with {missing[:4]}"
            )

        if self.entities != other.entities:
            moved = sum(
                1
                for recorded, derived in zip(self.entities, other.entities)
                if recorded != derived
            )
            return (
                f"same {len(self.entities)} entities in a different order: "
                f"{moved} columns moved"
            )

        return (
            f"classes differ: recorded {list(self.class_names)}, "
            f"derived {list(other.class_names)}"
        )

    def to_payload(self) -> Payload:
        """The plain-builtin form written into a checkpoint.

        :return: the payload to store.
        """
        return {
            "entities": list(self.entities),
            "class_map": {
                name: list(entity_ids)
                for name, entity_ids in self.class_map.items()
            },
        }

    @classmethod
    def from_payload(cls, payload: Payload) -> "Vocabulary":
        """Read a vocabulary back out of a checkpoint.

        :param payload: the stored plain-builtin form.
        :return: the vocabulary it describes.
        :raises ValueError: if a key is missing or holds the wrong shape. This
            runs on data that came off disk, so it states what is wrong rather
            than raising from the conversion.
        """
        try:
            entities = payload["entities"]
            class_map = payload["class_map"]
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"checkpoint vocabulary is missing {error}; expected the keys "
                "'entities' and 'class_map'"
            ) from None

        if not isinstance(entities, Sequence) or isinstance(entities, str):
            raise ValueError(
                f"checkpoint vocabulary's 'entities' is {type(entities)!r}, "
                "expected a sequence of entity IDs"
            )
        if not isinstance(class_map, Mapping):
            raise ValueError(
                f"checkpoint vocabulary's 'class_map' is {type(class_map)!r}, "
                "expected a mapping of class name to entity IDs"
            )

        return cls(
            entities=tuple(entities),
            class_map={
                name: tuple(entity_ids)
                for name, entity_ids in class_map.items()
            },
        )

    def __len__(self) -> int:
        return len(self.entities)


def _reject_duplicates(names: tuple[str, ...], what: str) -> None:
    """:raises ValueError: if `names` repeats a value.

    Counted rather than `names.count(name)`-ed per element as `schema.py` does:
    the entity list runs to thousands of IDs and this is on the path of every
    `Vocabulary` construction.
    """
    duplicates = sorted(
        name for name, count in collections.Counter(names).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"duplicate {what}: {duplicates}")
