"""The label vocabulary a model's heads were sized to.

An entity head has one column per training-split entity ID and a class head one
per entity type, and both are **positional**: nothing in a `state_dict` records
which ID owns which column. `train` used to save the weights alone, so
`evaluate` had to rebuild that order from the corpus and land on it by luck.
Anything that moves the training split moves the columns with it — a different
`--limit`, a changed `noise=`, a `brenda_references` refresh. A *width* change
fails loudly on `load_state_dict`; a same-width repermutation does not, and
scores every entity against another entity's logits, reading as a mediocre
model rather than a broken one (BUG-19, BUG-21).

`Vocabulary` is that order made explicit, so it can be written into the
checkpoint beside the weights and *read back* at evaluation instead of
re-derived. It is the whole of what a checkpoint needs to be interpreted: the
entity column order, and the class columns with their members — enough to
rebuild `entity_index` and `class_matrix` without consulting the corpus at all.

Leaf module: torch and `d3text.schema` only. `d3text.checkpoint` and the
dataset adapters sit above it.
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

    :param entities: Entity IDs in entity-head column order. The head's extra
        trailing ``UNK`` column is deliberately absent: it is a property of the
        head, not of the data, exactly as `Schema.class_names` omits ``OOS``.
    :param class_map: Class name -> the entity IDs of that class, in class-head
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

        The types are walked in `class_map`'s order — the schema's declaration
        order — and each type's IDs are **sorted** before they are laid down,
        so one training split yields one column order in every process. A
        `set` of strings iterates in an order that depends on `PYTHONHASHSEED`,
        which CPython randomizes per process (BUG-19).
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

        `entity_index` is authoritative for the column order — it is what the
        labels were encoded against — while `class_map` contributes only
        membership, so its `set`s are sorted here and their iteration order
        never reaches the checkpoint.

        :raises ValueError: if `entity_index` is not a bijection onto
            ``range(len(entity_index))``; a head cannot be built from an index
            with a gap or a repeat in it, so a caller holding one is already
            wrong and must not have it written to disk.
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
        """Entity ID -> the column it owns in the entity head's output."""
        return {
            entity_id: column for column, entity_id in enumerate(self.entities)
        }

    @property
    def class_names(self) -> tuple[str, ...]:
        """Class labels in class-head column order."""
        return tuple(self.class_map)

    def class_matrix(self) -> Float[Tensor, "entities classes"]:
        """One-hot rows mapping each entity onto the classes it belongs to.

        Built by walking the classes rather than by inverting them into an
        entity -> class dict, so an ID declared under two types lights both
        columns instead of whichever the inversion happened to write last.
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
        """`class_map` in the `set`-valued shape the model constructors take."""
        return {
            name: set(entity_ids) for name, entity_ids in self.class_map.items()
        }

    def validate(self) -> None:
        """Check the vocabulary's internal consistency.

        Called from `__post_init__`, so an inconsistent `Vocabulary` cannot be
        built and no consumer has to remember to ask; public so that one read
        back off a checkpoint can be re-checked at the boundary.

        :raises ValueError: on a repeated entity ID or class name, or on a
            class naming an entity that owns no column. The last is what a
            truncated or hand-edited payload looks like, and it would surface
            as a `KeyError` deep inside `class_matrix` instead.
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

        The class head's targets are built in *schema* order (`encode_split`)
        while its columns are built in *vocabulary* order (`class_matrix`), so
        the two orders being equal is what keeps a class scored against its own
        column. Equal sets in a different order is the dangerous case and is
        rejected with the rest.

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

        A one-line report rather than a bool: the two ways a corpus can drift
        away from a checkpoint — resized and repermuted — call for different
        responses from the operator, and only the first is visible in the
        shapes.
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
        """The plain-builtin form written into a checkpoint."""
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

        :raises ValueError: if a key is missing or holds the wrong shape. This
            runs on data that came off disk, so it states what is wrong rather
            than raising `KeyError` or `TypeError` from the conversion.
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
    the entity list runs to thousands of IDs on the full corpus, and this is on
    the path of every `Vocabulary` construction.
    """
    duplicates = sorted(
        name for name, count in collections.Counter(names).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"duplicate {what}: {duplicates}")
