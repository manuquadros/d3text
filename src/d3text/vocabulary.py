"""The label vocabulary a model's head was sized to.

The class head is positional and nothing in a `state_dict` records which class
owns which column, so a same-width repermutation scores every class against
another class's logits and reads as a mediocre model rather than a broken one.
`Vocabulary` is that order made explicit, written into the checkpoint and read
back. Leaf module: `d3text.schema` only.
"""

import dataclasses
from collections.abc import Mapping, Sequence, Set
from typing import Any

from d3text.schema import Schema, _reject_duplicates

# `torch.load` defaults to `weights_only=True`, which admits tensors and plain
# builtins and nothing else, so the payload is lists and dicts rather than a
# pickled `Vocabulary`. Keeping the checkpoint loadable without trusting it is
# worth more than the convenience of pickling the dataclass.
Payload = dict[str, Any]


@dataclasses.dataclass(frozen=True)
class Vocabulary:
    """The class columns a checkpoint's head was trained on, and their members.

    :param class_map: class name -> the entity IDs of that class, in class-head
        column order. A class with no groundable instances still holds its key,
        because the class head is sized from this mapping. The members are what
        say which entities the training split named, which is what splits the
        span tagger's detection recall into known and novel.
    """

    class_map: dict[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def from_class_map(cls, class_map: Mapping[str, Set[str]]) -> "Vocabulary":
        """The vocabulary a corpus's class map implies.

        Each type's IDs are sorted before they are laid down, so one training
        split yields one payload in every process: a `set` of strings iterates
        in an order that depends on `PYTHONHASHSEED`.

        :param class_map: class name -> its entity IDs, in the schema's order.
        :return: the vocabulary those columns define.
        """
        return cls(
            class_map={
                name: tuple(sorted(entity_ids))
                for name, entity_ids in class_map.items()
            }
        )

    @property
    def class_names(self) -> tuple[str, ...]:
        """Class labels in class-head column order.

        :return: the labels in column order.
        """
        return tuple(self.class_map)

    @property
    def entity_ids(self) -> frozenset[str]:
        """Every entity ID the recorded classes name.

        :return: the training split's entity vocabulary, unordered — nothing
            is positional in it now that no head has a column per entity.
        """
        return frozenset(
            entity_id
            for entity_ids in self.class_map.values()
            for entity_id in entity_ids
        )

    def as_class_map(self) -> dict[str, set[str]]:
        """`class_map` in the `set`-valued shape the dataset takes.

        :return: class name -> its entity IDs.
        """
        return {
            name: set(entity_ids) for name, entity_ids in self.class_map.items()
        }

    def validate(self) -> None:
        """Check the vocabulary's internal consistency.

        Called from `__post_init__`; public so one read back off a checkpoint
        can be re-checked at the boundary.

        :raises ValueError: on a repeated class name, or on a class repeating
            an entity ID — what a truncated or hand-edited payload looks like.
        """
        _reject_duplicates(tuple(self.class_map), "class names")

        for name, entity_ids in self.class_map.items():
            _reject_duplicates(entity_ids, f"entity IDs under class {name!r}")

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

    def to_payload(self) -> Payload:
        """The plain-builtin form written into a checkpoint.

        :return: the payload to store.
        """
        return {
            "class_map": {
                name: list(entity_ids)
                for name, entity_ids in self.class_map.items()
            }
        }

    @classmethod
    def from_payload(cls, payload: Payload) -> "Vocabulary":
        """Read a vocabulary back out of a checkpoint.

        :param payload: the stored plain-builtin form.
        :return: the vocabulary it describes.
        :raises ValueError: if the key is missing or holds the wrong shape.
            This runs on data that came off disk, so it states what is wrong
            rather than raising from the conversion.
        """
        try:
            class_map = payload["class_map"]
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"checkpoint vocabulary is missing {error}; expected the key "
                "'class_map'"
            ) from None

        if not isinstance(class_map, Mapping):
            raise ValueError(
                f"checkpoint vocabulary's 'class_map' is {type(class_map)!r}, "
                "expected a mapping of class name to entity IDs"
            )

        for name, entity_ids in class_map.items():
            if not isinstance(entity_ids, Sequence) or isinstance(
                entity_ids, str
            ):
                raise ValueError(
                    f"checkpoint vocabulary's class {name!r} holds "
                    f"{type(entity_ids)!r}, expected a sequence of entity IDs"
                )

        return cls(
            class_map={
                name: tuple(entity_ids)
                for name, entity_ids in class_map.items()
            }
        )

    def __len__(self) -> int:
        return len(self.class_map)
