"""The BRENDA corpus, declared as a `Schema` and indexed from it.

`BRENDA_SCHEMA` is the single place that says which entity types the corpus
carries and which prefix their database IDs wear; `brenda_dataset` derives from
it everything the loader used to spell out inline — the column list, the ID
prefixes, the class-matrix column order and the per-document class labels.
Adding a fifth entity type is now a line in the schema rather than four edits
that have to agree.

The schema declares no relation types yet: `ETEBrendaModel` still hardcodes
them, and `RelationType` cannot express `HasEnzyme` as the corpus builds it —
its subject is a bacterium, a strain *or* an other-organism, while a
`RelationType` names one subject type. Resolving that is SCHEMA-03's business.
"""

import os
import pathlib
from collections.abc import Iterable, Mapping, Set
from numbers import Real

import numpy
import pandas as pd
import torch
import xmlparser
from brenda_references import brenda_references
from jaxtyping import Float
from ordered_set import OrderedSet
from torch import Tensor

from d3text.data.data import (
    DATA_DIR,
    BrendaDataset,
    EntityRelationDataset,
    multi_hot_encode_series,
)
from d3text.schema import EntityType, Schema

# Declaration order is the class head's column order and the class matrix's,
# so it is not free to change: a checkpoint's class logits are positional.
# The prefixes are the ones `brenda_references.preprocess_labels` stamps onto
# the numeric BRENDA IDs — they happen to equal `name[:3]`, which is what the
# loader used to slice, but that is a coincidence this schema no longer relies
# on.
BRENDA_SCHEMA = Schema(
    entity_types=(
        EntityType(name="strains", prefix="str"),
        EntityType(name="bacteria", prefix="bac"),
        EntityType(name="other_organisms", prefix="oth"),
        EntityType(name="enzymes", prefix="enz"),
    )
)

Relations = list[dict[tuple[str, str], Iterable[Real]]]


def brenda_dataset(
    schema: Schema,
    encodings: str | os.PathLike[str],
    limit: int = 0,
) -> EntityRelationDataset:
    """The BRENDA splits, indexed under `schema`.

    :param schema: The entity types to index the corpus under. Every type's
        `name` must be a column of the split frames.
    :param encodings: Precomputed encodings HDF5, relative to `DATA_DIR`.
    :param limit: Truncate the training split to this many documents (0: all).
    """
    return build_dataset(
        schema=schema,
        splits={
            "train": brenda_references.training_data(noise=450, limit=limit),
            "val": brenda_references.validation_data(noise=100),
            "test": brenda_references.test_data(noise=50),
        },
        encodings=pathlib.Path(DATA_DIR / encodings),
    )


def build_dataset(
    schema: Schema,
    splits: Mapping[str, pd.DataFrame],
    encodings: pathlib.Path,
) -> EntityRelationDataset:
    """Index `splits` under `schema` and wrap each in a `BrendaDataset`.

    The entity vocabulary comes from the **training** split alone: an entity
    seen only in validation or test has no column of its own and is scored as
    `UNK`, which is the point of the `UNK` column.
    """
    class_map = entity_ids_by_class(schema, splits["train"])
    entity_index = build_entity_index(class_map)
    known_entities = entity_index.keys()
    check_relation_ids(splits["train"], known_entities)

    return EntityRelationDataset(
        data={
            name: BrendaDataset(
                encode_split(schema, split, entity_index, known_entities),
                encodings=encodings,
            )
            for name, split in splits.items()
        },
        entity_index=entity_index,
        class_map=class_map,
        class_matrix=class_matrix(schema, class_map, entity_index),
    )


def build_entity_index(class_map: Mapping[str, Set[str]]) -> dict[str, int]:
    """Entity ID -> the column it owns in the entity head's output.

    The types are walked in `class_map`'s order — which is the schema's
    declaration order — and each type's IDs are **sorted** before they are laid
    down, so one training split yields one column order in every process.
    `entity_ids_by_class` returns plain `set`s, and a `set` of strings iterates
    in an order that depends on `PYTHONHASHSEED`, which CPython randomizes per
    process.

    Nothing else records this order: `train` saves only a `state_dict`, and
    `evaluate` rebuilds the index in a *new* process. A column order that moved
    between the two would not fail — the head has the same width either way —
    it would score every entity against another entity's logits, within its own
    type block, and read as a mediocre model rather than a broken one.
    """
    ordered = OrderedSet[str]().union(
        *(sorted(entity_ids) for entity_ids in class_map.values())
    )
    return {entity_id: column for column, entity_id in enumerate(ordered)}


def entity_ids_by_class(
    schema: Schema, split: pd.DataFrame
) -> dict[str, set[str]]:
    """Entity-type name -> the prefixed IDs of that type occurring in `split`.

    Every type gets a key, including one that declares `has_ids=False`: the
    class head is sized from this mapping, so a type with no groundable
    instances must still hold its column.
    """
    return {
        entity_type.name: {
            entity_type.prefix + str(entity_id)
            for row in split[entity_type.name]
            for entity_id in row
        }
        if entity_type.has_ids
        else set()
        for entity_type in schema.entity_types
    }


def class_matrix(
    schema: Schema,
    class_map: Mapping[str, set[str]],
    entity_index: Mapping[str, int],
) -> Float[Tensor, "entities classes"]:
    """One-hot rows mapping each entity onto the classes it belongs to.

    Built by walking the classes rather than by inverting them into an
    entity -> class dict, so an ID declared under two types would light both
    columns instead of whichever the inversion happened to write last. The
    BRENDA prefixes make an entity's type unambiguous, so the two agree here.
    """
    matrix = torch.zeros(
        len(entity_index), len(schema.entity_types), dtype=torch.float32
    )
    for column, entity_type in enumerate(schema.entity_types):
        for entity_id in class_map[entity_type.name]:
            matrix[entity_index[entity_id], column] = 1.0
    return matrix


def encode_split(
    schema: Schema,
    split: pd.DataFrame,
    entity_index: Mapping[str, int],
    known_entities: Set[str],
) -> pd.DataFrame:
    """Encode one split's labels in place: entities, classes and relations."""
    split["entities"] = multi_hot_encode_series(
        series=split["entities"], index=entity_index
    )

    # Computed from the schema's columns rather than from the encoded
    # `entities` vector, so that a document whose entities are all UNK in
    # validation and evaluation still counts towards its classes.
    cls_array = numpy.stack(
        [
            numpy.array(
                [1 if len(row[name]) > 0 else 0 for name in schema.class_names],
                dtype=numpy.float32,
            )
            for _, row in split.iterrows()
        ]
    )
    split["relations"] = split["relations"].apply(
        lambda relations: filter_relations(relations, known_entities)
    )
    # A plain list is assigned positionally; a `Series` would be aligned on
    # `split`'s index, and the splits do not carry a `RangeIndex` — the corpus
    # loaders boolean-filter them without resetting. Under alignment every row
    # after the first dropped one takes some other row's labels, and the rows
    # whose label runs past the filtered length get `NaN`.
    split["classes"] = list(cls_array)
    split["fulltext"] = split["fulltext"].apply(xmlparser.remove_tags)

    return split


def filter_relations(
    relations: Relations, known_entities: Set[str]
) -> Relations:
    """Drop pairs naming an entity outside the index, and empty dicts with
    them: an empty dict is not the same as no relations, and the relation head
    would be handed a candidate list with a hole in it."""
    filtered = [
        {
            pair: relation
            for pair, relation in pairs.items()
            if all(argument in known_entities for argument in pair)
        }
        for pairs in relations
    ]

    if not filtered or not filtered[0]:
        return []
    return filtered


def check_relation_ids(split: pd.DataFrame, known_entities: Set[str]) -> None:
    """Fail loudly when the schema's ID prefixes miss the corpus's.

    The relation pairs are keyed by IDs that `brenda_references` prefixes
    itself, while `known_entities` is built from the schema's prefixes. Let the
    two disagree and every pair fails the `filter_relations` membership test —
    the run trains on zero relations and reports it as a clean loss.

    Returns as soon as one pair lands, so the healthy case pays for a single
    lookup.

    :raises ValueError: if the split declares relations and not one of them
        names an entity in the index.
    """
    saw_relation = False
    for relations in split["relations"]:
        for pairs in relations:
            for pair in pairs:
                saw_relation = True
                if any(argument in known_entities for argument in pair):
                    return

    if saw_relation:
        raise ValueError(
            "no relation in the training split names an entity in the index: "
            "the schema's ID prefixes do not match the corpus's"
        )
