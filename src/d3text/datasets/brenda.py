"""The BRENDA extraction target: the `Schema` describing it, and the loader
that reads the corpus against that schema.

Every index the loader hands the model — the class columns, the entity ID
prefixes, the rows of the class matrix — is derived from the `Schema`, so a
corpus annotating other entity types needs a new schema, not a new loader.

Sits above `d3text.data`, which holds the corpus-agnostic machinery (the HDF5
split, the samplers, the multi-hot encoders) and knows nothing of BRENDA's
columns.
"""

import itertools
import pathlib
from collections.abc import Iterable, Mapping
from numbers import Real

import numpy
import pandas as pd
import torch
import xmlparser
from brenda_references import brenda_references
from jaxtyping import Float
from torch import Tensor

from d3text.data.data import (
    DATA_DIR,
    BrendaDataset,
    EntityRelationDataset,
    multi_hot_encode_series,
)
from d3text.schema import EntityType, RelationType, Schema

__all__ = [
    "BRENDA_SCHEMA",
    "brenda_dataset",
    "class_labels",
    "class_matrix",
    "entity_vocabulary",
]

BRENDA_SCHEMA = Schema(
    # Entity order is the class head's column order, and the corpus columns are
    # read in this order to build the class matrix.
    entity_types=(
        EntityType(name="strains", prefix="str", vocab_path="strains.txt"),
        EntityType(name="bacteria", prefix="bac", vocab_path="bacteria.txt"),
        EntityType(name="other_organisms", prefix="oth"),
        EntityType(name="enzymes", prefix="enz", vocab_path="enzymes.txt"),
    ),
    # Relation order is pinned by the corpus: `brenda_references` hands each
    # candidate pair a one-hot label vector over exactly these labels in
    # exactly this order, and the models take its argmax as the target index.
    relation_types=(
        RelationType(
            name="HasEnzyme",
            subject_types=("bacteria", "strains", "other_organisms"),
            object_types=("enzymes",),
        ),
        RelationType(
            name="HasSpecies",
            subject_types=("strains",),
            object_types=("bacteria",),
        ),
        RelationType(name="none", is_none=True),
    ),
)

# The BRENDA splits are padded with entity-free psycholinguistics articles, so
# that the heads see documents in which none of their classes occurs.
NOISE = {"train": 450, "val": 100, "test": 50}


def entity_vocabulary(
    schema: Schema,
    columns: Mapping[str, Iterable[Iterable[object]]],
) -> tuple[dict[str, set[str]], dict[str, int]]:
    """The entities a corpus mentions, grouped by class and indexed.

    :param columns: The corpus' entity ID column per entity type, as it names
        them (`{"enzymes": [[26836], [], ...], ...}`). The *training* split's:
        an entity absent from it has no column of the entity head to be
        predicted in, and is UNK everywhere.
    :returns: The class map (type name -> its entities) and the entity index
        (`enz26836` -> its column of the entity head).

    Both are keyed by every class the schema declares, in its column order: a
    type the corpus never mentions — or one carrying no IDs at all — still owns
    a column of the class head, and the model reads its class names off the map.

    The index blocks the types in schema order and sorts the IDs within a
    block, which makes it a pure function of the corpus and the schema. Reading
    it off the `set`s instead — as this did — keys the entity head on Python's
    per-process string hash seed: the order then differs between the process
    that trains a checkpoint and the process that evaluates it, so every entity
    is silently scored against some other entity's column.
    """
    classes = {
        entity_type.name: _entity_ids(entity_type, columns[entity_type.name])
        if entity_type.has_ids
        else set()
        for entity_type in schema.entity_types
    }
    index = {
        entity: column
        for column, entity in enumerate(
            itertools.chain.from_iterable(
                sorted(classes[name]) for name in schema.class_names
            )
        )
    }
    return classes, index


def _entity_ids(
    entity_type: EntityType, column: Iterable[Iterable[object]]
) -> set[str]:
    """The distinct entity IDs one corpus column mentions, prefixed."""
    return {
        f"{entity_type.prefix}{entid}"
        for entid in itertools.chain.from_iterable(column)
    }


def class_matrix(
    schema: Schema, classes: Mapping[str, set[str]], index: Mapping[str, int]
) -> Float[Tensor, "entities classes"]:
    """One row per indexed entity, one-hot over the schema's class columns."""
    matrix = torch.zeros(len(index), len(schema.class_names))
    for column, class_name in enumerate(schema.class_names):
        rows = [index[entity] for entity in classes[class_name]]
        matrix[torch.tensor(rows, dtype=torch.long), column] = 1.0
    return matrix


def class_labels(schema: Schema, df: pd.DataFrame) -> pd.Series:
    """A multi-hot row per document, over the schema's class columns.

    Read off the corpus columns rather than off the encoded `entities` column,
    so that a document whose only mention is an entity missing from the
    training vocabulary still supervises its class.

    Carries `df`'s own row labels: a split is filtered *after* its rows were
    renumbered, so its labels have gaps, and a fresh `RangeIndex` would align
    the labels against the wrong documents and leave the tail of the split with
    no labels at all.
    """
    rows = [
        numpy.array(
            [
                1.0 if len(row[class_name]) > 0 else 0.0
                for class_name in schema.class_names
            ],
            dtype=numpy.float32,
        )
        for _, row in df.iterrows()
    ]
    return pd.Series(rows, index=df.index, dtype=object)


def brenda_dataset(
    schema: Schema,
    encodings: str,
    limit: int = 0,
) -> EntityRelationDataset:
    """The BRENDA splits, read against `schema` and keyed to `encodings`.

    :param schema: The extraction target. `BRENDA_SCHEMA`, unless a caller is
        deliberately training on a subset of it.
    :param encodings: The HDF5 of precomputed token sequences, named relative
        to the data directory.
    :param limit: Truncate the training split to this many documents. `0` reads
        all of it.
    """
    splits = {
        "train": brenda_references.training_data(
            noise=NOISE["train"], limit=limit
        ),
        "val": brenda_references.validation_data(noise=NOISE["val"]),
        "test": brenda_references.test_data(noise=NOISE["test"]),
    }

    # The vocabulary is the training split's: the model can only ever predict
    # an entity it has a column for.
    classes, index = entity_vocabulary(
        schema,
        {
            entity_type.name: splits["train"][entity_type.name]
            for entity_type in schema.entity_types
        },
    )

    def preprocess(df: pd.DataFrame) -> pd.DataFrame:
        df["entities"] = multi_hot_encode_series(
            series=df["entities"], index=index
        )
        df["classes"] = class_labels(schema, df)
        df["relations"] = df["relations"].apply(_filter_relations)
        df["fulltext"] = df["fulltext"].apply(xmlparser.remove_tags)
        return df

    def _filter_relations(
        rels: list[dict[tuple[str, str], Iterable[Real]]],
    ) -> list[dict[tuple[str, str], Iterable[Real]]]:
        filtered = [
            {
                pair: rel
                for pair, rel in d.items()
                if all(argument in index for argument in pair)
            }
            for d in rels
        ]

        if not filtered or not filtered[0]:
            # Prevent lists containing empty dicts
            return []
        return filtered

    encodings_path = pathlib.Path(DATA_DIR / encodings)
    return EntityRelationDataset(
        data={
            name: BrendaDataset(preprocess(split), encodings=encodings_path)
            for name, split in splits.items()
        },
        schema=schema,
        entity_index=index,
        class_map=classes,
        class_matrix=class_matrix(schema, classes, index),
    )
