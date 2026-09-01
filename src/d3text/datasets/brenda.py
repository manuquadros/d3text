"""The BRENDA corpus, declared as a `Schema` and indexed from it.

`BRENDA_SCHEMA` is the single place that says which entity types the corpus
carries and which prefix their IDs wear; the column list, the ID prefixes, the
class-matrix column order and the per-document class labels are all derived
from it.
"""

import os
import pathlib
from collections.abc import Callable, Iterable, Mapping, Sequence, Set
from numbers import Real

import numpy
import pandas as pd
import xmlparser
from brenda_references import brenda_references

from d3text.data.data import (
    DATA_DIR,
    BrendaDataset,
    EntityRelationDataset,
    multi_hot_encode_series,
)

# `BRENDA_SCHEMA` is declared in `d3text.schema`, not here: `d3text.corpus`,
# `d3text.surface_forms` and `d3text.token_labels` all need the entity types
# and their prefixes, and none of them may import this module, which reaches
# the BRENDA data layer. Re-exported so the old spelling keeps resolving.
from d3text.schema import (
    BRENDA_SCHEMA as BRENDA_SCHEMA,
    Schema,
)
from d3text.vocabulary import Vocabulary

Relations = list[dict[tuple[str, str], Iterable[Real]]]


SPLIT_LOADERS: dict[str, Callable[[int], pd.DataFrame]] = {
    "train": lambda limit: brenda_references.training_data(
        noise=450, limit=limit
    ),
    "val": lambda limit: brenda_references.validation_data(noise=100),
    "test": lambda limit: brenda_references.test_data(noise=50),
}


def brenda_dataset(
    schema: Schema,
    encodings: str | os.PathLike[str],
    limit: int | None = None,
    vocabulary: Vocabulary | None = None,
    split_names: Sequence[str] = ("train", "val", "test"),
    base_model: str | None = None,
) -> EntityRelationDataset:
    """The BRENDA splits, indexed under `schema`.

    :param schema: the entity types to index the corpus under. Every type's
        `name` must be a column of the split frames.
    :param encodings: precomputed encodings HDF5, relative to `DATA_DIR`.
    :param limit: truncate the training split to this many documents; `None`
        and 0 both mean all of it. It selects the entity vocabulary along with
        the documents, so it is a property of a training run and of any run
        that must reproduce one — which is why a recorded `vocabulary` makes it
        irrelevant.
    :param vocabulary: index the splits under this recorded column order
        instead of deriving one from the training split. This is what a
        checkpoint carries, and what makes an evaluation reproduce the run it
        is evaluating rather than the corpus as it stands today.
    :param split_names: which splits to load. Loading one costs a pass over its
        CSV, so an evaluation should ask only for the split it scores.
    :param base_model: the model this run will feed the encodings to, passed
        through to `BrendaDataset`; `None` skips that check.
    :return: the indexed splits.
    :raises ValueError: if `split_names` names a split the corpus has not got.
    """
    unknown = [name for name in split_names if name not in SPLIT_LOADERS]
    if unknown:
        raise ValueError(
            f"no such BRENDA split: {unknown}; "
            f"expected some of {sorted(SPLIT_LOADERS)}"
        )

    return build_dataset(
        schema=schema,
        splits={name: SPLIT_LOADERS[name](limit or 0) for name in split_names},
        encodings=pathlib.Path(DATA_DIR / encodings),
        vocabulary=vocabulary,
        base_model=base_model,
    )


def build_dataset(
    schema: Schema,
    splits: Mapping[str, pd.DataFrame],
    encodings: pathlib.Path,
    vocabulary: Vocabulary | None = None,
    base_model: str | None = None,
) -> EntityRelationDataset:
    """Index `splits` under `schema` and wrap each in a `BrendaDataset`.

    Without a `vocabulary` the entity columns come from the training split
    alone, so an entity seen only in validation or test is scored as `UNK`.
    With one, that order is used for *every* split, labels included: pinning
    only the model's geometry would leave the targets following the corpus,
    which is the failure this exists to prevent.

    :param schema: the entity types to index under.
    :param splits: the split frames, by name.
    :param encodings: the precomputed encodings file.
    :param vocabulary: the recorded column order to index under, if any.
    :param base_model: the model this run will feed the encodings to.
    :return: the indexed splits.
    :raises ValueError: if no `vocabulary` is given and no training split is
        there to derive one from, or if a given one does not fit `schema`.
    """
    if vocabulary is None:
        if "train" not in splits:
            raise ValueError(
                "deriving an entity vocabulary needs the 'train' split; pass "
                f"a recorded `vocabulary` to index {sorted(splits)} without it"
            )
        vocabulary = Vocabulary.from_class_map(
            entity_ids_by_class(schema, splits["train"])
        )
    else:
        vocabulary.check_fits(schema)

    entity_index = vocabulary.entity_index
    known_entities = entity_index.keys()
    if splits:
        check_relation_ids(_reference_split(splits), known_entities)

    return EntityRelationDataset(
        data={
            name: BrendaDataset(
                encode_split(schema, split, entity_index, known_entities),
                encodings=encodings,
                base_model=base_model,
            )
            for name, split in splits.items()
        },
        entity_index=entity_index,
        class_map=vocabulary.as_class_map(),
        class_matrix=vocabulary.class_matrix(),
    )


def _reference_split(splits: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """The split `check_relation_ids` reads the corpus's ID spelling off.

    The training split when there is one. An evaluation build has none and
    needs the check just as much: a recorded vocabulary written under different
    prefixes fails the same way, and scores a relation head on nothing at all.
    """
    if "train" in splits:
        return splits["train"]
    return next(iter(splits.values()))


def build_entity_index(class_map: Mapping[str, Set[str]]) -> dict[str, int]:
    """Entity ID -> the column it owns in the entity head's output.

    The ordering itself lives in `Vocabulary.from_class_map`, which is also
    what a checkpoint records.

    :param class_map: class name -> its entity IDs.
    :return: the index the labels are encoded against.
    """
    return Vocabulary.from_class_map(class_map).entity_index


def entity_ids_by_class(
    schema: Schema, split: pd.DataFrame
) -> dict[str, set[str]]:
    """Entity-type name -> the prefixed IDs of that type occurring in `split`.

    Every type gets a key, including one that declares `has_ids=False`: the
    class head is sized from this mapping, so a type with no groundable
    instances must still hold its column.

    :param schema: declares the types and their prefixes.
    :param split: the frame to read.
    :return: each type's IDs.
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


def encode_split(
    schema: Schema,
    split: pd.DataFrame,
    entity_index: Mapping[str, int],
    known_entities: Set[str],
) -> pd.DataFrame:
    """Encode one split's labels in place: entities, classes and relations.

    :param schema: declares the class column order.
    :param split: the frame to encode.
    :param entity_index: entity ID -> its column.
    :param known_entities: the IDs that own a column.
    :return: the frame, labels encoded.
    """
    split["entities"] = multi_hot_encode_series(
        series=split["entities"], index=entity_index
    )

    # Computed from the schema's columns rather than from the encoded
    # `entities` vector, so that a document whose entities are all UNK in
    # validation and evaluation still counts towards its classes.
    class_targets = [
        numpy.array(
            [1 if len(row[name]) > 0 else 0 for name in schema.class_names],
            dtype=numpy.float32,
        )
        for _, row in split.iterrows()
    ]
    split["relations"] = split["relations"].apply(
        lambda relations: filter_relations(relations, known_entities)
    )
    if class_targets:
        # A plain list is assigned positionally; a `Series` would be aligned on
        # `split`'s index, and the splits do not carry a `RangeIndex` — the
        # corpus loaders boolean-filter them without resetting. Under alignment
        # every row after the first dropped one takes some other row's labels,
        # and the rows whose label runs past the filtered length get `NaN`.
        split["classes"] = list(numpy.stack(class_targets))
    else:
        # `numpy.stack` refuses an empty sequence, and a split filtered down
        # to no row is a legal split — `limit` interacting with the corpus
        # loaders' `dropna` reaches it too. The column is built directly rather
        # than from an empty list, which pandas would type `float64` where the
        # populated case and every other label column here are `object`.
        split["classes"] = pd.Series(index=split.index, dtype=object)
    split["fulltext"] = split["fulltext"].apply(xmlparser.remove_tags)

    return split


def filter_relations(
    relations: Relations, known_entities: Set[str]
) -> Relations:
    """Drop pairs naming an entity outside the index, and empty dicts too.

    An empty dict is not the same as no relations, and the relation head would
    be handed a candidate list with a hole in it. Each element is judged on its
    own, so a document whose first dict loses every pair keeps what the later
    ones hold.

    :param relations: the document's relation dicts.
    :param known_entities: the IDs that own a column.
    :return: the surviving dicts, empty only when nothing survived anywhere.
    """
    return [
        kept
        for pairs in relations
        if (
            kept := {
                pair: relation
                for pair, relation in pairs.items()
                if all(argument in known_entities for argument in pair)
            }
        )
    ]


def check_relation_ids(split: pd.DataFrame, known_entities: Set[str]) -> None:
    """Fail loudly when the schema's ID prefixes miss the corpus's.

    `brenda_references` prefixes the relation pairs itself while
    `known_entities` is built from the schema, and if the two disagree every
    pair fails the membership test — the run trains on zero relations and
    reports a clean loss. Returns as soon as one pair lands.

    :param split: the frame to check.
    :param known_entities: the IDs that own a column.
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
