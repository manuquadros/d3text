"""The BRENDA corpus, declared as a `Schema` and indexed from it.

`BRENDA_SCHEMA` is the single place that says which entity types the corpus
carries and which prefix their IDs wear; the column list, the ID prefixes, the
class column order and the per-document class labels are all derived from it.
"""

import os
import pathlib
from collections.abc import Callable, Iterable, Mapping, Sequence, Set
from numbers import Real

import numpy
import pandas as pd
from brenda_references import brenda_references

from d3text.constraints import NonNegative
from d3text.data.data import BrendaDataset, EntityRelationDataset

# Declared in `d3text.schema` so modules that must not reach the BRENDA data
# layer can read it; re-exported so the old spelling keeps resolving.
from d3text.schema import (
    BRENDA_SCHEMA as BRENDA_SCHEMA,
    Schema,
)
from d3text.vocabulary import Vocabulary

Relations = list[dict[tuple[str, str], Iterable[Real]]]


SPLIT_LOADERS: dict[str, Callable[[int], pd.DataFrame]] = {
    "train": lambda limit: brenda_references.training_data(
        noise=450, enzyme_noise=150, limit=limit
    ),
    "val": lambda limit: brenda_references.validation_data(
        noise=100, enzyme_noise=30, limit=limit
    ),
    "test": lambda limit: brenda_references.test_data(
        noise=50, enzyme_noise=15, limit=limit
    ),
}


def brenda_dataset(
    schema: Schema,
    encodings: str | os.PathLike[str],
    limit: NonNegative | None = None,
    vocabulary: Vocabulary | None = None,
    split_names: Sequence[str] = ("train", "val", "test"),
    base_model: str | None = None,
) -> EntityRelationDataset:
    """The BRENDA splits, indexed under `schema`.

    :param schema: the entity types to index the corpus under. Every type's
        `name` must be a column of the split frames.
    :param encodings: precomputed encodings store, as
        `models.config.encodings_path` resolves it for the run's base model.
    :param limit: keep this many text-carrying documents of *every* split,
        the synthetic noise each one appends scaled by the same fraction;
        `None` and 0 both mean all of it. A short run is then short in its
        validation pass too, which is the half that costs the most.
    :param vocabulary: index the splits under this recorded class order
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
        encodings=pathlib.Path(encodings),
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

    Without a `vocabulary` the class columns and their members come from the
    training split alone. With one, that order is used for *every* split,
    labels included: pinning only the model's geometry would leave the targets
    following the corpus, which is the failure this exists to prevent.

    :param schema: the entity types to index under.
    :param splits: the split frames, by name.
    :param encodings: the precomputed encodings file.
    :param vocabulary: the recorded column order to index under, if any.
    :param base_model: the model this run will feed the encodings to.
    :return: the indexed splits.
    :raises ValueError: if no `vocabulary` is given and no training split is
        there to derive one from, if a given one does not fit `schema`, or if
        a split frame carries no `source` column.
    """
    untagged = sorted(
        name for name, split in splits.items() if "source" not in split.columns
    )
    if untagged:
        raise ValueError(
            f"split(s) {untagged} carry no `source` column, so "
            "`BrendaDataset` can never check whether a whole corpus file "
            "went unbuilt for them; tag every row with its corpus source "
            "before calling `build_dataset` — `brenda_references.load_split` "
            "already does this for the production splits"
        )

    if vocabulary is None:
        if "train" not in splits:
            raise ValueError(
                "deriving a class vocabulary needs the 'train' split; pass "
                f"a recorded `vocabulary` to index {sorted(splits)} without it"
            )
        vocabulary = Vocabulary.from_class_map(
            entity_ids_by_class(schema, splits["train"])
        )
    else:
        vocabulary.check_fits(schema)

    if splits:
        check_relation_ids(
            _reference_split(splits), vocabulary.entity_ids, schema
        )

    return EntityRelationDataset(
        data={
            name: BrendaDataset(
                encode_split(schema, split),
                encodings=encodings,
                base_model=base_model,
            )
            for name, split in splits.items()
        },
        class_map=vocabulary.as_class_map(),
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


def entity_ids_by_class(
    schema: Schema, split: pd.DataFrame
) -> dict[str, set[str]]:
    """Entity-type name -> the prefixed IDs of that type occurring in `split`.

    Every type gets a key, including one that declares `has_ids=False`:
    `Vocabulary.check_fits` requires the recorded class names to equal the
    schema's, and `dataset/classes` counts the class-map keys.

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
) -> pd.DataFrame:
    """Encode one split's labels in place: classes and relations.

    The text columns are handed on exactly as the corpus gave them. The
    encodings are built separately, from `corpus.document_text`'s join of
    abstract and body, so any string rendered here would be a second, different
    version of the document that no stored offset addresses.

    :param schema: declares the class column order.
    :param split: the frame to encode.
    :return: the frame, labels encoded.
    """
    class_targets = list(
        numpy.column_stack(
            [
                split[name].map(len).gt(0).to_numpy(dtype=numpy.float32)
                for name in schema.class_names
            ]
        )
    )
    split["relations"] = split["relations"].apply(
        lambda relations: filter_relations(relations, schema)
    )
    if class_targets:
        # A list, not a `Series`: the splits keep a filtered, non-`RangeIndex`
        # index, and index alignment would hand rows each other's labels.
        split["classes"] = class_targets
    else:
        # An empty split is legal. Built directly because pandas would type
        # an empty list `float64`, where every other label column is `object`.
        split["classes"] = pd.Series(index=split.index, dtype=object)

    return split


def _typed_by(schema: Schema, entity_id: str) -> bool:
    """Whether `schema` declares a prefix `entity_id` wears."""
    return any(entity_id.startswith(prefix) for prefix in schema.prefix_to_type)


def filter_relations(relations: Relations, schema: Schema) -> Relations:
    """Drop pairs the schema could never score, and empty dicts too.

    A pair is dropped if no relation type admits its argument types, or if
    the schema cannot type an argument; membership of the training split's
    entity set is deliberately not a condition. Each dict is judged on its
    own. The rationale is in the dataset-adapter explanation.

    :param relations: the document's relation dicts.
    :param schema: declares which entity-type pairs a relation admits.
    :return: the surviving dicts, empty only when nothing survived anywhere.
    """
    return [
        kept
        for pairs in relations
        if (
            kept := {
                pair: relation
                for pair, relation in pairs.items()
                if all(_typed_by(schema, argument) for argument in pair)
                and schema.admits_relation(*pair)
            }
        )
    ]


def check_relation_ids(
    split: pd.DataFrame, known_entities: Set[str], schema: Schema
) -> None:
    """Fail loudly when the schema's ID prefixes miss the corpus's, per type.

    A disagreement is silent everywhere else. Checked per `has_ids` type,
    since a split-wide check passes on one correct type's pairs while every
    pair of another type goes unmatched.

    :param split: the frame to check.
    :param known_entities: the IDs the corpus's classes name.
    :param schema: declares which entity types carry a database ID and what
        prefix each wears.
    :raises ValueError: if the split declares relations and, for some
        `has_ids` entity type, not one relation argument wearing that type's
        prefix is a known entity.
    """
    matched = {
        entity_type.prefix: False
        for entity_type in schema.entity_types
        if entity_type.has_ids
    }
    saw_relation = False
    for relations in split["relations"]:
        for pairs in relations:
            for pair in pairs:
                saw_relation = True
                for argument in pair:
                    if argument in known_entities:
                        matched[schema.type_of(argument).prefix] = True

    missing = sorted(prefix for prefix, hit in matched.items() if not hit)
    if saw_relation and missing:
        raise ValueError(
            "no relation in the reference split names a known entity of "
            f"type prefix {missing}: the schema's ID prefixes do not match "
            "the corpus's"
        )
