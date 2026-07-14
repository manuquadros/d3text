"""The BRENDA adapter: how a corpus column becomes an entity index, a class
map, a class matrix and a document's class labels.

Every test here is corpus-free — it feeds the builders synthetic columns — so
it runs in the default suite. What they pin are the two orders the model reads
back out and cannot re-derive: the column an entity occupies in the entity
head, and the row a document's class labels belong to.
"""

import subprocess
import sys

import numpy
import pandas as pd
import pytest
import torch
from d3text.datasets.brenda import (
    BRENDA_SCHEMA,
    class_labels,
    class_matrix,
    entity_vocabulary,
)
from d3text.schema import EntityType, RelationType, Schema

_NONE = RelationType(name="none", is_none=True)
_HAS_ENZYME = RelationType(
    name="HasEnzyme", subject_types=("bacteria",), object_types=("enzymes",)
)

# Entity order here is the class-head column order, so the tests below can read
# a class column off the schema rather than a literal.
SCHEMA = Schema(
    entity_types=(
        EntityType(name="bacteria", prefix="bac"),
        EntityType(name="enzymes", prefix="enz"),
    ),
    relation_types=(_HAS_ENZYME, _NONE),
)

# Two documents' worth of corpus columns: raw IDs, one list per document.
COLUMNS: dict[str, list[list[int]]] = {
    "bacteria": [[42, 7], [7]],
    "enzymes": [[26836], [26836, 11]],
}


def test_entity_ids_carry_their_type_prefix() -> None:
    _, index = entity_vocabulary(SCHEMA, COLUMNS)

    assert set(index) == {"bac7", "bac42", "enz11", "enz26836"}


def test_index_blocks_the_types_in_schema_order() -> None:
    """The class matrix is one-hot per row, so a model that pooled entity
    logits into class logits by blocks would read the wrong block if the index
    interleaved the types."""
    _, index = entity_vocabulary(SCHEMA, COLUMNS)

    bacteria = [index[entity] for entity in ("bac7", "bac42")]
    enzymes = [index[entity] for entity in ("enz11", "enz26836")]

    assert max(bacteria) < min(enzymes)
    assert sorted(index.values()) == list(range(len(index)))


def test_ids_are_sorted_within_a_block() -> None:
    _, index = entity_vocabulary(SCHEMA, COLUMNS)

    assert list(index) == ["bac42", "bac7", "enz11", "enz26836"]


def test_the_class_map_keeps_a_key_per_declared_class() -> None:
    """A type the corpus never mentions still owns a column of the class head,
    and the model reads its class names — hence their order — off this map."""
    schema = Schema(
        entity_types=(
            *SCHEMA.entity_types,
            EntityType(name="strains", prefix="str"),
        ),
        relation_types=SCHEMA.relation_types,
    )
    classes, _ = entity_vocabulary(schema, {**COLUMNS, "strains": [[], []]})

    assert list(classes) == ["bacteria", "enzymes", "strains"]
    assert classes["strains"] == set()


def test_a_type_without_ids_is_classified_but_never_indexed() -> None:
    schema = Schema(
        entity_types=(
            *SCHEMA.entity_types,
            EntityType(name="processes", prefix="prc", has_ids=False),
        ),
        relation_types=SCHEMA.relation_types,
    )
    classes, index = entity_vocabulary(
        schema, {**COLUMNS, "processes": [[1], [2]]}
    )

    assert classes["processes"] == set()
    assert not any(entity.startswith("prc") for entity in index)


def test_class_matrix_is_one_hot_on_the_type_of_each_entity() -> None:
    classes, index = entity_vocabulary(SCHEMA, COLUMNS)

    matrix = class_matrix(SCHEMA, classes, index)

    assert matrix.shape == (len(index), len(SCHEMA.class_names))
    assert torch.equal(matrix.sum(dim=1), torch.ones(len(index)))
    for class_column, class_name in enumerate(SCHEMA.class_names):
        for entity in classes[class_name]:
            assert matrix[index[entity], class_column] == 1.0


def test_class_labels_mark_the_classes_a_document_mentions() -> None:
    df = pd.DataFrame({"bacteria": [[42], []], "enzymes": [[], []]})

    labels = class_labels(SCHEMA, df)

    assert numpy.array_equal(labels.iloc[0], numpy.array([1.0, 0.0]))
    assert numpy.array_equal(labels.iloc[1], numpy.array([0.0, 0.0]))


def test_class_labels_follow_the_rows_of_a_filtered_split() -> None:
    """`brenda_references` filters each split *after* renumbering its rows, so
    the labels have to carry the frame's own row labels. Built against a fresh
    `RangeIndex`, they land on whichever document happens to hold that label —
    and the rows past the end of the range get no labels at all."""
    df = pd.DataFrame(
        {"bacteria": [[42], [], []], "enzymes": [[], [], [26836]]},
        index=[0, 3, 7],
    )

    labels = class_labels(SCHEMA, df)

    assert not labels.isna().any()
    assert list(labels.index) == [0, 3, 7]
    assert numpy.array_equal(labels.loc[0], numpy.array([1.0, 0.0]))
    assert numpy.array_equal(labels.loc[3], numpy.array([0.0, 0.0]))
    assert numpy.array_equal(labels.loc[7], numpy.array([0.0, 1.0]))


def test_brenda_schema_indexes_the_corpus_columns() -> None:
    columns: dict[str, list[list[int]]] = {
        "strains": [[1]],
        "bacteria": [[2]],
        "other_organisms": [[3]],
        "enzymes": [[4]],
    }

    classes, index = entity_vocabulary(BRENDA_SCHEMA, columns)

    assert list(index) == ["str1", "bac2", "oth3", "enz4"]
    assert classes["other_organisms"] == {"oth3"}


_INDEX_IN_SUBPROCESS = """
from d3text.datasets.brenda import entity_vocabulary
from d3text.schema import EntityType, RelationType, Schema

schema = Schema(
    entity_types=(
        EntityType(name="bacteria", prefix="bac"),
        EntityType(name="enzymes", prefix="enz"),
    ),
    relation_types=(RelationType(name="none", is_none=True),),
)
columns = {
    "bacteria": [[b for b in range(20)]],
    "enzymes": [[e for e in range(20)]],
}
_, index = entity_vocabulary(schema, columns)
print(",".join(index))
"""


@pytest.mark.parametrize("hash_seed", ["1", "2"])
def test_the_index_does_not_depend_on_the_hash_seed(
    hash_seed: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The entity head is keyed by position, and only the training process ever
    sees the mapping: `evaluate` rebuilds it from the corpus and loads a
    checkpoint into it. Reading the index off a `set` makes that mapping a
    function of PYTHONHASHSEED, so the two processes disagree and every entity
    is scored against another entity's column — silently, since the shapes
    still match.

    Hence a subprocess: within one process the order is stable however it is
    built, so an in-process test cannot see the bug.
    """
    monkeypatch.setenv("PYTHONHASHSEED", hash_seed)
    result = subprocess.run(
        [sys.executable, "-c", _INDEX_IN_SUBPROCESS],
        capture_output=True,
        check=True,
        text=True,
    )

    entities = result.stdout.strip().splitlines()[-1].split(",")
    expected = sorted(f"bac{n}" for n in range(20)) + sorted(
        f"enz{n}" for n in range(20)
    )
    assert entities == expected
