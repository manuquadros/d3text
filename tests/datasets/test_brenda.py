"""The schema-driven BRENDA adapter, on synthetic splits.

None of these tests touch the ~300 MB BRENDA files: `build_dataset` takes the
split frames, so the corpus loaders are only reached through the one test that
exercises the `d3text.data.brenda_dataset` shim, and there they are stubbed.

What is pinned here is that every fact the loader used to spell out inline is
now read off the `Schema`: the columns it indexes, the prefix each entity ID
wears, and which column of the class matrix a type owns. The old loader hard-
coded a four-name list, sliced the prefix out of the column name (`col[:3]`)
and located the class column with `entity_cols.index`, so a schema declaring
different names, prefixes or order would have been ignored — which is what the
schemas below declare.
"""

import numpy as np
import pandas as pd
import pytest

from d3text.datasets import brenda
from d3text.schema import EntityType, Schema

# name[:3] is deliberately *not* the prefix for either type, and the
# declaration order is not the frame's column order.
TOY_SCHEMA = Schema(
    entity_types=(
        EntityType(name="enzymes", prefix="ec"),
        EntityType(name="bacteria", prefix="taxon"),
    )
)

BRENDA_CLASSES = ("strains", "bacteria", "other_organisms", "enzymes")
BRENDA_PREFIXES = ("str", "bac", "oth", "enz")

HAS_ENZYME = np.array([1, 0, 0], dtype=np.float16)


def frame(rows: list[dict], schema: Schema = TOY_SCHEMA) -> pd.DataFrame:
    """A split frame in the shape `brenda_references` hands over.

    Each row gives the per-type ID columns; `entities` (the flat, prefixed
    list) and `relations` are derived here the way the corpus builder derives
    them, so a test only has to state the IDs.
    """
    records = []
    for row in rows:
        record = {
            "pubmed_id": row["pubmed_id"],
            "fulltext": row.get("fulltext", "<p>body</p>"),
            "relations": row.get("relations", []),
        }
        entities = []
        for entity_type in schema.entity_types:
            ids = row.get(entity_type.name, [])
            record[entity_type.name] = ids
            entities += [entity_type.prefix + str(i) for i in ids]
        record["entities"] = row.get("entities", entities)
        records.append(record)
    return pd.DataFrame(records)


def splits(train: pd.DataFrame, **others: pd.DataFrame) -> dict:
    """`train` plus any other split; the missing ones repeat `train`."""
    return {
        "train": train,
        "val": others.get("val", train.copy()),
        "test": others.get("test", train.copy()),
    }


@pytest.fixture
def toy(tmp_path):
    """Index a two-document toy corpus under `TOY_SCHEMA`."""
    train = frame(
        [
            {"pubmed_id": 10, "enzymes": [7], "bacteria": [42]},
            {"pubmed_id": 20, "enzymes": [8], "bacteria": []},
        ]
    )
    return brenda.build_dataset(
        schema=TOY_SCHEMA,
        splits=splits(train),
        encodings=tmp_path / "encodings.hdf5",
    )


def test_brenda_schema_declares_the_corpus_columns_and_prefixes():
    """Both are positional: the class head's columns follow this order, and
    the IDs these prefixes build are what the relation pairs are keyed by."""
    assert brenda.BRENDA_SCHEMA.class_names == BRENDA_CLASSES
    assert (
        tuple(
            entity_type.prefix
            for entity_type in brenda.BRENDA_SCHEMA.entity_types
        )
        == BRENDA_PREFIXES
    )


def test_entity_ids_carry_the_schema_prefix_not_the_column_name(toy):
    # `col[:3]` would have produced "enz7" / "bac42" from the column names.
    assert set(toy.entity_index) == {"ec7", "ec8", "taxon42"}


def test_class_map_is_keyed_by_the_schema_class_names(toy):
    assert list(toy.class_map) == list(TOY_SCHEMA.class_names)
    assert toy.class_map["enzymes"] == {"ec7", "ec8"}
    assert toy.class_map["bacteria"] == {"taxon42"}


def test_class_matrix_columns_follow_the_schema_declaration_order(toy):
    # enzymes is declared first, so it owns column 0 — even though the frame
    # and the BRENDA schema both order the columns differently.
    for entity_id in ("ec7", "ec8"):
        assert toy.class_matrix[toy.entity_index[entity_id]].tolist() == [
            1.0,
            0.0,
        ]
    assert toy.class_matrix[toy.entity_index["taxon42"]].tolist() == [0.0, 1.0]
    assert toy.class_matrix.shape == (3, 2)


def test_a_type_without_ids_keeps_its_class_column_but_indexes_nothing(
    tmp_path,
):
    """`has_ids=False` means detected but never linked: the class head still
    needs the column, the entity head must not grow one."""
    schema = Schema(
        entity_types=(
            EntityType(name="enzymes", prefix="ec"),
            EntityType(name="processes", prefix="pro", has_ids=False),
        )
    )
    train = frame(
        [{"pubmed_id": 10, "enzymes": [7], "processes": [1]}], schema=schema
    )

    dataset = brenda.build_dataset(
        schema=schema,
        splits=splits(train),
        encodings=tmp_path / "encodings.hdf5",
    )

    assert set(dataset.entity_index) == {"ec7"}
    assert dataset.class_map["processes"] == set()
    assert dataset.class_matrix.shape == (1, 2)


def test_document_classes_are_multi_hot_over_the_schema_columns(toy):
    classes = list(toy.data["train"].data["classes"])
    assert classes[0].tolist() == [1.0, 1.0]  # an enzyme and a bacterium
    assert classes[1].tolist() == [1.0, 0.0]  # an enzyme only


def test_class_targets_follow_row_position_not_index_label():
    """The splits reach `encode_split` boolean-filtered and never reset, so
    their index is non-contiguous. Assigning the class matrix as a `Series`
    aligns it on those labels: every row after the first dropped one takes
    another row's labels, and a label running past the filtered length gets
    `NaN`."""
    rows = [
        {"pubmed_id": 10, "enzymes": [7]},
        {"pubmed_id": 20, "bacteria": [42]},
        {"pubmed_id": 30, "enzymes": [8], "bacteria": [43]},
        {"pubmed_id": 40, "enzymes": [9]},
    ]
    split = frame(rows).set_index(pd.Index([0, 1, 3, 5]))

    encoded = brenda.encode_split(
        TOY_SCHEMA, split, entity_index={"ec7": 0}, known_entities={"ec7"}
    )

    expected = [
        [1.0, 0.0],  # enzyme only
        [0.0, 1.0],  # bacterium only
        [1.0, 1.0],  # both — under alignment this took row 3's labels
        [1.0, 0.0],  # enzyme only — under alignment this was NaN
    ]
    assert [np.asarray(row).tolist() for row in encoded["classes"]] == expected


def test_entities_are_encoded_against_the_entity_index(toy):
    encoded = list(toy.data["train"].data["entities"])
    assert encoded[0][toy.entity_index["ec7"]] == 1
    assert encoded[0][toy.entity_index["taxon42"]] == 1
    assert encoded[0][toy.entity_index["ec8"]] == 0


def test_the_entity_index_is_built_from_the_training_split_alone(tmp_path):
    """An entity seen only in validation has no column of its own; it is what
    the `UNK` column is for."""
    train = frame([{"pubmed_id": 10, "enzymes": [7]}])
    val = frame([{"pubmed_id": 20, "enzymes": [99]}])

    dataset = brenda.build_dataset(
        schema=TOY_SCHEMA,
        splits=splits(train, val=val),
        encodings=tmp_path / "encodings.hdf5",
    )

    assert set(dataset.entity_index) == {"ec7"}
    assert list(dataset.data["val"].data["entities"])[0].tolist() == [0]


def test_relations_naming_an_unindexed_entity_are_dropped(tmp_path):
    train = frame(
        [
            {
                "pubmed_id": 10,
                "enzymes": [7],
                "bacteria": [42],
                "relations": [
                    {
                        ("taxon42", "ec7"): HAS_ENZYME,
                        ("taxon42", "ec99"): HAS_ENZYME,
                    }
                ],
            }
        ]
    )

    dataset = brenda.build_dataset(
        schema=TOY_SCHEMA,
        splits=splits(train),
        encodings=tmp_path / "encodings.hdf5",
    )

    kept = list(dataset.data["train"].data["relations"])[0]
    assert [sorted(pairs) for pairs in kept] == [[("taxon42", "ec7")]]


def test_a_schema_whose_prefixes_miss_the_corpus_is_rejected(tmp_path):
    """The relation pairs are keyed by the prefixes `brenda_references` stamps
    on. A schema that prefixes its IDs differently silently loses every pair —
    the run trains on no relations at all and reports a clean loss."""
    train = frame(
        [
            {
                "pubmed_id": 10,
                "enzymes": [7],
                "bacteria": [42],
                # keyed the way the corpus keys them, not the way TOY_SCHEMA
                # would: "bac"/"enz" against "taxon"/"ec".
                "relations": [{("bac42", "enz7"): HAS_ENZYME}],
            }
        ]
    )

    with pytest.raises(ValueError, match="prefixes do not match"):
        brenda.build_dataset(
            schema=TOY_SCHEMA,
            splits=splits(train),
            encodings=tmp_path / "encodings.hdf5",
        )


def test_data_brenda_dataset_delegates_with_the_brenda_schema(
    tmp_path, monkeypatch
):
    """The shim the console scripts still call: same three splits, indexed
    under `BRENDA_SCHEMA`, with `limit` reaching the training loader."""
    from d3text import data

    train = frame(
        [{"pubmed_id": 10, "strains": [1], "enzymes": [7]}],
        schema=brenda.BRENDA_SCHEMA,
    )
    calls = {}

    def loader(split):
        def load(noise=0, limit=0):
            calls[split] = {"noise": noise, "limit": limit}
            return train.copy()

        return load

    for split in ("training", "validation", "test"):
        monkeypatch.setattr(
            brenda.brenda_references, f"{split}_data", loader(split)
        )

    dataset = data.brenda_dataset(encodings="nowhere.hdf5", limit=3)

    assert calls["training"]["limit"] == 3
    assert set(dataset.data) == {"train", "val", "test"}
    assert list(dataset.class_map) == list(BRENDA_CLASSES)
    assert set(dataset.entity_index) == {"str1", "enz7"}
