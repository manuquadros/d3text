"""The schema-driven BRENDA adapter, on synthetic splits.

None of these touch the ~300 MB BRENDA files. What is pinned is that every fact
the loader used to spell out inline is now read off the `Schema`: the columns
it indexes, the prefix each ID wears, and which class-matrix column a type
owns. The old loader hardcoded a four-name list, sliced the prefix out of the
column name and located the class column by index, so a schema declaring
different names, prefixes or order would have been ignored.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from d3text.datasets import brenda
from d3text.schema import BRENDA_SCHEMA, EntityType, Schema
from d3text.vocabulary import Vocabulary

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

# Built from the schema rather than written out, so this file is not a third
# restatement of the column order the corpus and the relation head share.
HAS_ENZYME = np.eye(len(BRENDA_SCHEMA.relation_types), dtype=np.float16)[
    BRENDA_SCHEMA.relation_names.index("HasEnzyme")
]


def frame(rows: list[dict], schema: Schema = TOY_SCHEMA) -> pd.DataFrame:
    """A split frame in the shape `brenda_references` hands over.

    `entities` and `relations` are derived here the way the corpus builder
    derives them, so a test only has to state the IDs.
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


def test_a_split_filtered_down_to_no_row_encodes_to_an_empty_frame():
    """`numpy.stack` refuses an empty sequence, so a split whose filter kept
    no row died here rather than encoding to nothing. The class column has to
    come out empty and `object`-typed, like the populated case and like the
    other label columns, since a split reaching this state is a legal one."""
    rows = [{"pubmed_id": 10, "enzymes": [7]}, {"pubmed_id": 20}]
    split = frame(rows).iloc[:0].copy()

    encoded = brenda.encode_split(
        TOY_SCHEMA, split, entity_index={"ec7": 0}, known_entities={"ec7"}
    )

    assert len(encoded) == 0
    assert list(encoded["classes"]) == []
    assert encoded["classes"].dtype == encoded["entities"].dtype == object


def test_an_empty_split_indexes_alongside_a_populated_one(tmp_path):
    """The reachable case: one split filtered down to nothing must not take
    the build of the others with it."""
    train = frame([{"pubmed_id": 10, "enzymes": [7], "bacteria": [42]}])

    dataset = brenda.build_dataset(
        schema=TOY_SCHEMA,
        splits=splits(train, val=train.iloc[:0].copy()),
        encodings=tmp_path / "encodings.hdf5",
    )

    assert len(dataset.data["val"]) == 0
    assert len(dataset.data["train"]) == 1


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


def test_a_dict_that_filters_to_empty_does_not_veto_the_later_ones():
    """The row is a *list* of pair-dicts, and each is judged on its own.

    The filter used to decide the whole row from `filtered[0]`: unreachable
    while the corpus emits one dict per document, and silent relation loss the
    moment it emits two.
    """
    relations = [
        {("taxon42", "ec99"): HAS_ENZYME},
        {("taxon42", "ec7"): HAS_ENZYME, ("taxon99", "ec7"): HAS_ENZYME},
    ]

    kept = brenda.filter_relations(relations, {"taxon42", "ec7"})

    assert [sorted(pairs) for pairs in kept] == [[("taxon42", "ec7")]]


def test_relations_are_empty_only_when_no_dict_survives():
    relations = [
        {("taxon42", "ec99"): HAS_ENZYME},
        {("taxon99", "ec7"): HAS_ENZYME},
    ]

    assert brenda.filter_relations(relations, {"taxon42", "ec7"}) == []
    assert brenda.filter_relations([], {"taxon42", "ec7"}) == []


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


def test_brenda_dataset_indexes_under_the_brenda_schema(tmp_path, monkeypatch):
    """The console scripts' entry point: same three splits, indexed under
    `BRENDA_SCHEMA`, with `limit` reaching the training loader."""
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

    dataset = brenda.brenda_dataset(
        schema=brenda.BRENDA_SCHEMA, encodings="nowhere.hdf5", limit=3
    )

    assert calls["training"]["limit"] == 3
    assert set(dataset.data) == {"train", "val", "test"}
    assert list(dataset.class_map) == list(BRENDA_CLASSES)
    assert set(dataset.entity_index) == {"str1", "enz7"}


def test_a_recorded_vocabulary_overrides_the_one_the_split_implies(tmp_path):
    """A checkpoint's columns outlive the split they were derived from. The
    corpus here would index three entities; the recorded vocabulary knows one,
    and the labels must be encoded against *that* — a model built on the
    checkpoint's columns and targets built on the corpus's disagree silently.
    """
    recorded = Vocabulary.from_class_map(
        {"enzymes": {"ec7"}, "bacteria": set()}
    )
    grown = frame(
        [
            {"pubmed_id": 10, "enzymes": [7]},
            {"pubmed_id": 20, "enzymes": [8], "bacteria": [42]},
        ]
    )

    dataset = brenda.build_dataset(
        schema=TOY_SCHEMA,
        splits=splits(grown),
        encodings=tmp_path / "encodings.hdf5",
        vocabulary=recorded,
    )

    assert dataset.entity_index == {"ec7": 0}
    assert dataset.class_matrix.shape == (1, 2)
    encoded = list(dataset.data["train"].data["entities"])
    assert encoded[0].tolist() == [1]
    # ec8 and taxon42 own no column: outside the vocabulary is what UNK is for.
    assert encoded[1].tolist() == [0]


def test_the_recorded_column_order_wins_over_the_corpus_order(tmp_path):
    """The dangerous half of the drift: same width, so `load_state_dict`
    raises nothing and every entity is scored on another entity's column."""
    train = frame(
        [{"pubmed_id": 10, "enzymes": [7]}, {"pubmed_id": 20, "enzymes": [8]}]
    )
    reversed_order = Vocabulary(
        entities=("ec8", "ec7"),
        class_map={"enzymes": ("ec7", "ec8"), "bacteria": ()},
    )

    dataset = brenda.build_dataset(
        schema=TOY_SCHEMA,
        splits=splits(train),
        encodings=tmp_path / "encodings.hdf5",
        vocabulary=reversed_order,
    )

    assert dataset.entity_index == {"ec8": 0, "ec7": 1}
    encoded = list(dataset.data["train"].data["entities"])
    assert encoded[0].tolist() == [0, 1]
    assert encoded[1].tolist() == [1, 0]


def test_a_recorded_vocabulary_that_misses_the_schema_is_rejected(tmp_path):
    """Class targets are built in schema order and class columns in vocabulary
    order. Letting the two differ scores every class on another's column."""
    train = frame([{"pubmed_id": 10, "enzymes": [7]}])
    reordered = Vocabulary(
        entities=("ec7",), class_map={"bacteria": (), "enzymes": ("ec7",)}
    )

    with pytest.raises(ValueError, match="do not match the schema"):
        brenda.build_dataset(
            schema=TOY_SCHEMA,
            splits=splits(train),
            encodings=tmp_path / "encodings.hdf5",
            vocabulary=reordered,
        )


def test_indexing_without_a_training_split_needs_a_recorded_vocabulary(
    tmp_path,
):
    """Evaluation loads the test split alone. Without a vocabulary there is
    nothing to derive the columns from, and it must say so rather than raise
    `KeyError: 'train'` from inside the builder."""
    test = frame([{"pubmed_id": 10, "enzymes": [7]}])

    with pytest.raises(ValueError, match="needs the 'train' split"):
        brenda.build_dataset(
            schema=TOY_SCHEMA,
            splits={"test": test},
            encodings=tmp_path / "encodings.hdf5",
        )


def test_the_test_split_alone_can_be_indexed_under_a_recorded_vocabulary(
    tmp_path,
):
    recorded = Vocabulary.from_class_map(
        {"enzymes": {"ec7"}, "bacteria": set()}
    )

    dataset = brenda.build_dataset(
        schema=TOY_SCHEMA,
        splits={"test": frame([{"pubmed_id": 10, "enzymes": [7]}])},
        encodings=tmp_path / "encodings.hdf5",
        vocabulary=recorded,
    )

    assert set(dataset.data) == {"test"}
    assert dataset.entity_index == {"ec7": 0}


def test_only_the_named_splits_are_loaded(tmp_path, monkeypatch):
    """Each split costs a pass over its CSV — the training one runs to
    hundreds of MB. Once the vocabulary is recorded, evaluation has no reason
    to read it, and this pins that it does not."""
    train = frame(
        [{"pubmed_id": 10, "strains": [1], "enzymes": [7]}],
        schema=brenda.BRENDA_SCHEMA,
    )
    loaded = []

    def loader(split):
        def load(noise=0, limit=0):
            loaded.append(split)
            return train.copy()

        return load

    for split in ("training", "validation", "test"):
        monkeypatch.setattr(
            brenda.brenda_references, f"{split}_data", loader(split)
        )
    recorded = Vocabulary.from_class_map(
        {name: set() for name in brenda.BRENDA_SCHEMA.class_names}
        | {"enzymes": {"enz7"}}
    )

    dataset = brenda.brenda_dataset(
        schema=brenda.BRENDA_SCHEMA,
        encodings="nowhere.hdf5",
        vocabulary=recorded,
        split_names=("test",),
    )

    assert loaded == ["test"]
    assert set(dataset.data) == {"test"}
    assert dataset.entity_index == {"enz7": 0}


def test_an_unknown_split_name_is_rejected():
    with pytest.raises(ValueError, match="no such BRENDA split"):
        brenda.brenda_dataset(
            schema=TOY_SCHEMA,
            encodings="nowhere.hdf5",
            split_names=("trian",),
        )


# Run in a subprocess: `PYTHONHASHSEED` is read once, at interpreter start-up,
# so the only way to observe a second hash seed is a second process. Fifty IDs
# per type make the two seeds' raw set iteration differ with certainty; the
# probe reports that raw order too, so the test can prove the seeds really do
# disagree rather than pass because both processes iterated the same way.
_INDEX_ORDER_PROBE = """
import json

from d3text.datasets.brenda import build_entity_index

class_map = {
    "strains": {"str%d" % i for i in range(50)},
    "bacteria": {"bac%d" % i for i in range(50)},
    "enzymes": {"enz%d" % i for i in range(50)},
}
print(json.dumps({
    "index": list(build_entity_index(class_map)),
    "raw": [entity_id for ids in class_map.values() for entity_id in ids],
}))
"""


def _probe_index_order(tmp_path, hash_seed: str) -> dict[str, list[str]]:
    """`build_entity_index`'s output in a process run under `hash_seed`.

    `cwd` is a tmp dir because importing the adapter reaches `lpsn_interface`,
    which opens `lpsn.log` relative to the working directory.
    """
    result = subprocess.run(
        [sys.executable, "-c", _INDEX_ORDER_PROBE],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=True,
        env={**os.environ, "PYTHONHASHSEED": hash_seed},
    )
    return json.loads(result.stdout.splitlines()[-1])


def test_the_entity_index_order_survives_a_new_process(tmp_path):
    """The entity head's columns are positional and the checkpoint records
    only weights, so `evaluate` — a *new* process — must rebuild the same
    order `train` built, or every entity metric is read off another entity's
    column."""
    first = _probe_index_order(tmp_path, "1")
    second = _probe_index_order(tmp_path, "2")

    assert first["raw"] != second["raw"], (
        "the two seeds iterated the sets identically: the probe is not "
        "exercising hash randomization any more"
    )
    assert first["index"] == second["index"]


def test_the_entity_index_is_sorted_within_each_declaration_block(tmp_path):
    """Order in full: the types in schema declaration order, each type's IDs
    sorted within its block."""
    schema = Schema(
        entity_types=(
            EntityType(name="bacteria", prefix="taxon"),
            EntityType(name="processes", prefix="pro", has_ids=False),
            EntityType(name="enzymes", prefix="ec"),
        )
    )
    train = frame(
        [
            {"pubmed_id": 10, "bacteria": [42, 7], "enzymes": [3, 11]},
            {
                "pubmed_id": 20,
                "bacteria": [8],
                "enzymes": [2],
                "processes": [1],
            },
        ],
        schema=schema,
    )

    dataset = brenda.build_dataset(
        schema=schema,
        splits=splits(train),
        encodings=tmp_path / "encodings.hdf5",
    )

    assert list(dataset.entity_index) == [
        "taxon42",
        "taxon7",
        "taxon8",
        "ec11",
        "ec2",
        "ec3",
    ]
    assert list(dataset.entity_index.values()) == list(range(6))


def _record_split_limits(monkeypatch, split_frame) -> dict[str, int]:
    """Patch the three split loaders to record the `limit` each is called with.

    Keyed by split, because a flat list of the values cannot tell a limit that
    reached the right loader from one that also truncated validation.
    """
    limits: dict[str, int] = {}

    def loader(split):
        def load(noise=0, limit=0):
            limits[split] = limit
            return split_frame.copy()

        return load

    for split in ("training", "validation", "test"):
        monkeypatch.setattr(
            brenda.brenda_references, f"{split}_data", loader(split)
        )

    return limits


@pytest.mark.parametrize("limit", [None, 0])
def test_an_absent_limit_loads_the_whole_split(monkeypatch, limit):
    """`None` and 0 both mean "all of it".

    `--limit` is unset as `None` while the loaders spell "no limit" as 0, so
    a caller holding one had to translate it or branch around the parameter.
    """
    train = frame(
        [{"pubmed_id": 10, "strains": [1], "enzymes": [7]}],
        schema=brenda.BRENDA_SCHEMA,
    )
    limits = _record_split_limits(monkeypatch, train)

    dataset = brenda.brenda_dataset(
        schema=brenda.BRENDA_SCHEMA, encodings="nowhere.hdf5", limit=limit
    )

    assert limits == {"training": 0, "validation": 0, "test": 0}
    assert set(dataset.data) == {"train", "val", "test"}


def test_a_limit_truncates_the_training_split_alone(monkeypatch):
    """A real limit still reaches the training loader — it selects the entity
    vocabulary, so it is part of a run's identity — and reaches no other one.
    Truncating validation or test would move every metric a run reports
    without changing anything the run is asked for."""
    train = frame(
        [{"pubmed_id": 10, "strains": [1], "enzymes": [7]}],
        schema=brenda.BRENDA_SCHEMA,
    )
    limits = _record_split_limits(monkeypatch, train)

    brenda.brenda_dataset(
        schema=brenda.BRENDA_SCHEMA, encodings="nowhere.hdf5", limit=250
    )

    assert limits == {"training": 250, "validation": 0, "test": 0}
