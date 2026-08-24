"""The column order a checkpoint records, and what it refuses to record.

Everything here is about *positions*. An entity head is a matrix of the right
width and nothing more, so a vocabulary that comes back off disk in a different
order than it went in is not a loud failure — it is a model scoring every
entity against another entity's column.
"""

import pytest
import torch

from d3text.schema import EntityType, Schema
from d3text.vocabulary import Vocabulary

SCHEMA = Schema(
    entity_types=(
        EntityType(name="enzymes", prefix="ec"),
        EntityType(name="bacteria", prefix="taxon"),
    )
)

CLASS_MAP = {"enzymes": {"ec7", "ec11", "ec2"}, "bacteria": {"taxon42"}}


def test_the_column_order_is_declaration_order_then_sorted_within_it():
    """The types in `class_map`'s order — the schema's — each type's IDs
    sorted within its block. Sorted lexically, so "ec11" precedes "ec2"."""
    vocabulary = Vocabulary.from_class_map(CLASS_MAP)

    assert vocabulary.entities == ("ec11", "ec2", "ec7", "taxon42")
    assert vocabulary.entity_index == {
        "ec11": 0,
        "ec2": 1,
        "ec7": 2,
        "taxon42": 3,
    }


def test_a_class_with_no_instances_keeps_its_column():
    """The class head is sized from `class_map`, so a type nothing grounds
    still owns a column; the entity head must not grow one for it."""
    vocabulary = Vocabulary.from_class_map(
        {"enzymes": {"ec7"}, "processes": set()}
    )

    assert vocabulary.entities == ("ec7",)
    assert vocabulary.class_names == ("enzymes", "processes")
    assert vocabulary.class_matrix().shape == (1, 2)


def test_from_index_takes_its_order_from_the_index_not_the_class_map():
    """`entity_index` is what the labels were encoded against, so it is
    authoritative. `class_map`'s `set`s iterate in a `PYTHONHASHSEED`-dependent
    order and must contribute membership only."""
    vocabulary = Vocabulary.from_index(
        entity_index={"taxon42": 0, "ec7": 1, "ec2": 2},
        class_map=CLASS_MAP | {"enzymes": {"ec7", "ec2"}},
    )

    assert vocabulary.entities == ("taxon42", "ec7", "ec2")
    assert vocabulary.class_map["enzymes"] == ("ec2", "ec7")


def test_from_index_rejects_an_index_that_is_not_a_column_numbering():
    """A gap or a repeat means some column has no entity or two. No head can
    be built from it, so it must not reach a checkpoint either."""
    with pytest.raises(ValueError, match="0..n-1"):
        Vocabulary.from_index(
            entity_index={"ec7": 0, "ec2": 2}, class_map={"enzymes": {"ec7"}}
        )


def test_the_class_matrix_rows_follow_the_entity_columns():
    vocabulary = Vocabulary.from_class_map(CLASS_MAP)

    matrix = vocabulary.class_matrix()

    for entity_id in ("ec11", "ec2", "ec7"):
        row = matrix[vocabulary.entity_index[entity_id]]
        assert row.tolist() == [1.0, 0.0]
    assert matrix[vocabulary.entity_index["taxon42"]].tolist() == [0.0, 1.0]


def test_an_entity_in_two_classes_lights_both_columns():
    """Walking the classes rather than inverting them into an entity -> class
    dict: an inversion keeps whichever class it wrote last."""
    vocabulary = Vocabulary(
        entities=("ec7",),
        class_map={"enzymes": ("ec7",), "bacteria": ("ec7",)},
    )

    assert vocabulary.class_matrix().tolist() == [[1.0, 1.0]]


def test_a_class_naming_an_entity_with_no_column_is_rejected():
    """What a truncated payload looks like. Unchecked it surfaces as a
    `KeyError` from inside `class_matrix`, long after the file was read."""
    with pytest.raises(ValueError, match="own no column"):
        Vocabulary(entities=("ec7",), class_map={"enzymes": ("ec7", "ec9")})


def test_a_repeated_entity_is_rejected():
    with pytest.raises(ValueError, match="duplicate entity IDs"):
        Vocabulary(entities=("ec7", "ec7"), class_map={"enzymes": ("ec7",)})


def test_the_payload_round_trips_the_order():
    original = Vocabulary.from_class_map(CLASS_MAP)

    assert Vocabulary.from_payload(original.to_payload()) == original


def test_the_payload_is_plain_builtins():
    """`torch.load` defaults to `weights_only=True`, which admits tensors and
    builtins and nothing else; a pickled dataclass would make the checkpoint
    unreadable without trusting it."""
    payload = Vocabulary.from_class_map(CLASS_MAP).to_payload()

    assert type(payload) is dict
    assert type(payload["entities"]) is list
    assert type(payload["class_map"]) is dict
    assert all(type(name) is str for name in payload["entities"])
    assert all(type(ids) is list for ids in payload["class_map"].values())


@pytest.mark.parametrize(
    "payload",
    [
        {"entities": ["ec7"]},
        {"class_map": {"enzymes": ["ec7"]}},
        {"entities": "ec7", "class_map": {"enzymes": ["ec7"]}},
        {"entities": ["ec7"], "class_map": ["enzymes"]},
    ],
)
def test_a_malformed_payload_is_rejected_by_name(payload):
    """This runs on bytes that came off disk, so it must say what is wrong
    rather than raise `KeyError` or `TypeError` from the conversion."""
    with pytest.raises(ValueError, match="checkpoint vocabulary"):
        Vocabulary.from_payload(payload)


def test_a_vocabulary_whose_classes_differ_from_the_schema_does_not_fit():
    """Targets are built in schema order and columns in vocabulary order, so
    the two disagreeing means every class is scored on another's column."""
    reordered = Vocabulary(
        entities=("ec7",),
        class_map={"bacteria": (), "enzymes": ("ec7",)},
    )

    with pytest.raises(ValueError, match="do not match the schema"):
        reordered.check_fits(SCHEMA)


def test_a_vocabulary_matching_the_schema_fits():
    Vocabulary.from_class_map(CLASS_MAP).check_fits(SCHEMA)


def test_disagreement_reports_a_resize():
    recorded = Vocabulary.from_class_map(CLASS_MAP)
    derived = Vocabulary.from_class_map(
        {"enzymes": {"ec7", "ec11", "ec2", "ec3"}, "bacteria": {"taxon42"}}
    )

    assert "4 entities recorded against 5" in recorded.disagreement_with(
        derived
    )


def test_disagreement_reports_a_repermutation_of_the_same_entities():
    """The dangerous case: same width, so `load_state_dict` accepts it."""
    recorded = Vocabulary.from_class_map(CLASS_MAP)
    shuffled = Vocabulary(
        entities=("ec2", "ec11", "ec7", "taxon42"),
        class_map=recorded.class_map,
    )

    report = recorded.disagreement_with(shuffled)

    assert "different order" in report
    assert "2 columns moved" in report


def test_an_identical_vocabulary_reports_no_disagreement():
    recorded = Vocabulary.from_class_map(CLASS_MAP)

    assert recorded.disagreement_with(Vocabulary.from_class_map(CLASS_MAP)) is (
        None
    )


def test_as_class_map_hands_back_the_shape_the_models_take():
    vocabulary = Vocabulary.from_class_map(CLASS_MAP)

    assert vocabulary.as_class_map() == CLASS_MAP
    assert list(vocabulary.as_class_map()) == list(CLASS_MAP)


def test_the_class_matrix_is_float32():
    """It is multiplied into the class logits; an integer matrix would upcast
    or fail depending on the operand."""
    matrix = Vocabulary.from_class_map(CLASS_MAP).class_matrix()

    assert matrix.dtype == torch.float32
