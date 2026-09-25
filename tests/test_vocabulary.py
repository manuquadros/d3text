"""The column order a checkpoint records, and what it refuses to record.

Everything here is about *positions*. A class head is a matrix of the right
width and nothing more, so a vocabulary that comes back off disk in a different
order than it went in is not a loud failure — it is a model scoring every class
against another class's column. The members are the one part that is not
positional: they are the training split's entity vocabulary, which is what
splits detection recall by novelty.
"""

import pytest

from d3text.schema import EntityType, Schema
from d3text.vocabulary import Vocabulary

SCHEMA = Schema(
    entity_types=(
        EntityType(name="enzymes", prefix="ec"),
        EntityType(name="bacteria", prefix="taxon"),
    )
)

CLASS_MAP = {"enzymes": {"ec7", "ec11", "ec2"}, "bacteria": {"taxon42"}}


def test_the_class_order_is_the_declaration_order_it_was_given():
    """`class_map`'s order is the class head's column order, so a dict that
    arrived reordered must come back reordered rather than normalised.

    Declared enzymes-first, which is *not* alphabetical order, so a
    `from_class_map` that sorted its keys is caught rather than agreeing with
    the schema by luck.
    """
    vocabulary = Vocabulary.from_class_map(
        {"enzymes": {"ec7"}, "bacteria": {"taxon42"}}
    )

    assert vocabulary.class_names == ("enzymes", "bacteria")
    assert tuple(sorted(vocabulary.class_names)) != vocabulary.class_names


def test_the_members_are_sorted_within_each_class():
    """Sorted lexically, so "ec11" precedes "ec2"."""
    vocabulary = Vocabulary.from_class_map(CLASS_MAP)

    assert vocabulary.class_map["enzymes"] == ("ec11", "ec2", "ec7")
    assert vocabulary.class_map["bacteria"] == ("taxon42",)


def test_the_sort_holds_for_a_block_too_large_to_match_by_chance():
    """Fifty IDs per type make an unsorted `set`'s iteration order coincide
    with the sorted order with vanishing probability (~1/50!), so this goes
    red deterministically, in one process, if the sort in `from_class_map`
    is ever dropped — no second process or hash seed needed to prove it."""
    class_map = {
        "enzymes": {f"ec{i}" for i in range(50)},
        "bacteria": {f"taxon{i}" for i in range(50)},
    }
    vocabulary = Vocabulary.from_class_map(class_map)

    assert vocabulary.class_map["enzymes"] == tuple(
        sorted(class_map["enzymes"])
    )
    assert vocabulary.class_map["bacteria"] == tuple(
        sorted(class_map["bacteria"])
    )


def test_a_class_with_no_instances_keeps_its_column():
    """`check_fits` requires the recorded class names to equal the schema's,
    so a type nothing grounds still owns a column."""
    vocabulary = Vocabulary.from_class_map(
        {"enzymes": {"ec7"}, "processes": set()}
    )

    assert vocabulary.class_names == ("enzymes", "processes")
    assert vocabulary.class_map["processes"] == ()
    assert len(vocabulary) == 2


def test_entity_ids_unions_every_class_s_members():
    """`evaluate` hands this to the detection accumulator as the training
    split's vocabulary, so a class left out of the union would make its
    entities read as novel."""
    assert Vocabulary.from_class_map(CLASS_MAP).entity_ids == frozenset(
        {"ec7", "ec11", "ec2", "taxon42"}
    )


def test_entity_ids_counts_an_entity_in_two_classes_once():
    """It is a set, not a concatenation: an ID declared under two types is
    one entity the training split named, not two."""
    vocabulary = Vocabulary(
        class_map={"enzymes": ("ec7",), "bacteria": ("ec7",)}
    )

    assert vocabulary.entity_ids == frozenset({"ec7"})


def test_a_class_repeating_an_entity_is_rejected():
    """What a truncated or hand-edited payload looks like."""
    with pytest.raises(ValueError, match="duplicate entity IDs under class"):
        Vocabulary(class_map={"enzymes": ("ec7", "ec7")})


def test_the_payload_round_trips_the_order():
    original = Vocabulary.from_class_map(CLASS_MAP)

    assert Vocabulary.from_payload(original.to_payload()) == original


def test_the_payload_is_plain_builtins():
    """`torch.load` defaults to `weights_only=True`, which admits tensors and
    builtins and nothing else; a pickled dataclass would make the checkpoint
    unreadable without trusting it.

    The types are compared exactly rather than with `isinstance`, because a
    subclass of `dict` is as unreadable under `weights_only` as a dataclass.
    """
    payload = Vocabulary.from_class_map(CLASS_MAP).to_payload()

    assert payload.__class__ is dict
    assert payload["class_map"].__class__ is dict
    assert all(ids.__class__ is list for ids in payload["class_map"].values())


def test_the_payload_carries_no_entity_column_order():
    """Format 2 records no per-entity column, so a payload that still carried
    one would be a format-1 file wearing the new version number."""
    assert set(Vocabulary.from_class_map(CLASS_MAP).to_payload()) == {
        "class_map"
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"entities": ["ec7"]},
        {},
        {"class_map": ["enzymes"]},
        {"class_map": {"enzymes": "ec7"}},
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
    reordered = Vocabulary(class_map={"bacteria": (), "enzymes": ("ec7",)})

    with pytest.raises(ValueError, match="do not match the schema"):
        reordered.check_fits(SCHEMA)


def test_a_vocabulary_matching_the_schema_fits():
    Vocabulary.from_class_map(CLASS_MAP).check_fits(SCHEMA)


def test_as_class_map_hands_back_the_shape_the_dataset_takes():
    vocabulary = Vocabulary.from_class_map(CLASS_MAP)

    assert vocabulary.as_class_map() == CLASS_MAP
    assert list(vocabulary.as_class_map()) == list(CLASS_MAP)
