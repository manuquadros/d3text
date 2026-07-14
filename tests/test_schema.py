"""The schema dataclasses, and the BRENDA schema built from them.

`Schema` validates on construction, so every consumer downstream may read
`class_names`, `prefix_to_type` and `none_relation_index` without re-checking
them. The tests below pin the invariants that makes true, and pin the BRENDA
schema against the two orders the corpus and the heads already agree on: the
class columns are the corpus entity columns, and the relation columns are the
positions of the one-hot label vector `brenda_references` emits per candidate
pair — the models take its argmax as the target index, so a permutation here
would silently relabel every gold relation.
"""

import dataclasses

import pytest

from d3text.datasets.brenda import BRENDA_SCHEMA
from d3text.schema import EntityType, RelationType, Schema

_ENZYMES = EntityType(name="enzymes", prefix="enz")
_BACTERIA = EntityType(name="bacteria", prefix="bac")
_NONE = RelationType(name="none", is_none=True)
_HAS_ENZYME = RelationType(
    name="HasEnzyme",
    subject_types=("bacteria",),
    object_types=("enzymes",),
)


def _schema(
    entity_types: tuple[EntityType, ...] = (_ENZYMES, _BACTERIA),
    relation_types: tuple[RelationType, ...] = (_HAS_ENZYME, _NONE),
) -> Schema:
    return Schema(entity_types=entity_types, relation_types=relation_types)


def test_class_names_follow_entity_order() -> None:
    assert _schema().class_names == ("enzymes", "bacteria")
    assert _schema(entity_types=(_BACTERIA, _ENZYMES)).class_names == (
        "bacteria",
        "enzymes",
    )


def test_prefix_to_type_maps_each_prefix_to_its_type() -> None:
    assert _schema().prefix_to_type == {"enz": _ENZYMES, "bac": _BACTERIA}


def test_none_relation_index_is_the_column_of_the_none_label() -> None:
    schema = _schema()
    assert schema.relation_names == ("HasEnzyme", "none")
    assert schema.none_relation_index == 1
    assert schema.relation_types[schema.none_relation_index].is_none


def test_schema_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        _schema().entity_types = ()  # type: ignore[misc]


def test_entity_types_cannot_be_empty() -> None:
    with pytest.raises(ValueError, match="at least one entity type"):
        _schema(entity_types=())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        pytest.param(
            {"entity_types": (_ENZYMES, EntityType("enzymes", "enz2"))},
            "Duplicate entity type name",
            id="duplicate-name",
        ),
        pytest.param(
            {"entity_types": (_ENZYMES, EntityType("enzyme_names", "enz"))},
            "Duplicate entity ID prefix",
            id="duplicate-prefix",
        ),
        pytest.param(
            {"relation_types": (_HAS_ENZYME, _HAS_ENZYME, _NONE)},
            "Duplicate relation type name",
            id="duplicate-relation",
        ),
    ],
)
def test_names_and_prefixes_are_unique(
    kwargs: dict[str, tuple], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _schema(**kwargs)


def test_relation_arguments_must_be_known_entity_types() -> None:
    strains = RelationType(
        name="HasSpecies",
        subject_types=("strains",),
        object_types=("bacteria",),
    )
    with pytest.raises(
        ValueError, match="unknown entity types \\['strains'\\]"
    ):
        _schema(relation_types=(strains, _NONE))


def test_a_relation_needs_both_a_subject_and_an_object_type() -> None:
    subjectless = RelationType(name="HasEnzyme", object_types=("enzymes",))
    with pytest.raises(ValueError, match="both a subject and an object"):
        _schema(relation_types=(subjectless, _NONE))


def test_the_none_relation_takes_no_arguments() -> None:
    armed = RelationType(
        name="none",
        subject_types=("bacteria",),
        object_types=("enzymes",),
        is_none=True,
    )
    with pytest.raises(ValueError, match="takes no arguments"):
        _schema(relation_types=(_HAS_ENZYME, armed))


@pytest.mark.parametrize(
    "relation_types",
    [
        pytest.param((_HAS_ENZYME,), id="no-none"),
        pytest.param(
            (_HAS_ENZYME, _NONE, RelationType(name="nil", is_none=True)),
            id="two-nones",
        ),
    ],
)
def test_exactly_one_none_relation(
    relation_types: tuple[RelationType, ...],
) -> None:
    with pytest.raises(ValueError, match="exactly one `none` relation"):
        _schema(relation_types=relation_types)


def test_the_none_relation_comes_last() -> None:
    with pytest.raises(ValueError, match="must come last"):
        _schema(relation_types=(_NONE, _HAS_ENZYME))


def test_brenda_classes_are_the_corpus_entity_columns() -> None:
    assert BRENDA_SCHEMA.class_names == (
        "strains",
        "bacteria",
        "other_organisms",
        "enzymes",
    )


def test_brenda_prefixes_are_the_first_three_letters_of_the_column() -> None:
    """The corpus builds an entity ID as `column[:3] + str(id)`, so a prefix
    that is anything else silently names entities the corpus never mentions."""
    for entity_type in BRENDA_SCHEMA.entity_types:
        assert entity_type.prefix == entity_type.name[:3]


def test_brenda_relations_are_the_one_hot_label_columns() -> None:
    assert BRENDA_SCHEMA.relation_names == ("HasEnzyme", "HasSpecies", "none")
    assert BRENDA_SCHEMA.none_relation_index == 2


def test_brenda_vocabularies_exist() -> None:
    from d3text.data.data import DATA_DIR

    named = [et for et in BRENDA_SCHEMA.entity_types if et.vocab_path]
    assert named, "the dictionary tagger has no term list to read"
    for entity_type in named:
        assert (DATA_DIR / entity_type.vocab_path).is_file()
