"""Pure unit tests for `d3text.schema`.

No data files, no network, no GPU: the schema is a leaf module over plain
dataclasses.
"""

import dataclasses
import pathlib

import pytest

from d3text.schema import EntityType, RelationType, Schema

ENZYMES = EntityType(name="enzymes", prefix="enz")
BACTERIA = EntityType(name="bacteria", prefix="bac")
STRAINS = EntityType(name="strains", prefix="str")

HAS_ENZYME = RelationType(
    name="HasEnzyme", subject_type="bacteria", object_type="enzymes"
)
NONE = RelationType(name="none", is_none=True)


def brenda_like() -> Schema:
    return Schema(
        entity_types=(STRAINS, BACTERIA, ENZYMES),
        relation_types=(HAS_ENZYME, NONE),
    )


def test_class_names_follow_declaration_order():
    assert brenda_like().class_names == ("strains", "bacteria", "enzymes")


def test_class_names_exclude_the_oos_sentinel():
    # The head appends OOS itself; a schema that carried it would make the
    # class head one column too wide.
    assert "OOS" not in brenda_like().class_names


def test_relation_names_follow_declaration_order():
    assert brenda_like().relation_names == ("HasEnzyme", "none")


def test_prefix_to_type_maps_each_prefix_to_its_type():
    assert brenda_like().prefix_to_type == {
        "str": STRAINS,
        "bac": BACTERIA,
        "enz": ENZYMES,
    }


def test_none_relation_index_is_found_by_flag_not_by_position():
    schema = Schema(
        entity_types=(BACTERIA, ENZYMES),
        relation_types=(
            RelationType(name="nothing_here", is_none=True),
            HAS_ENZYME,
        ),
    )
    assert schema.none_relation_index == 0
    assert schema.relation_names[schema.none_relation_index] == "nothing_here"


def test_none_relation_index_rejects_a_schema_without_relations():
    schema = Schema(entity_types=(BACTERIA,))
    with pytest.raises(ValueError, match="no relation types"):
        schema.none_relation_index


def test_classification_only_schema_needs_no_none_relation():
    schema = Schema(entity_types=(BACTERIA, ENZYMES))
    assert schema.relation_names == ()
    schema.validate()


def test_validate_rejects_relation_naming_unknown_entity_type():
    with pytest.raises(ValueError, match="unknown entity type 'bacteria'"):
        Schema(
            entity_types=(ENZYMES,),
            relation_types=(HAS_ENZYME, NONE),
        )


def test_validate_rejects_none_relation_naming_unknown_entity_type():
    with pytest.raises(ValueError, match="unknown entity type 'viruses'"):
        Schema(
            entity_types=(BACTERIA, ENZYMES),
            relation_types=(
                HAS_ENZYME,
                RelationType(
                    name="none",
                    subject_type="viruses",
                    object_type="enzymes",
                    is_none=True,
                ),
            ),
        )


def test_validate_rejects_relation_without_argument_types():
    with pytest.raises(ValueError, match="must declare both"):
        Schema(
            entity_types=(BACTERIA, ENZYMES),
            relation_types=(RelationType(name="HasEnzyme"), NONE),
        )


def test_validate_rejects_zero_none_relation_types():
    with pytest.raises(ValueError, match="exactly one `is_none`"):
        Schema(entity_types=(BACTERIA, ENZYMES), relation_types=(HAS_ENZYME,))


def test_validate_rejects_multiple_none_relation_types():
    with pytest.raises(ValueError, match="exactly one `is_none`"):
        Schema(
            entity_types=(BACTERIA, ENZYMES),
            relation_types=(
                HAS_ENZYME,
                NONE,
                RelationType(name="no_relation", is_none=True),
            ),
        )


def test_validate_rejects_duplicate_entity_type_names():
    with pytest.raises(ValueError, match="duplicate entity type names"):
        Schema(
            entity_types=(BACTERIA, EntityType(name="bacteria", prefix="bct"))
        )


def test_validate_rejects_duplicate_prefixes():
    # prefix_to_type would silently keep only the last type under "bac".
    with pytest.raises(ValueError, match="duplicate entity ID prefixes"):
        Schema(
            entity_types=(BACTERIA, EntityType(name="bacilli", prefix="bac"))
        )


def test_validate_rejects_duplicate_relation_names():
    with pytest.raises(ValueError, match="duplicate relation type names"):
        Schema(
            entity_types=(BACTERIA, ENZYMES),
            relation_types=(HAS_ENZYME, HAS_ENZYME, NONE),
        )


def test_validate_rejects_empty_entity_types():
    with pytest.raises(ValueError, match="at least one entity type"):
        Schema(entity_types=())


@pytest.mark.parametrize("blank", [{"name": ""}, {"prefix": ""}])
def test_validate_rejects_blank_entity_names_and_prefixes(blank):
    with pytest.raises(ValueError, match="non-empty name and prefix"):
        Schema(entity_types=(dataclasses.replace(BACTERIA, **blank),))


def test_schema_is_frozen_and_hashable():
    schema = brenda_like()
    with pytest.raises(dataclasses.FrozenInstanceError):
        schema.entity_types = ()
    assert hash(schema) == hash(brenda_like())


def test_entity_type_carries_optional_vocab_and_abbreviation():
    entity_type = EntityType(
        name="bacteria",
        prefix="bac",
        vocab_path=pathlib.Path("data/bacteria.txt"),
        abbreviation_fn=lambda name: name[0] + ".",
    )
    assert entity_type.vocab_path == pathlib.Path("data/bacteria.txt")
    assert entity_type.abbreviation_fn is not None
    assert entity_type.abbreviation_fn("Escherichia") == "E."
    assert BACTERIA.vocab_path is None and BACTERIA.abbreviation_fn is None
    assert BACTERIA.has_ids
