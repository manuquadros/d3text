"""The BRENDA extraction target, as a `Schema`."""

from d3text.schema import EntityType, RelationType, Schema

__all__ = ["BRENDA_SCHEMA"]

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
