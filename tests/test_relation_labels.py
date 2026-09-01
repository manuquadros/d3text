"""The relation one-hot's column meaning, end to end.

`preprocess_relations` writes literal one-hot vectors and `ground_truth` reads
them back with `argmax` against the schema's declaration order. Neither side
imports the other, so they agree only by coincidence: reorder the tuple, or
swap the literals, and every `HasEnzyme` gold trains and scores as a
`HasSpecies` — wrong numbers, no crash.
"""

import numpy as np
import pandas as pd
import pytest
from brenda_references.brenda_references import preprocess_relations

from d3text.schema import BRENDA_SCHEMA

BACTERIUM, STRAIN, ENZYME = 42, 7, 26836

BACTERIUM_ID = f"bac{BACTERIUM}"
STRAIN_ID = f"str{STRAIN}"
ENZYME_ID = f"enz{ENZYME}"


def synthetic_row() -> pd.Series:
    """One document declaring one relation of every non-null type.

    `preprocess_relations` takes its unrelated pairs from combinations of the
    `entities` column, which is how the null class gets exercised too.
    """
    return pd.Series(
        {
            "bacteria": [BACTERIUM],
            "strains": [STRAIN],
            "other_organisms": [],
            "enzymes": [ENZYME],
            "entities": sorted([BACTERIUM_ID, STRAIN_ID, ENZYME_ID]),
            "relations": repr(
                {
                    "HasEnzyme": [
                        {"subject": BACTERIUM, "object": ENZYME},
                    ],
                    "HasSpecies": [
                        {"subject": STRAIN, "object": BACTERIUM},
                    ],
                }
            ),
        }
    )


def relation_name(label: np.ndarray) -> str:
    """The label a one-hot vector names, decoded the way the model decodes it."""
    return BRENDA_SCHEMA.relation_names[int(label.argmax())]


@pytest.fixture
def preprocessed() -> dict[tuple[str, str], np.ndarray]:
    (pairs,) = preprocess_relations(synthetic_row())["relations"]
    return pairs


@pytest.mark.parametrize(
    ("arguments", "declared"),
    [
        ((BACTERIUM_ID, ENZYME_ID), "HasEnzyme"),
        ((BACTERIUM_ID, STRAIN_ID), "HasSpecies"),
        ((ENZYME_ID, STRAIN_ID), "none"),
    ],
)
def test_one_hot_column_names_the_declared_relation(
    preprocessed: dict[tuple[str, str], np.ndarray],
    arguments: tuple[str, str],
    declared: str,
) -> None:
    assert relation_name(preprocessed[arguments]) == declared


def test_one_hot_width_matches_the_relation_head(
    preprocessed: dict[tuple[str, str], np.ndarray],
) -> None:
    """A vector narrower than the schema decodes silently, not loudly: an
    argmax over three columns is in range for a head of four.
    """
    widths = {label.shape for label in preprocessed.values()}
    assert widths == {(len(BRENDA_SCHEMA.relation_types),)}


def test_null_class_lands_on_the_schema_null_column(
    preprocessed: dict[tuple[str, str], np.ndarray],
) -> None:
    unrelated = preprocessed[(ENZYME_ID, STRAIN_ID)]
    assert int(unrelated.argmax()) == BRENDA_SCHEMA.none_relation_index
