"""The committed organism bridge keys each organism under its current class.

The bridge is keyed by prefixed entity id, and the class prefix is part of the
key; a table built before organisms were re-sorted between `bacteria` and
`other_organisms` leaves a moved organism bridged only under the prefix of the
class it left, which the linking evaluation reads as unbridged.
"""

import ast
import pathlib

import polars as pl
import pytest
from brenda_references.data_paths import resolve_data_dir

from d3text.identifier_bridge import load_bridge
from d3text.schema import BRENDA_SCHEMA

_SPLITS = ("training_data", "validation_data", "test_data")
_DATA = resolve_data_dir()
_BRIDGE = (
    pathlib.Path(__file__).resolve().parents[1] / "data/organism_taxids.tsv"
)
_PREFIX = {t.name: t.prefix for t in BRENDA_SCHEMA.entity_types}
_MISSING = [s for s in _SPLITS if not (_DATA / f"{s}.csv").is_file()]


def _ids(split: str, column: str) -> set[str]:
    cells = pl.read_csv(_DATA / f"{split}.csv", columns=[column])[column]
    return {
        f"{_PREFIX[column]}{int(i)}"
        for cell in cells.drop_nulls()
        for i in ast.literal_eval(cell)
    }


@pytest.mark.integration
@pytest.mark.skipif(
    bool(_MISSING), reason=f"splits absent under {_DATA}: {_MISSING}"
)
@pytest.mark.parametrize("split", _SPLITS)
def test_no_organism_is_bridged_only_under_the_other_class_prefix(
    split: str,
) -> None:
    """An organism the bridge knows must be keyed by its current class."""
    bridge = load_bridge(_BRIDGE)
    swap = {"bacteria": "other_organisms", "other_organisms": "bacteria"}
    stranded = {}
    for column, other in swap.items():
        stranded[column] = sorted(
            entity
            for entity in _ids(split, column)
            if not bridge.external_ids(entity)
            and bridge.external_ids(
                _PREFIX[other] + entity.removeprefix(_PREFIX[column])
            )
        )
    assert stranded == {"bacteria": [], "other_organisms": []}
