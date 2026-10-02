"""Split a BRENDA frame's other organisms into `KINGDOM_SCHEMA`'s columns.

Kingdom is read off the NCBI lineage of the taxid `data/organism_taxids.tsv`
bridges an organism to; an organism with no bridged taxid, or one under no
kingdom's ancestor, stays in `other_organisms`.
"""

import functools

import pandas as pd
from taxonomy.ncbitax import ncbitax

from d3text.identifier_bridge import NCBI_TAXID, load_bridge
from d3text.schema import DATA_DIR, KINGDOMS
from d3text.taxonomy import merged_taxids

_OTHER = "other_organisms"
_OTHER_PREFIX = "oth"
ORGANISM_BRIDGE = DATA_DIR / "organism_taxids.tsv"


@functools.cache
def _kingdom_by_id() -> dict[int, tuple[str, str]]:
    """Other-organism ID -> `(kingdom type name, prefix)`, for those placed."""
    bridge = load_bridge(ORGANISM_BRIDGE, expect=NCBI_TAXID)
    merged = merged_taxids()
    placed: dict[int, tuple[str, str]] = {}
    for entity_id in bridge.by_entity:
        if not entity_id.startswith(_OTHER_PREFIX):
            continue
        external = bridge.external_id(entity_id)
        if external is None:
            continue
        tax_id = merged.get(int(external), int(external))
        for name, prefix, ancestor in KINGDOMS:
            if ncbitax.is_descendant(tax_id, ancestor):
                placed[int(entity_id.removeprefix(_OTHER_PREFIX))] = (
                    name,
                    prefix,
                )
                break
    return placed


def _rename(entity_id: str, placed: dict[int, tuple[str, str]]) -> str:
    if not entity_id.startswith(_OTHER_PREFIX):
        return entity_id
    kingdom = placed.get(int(entity_id.removeprefix(_OTHER_PREFIX)))
    if kingdom is None:
        return entity_id
    return kingdom[1] + entity_id.removeprefix(_OTHER_PREFIX)


def split_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """A copy of `frame` with its other organisms moved to kingdom columns.

    Relation arguments naming a moved organism are re-prefixed to match, so
    `filter_relations` still types them.

    :param frame: a split as `brenda_references.load_split` returns it.
    :return: the copy, carrying one list column per kingdom.
    """
    placed = _kingdom_by_id()
    out = frame.copy()
    for name, _, _ in KINGDOMS:
        out[name] = [
            [i for i in ids if placed.get(i, ("",))[0] == name]
            for ids in frame[_OTHER]
        ]
    out[_OTHER] = [[i for i in ids if i not in placed] for ids in frame[_OTHER]]
    out["relations"] = [
        [
            {
                tuple(_rename(arg, placed) for arg in pair): value
                for pair, value in pairs.items()
            }
            for pairs in relations
        ]
        for relations in frame["relations"]
    ]
    return out
