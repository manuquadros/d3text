"""The bridge table's on-disk contract and its two-way reading.

The table is the only thing standing between an evaluation and a 176 MB NCBI
dump, so what it says has to survive a round trip and refuse to be read as
identifiers of another kind.
"""

import collections
import pathlib

import pytest
from d3text.identifier_bridge import (
    EC_NUMBER,
    NCBI_TAXID,
    BridgeRow,
    IdentifierBridge,
    load_bridge,
    write_bridge,
)

ROWS = [
    BridgeRow("bac1", "562", "organism"),
    BridgeRow("bac2", "562", "synonym"),
    BridgeRow("bac3", "1423", "organism"),
]

ROOT = pathlib.Path(__file__).resolve().parents[1]
COMMITTED = ROOT / "data/organism_taxids.tsv"
COMMITTED_EC = ROOT / "data/enzyme_ec_numbers.tsv"


def _written(tmp_path: pathlib.Path, namespace: str = NCBI_TAXID):
    path = tmp_path / "bridge.tsv"
    write_bridge(path, namespace, ROWS)
    return path


def test_a_written_table_reads_back_unchanged(tmp_path: pathlib.Path) -> None:
    bridge = load_bridge(_written(tmp_path))

    assert bridge.namespace == NCBI_TAXID
    assert len(bridge) == 3
    assert bridge.external_id("bac3") == "1423"
    assert bridge.sources["bac2"] == "synonym"


def test_the_table_reads_in_both_directions(tmp_path: pathlib.Path) -> None:
    bridge = load_bridge(_written(tmp_path))

    assert bridge.entity_ids("562") == {"bac1", "bac2"}
    assert bridge.entity_ids("1423") == {"bac3"}
    assert bridge.entity_ids("999") == frozenset()


def test_an_identifier_two_entities_share_has_no_sole_entity(
    tmp_path: pathlib.Path,
) -> None:
    """BRENDA curates the same taxon twice, and nothing in a linker's answer
    could choose between the rows — so the identifier is not gold for a
    strict score and must not be treated as one."""
    bridge = load_bridge(_written(tmp_path))

    assert bridge.sole_entity("562") is None
    assert bridge.sole_entity("1423") == "bac3"
    assert bridge.sole_entity("999") is None


def test_a_table_of_other_identifiers_is_refused(
    tmp_path: pathlib.Path,
) -> None:
    """An EC table and a taxid table are both `entity_id -> string`, so
    reading one for the other produces a score rather than an error."""
    path = _written(tmp_path, namespace="ec_number")

    with pytest.raises(ValueError, match="ec_number"):
        load_bridge(path, expect=NCBI_TAXID)


def test_rows_are_written_sorted(tmp_path: pathlib.Path) -> None:
    """A rebuild against a refreshed dump has to diff as what changed, not as
    a dictionary's iteration order."""
    path = tmp_path / "bridge.tsv"
    write_bridge(path, NCBI_TAXID, reversed(ROWS))

    assert path.read_text(encoding="utf8").splitlines()[2:] == [
        "bac1\t562\torganism",
        "bac2\t562\tsynonym",
        "bac3\t1423\torganism",
    ]


def test_an_entity_with_two_identifiers_is_refused() -> None:
    with pytest.raises(ValueError, match="two ncbi_taxid identifiers"):
        IdentifierBridge.from_rows(
            NCBI_TAXID,
            [BridgeRow("bac1", "562", "organism"), BridgeRow("bac1", "9", "x")],
        )


def test_a_field_carrying_the_separator_is_refused(
    tmp_path: pathlib.Path,
) -> None:
    with pytest.raises(ValueError, match="separator or newline"):
        write_bridge(
            tmp_path / "bridge.tsv",
            NCBI_TAXID,
            [BridgeRow("bac1", "5\t62", "organism")],
        )


def test_a_file_declaring_no_namespace_is_refused(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "bridge.tsv"
    path.write_text(
        "entity_id\texternal_id\tsource\nbac1\t562\torganism\n",
        encoding="utf8",
    )

    with pytest.raises(ValueError, match="namespace"):
        load_bridge(path)


def test_a_file_with_no_header_at_all_is_refused(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "bridge.tsv"
    path.write_text("bac1\t562\torganism\n", encoding="utf8")

    with pytest.raises(ValueError, match="no bridge header"):
        load_bridge(path)


def test_a_short_row_is_refused(tmp_path: pathlib.Path) -> None:
    path = _written(tmp_path)
    path.write_text(
        path.read_text(encoding="utf8") + "bac4\t99\n", encoding="utf8"
    )

    with pytest.raises(ValueError, match="2 fields"):
        load_bridge(path)


def test_the_committed_table_carries_both_organism_halves() -> None:
    """The table is the whole of what the evaluation reads, and building it
    needs a 176 MB dump nothing here has: a duplicated entity or a field that
    grew a tab would otherwise surface only on the next machine that has one.

    Both prefixes have to be present, because a taxid carried by a bacterium
    and by an other organism is gold for neither, and half a table cannot see
    that. So does every route the builder pairs by: a rebuild that lost the
    identifier join would silently fall back to resolving names, which
    answers with the parent species wherever the entity is a subspecies.
    """
    bridge = load_bridge(COMMITTED, expect=NCBI_TAXID)
    prefixes = collections.Counter(entity[:3] for entity in bridge.by_entity)

    assert set(prefixes) == {"bac", "oth"}
    assert all(taxid.isdigit() for taxid in bridge.by_entity.values())
    assert set(bridge.sources.values()) == {
        "lpsn_id",
        "organism",
        "organism_all_divisions",
        "synonym",
        "synonym_all_divisions",
        "inline_name",
    }


def test_the_committed_ec_table_names_one_enzyme_per_number() -> None:
    """The enzyme bridge is a curated identifier column, not a resolution, so
    it is expected to be exact — and that is what makes it useless as a
    filter. `sole_entity` excludes nothing here, so the whole of the judged
    subset rests on the outside nomenclature; a table where that stopped
    being true would move the subset without moving a score.
    """
    bridge = load_bridge(COMMITTED_EC, expect=EC_NUMBER)
    numbers = collections.Counter(bridge.by_entity.values())

    assert set(entity[:3] for entity in bridge.by_entity) == {"enz"}
    assert set(bridge.sources.values()) == {"ec_class"}
    assert [number for number, count in numbers.items() if count > 1] == []


def test_the_two_committed_tables_are_not_interchangeable() -> None:
    """Both are `entity_id -> string`, so the namespace stamp is the only
    thing that stops one being scored as the other — which raises nothing on
    its own and produces a number."""
    with pytest.raises(ValueError, match="ec_number"):
        load_bridge(COMMITTED_EC, expect=NCBI_TAXID)

    with pytest.raises(ValueError, match="ncbi_taxid"):
        load_bridge(COMMITTED, expect=EC_NUMBER)
