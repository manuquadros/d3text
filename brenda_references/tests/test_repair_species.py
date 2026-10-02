"""Pin `repair_species` restoring a lost species without inventing one.

The invariant: after a repair, a `HasSpecies` object BRENDA attests for a
document names an organism in that document's columns and in its split
row, under the class BRENDA's relations put it in, and nothing else in
either resolves differently. BRENDA and LPSN are faked, so nothing here
reads the SQL database or real data.
"""

from __future__ import annotations

import ast
import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

import pytest
from brenda_references.docdb import BrendaDocDB
from d3types import HasSpecies, Organism
from scripts import repair_species as rs
from tinydb.table import Document as TinyDBDoc

BAC, OTH = "bacteria", "other_organisms"


def _doc(
    bacteria: dict[str, str],
    other: dict[str, str],
    strains: list[int],
    has_species: Sequence[tuple[int, int]],
    has_enzyme: Sequence[tuple[int, int]] = (),
) -> dict[str, Any]:
    return {
        BAC: bacteria,
        OTH: other,
        "strains": strains,
        "relations": {
            "HasEnzyme": [{"subject": s, "object": o} for s, o in has_enzyme],
            "HasSpecies": [{"subject": s, "object": o} for s, o in has_species],
        },
    }


# Reference id -> document. 1 restores a bacterium under its own id. 2's
# object 20 is a different `bacteria` record, 3's object 30 names another
# organism in other_organisms (document 4), 6's object 70 is also a strain
# a HasEnzyme subject resolves to: each needs a fresh id. 4 already
# resolves. 7 and 8 already hold the organism BRENDA names for their object,
# 7 under its name and 8 under a synonym, each with another id. BRENDA lists 5's organism 60 but no HasSpecies pair naming it.
DOCS = {
    1: _doc({}, {}, [5], [(5, 14)], [(5, 90)]),
    2: _doc({}, {}, [6], [(6, 20)]),
    3: _doc({}, {}, [7], [(7, 30)]),
    4: _doc({"41": "Bacillus b"}, {"30": "Plant p"}, [8], [(8, 41)]),
    5: _doc({}, {}, [9], [(9, 60)]),
    6: _doc({}, {}, [70, 71], [(71, 70)], [(70, 90)]),
    7: _doc({"2026": "Escherichia coli"}, {}, [8], [(8, 396)]),
    8: _doc({"3445": "Mycobacterium tuberculosis"}, {}, [9], [(9, 3494)]),
}
BRENDA_SAYS = {
    1: ([Organism(id=14, organism="Bacillus x")], [], [(5, 14)]),
    2: ([Organism(id=20, organism="Clostridium y")], [], [(6, 20)]),
    3: ([], [Organism(id=30, organism="Plant q")], [(7, 30)]),
    4: ([Organism(id=41, organism="Bacillus b")], [], [(8, 41)]),
    5: ([Organism(id=60, organism="Bacillus w")], [], []),
    6: ([Organism(id=70, organism="Bacillus z")], [], [(71, 70)]),
    7: ([Organism(id=396, organism="Escherichia coli")], [], [(8, 396)]),
    8: (
        [Organism(id=3494, organism="Mycobacterium tuberculosis var. x")],
        [],
        [(9, 3494)],
    ),
}
TABLE: dict[int, dict[str, Any]] = {
    20: {"organism": "Unrelated", "synonyms": [], "lpsn_id": None},
    41: {"organism": "Bacillus b", "synonyms": [], "lpsn_id": None},
    3445: {
        "organism": "Mycobacterium tuberculosis",
        "synonyms": ["Mycobacterium tuberculosis var. x"],
        "lpsn_id": None,
    },
}
COLS = (BAC, OTH, "strains", "relations")


def _created(ref: int) -> str:
    return f"2025-01-0{ref}T00:00:00+00:00"


class Corpus(NamedTuple):
    """A doc db and one split, written to a temp dir, and BRENDA's calls."""

    docdb: Path
    split: Path
    calls: list[int]

    def relations(self, ref: int) -> dict[str, Any]:
        self.calls.append(ref)
        bacteria, other, species = BRENDA_SAYS.get(ref, ([], [], []))
        return {
            BAC: set(bacteria),
            OTH: set(other),
            "strains": set(),
            "enzymes": set(),
            "triples": {
                "HasSpecies": {
                    HasSpecies(subject=s, object=o) for s, o in species
                }
            },
        }

    def run(self, dry_run: bool = False) -> rs.RepairReport:
        return rs.repair_all(
            self.docdb,
            {"training": self.split},
            self.relations,
            lpsn=lambda name: (777, [f"Old {name}"]),
            dry_run=dry_run,
        )

    def docs(self) -> dict[int, dict[str, Any]]:
        with BrendaDocDB(str(self.docdb)) as docdb:
            return {d.doc_id: dict(d) for d in docdb.documents.all()}

    def table(self) -> dict[int, dict[str, Any]]:
        with BrendaDocDB(str(self.docdb)) as docdb:
            return {r.doc_id: dict(r) for r in docdb.bacteria.all()}

    def rows(self) -> dict[int, dict[str, Any]]:
        """Split rows keyed by the reference whose `created` they carry."""
        refs = {_created(ref): ref for ref in DOCS}
        with self.split.open(newline="") as handle:
            return {
                refs[row["created"]]: {
                    c: ast.literal_eval(row[c]) for c in COLS
                }
                for row in csv.DictReader(handle)
            }

    def snapshot(self) -> dict[str, bytes]:
        return {p.name: p.read_bytes() for p in self.docdb.parent.iterdir()}


@pytest.fixture
def corpus(tmp_path: Path) -> Corpus:
    out = Corpus(tmp_path / "documents.json", tmp_path / "split.csv", [])
    with BrendaDocDB(str(out.docdb), create=True) as docdb:
        for ref, doc in DOCS.items():
            docdb.documents.insert(
                TinyDBDoc({**doc, "created": _created(ref)}, ref)
            )
        for ident, record in TABLE.items():
            docdb.bacteria.insert(TinyDBDoc(record, ident))
    with out.split.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["", *COLS, "created"])
        # Reversed, so a row is matched to its document by `created`, not
        # by position.
        for i, ref in enumerate(reversed(DOCS)):
            writer.writerow(
                [i, *(repr(DOCS[ref][c]) for c in COLS), _created(ref)]
            )
    return out


def _species(unit: dict[str, Any]) -> list[tuple[str, str] | None]:
    """Each HasSpecies object's `(class, name)`, as preprocessing finds it."""
    out: list[tuple[str, str] | None] = []
    for pair in unit["relations"]["HasSpecies"]:
        key = str(pair["object"])
        found = [
            (kind, unit[kind][key]) for kind in (BAC, OTH) if key in unit[kind]
        ]
        out.append(found[0] if found else None)
    return out


def test_attested_species_is_restored_in_its_class(corpus: Corpus) -> None:
    corpus.run()
    expected = {
        1: (BAC, "Bacillus x"),
        2: (BAC, "Clostridium y"),
        3: (OTH, "Plant q"),
        6: (BAC, "Bacillus z"),
    }
    for units in (corpus.docs(), corpus.rows()):
        for ref, species in expected.items():
            assert _species(units[ref]) == [species]
    assert corpus.docs()[1][BAC] == {"14": "Bacillus x"}


def test_clashing_id_gets_one_fresh_id_in_db_and_split(
    corpus: Corpus,
) -> None:
    report = corpus.run()
    docs, rows = corpus.docs(), corpus.rows()
    for ref in (2, 3, 6):
        (new,) = (p["object"] for p in docs[ref]["relations"]["HasSpecies"])
        assert new > 90, "a fresh id clears every id the corpus holds"
        assert rows[ref]["relations"] == docs[ref]["relations"]
    assert {why for *_, why in report.reids} == {
        "id is a different bacteria record",
        "id also names another organism in other_organisms",
        "id would change what another relation position names",
    }


def test_other_relation_positions_resolve_as_before(corpus: Corpus) -> None:
    """Document 6's HasEnzyme subject 70 still names its strain."""
    corpus.run()
    for units in (corpus.docs(), corpus.rows()):
        assert "70" not in units[6][BAC]
        assert units[6]["relations"]["HasEnzyme"] == [
            {"subject": 70, "object": 90}
        ]


def test_resolving_document_is_untouched(corpus: Corpus) -> None:
    corpus.run()
    assert 4 not in corpus.calls
    assert {k: corpus.docs()[4][k] for k in COLS} == DOCS[4]
    assert corpus.rows()[4] == DOCS[4]


def test_unattested_species_is_counted_not_invented(corpus: Corpus) -> None:
    report = corpus.run()
    assert report.unattested == 1
    for units in (corpus.docs(), corpus.rows()):
        assert _species(units[5]) == [None]
        assert {k: units[5][k] for k in COLS} == DOCS[5]


def test_restored_bacteria_get_a_table_record(corpus: Corpus) -> None:
    corpus.run()
    table = corpus.table()
    docs = corpus.docs()
    for ref in (1, 2, 6):
        (ident,) = docs[ref][BAC]
        assert table[int(ident)]["organism"] == docs[ref][BAC][ident]
        assert table[int(ident)]["lpsn_id"] == 777
    assert table[20] == TABLE[20]


def test_dry_run_writes_nothing(corpus: Corpus) -> None:
    before = corpus.snapshot()
    report = corpus.run(dry_run=True)
    assert corpus.snapshot() == before
    assert report.counts["documents.json"].to_bacteria == 3


def test_organism_already_held_is_pointed_at_not_added(
    corpus: Corpus,
) -> None:
    """An object BRENDA names as an organism the unit holds follows that id.

    A second id would give the one organism two `bac` columns.
    """
    report = corpus.run()
    held = {7: ("2026", 396), 8: ("3445", 3494)}
    table = corpus.table()
    for units in (corpus.docs(), corpus.rows()):
        for ref, (ident, dangling) in held.items():
            assert units[ref][BAC] == DOCS[ref][BAC]
            assert units[ref]["relations"]["HasSpecies"] == [
                {"subject": DOCS[ref]["strains"][0], "object": int(ident)}
            ]
            assert dangling not in table
    assert report.counts["documents.json"].repointed == 2
    assert report.counts["split.csv"].repointed == 2


def test_shared_created_raises_before_anything_is_written(
    corpus: Corpus,
) -> None:
    """Two documents sharing `created` would match one split row twice."""
    with BrendaDocDB(str(corpus.docdb)) as docdb:
        docdb.documents.update({"created": _created(1)}, doc_ids=[2])
    before = corpus.snapshot()
    with pytest.raises(ValueError, match="share created"):
        corpus.run()
    assert corpus.snapshot() == before
