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
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import pytest
from brenda_references.docdb import BrendaDocDB
from d3types import HasSpecies, Organism
from scripts import repair_species as rs
from tinydb.table import Document as TinyDBDoc

BAC, OTH = "bacteria", "other_organisms"
Said = tuple[list[Organism], list[Organism], list[tuple[int, int]]]


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
BRENDA_SAYS: dict[int, Said] = {
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
    return f"2025-01-{ref:02d}T00:00:00+00:00"


class Corpus(NamedTuple):
    """A doc db and one split, written to a temp dir, and BRENDA's calls."""

    docdb: Path
    split: Path
    calls: list[int]
    source: Mapping[int, dict[str, Any]] = DOCS
    brenda: Mapping[int, Said] = BRENDA_SAYS

    def relations(self, ref: int) -> dict[str, Any]:
        self.calls.append(ref)
        bacteria, other, species = self.brenda.get(ref, ([], [], []))
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
        refs = {_created(ref): ref for ref in self.source}
        with self.split.open(newline="") as handle:
            return {
                refs[row["created"]]: {
                    c: ast.literal_eval(row[c]) for c in COLS
                }
                for row in csv.DictReader(handle)
            }

    def snapshot(self) -> dict[str, bytes]:
        return {p.name: p.read_bytes() for p in self.docdb.parent.iterdir()}


def _write_corpus(
    tmp_path: Path,
    docs: Mapping[int, dict[str, Any]],
    brenda: Mapping[int, Said],
    table: Mapping[int, dict[str, Any]],
) -> Corpus:
    out = Corpus(
        tmp_path / "documents.json", tmp_path / "split.csv", [], docs, brenda
    )
    with BrendaDocDB(str(out.docdb), create=True) as docdb:
        for ref, doc in docs.items():
            docdb.documents.insert(
                TinyDBDoc({**doc, "created": _created(ref)}, ref)
            )
        for ident, record in table.items():
            docdb.bacteria.insert(TinyDBDoc(record, ident))
    with out.split.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["", *COLS, "created"])
        # Reversed, so a row is matched to its document by `created`, not
        # by position.
        for i, ref in enumerate(reversed(list(docs))):
            writer.writerow(
                [i, *(repr(docs[ref][c]) for c in COLS), _created(ref)]
            )
    return out


@pytest.fixture
def corpus(tmp_path: Path) -> Corpus:
    return _write_corpus(tmp_path, DOCS, BRENDA_SAYS, TABLE)


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


def _bac(ident: int, name: str, ref: int) -> Said:
    return ([Organism(id=ident, organism=name)], [], [(ref, ident)])


# One organism under several BRENDA ids. 11 and 12 dangle one under 110 and
# 120; 13 dangles it twice, under 130 and 131. 16's object names what 15
# holds as 150, and 19's what table record 50 names. 22's strain 210 is a
# HasEnzyme subject, so 210, 21's object, cannot be restored in 22. 17 and
# 18 already hold one organism under two ids.
SHARED_DOCS = {
    11: _doc({}, {}, [11], [(11, 110)]),
    12: _doc({}, {}, [12], [(12, 120)]),
    13: _doc({}, {}, [13, 14], [(13, 130), (14, 131)]),
    15: _doc({"150": "Gloeobacter violaceus"}, {}, [15], [(15, 150)]),
    16: _doc({}, {}, [16], [(16, 160)]),
    17: _doc({"170": "Acaryochloris marina"}, {}, [17], []),
    18: _doc({"180": "Acaryochloris marina"}, {}, [18], []),
    19: _doc({}, {}, [19], [(19, 190)]),
    21: _doc({}, {}, [21], [(21, 210)]),
    22: _doc({}, {}, [210, 22], [(22, 211)], [(210, 90)]),
}
SHARED_BRENDA: dict[int, Said] = {
    11: _bac(110, "Synechococcus elongatus", 11),
    12: _bac(120, "Synechococcus elongatus", 12),
    13: (
        [
            Organism(id=130, organism="Prochlorococcus marinus"),
            Organism(id=131, organism="Prochlorococcus marinus"),
        ],
        [],
        [(13, 130), (14, 131)],
    ),
    16: _bac(160, "Gloeobacter violaceus", 16),
    19: _bac(190, "Bacillus t", 19),
    21: _bac(210, "Bacillus q", 21),
    22: _bac(211, "Bacillus q", 22),
}
SHARED_TABLE: dict[int, dict[str, Any]] = {
    50: {"organism": "Bacillus t", "synonyms": [], "lpsn_id": None},
    150: {"organism": "Gloeobacter violaceus", "synonyms": [], "lpsn_id": None},
}


@pytest.fixture
def shared(tmp_path: Path) -> Corpus:
    return _write_corpus(tmp_path, SHARED_DOCS, SHARED_BRENDA, SHARED_TABLE)


def _objects(unit: dict[str, Any]) -> list[int]:
    return [pair["object"] for pair in unit["relations"]["HasSpecies"]]


def test_one_organism_takes_one_id_across_documents(shared: Corpus) -> None:
    """Two BRENDA ids naming one organism would make it two entities."""
    shared.run()
    for units in (shared.docs(), shared.rows()):
        assert _objects(units[11]) == _objects(units[12]) == [110]
        assert units[12][BAC] == {"110": "Synechococcus elongatus"}
    assert shared.table().keys() == SHARED_TABLE.keys() | {110, 130, 211}


def test_two_objects_of_one_unit_naming_one_organism_share_its_id(
    shared: Corpus,
) -> None:
    shared.run()
    for units in (shared.docs(), shared.rows()):
        assert units[13][BAC] == {"130": "Prochlorococcus marinus"}
        assert _objects(units[13]) == [130, 130]


def test_organism_another_document_holds_takes_its_id(shared: Corpus) -> None:
    shared.run()
    for units in (shared.docs(), shared.rows()):
        assert units[16][BAC] == {"150": "Gloeobacter violaceus"}
        assert _objects(units[16]) == [150]
    assert 160 not in shared.table()


def test_bacteria_record_naming_the_organism_gives_its_id(
    shared: Corpus,
) -> None:
    shared.run()
    for units in (shared.docs(), shared.rows()):
        assert units[19][BAC] == {"50": "Bacillus t"}
    assert shared.table()[50] == SHARED_TABLE[50]
    assert 190 not in shared.table()


def test_shared_id_is_checked_against_each_units_own_objects(
    shared: Corpus,
) -> None:
    """210 is restorable in 21 but would turn 22's strain into an organism."""
    report = shared.run()
    for units in (shared.docs(), shared.rows()):
        assert _objects(units[21]) == _objects(units[22]) == [211]
        assert units[22][BAC] == {"211": "Bacillus q"}
        assert units[22]["relations"]["HasEnzyme"] == [
            {"subject": 210, "object": 90}
        ]
    assert (210, 211, "Bacillus q") in {r[:3] for r in report.reids}


def test_report_counts_names_under_more_than_one_id(shared: Corpus) -> None:
    """Only 17 and 18's organism, there before the run, has two ids."""
    lines = shared.run(dry_run=True).lines()
    assert "names under more than one id: 1 before, 1 after" in lines
    assert "bacteria table: 3 added" in lines


TWO_DOC_DOCS = {
    31: _doc({}, {}, [31], [(31, 130), (31, 120)]),
    32: _doc({}, {}, [32], [(32, 120)]),
}
TWO_DOC_BRENDA: dict[int, Said] = {
    31: _bac(130, "Bacillus z", 31),
    32: _bac(120, "Bacillus z", 32),
}
TWO_DOC_TABLE: dict[int, dict[str, Any]] = {}


@pytest.fixture
def two_doc(tmp_path: Path) -> Corpus:
    return _write_corpus(tmp_path, TWO_DOC_DOCS, TWO_DOC_BRENDA, TWO_DOC_TABLE)


def test_restores_per_unit_keys_only(two_doc: Corpus) -> None:
    """Each unit is checked only against the ids its own document attests.

    Reference 31 attests 130 but not 120 as "Bacillus z", and 32 attests
    120: 31's object 120 stays dangling, and both end on 130.
    """
    two_doc.run()
    for units in (two_doc.docs(), two_doc.rows()):
        assert _objects(units[31]) == [130, 120]
        assert units[31][BAC] == {"130": "Bacillus z"}
        assert _objects(units[32]) == [130]
        assert units[32][BAC] == {"130": "Bacillus z"}


# 41's two objects name organisms of different classes, and the corpus
# holds both names under one id, 500: 42 in other_organisms, 43 in bacteria.
TWO_CLASS_DOCS = {
    41: _doc({}, {}, [41, 42], [(41, 410), (42, 411)]),
    42: _doc({}, {"500": "Plant a"}, [43], []),
    43: _doc({"500": "Bact b"}, {}, [44], []),
}
TWO_CLASS_BRENDA: dict[int, Said] = {
    41: (
        [Organism(id=411, organism="Bact b")],
        [Organism(id=410, organism="Plant a")],
        [(41, 410), (42, 411)],
    ),
}


@pytest.fixture
def two_class(tmp_path: Path) -> Corpus:
    return _write_corpus(tmp_path, TWO_CLASS_DOCS, TWO_CLASS_BRENDA, {})


def test_ids_of_two_classes_are_checked_together(two_class: Corpus) -> None:
    """Each class's id passes alone; restored together, 500 is two entities."""
    two_class.run()
    for units in (two_class.docs(), two_class.rows()):
        assert _species(units[41]) == [(OTH, "Plant a"), (BAC, "Bact b")]
        assert len(set(_objects(units[41]))) == 2
