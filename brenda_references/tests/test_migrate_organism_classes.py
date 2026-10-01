"""Pin `migrate_organism_classes` moving organisms without moving relations.

The invariant: after the migration, every pair `preprocess_labels` builds
for a document or split row names the same entities (by organism name) as
before, whatever ids the moves had to change, and a run that stops early
leaves the doc db and the splits in step. `classify` and the LPSN lookup are
injected, so nothing here reads the NCBI taxonomy or real data.
"""

from __future__ import annotations

import ast
import csv
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd
import pytest
from brenda_references.docdb import BrendaDocDB
from scripts import migrate_organism_classes as mig
from tinydb.table import Document as TinyDBDoc

from brenda_references.brenda_references import preprocess_labels
from brenda_references.db import Classification

BACTERIA_NAMES = {"Bacillus b", "Clostridium genus"}
BY_NAME_LIST = {"Archaeon d"}


def is_bac(name: str) -> bool:
    return name in BACTERIA_NAMES


def classify(name: str) -> Classification:
    return Classification(is_bac(name), name not in BY_NAME_LIST)


def _unit(
    bacteria: dict[str, str],
    other: dict[str, str],
    strains: list[int],
    has_enzyme: Sequence[tuple[int, int]] = (),
    has_species: Sequence[tuple[int, int]] = (),
) -> dict[str, Any]:
    return {
        "bacteria": bacteria,
        "other_organisms": other,
        "strains": strains,
        "enzymes": [90],
        "relations": {
            "HasEnzyme": [{"subject": s, "object": o} for s, o in has_enzyme],
            "HasSpecies": [{"subject": s, "object": o} for s, o in has_species],
        },
    }


# 10: archaeon moving to other keeps its id. Genus 20 moving into bacteria
# meets a different bacteria record at 20. 30 meets a non-moving organism at
# 30 in other. 50 is also a strain id in its own document. 51 is a dangling
# HasEnzyme subject, the first id above every organism and strain id.
UNITS = [
    _unit(
        {"10": "Archaeon a"},
        {"20": "Clostridium genus"},
        [5],
        has_enzyme=[(10, 90), (20, 90), (5, 90), (51, 90)],
        has_species=[(5, 20)],
    ),
    _unit(
        {"30": "Archaeon c"},
        {"31": "Plant p", "20": "Clostridium genus"},
        [6],
        has_enzyme=[(30, 90), (31, 90)],
    ),
    _unit({"41": "Bacillus b"}, {"30": "Plant q"}, [7], has_enzyme=[(30, 90)]),
    _unit(
        {"50": "Archaeon d"},
        {},
        [50],
        has_enzyme=[(50, 90)],
        has_species=[(50, 50)],
    ),
]
TABLE = {
    10: {"organism": "Archaeon a", "synonyms": []},
    20: {"organism": "Unrelated", "synonyms": []},
}


def _labels(unit: dict[str, Any]) -> dict[frozenset[str], tuple[float, ...]]:
    """`preprocess_labels`' pairs, with each entity replaced by its name."""
    cols = ("bacteria", "other_organisms", "strains", "enzymes", "relations")
    frame = pd.DataFrame([{col: repr(unit[col]) for col in cols}])
    row = preprocess_labels(frame).iloc[0]
    names = {
        **{f"bac{k}": v for k, v in unit["bacteria"].items()},
        **{f"oth{k}": v for k, v in unit["other_organisms"].items()},
        **{f"str{s}": f"strain {s}" for s in unit["strains"]},
        **{f"enz{e}": f"enzyme {e}" for e in unit["enzymes"]},
    }
    return {
        frozenset(names[e] for e in key): tuple(label)
        for key, label in row["relations"][0].items()
    }


def _write_csv(path: Path, units: list[dict[str, Any]]) -> None:
    cols = ["bacteria", "other_organisms", "strains", "enzymes", "relations"]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["", *cols, "pubmed_id"])
        for i, unit in enumerate(units):
            writer.writerow([i, *(repr(unit[c]) for c in cols), 100 + i])


def _read_csv(path: Path) -> list[dict[str, Any]]:
    cols = ("bacteria", "other_organisms", "strains", "enzymes", "relations")
    frame = pd.read_csv(path, index_col=0)
    return [
        {c: ast.literal_eval(row[c]) for c in cols}
        for _, row in frame.iterrows()
    ]


def _write_docdb(
    path: Path,
    units: list[dict[str, Any]],
    table: dict[int, dict[str, Any]],
) -> None:
    with BrendaDocDB(str(path), create=True) as docdb:
        for i, unit in enumerate(units, start=1):
            docdb.documents.insert(TinyDBDoc(dict(unit), i))
        for ident, record in table.items():
            docdb.bacteria.insert(TinyDBDoc(record, ident))


def _fake_lpsn(name: str) -> tuple[int | None, list[str]]:
    return (777, ["Old " + name]) if name == "Clostridium genus" else (None, [])


class Corpus(NamedTuple):
    """A doc db and one split, written to a temp dir."""

    docdb: Path
    split: Path

    def snapshot(self) -> dict[str, bytes]:
        return {p.name: p.read_bytes() for p in self.docdb.parent.iterdir()}

    def run(self, **kwargs: Any) -> mig.Report:
        kwargs.setdefault("lpsn", _fake_lpsn)
        return mig.migrate_all(
            self.docdb, {"training": self.split}, classify, **kwargs
        )


def _corpus(root: Path, units: list[dict[str, Any]] = UNITS) -> Corpus:
    root.mkdir(exist_ok=True)
    corpus = Corpus(root / "documents.json", root / "split.csv")
    _write_docdb(corpus.docdb, units, TABLE)
    _write_csv(corpus.split, units)
    return corpus


class Migrated(NamedTuple):
    """The migrated documents, split rows, `bacteria` table and report."""

    docs: list[dict[str, Any]]
    rows: list[dict[str, Any]]
    table: dict[int, dict[str, Any]]
    report: mig.Report


@pytest.fixture
def migrated(tmp_path: Path) -> Migrated:
    corpus = _corpus(tmp_path)
    report = corpus.run()
    with BrendaDocDB(str(corpus.docdb)) as docdb:
        docs = [dict(d) for d in docdb.documents.all()]
        table = {r.doc_id: dict(r) for r in docdb.bacteria.all()}
    return Migrated(docs, _read_csv(corpus.split), table, report)


def test_every_pair_resolves_to_the_same_entity(migrated: Migrated) -> None:
    for migrated_units in (migrated.docs, migrated.rows):
        for before, after in zip(UNITS, migrated_units, strict=True):
            assert _labels(after) == _labels(before)


def test_organisms_land_in_their_class(migrated: Migrated) -> None:
    for doc in migrated.docs:
        assert all(is_bac(n) for n in doc["bacteria"].values())
        assert not any(is_bac(n) for n in doc["other_organisms"].values())


def _ids(units: list[dict[str, Any]], name: str) -> set[str]:
    return {
        i
        for u in units
        for kind in ("bacteria", "other_organisms")
        for i, n in u[kind].items()
        if n == name
    }


def test_unclashing_move_keeps_its_id(migrated: Migrated) -> None:
    for units in (migrated.docs, migrated.rows):
        assert units[0]["other_organisms"] == {"10": "Archaeon a"}


def test_clashing_moves_get_one_fresh_id_in_db_and_split(
    migrated: Migrated,
) -> None:
    for name in ("Clostridium genus", "Archaeon c", "Archaeon d"):
        fresh = _ids(migrated.docs, name)
        assert len(fresh) == 1
        assert int(fresh.pop()) > 51
        assert _ids(migrated.docs, name) == _ids(migrated.rows, name)
    assert {why for *_, why in migrated.report.reids} == {
        "id is a different bacteria record",
        "id also names another organism in other_organisms",
        "id is also a strain id in a document listing it",
    }


def test_dangling_relation_id_stays_dangling(migrated: Migrated) -> None:
    """A fresh id equal to a dangling relation id would make it a live pair."""
    for units in (migrated.docs, migrated.rows):
        entities = {
            *map(int, units[0]["bacteria"]),
            *map(int, units[0]["other_organisms"]),
            *units[0]["strains"],
        }
        assert 51 not in entities


def test_fresh_ids_clear_every_relation_id() -> None:
    survey = mig.Survey()
    survey.add(
        _unit({}, {"20": "Clostridium genus"}, [], has_enzyme=[(99, 90)])
    )
    plan = mig.plan_ids(survey, is_bac, {20: {"Unrelated"}})
    assert plan and all(new > 99 for new, _ in plan.values())


def test_migrate_unit_refuses_to_revive_a_dangling_id() -> None:
    unit = _unit({}, {"20": "Clostridium genus"}, [], has_enzyme=[(51, 90)])
    with pytest.raises(ValueError, match="changes entity"):
        mig.migrate_unit(unit, is_bac, {(20, "Clostridium genus"): 51})


def test_table_gains_moved_in_records_and_loses_orphans(
    migrated: Migrated,
) -> None:
    (genus_id,) = _ids(migrated.docs, "Clostridium genus")
    assert migrated.table[int(genus_id)] == {
        "organism": "Clostridium genus",
        "synonyms": ["Old Clostridium genus"],
        "lpsn_id": 777,
    }
    assert 10 not in migrated.table
    assert migrated.table[20]["organism"] == "Unrelated"
    assert "Archaeon c" not in migrated.report.no_lpsn


def test_report_separates_lineage_from_name_list(migrated: Migrated) -> None:
    lines = migrated.report.lines()
    assert "moved by the name list: 'Archaeon d'" in lines
    assert any("by the name list (1 of them moved)" in ln for ln in lines)


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    before = corpus.snapshot()
    report = corpus.run(dry_run=True)
    assert corpus.snapshot() == before
    assert report.counts["split.csv"].to_other > 0


def test_lookup_failure_leaves_every_file_unchanged(tmp_path: Path) -> None:
    """Every LPSN lookup runs before the first write."""
    corpus = _corpus(tmp_path)
    before = corpus.snapshot()

    def down(name: str) -> tuple[int | None, list[str]]:
        raise RuntimeError("LPSN unavailable")

    with pytest.raises(RuntimeError):
        corpus.run(lpsn=down)
    assert corpus.snapshot() == before


def test_docdb_is_replaced_never_rewritten_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A kill before the first replace leaves every original intact."""
    corpus = _corpus(tmp_path)
    before = corpus.snapshot()
    replaced: list[Path] = []

    def stop(src: Path, dst: Path) -> None:
        replaced.append(Path(dst))
        raise SystemExit("killed")

    monkeypatch.setattr(mig.os, "replace", stop)
    with pytest.raises(SystemExit):
        corpus.run()
    assert replaced == [corpus.docdb]
    assert {k: v for k, v in corpus.snapshot().items() if k in before} == (
        before
    )


def test_interrupted_commit_resumes_to_the_same_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A kill between two replaces is finished, not re-planned, on re-run."""
    clean = _corpus(tmp_path / "clean")
    clean.run()
    expected = clean.snapshot()

    corpus = _corpus(tmp_path / "killed")
    real_replace = mig.os.replace
    calls = 0

    def stop_second(src: Path, dst: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise SystemExit("killed")
        real_replace(src, dst)

    monkeypatch.setattr(mig.os, "replace", stop_second)
    with pytest.raises(SystemExit):
        corpus.run()
    monkeypatch.setattr(mig.os, "replace", real_replace)

    assert corpus.run().resumed
    assert corpus.snapshot() == expected
    assert not corpus.run().resumed
    assert corpus.snapshot() == expected


def test_untouched_rows_and_columns_keep_their_bytes(tmp_path: Path) -> None:
    """Only a moved row's organism cells change in a pandas-written split."""
    units = [UNITS[2], _unit({"10": "Archaeon a"}, {}, [5], [(10, 90)])]
    corpus = _corpus(tmp_path, units)
    frame = pd.DataFrame(
        [
            {
                **{c: repr(u[c]) for c in u},
                "authors": 'Smith, J. "Jr"',
                "fulltext": "line one\r\nline, two",
                "score": math.nan if i else 1.5,
            }
            for i, u in enumerate(units)
        ]
    )
    frame.to_csv(corpus.split)
    corpus.run()

    expected = frame.copy()
    expected.loc[1, "bacteria"] = repr({})
    expected.loc[1, "other_organisms"] = repr({"10": "Archaeon a"})
    assert corpus.split.read_bytes() == expected.to_csv().encode()
