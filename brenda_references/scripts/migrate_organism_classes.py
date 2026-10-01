"""Re-sort the organisms of the doc db and the split CSVs by NCBI lineage.

`db.is_bacteria` decides by descent from NCBITaxon:2, but `documents.json`
and the splits still hold the classes an older name list decided. This moves
each organism between `bacteria` and `other_organisms` to where
`db.classify_organism` now puts it, patching the split CSVs row by row
(regenerating them can redraw the partition).

An id is kept unless keeping it changes what a relation points at.
`preprocess_relations` resolves a HasSpecies object in `bacteria`, then
`other_organisms`, and a HasEnzyme subject in `bacteria`, then `strains`,
then `other_organisms` (`_RELATION_SLOTS`); BRENDA organism ids share a
numeric range with the ids minted for the `bacteria` table. A move whose id
would name two organisms in its new class, equal a strain id of a document
that lists it, or name a different `bacteria` record gets a fresh id, the
same one everywhere, counted up from above every organism, strain, table and
relation id. The relation positions that named it are rewritten, and every
changed document and row is checked to resolve each relation position to
the same entity as before, dangling ones to nothing.

Every read, LPSN lookup and check runs before anything is written. Each file
is then written to a `.migrating` sibling, an empty `.commit` marker is
created beside the doc db, and the siblings replace the originals. A run
that finds the marker finishes that commit instead of planning again.
"""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import itertools
import os
import sys
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import lpsn_interface
from tinydb.storages import JSONStorage

if TYPE_CHECKING:
    from brenda_references.db import Classification

BAC = "bacteria"
OTH = "other_organisms"
_CSV_COLUMNS = (BAC, OTH, "strains", "relations")
_RELATION_SLOTS = (
    ("HasSpecies", "object", (BAC, OTH)),
    ("HasEnzyme", "subject", (BAC, "strains", OTH)),
)

Unit = Mapping[str, Any]
Key = tuple[int, str]
Lookup = Callable[[str], tuple[int | None, list[str]]]


def _organisms(unit: Unit, kind: str) -> dict[int, str]:
    return {int(k): v for k, v in (unit.get(kind) or {}).items()}


def _strains(unit: Unit) -> set[int]:
    return {int(s) for s in unit.get("strains") or ()}


def _relation_ids(unit: Unit) -> Iterator[int]:
    for pairs in (unit.get("relations") or {}).values():
        for pair in pairs:
            yield from (pair["subject"], pair["object"])


@dataclass
class Counts:
    """How many organisms one artifact moved into each class."""

    to_bacteria: int = 0
    to_other: int = 0


@dataclass
class Survey:
    """What planning needs from every document and row, read once."""

    entries: set[tuple[int, str, str]] = field(default_factory=set)
    strain_clash: set[Key] = field(default_factory=set)
    max_id: int = 0

    def add(self, unit: Unit) -> None:
        """Record `unit`'s organisms, and those whose id is also a strain.

        :param unit: a document or split row with its fields parsed.
        """
        strains = _strains(unit)
        self.max_id = max(self.max_id, *strains, *_relation_ids(unit), 0)
        for kind in (BAC, OTH):
            for key, name in _organisms(unit, kind).items():
                self.entries.add((key, name, kind))
                self.max_id = max(self.max_id, key)
                if key in strains:
                    self.strain_clash.add((key, name))


@dataclass
class Report:
    """What a run did, or with `--dry-run` would do."""

    counts: dict[str, Counts] = field(default_factory=dict)
    reids: list[tuple[int, int, str, str]] = field(default_factory=list)
    no_lpsn: list[str] = field(default_factory=list)
    records_added: int = 0
    records_removed: int = 0
    decided: dict[str, Classification] = field(default_factory=dict)
    moved: set[str] = field(default_factory=set)
    resumed: bool = False

    def lines(self) -> list[str]:
        """The report, one line per fact.

        :return: the lines, in no particular order beyond grouping.
        """
        if self.resumed:
            return [
                "an interrupted run's commit was pending: this run finishes "
                "it (with --dry-run, would) and plans nothing"
            ]
        out = [
            f"{name}: {c.to_bacteria} into bacteria, {c.to_other} into "
            "other_organisms"
            for name, c in self.counts.items()
        ]
        out += [
            f"re-id {old} -> {new} {name!r}: {why}"
            for old, new, name, why in self.reids
        ]
        out.append(
            f"bacteria table: {self.records_added} added, "
            f"{self.records_removed} removed"
        )
        out += [f"no LPSN record: {name!r}" for name in self.no_lpsn]
        by_list = sorted(n for n, c in self.decided.items() if not c.by_lineage)
        moved_by_list = [n for n in by_list if n in self.moved]
        out.append(
            f"classified: {len(self.decided) - len(by_list)} names by "
            f"lineage, {len(by_list)} by the name list "
            f"({len(moved_by_list)} of them moved)"
        )
        out += [f"moved by the name list: {n!r}" for n in moved_by_list]
        return out


def _class_of(classify: Callable[[str], bool], name: str) -> str:
    return BAC if classify(name) else OTH


def plan_ids(
    survey: Survey,
    classify: Callable[[str], bool],
    table: Mapping[int, set[str]],
) -> dict[Key, tuple[int, str]]:
    """Choose the fresh id every moved organism that cannot keep its own gets.

    :param survey: every organism in the corpus, from `Survey.add`.
    :param classify: whether a name is a bacterium.
    :param table: `bacteria` table id -> the names its record answers to.
    :return: `(old id, name)` -> `(new id, why)`; new ids count up from above
        every id the survey and the table hold, in `(old id, name)` order.
    """
    final: dict[str, dict[int, set[str]]] = {BAC: {}, OTH: {}}
    moved: set[tuple[int, str, str]] = set()
    for key, name, kind in survey.entries:
        new = _class_of(classify, name)
        final[new].setdefault(key, set()).add(name)
        if new != kind:
            moved.add((key, name, new))

    fresh = itertools.count(max(survey.max_id, *table, 0) + 1)
    plan: dict[Key, tuple[int, str]] = {}
    for key, name, new in sorted(moved):
        if final[new][key] != {name}:
            why = f"id also names another organism in {new}"
        elif (key, name) in survey.strain_clash:
            why = "id is also a strain id in a document listing it"
        elif new == BAC and key in table and name not in table[key]:
            why = "id is a different bacteria record"
        else:
            continue
        plan[key, name] = (next(fresh), why)
    return plan


def _entity(
    maps: Mapping[str, Mapping[int, str]],
    order: tuple[str, ...],
    ident: int,
) -> tuple[str, str | int] | None:
    """What `preprocess_relations` resolves `ident` to, in its search order."""
    for kind in order:
        if ident in maps[kind]:
            if kind == "strains":
                return ("strain", ident)
            return ("organism", maps[kind][ident])
    return None


@dataclass
class Migrated:
    """One unit's rewritten fields and how many organisms moved."""

    bacteria: dict[str, str]
    other_organisms: dict[str, str]
    relations: Mapping[str, Any]
    to_bacteria: int
    to_other: int


def migrate_unit(
    unit: Unit,
    classify: Callable[[str], bool],
    reid: Mapping[Key, int],
) -> Migrated | None:
    """Move `unit`'s organisms to their class, rewriting relations to match.

    :param unit: a document or split row with `bacteria`, `other_organisms`,
        `strains` and `relations` parsed.
    :param classify: whether a name is a bacterium.
    :param reid: `(old id, name)` -> the id a moved organism takes.
    :return: the new fields, or `None` when nothing moves.
    :raises ValueError: if two organisms land on one id, or a relation
        position would resolve to a different entity than before, a dangling
        one included.
    """
    old = {
        BAC: _organisms(unit, BAC),
        OTH: _organisms(unit, OTH),
        "strains": dict.fromkeys(_strains(unit), ""),
    }
    new: dict[str, dict[int, str]] = {BAC: {}, OTH: {}}
    moved = Counts()
    for kind in (BAC, OTH):
        for ident, name in old[kind].items():
            target = _class_of(classify, name)
            if target != kind:
                ident = reid.get((ident, name), ident)
                if target == BAC:
                    moved.to_bacteria += 1
                else:
                    moved.to_other += 1
            if new[target].setdefault(ident, name) != name:
                msg = f"id {ident} would name two organisms in {target}"
                raise ValueError(msg)
    if not (moved.to_bacteria or moved.to_other):
        return None

    after = {**new, "strains": old["strains"]}
    relations = copy.deepcopy(unit.get("relations") or {})
    for predicate, slot, order in _RELATION_SLOTS:
        for pair in relations.get(predicate, []):
            before = _entity(old, order, pair[slot])
            if before is not None and before[0] == "organism":
                pair[slot] = reid.get((pair[slot], str(before[1])), pair[slot])
            if _entity(after, order, pair[slot]) != before:
                msg = f"{predicate} {slot} {pair[slot]} changes entity"
                raise ValueError(msg)
    return Migrated(
        bacteria={str(k): v for k, v in new[BAC].items()},
        other_organisms={str(k): v for k, v in new[OTH].items()},
        relations=relations,
        to_bacteria=moved.to_bacteria,
        to_other=moved.to_other,
    )


def _split_units(path: Path) -> Iterator[Unit]:
    csv.field_size_limit(sys.maxsize)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            yield {col: ast.literal_eval(row[col]) for col in _CSV_COLUMNS}


def migrate_split(
    path: Path,
    classify: Callable[[str], bool],
    reid: Mapping[Key, int],
    dst: Path | None,
) -> Counts:
    """Rewrite one split CSV's organism cells, leaving the rest untouched.

    :param path: the split CSV.
    :param classify: whether a name is a bacterium.
    :param reid: `(old id, name)` -> the id a moved organism takes.
    :param dst: where to write the result, or `None` to only check and count.
    :return: how many organisms moved into each class.
    :raises ValueError: from `migrate_unit`, for a row that cannot move.
    """
    csv.field_size_limit(sys.maxsize)
    total = Counts()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        out = dst.open("w", newline="") if dst else None
        try:
            writer = None
            if out is not None:
                assert reader.fieldnames is not None
                writer = csv.DictWriter(
                    out, reader.fieldnames, lineterminator="\n"
                )
                writer.writeheader()
            for row in reader:
                unit = {col: ast.literal_eval(row[col]) for col in _CSV_COLUMNS}
                result = migrate_unit(unit, classify, reid)
                if result is not None:
                    total.to_bacteria += result.to_bacteria
                    total.to_other += result.to_other
                    row[BAC] = repr(result.bacteria)
                    row[OTH] = repr(result.other_organisms)
                    if result.relations != unit["relations"]:
                        row["relations"] = repr(result.relations)
                if writer is not None:
                    writer.writerow(row)
        finally:
            if out is not None:
                out.close()
    return total


def lpsn_record(name: str) -> tuple[int | None, list[str]]:
    """The LPSN id and sorted synonyms of `name`, as `store_bacteria` stores.

    :param name: a bacterial name.
    :return: `(lpsn_id, synonyms)`, or `(None, [])` when LPSN has no record.
    """
    lpsn_id = lpsn_interface.lpsn_id(name)
    if lpsn_id is None:
        return None, []
    return lpsn_id, sorted(lpsn_interface.lpsn_synonyms(lpsn_id))


def plan_docdb(
    data: Mapping[str, Mapping[str, Any]],
    classify: Callable[[str], bool],
    reid: Mapping[Key, int],
    lpsn: Lookup,
    report: Report,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    """Compute the doc db's document updates and `bacteria` table changes.

    :param data: the doc db's tables as TinyDB stores them, read only.
    :param classify: whether a name is a bacterium.
    :param reid: `(old id, name)` -> the id a moved organism takes.
    :param lpsn: name -> `(lpsn_id, synonyms)` for a record being added.
    :param report: receives counts, added and removed records.
    :return: document id -> new fields; table id -> record to add; table
        ids to remove.
    :raises ValueError: from `migrate_unit`, for a document that cannot move.
    """
    counts = report.counts.setdefault("documents.json", Counts())
    before: set[int] = set()
    after: dict[int, str] = {}
    updates: dict[str, dict[str, Any]] = {}
    for doc_id, doc in data.get("documents", {}).items():
        before |= _organisms(doc, BAC).keys()
        result = migrate_unit(doc, classify, reid)
        if result is None:
            after.update(_organisms(doc, BAC))
            continue
        after.update({int(k): v for k, v in result.bacteria.items()})
        counts.to_bacteria += result.to_bacteria
        counts.to_other += result.to_other
        updates[doc_id] = {
            BAC: result.bacteria,
            OTH: result.other_organisms,
            "relations": result.relations,
        }
    table_ids = {int(k) for k in data.get(BAC, {})}
    gone = sorted((before - after.keys()) & table_ids)
    added: dict[str, dict[str, Any]] = {}
    for ident in sorted(after.keys() - table_ids):
        lpsn_id, synonyms = lpsn(after[ident])
        if lpsn_id is None:
            report.no_lpsn.append(after[ident])
        added[str(ident)] = {
            "organism": after[ident],
            "synonyms": synonyms,
            "lpsn_id": lpsn_id,
        }
    report.records_added = len(added)
    report.records_removed = len(gone)
    return updates, added, [str(g) for g in gone]


def _tmp(path: Path) -> Path:
    return path.with_name(path.name + ".migrating")


def _marker(docdb_path: Path) -> Path:
    return docdb_path.with_name(docdb_path.name + ".commit")


def _commit(targets: list[Path], marker: Path) -> None:
    """Replace each target whose temp sibling is still there, then unmark."""
    for path in targets:
        if _tmp(path).exists():
            os.replace(_tmp(path), path)
    marker.unlink()


def migrate_all(
    docdb_path: Path,
    splits: Mapping[str, Path],
    classify: Callable[[str], Classification],
    lpsn: Lookup = lpsn_record,
    dry_run: bool = False,
) -> Report:
    """Migrate the doc db and every split, planning ids across all of them.

    :param docdb_path: the TinyDB JSON doc db.
    :param splits: split name -> CSV path.
    :param classify: a name's class and how it was decided; called once per
        name.
    :param lpsn: name -> `(lpsn_id, synonyms)` for a record being added.
    :param dry_run: write nothing.
    :return: what was, or would be, done.
    :raises ValueError: from `migrate_unit`, before anything is written.
    """
    targets = [docdb_path, *splits.values()]
    marker = _marker(docdb_path)
    if marker.exists():
        if not dry_run:
            _commit(targets, marker)
        return Report(resumed=True)

    report = Report()

    def is_bac(name: str) -> bool:
        if name not in report.decided:
            report.decided[name] = classify(name)
        return report.decided[name].bacterium

    storage = JSONStorage(str(docdb_path), access_mode="r")
    try:
        data = storage.read() or {}
    finally:
        storage.close()

    survey = Survey()
    survey.max_id = max(map(int, data.get("strains", {})), default=0)
    table = {
        int(k): {rec["organism"], *rec["synonyms"]}
        for k, rec in data.get(BAC, {}).items()
    }
    for unit in itertools.chain(
        data.get("documents", {}).values(),
        *(_split_units(path) for path in splits.values()),
    ):
        survey.add(unit)

    plan = plan_ids(survey, is_bac, table)
    reid = {key: new for key, (new, _) in plan.items()}
    report.reids = [(k[0], new, k[1], why) for k, (new, why) in plan.items()]
    report.moved = {
        name
        for _, name, kind in survey.entries
        if _class_of(is_bac, name) != kind
    }
    updates, added, gone = plan_docdb(data, is_bac, reid, lpsn, report)
    for path in splits.values():
        report.counts[path.name] = migrate_split(path, is_bac, reid, None)
    if dry_run:
        return report

    for doc_id, fields in updates.items():
        data["documents"][doc_id].update(fields)
    data.setdefault(BAC, {}).update(added)
    for ident in gone:
        del data[BAC][ident]
    storage = JSONStorage(str(_tmp(docdb_path)))
    try:
        storage.write(data)
    finally:
        storage.close()
    for path in splits.values():
        migrate_split(path, is_bac, reid, _tmp(path))
    marker.touch()
    _commit(targets, marker)
    return report


def main(argv: list[str] | None = None) -> int:
    """Run the migration on the configured data.

    :param argv: arguments, `None` for `sys.argv`.
    :return: the exit status; 2 when an input file is missing.
    """
    from brenda_references.config import config
    from brenda_references.data_paths import documents_path, split_path
    from brenda_references.db import classify_organism

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    splits = {name: split_path(name) for name in config["datasets"]["splits"]}
    missing = [
        p for p in (documents_path(), *splits.values()) if not p.exists()
    ]
    if missing:
        print("missing input:", *missing, sep="\n  ", file=sys.stderr)
        return 2
    report = migrate_all(
        documents_path(), splits, classify_organism, dry_run=args.dry_run
    )
    print(*report.lines(), sep="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
