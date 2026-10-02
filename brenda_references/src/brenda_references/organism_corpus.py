"""Read, patch and commit the doc db and split CSVs as one organism corpus.

The shared ground of the scripts that rewrite organisms in place: the
parsed view of a document or split row, the survey ids are planned against,
row-by-row CSV patching, and the commit that writes each file to a
`.migrating` sibling, creates an empty `.commit` marker beside the doc db,
and then replaces the originals, so an interrupted run can be finished.
"""

from __future__ import annotations

import ast
import csv
import itertools
import os
import sys
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lpsn_interface
from tinydb.storages import JSONStorage

BAC = "bacteria"
OTH = "other_organisms"
_CSV_COLUMNS = (BAC, OTH, "strains", "relations")
RELATION_SLOTS = (
    ("HasSpecies", "object", (BAC, OTH)),
    ("HasEnzyme", "subject", (BAC, "strains", OTH)),
)

Unit = Mapping[str, Any]
Key = tuple[int, str]
Lookup = Callable[[str], tuple[int | None, list[str]]]


def organisms_of(unit: Unit, kind: str) -> dict[int, str]:
    """One organism column of `unit`, keyed by int id.

    :param unit: a document or split row with its fields parsed.
    :param kind: `BAC` or `OTH`.
    :return: id -> organism name.
    """
    return {int(k): v for k, v in (unit.get(kind) or {}).items()}


def strains_of(unit: Unit) -> set[int]:
    """The strain ids of `unit`.

    :param unit: a document or split row with its fields parsed.
    :return: those ids, as ints.
    """
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
        strains = strains_of(unit)
        self.max_id = max(self.max_id, *strains, *_relation_ids(unit), 0)
        for kind in (BAC, OTH):
            for key, name in organisms_of(unit, kind).items():
                self.entries.add((key, name, kind))
                self.max_id = max(self.max_id, key)
                if key in strains:
                    self.strain_clash.add((key, name))


def fresh_ids(survey: Survey, table: Mapping[int, object]) -> Iterator[int]:
    """Ids above every id `survey` and the `bacteria` table hold, ascending.

    :param survey: every organism in the corpus, from `Survey.add`.
    :param table: the `bacteria` table, keyed by id.
    :return: an endless iterator over the free ids.
    """
    return itertools.count(max(survey.max_id, *table, 0) + 1)


def entity_of(
    maps: Mapping[str, Mapping[int, str]],
    order: tuple[str, ...],
    ident: int,
) -> tuple[str, str | int] | None:
    """What `preprocess_relations` resolves `ident` to, in its search order.

    :param maps: column (`BAC`, `OTH` or `"strains"`) -> id -> name.
    :param order: the columns to search, first match wins.
    :param ident: a relation position's id.
    :return: `("organism", name)`, `("strain", id)`, or `None` if dangling.
    """
    for kind in order:
        if ident in maps[kind]:
            if kind == "strains":
                return ("strain", ident)
            return ("organism", maps[kind][ident])
    return None


def parse_row(row: Mapping[str, str]) -> Unit:
    """A split row's organism and relation cells, parsed.

    :param row: one row as `csv.DictReader` reads it.
    :return: the parsed `bacteria`, `other_organisms`, `strains` and
        `relations`.
    """
    return {col: ast.literal_eval(row[col]) for col in _CSV_COLUMNS}


def _split_units(path: Path) -> Iterator[Unit]:
    csv.field_size_limit(sys.maxsize)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            yield parse_row(row)


def patch_split(
    path: Path,
    rewrite: Callable[[Mapping[str, str]], Mapping[str, object] | None],
    dst: Path | None,
) -> None:
    """Run `rewrite` over every row of a split CSV, writing what it changes.

    :param path: the split CSV.
    :param rewrite: a row -> the cells to replace, each written as its
        `repr`, or `None` to keep the row as it is.
    :param dst: where to write the result, or `None` to only run `rewrite`.
    """
    csv.field_size_limit(sys.maxsize)
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
                cells = rewrite(row)
                if cells is not None:
                    row.update({col: repr(v) for col, v in cells.items()})
                if writer is not None:
                    writer.writerow(row)
        finally:
            if out is not None:
                out.close()


def lpsn_record(name: str) -> tuple[int | None, list[str]]:
    """The LPSN id and sorted synonyms of `name`, as `store_bacteria` stores.

    :param name: a bacterial name.
    :return: `(lpsn_id, synonyms)`, or `(None, [])` when LPSN has no record.
    """
    lpsn_id = lpsn_interface.lpsn_id(name)
    if lpsn_id is None:
        return None, []
    return lpsn_id, sorted(lpsn_interface.lpsn_synonyms(lpsn_id))


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


def resume_commit(
    docdb_path: Path, splits: Mapping[str, Path], dry_run: bool
) -> bool:
    """Finish an interrupted run's commit, if its marker is there.

    :param docdb_path: the TinyDB JSON doc db.
    :param splits: split name -> CSV path.
    :param dry_run: only report whether a commit is pending.
    :return: whether one was pending.
    """
    marker = _marker(docdb_path)
    if not marker.exists():
        return False
    if not dry_run:
        _commit([docdb_path, *splits.values()], marker)
    return True


def read_corpus(
    docdb_path: Path, splits: Mapping[str, Path]
) -> tuple[dict[str, Any], Survey, dict[int, set[str]]]:
    """Read the doc db, and survey it together with every split.

    :param docdb_path: the TinyDB JSON doc db.
    :param splits: split name -> CSV path.
    :return: the doc db's tables as TinyDB stores them; the survey; the
        `bacteria` table as id -> the names its record answers to.
    """
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
    return data, survey, table


def write_all(
    docdb_path: Path,
    data: Mapping[str, Any],
    splits: Mapping[str, Path],
    write_split: Callable[[Path, Path], object],
) -> None:
    """Write every file to its sibling, mark the commit, then replace.

    :param docdb_path: the TinyDB JSON doc db.
    :param data: the doc db's new tables.
    :param splits: split name -> CSV path.
    :param write_split: `(split, destination)` -> writes the new split.
    """
    storage = JSONStorage(str(_tmp(docdb_path)))
    try:
        storage.write(dict(data))
    finally:
        storage.close()
    for path in splits.values():
        write_split(path, _tmp(path))
    marker = _marker(docdb_path)
    marker.touch()
    _commit([docdb_path, *splits.values()], marker)


def configured_inputs() -> tuple[Path, dict[str, Path]] | None:
    """The configured doc db and splits, or `None` naming what is missing.

    :return: `(doc db path, split name -> CSV path)`, or `None` after
        printing every missing file to stderr.
    """
    from brenda_references.config import config
    from brenda_references.data_paths import documents_path, split_path

    splits = {name: split_path(name) for name in config["datasets"]["splits"]}
    missing = [
        p for p in (documents_path(), *splits.values()) if not p.exists()
    ]
    if missing:
        print("missing input:", *missing, sep="\n  ", file=sys.stderr)
        return None
    return documents_path(), splits
