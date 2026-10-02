"""Put back the species organism a strain-linked document has lost.

A document can carry a `HasSpecies` pair whose object is in neither its
`bacteria` nor its `other_organisms`; `preprocess_relations` drops such a
pair, so the species becomes a negative. For each such document this asks
BRENDA (`BRENDA.enzyme_relations`) for the organism behind the object id and
adds it to the column `enzyme_relations` files it under, in the doc db and
in the split row whose `created` matches the document's. A bacterium missing
from the `bacteria` table gets a record there. An object BRENDA does not
attest as a `HasSpecies` object of that reference is counted and left alone.

Ids follow `migrate_organism_classes`: an organism keeps its BRENDA id
unless that id names another organism in its class anywhere in the corpus,
names a different `bacteria` record, or would change what another relation
position of a document or row resolves to. It then gets a fresh id, the same
one everywhere, and the `HasSpecies` objects that named it follow it.

Every BRENDA query, LPSN lookup and check runs before anything is written,
and the files are written through the same sibling-and-marker commit as
`migrate_organism_classes`, whose interrupted commit either script finishes.
"""

from __future__ import annotations

import argparse
import copy
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from brenda_references.organism_corpus import (
    BAC,
    OTH,
    RELATION_SLOTS,
    Counts,
    Lookup,
    Survey,
    Unit,
    configured_inputs,
    entity_of,
    fresh_ids,
    lpsn_record,
    organisms_of,
    parse_row,
    patch_split,
    read_corpus,
    resume_commit,
    strains_of,
    write_all,
)

Relations = Callable[[int], Mapping[str, Any]]
Found = tuple[str, str]
OrgKey = tuple[int, str, str]
Fix = tuple[int, str, str]
Rewrite = Callable[[Mapping[str, str]], dict[str, Any] | None]


def dangling(unit: Unit) -> set[int]:
    """The `HasSpecies` objects of `unit` in neither organism column.

    :param unit: a document or split row with its fields parsed.
    :return: those object ids.
    """
    known = organisms_of(unit, BAC).keys() | organisms_of(unit, OTH).keys()
    return {
        pair["object"]
        for pair in (unit.get("relations") or {}).get("HasSpecies", [])
        if pair["object"] not in known
    }


def attested(
    relations: Mapping[str, Any], wanted: set[int]
) -> dict[int, Found]:
    """BRENDA's organism behind each wanted id it has as a `HasSpecies` object.

    :param relations: `BRENDA.enzyme_relations` for one reference.
    :param wanted: the object ids to look up.
    :return: id -> `(name, column)`, for the ids BRENDA attests.
    """
    objects = {t.object for t in relations["triples"].get("HasSpecies", ())}
    return {
        org.id: (org.organism, kind)
        for kind in (BAC, OTH)
        for org in relations[kind]
        if org.id in wanted & objects
    }


def restore_unit(unit: Unit, fixes: Mapping[int, Fix]) -> dict[str, Any] | None:
    """Add the organism behind each dangling `HasSpecies` object of `unit`.

    :param unit: a document or split row with its fields parsed.
    :param fixes: dangling id -> `(id to use, name, column)`.
    :return: the new `bacteria`, `other_organisms` and `relations`, or `None`
        when no dangling object of `unit` has a fix.
    :raises ValueError: if a relation position other than a restored object
        would resolve to a different entity than before.
    """
    todo = dangling(unit) & fixes.keys()
    if not todo:
        return None
    before = {
        BAC: organisms_of(unit, BAC),
        OTH: organisms_of(unit, OTH),
        "strains": dict.fromkeys(strains_of(unit), ""),
    }
    after = copy.deepcopy(before)
    for old in todo:
        new, name, kind = fixes[old]
        after[kind][new] = name
    relations = copy.deepcopy(unit.get("relations") or {})
    for predicate, slot, order in RELATION_SLOTS:
        for pair in relations.get(predicate, []):
            expect = entity_of(before, order, pair[slot])
            if predicate == "HasSpecies" and pair[slot] in todo:
                new, name, _ = fixes[pair[slot]]
                pair[slot], expect = new, ("organism", name)
            if entity_of(after, order, pair[slot]) != expect:
                msg = f"{predicate} {slot} {pair[slot]} changes entity"
                raise ValueError(msg)
    return {
        BAC: {str(k): v for k, v in after[BAC].items()},
        OTH: {str(k): v for k, v in after[OTH].items()},
        "relations": relations,
    }


@dataclass
class RepairReport:
    """What a run did, or with `--dry-run` would do."""

    counts: dict[str, Counts] = field(default_factory=dict)
    reids: list[tuple[int, int, str, str]] = field(default_factory=list)
    unattested: int = 0
    records_added: int = 0
    no_lpsn: list[str] = field(default_factory=list)
    resumed: bool = False

    def lines(self) -> list[str]:
        """The report, one line per fact.

        :return: the lines.
        """
        if self.resumed:
            return [
                "an interrupted run's commit was pending: this run finishes "
                "it (with --dry-run, would) and plans nothing"
            ]
        out = [
            f"{name}: {c.to_bacteria} restored into bacteria, {c.to_other} "
            "into other_organisms"
            for name, c in self.counts.items()
        ]
        out += [
            f"re-id {old} -> {new} {name!r}: {why}"
            for old, new, name, why in self.reids
        ]
        out.append(f"left dangling, not attested by BRENDA: {self.unattested}")
        out.append(f"bacteria table: {self.records_added} added")
        out += [f"no LPSN record: {name!r}" for name in self.no_lpsn]
        return out


def _count(counts: Counts, fixes: Mapping[int, Fix], unit: Unit) -> None:
    for old in dangling(unit) & fixes.keys():
        if fixes[old][2] == BAC:
            counts.to_bacteria += 1
        else:
            counts.to_other += 1


def repair_all(
    docdb_path: Path,
    splits: Mapping[str, Path],
    relations: Relations,
    lpsn: Lookup = lpsn_record,
    dry_run: bool = False,
) -> RepairReport:
    """Restore every attested dangling species in the doc db and the splits.

    :param docdb_path: the TinyDB JSON doc db.
    :param splits: split name -> CSV path.
    :param relations: reference id -> `BRENDA.enzyme_relations` for it.
    :param lpsn: name -> `(lpsn_id, synonyms)` for a record being added.
    :param dry_run: write nothing.
    :return: what was, or would be, done.
    :raises ValueError: if two documents share the `created` a split row is
        matched by, or from `restore_unit`; before anything is written.
    """
    if resume_commit(docdb_path, splits, dry_run):
        return RepairReport(resumed=True)

    report = RepairReport()
    data, survey, table = read_corpus(docdb_path, splits)
    docs: dict[str, dict[str, Any]] = data.get("documents", {})
    found: dict[str, dict[int, Found]] = {}
    for doc_id, doc in docs.items():
        wanted = dangling(doc)
        if wanted:
            found[doc_id] = attested(relations(int(doc_id)), wanted)
            report.unattested += len(wanted - found[doc_id].keys())

    by_created: dict[str, str] = {}
    for doc_id, doc in docs.items():
        created = doc.get("created")
        if created is not None and (
            by_created.setdefault(created, doc_id) != doc_id
        ):
            msg = f"documents {by_created[created]} and {doc_id} share "
            raise ValueError(msg + f"created {created!r}")

    rows: dict[str, list[Unit]] = {}

    def collect(row: Mapping[str, str]) -> None:
        doc_id = by_created.get(row["created"])
        if doc_id in found:
            rows.setdefault(doc_id, []).append(parse_row(row))

    for path in splits.values():
        patch_split(path, collect, None)

    units_of: dict[OrgKey, list[Unit]] = {}
    for doc_id, got in found.items():
        for key, (name, kind) in got.items():
            units_of.setdefault((key, name, kind), []).extend(
                [docs[doc_id], *rows.get(doc_id, [])]
            )
    ids = _choose_ids(units_of, survey, table, report)
    fixes = {
        doc_id: {k: (ids[k, n, c], n, c) for k, (n, c) in got.items()}
        for doc_id, got in found.items()
    }
    updates = {}
    counts = report.counts.setdefault(docdb_path.name, Counts())
    for doc_id, doc_fixes in fixes.items():
        result = restore_unit(docs[doc_id], doc_fixes)
        if result is not None:
            _count(counts, doc_fixes, docs[doc_id])
            updates[doc_id] = result

    def rewrite_row(counts: Counts) -> Rewrite:
        def rewrite(row: Mapping[str, str]) -> dict[str, Any] | None:
            doc_fixes = fixes.get(by_created.get(row["created"], ""), {})
            unit = parse_row(row)
            _count(counts, doc_fixes, unit)
            return restore_unit(unit, doc_fixes)

        return rewrite

    for path in splits.values():
        counts = report.counts.setdefault(path.name, Counts())
        patch_split(path, rewrite_row(counts), None)

    added: dict[str, dict[str, Any]] = {}
    for (_, name, kind), ident in sorted(ids.items(), key=lambda kv: kv[1]):
        if kind == BAC and ident not in table:
            lpsn_id, synonyms = lpsn(name)
            if lpsn_id is None:
                report.no_lpsn.append(name)
            added[str(ident)] = {
                "organism": name,
                "synonyms": synonyms,
                "lpsn_id": lpsn_id,
            }
    report.records_added = len(added)
    if dry_run:
        return report

    for doc_id, fields in updates.items():
        docs[doc_id].update(fields)
    data.setdefault(BAC, {}).update(added)
    write_all(
        docdb_path,
        data,
        splits,
        lambda path, dst: patch_split(path, rewrite_row(Counts()), dst),
    )
    return report


def _choose_ids(
    units_of: Mapping[OrgKey, list[Unit]],
    survey: Survey,
    table: Mapping[int, set[str]],
    report: RepairReport,
) -> dict[OrgKey, int]:
    """The id each restored organism takes, recording each fresh one."""
    in_class: dict[tuple[int, str], set[str]] = {}
    for key, name, kind in survey.entries:
        in_class.setdefault((key, kind), set()).add(name)
    fresh = fresh_ids(survey, table)
    ids: dict[OrgKey, int] = {}
    for (key, name, kind), units in sorted(units_of.items()):
        if in_class.setdefault((key, kind), {name}) != {name}:
            why = f"id also names another organism in {kind}"
        elif kind == BAC and key in table and name not in table[key]:
            why = "id is a different bacteria record"
        elif not _keeps_others(units, key, name, kind):
            why = "id would change what another relation position names"
        else:
            ids[key, name, kind] = key
            continue
        ids[key, name, kind] = next(fresh)
        report.reids.append((key, ids[key, name, kind], name, why))
    return ids


def _keeps_others(units: list[Unit], key: int, name: str, kind: str) -> bool:
    """Whether `key` can be restored under its own id in every unit."""
    try:
        for unit in units:
            restore_unit(unit, {key: (key, name, kind)})
    except ValueError:
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    """Run the repair on the configured data against BRENDA.

    :param argv: arguments, `None` for `sys.argv`.
    :return: the exit status; 2 when an input file is missing.
    """
    from brenda_references.db import BRENDA

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    inputs = configured_inputs()
    if inputs is None:
        return 2
    brenda = BRENDA()
    try:
        report = repair_all(
            *inputs, brenda.enzyme_relations, dry_run=args.dry_run
        )
    finally:
        brenda.session.close()
    print(*report.lines(), sep="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
