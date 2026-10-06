"""Put back the species organism a strain-linked document has lost.

A document can carry a `HasSpecies` pair whose object is in neither its
`bacteria` nor its `other_organisms`; `preprocess_relations` drops such a
pair, so the species becomes a negative. For each such document this asks
BRENDA (`BRENDA.enzyme_relations`) for the organism behind the object id and
adds it to the column `enzyme_relations` files it under, in the doc db and
in the split row whose `created` matches the document's. A bacterium missing
from the `bacteria` table gets a record there. A document or row whose column
already holds that organism, under its name or one the `bacteria` table lists
for it, has the object pointed at the id it holds instead, and is counted
apart. An object BRENDA does not
attest as a `HasSpecies` object of that reference is counted and left alone.

An organism gets one id per class for the whole run, whichever BRENDA ids
name it: the lowest id the corpus already gives that name (for a bacterium,
also one whose `bacteria` record answers to it), else its lowest BRENDA id,
skipping an id that names another organism in its class anywhere in the
corpus, names a different `bacteria` record, or would change what another
relation position of a document or row resolves to. With none left it gets a
fresh id, and the `HasSpecies` objects that named it follow it.

Every BRENDA query, LPSN lookup and check runs before anything is written,
and the files are written through the same sibling-and-marker commit as
`migrate_organism_classes`, whose interrupted commit either script finishes.
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from collections.abc import Callable, Mapping, Set
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


def held_id(
    unit: Unit, name: str, kind: str, table: Mapping[int, set[str]]
) -> int | None:
    """The lowest id in `unit`'s `kind` column that already names `name`.

    :param unit: a document or split row with its fields parsed.
    :param name: an organism name BRENDA attests.
    :param kind: `BAC` or `OTH`.
    :param table: the `bacteria` table, id -> the names its record answers to.
    :return: that id, or `None` if the column holds no such organism.
    """
    return min(
        (
            ident
            for ident, held in organisms_of(unit, kind).items()
            if held == name or (kind == BAC and name in table.get(ident, ()))
        ),
        default=None,
    )


def restore_unit(
    unit: Unit, fixes: Mapping[int, Fix], table: Mapping[int, set[str]]
) -> dict[str, Any] | None:
    """Add the organism behind each dangling `HasSpecies` object of `unit`.

    An organism the column already holds (`held_id`) is not added again: the
    object is pointed at the id held.

    :param unit: a document or split row with its fields parsed.
    :param fixes: dangling id -> `(id to use, name, column)`.
    :param table: the `bacteria` table, id -> the names its record answers to.
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
    target = {}
    for old in todo:
        new, name, kind = fixes[old]
        held = held_id(unit, name, kind, table)
        if held is None:
            after[kind][new] = name
            target[old] = new
        else:
            target[old] = held
    relations = copy.deepcopy(unit.get("relations") or {})
    for predicate, slot, order in RELATION_SLOTS:
        for pair in relations.get(predicate, []):
            expect = entity_of(before, order, pair[slot])
            if predicate == "HasSpecies" and pair[slot] in todo:
                _, name, kind = fixes[pair[slot]]
                pair[slot] = target[pair[slot]]
                expect = ("organism", after[kind][pair[slot]])
            if entity_of(after, order, pair[slot]) != expect:
                msg = f"{predicate} {slot} {pair[slot]} changes entity"
                raise ValueError(msg)
    return {
        BAC: {str(k): v for k, v in after[BAC].items()},
        OTH: {str(k): v for k, v in after[OTH].items()},
        "relations": relations,
    }


@dataclass
class Tally(Counts):
    """`Counts`, and the objects pointed at an organism already held."""

    repointed: int = 0


@dataclass
class RepairReport:
    """What a run did, or with `--dry-run` would do."""

    counts: dict[str, Tally] = field(default_factory=dict)
    reids: list[tuple[int, int, str, str]] = field(default_factory=list)
    unattested: int = 0
    records_added: int = 0
    names_split: tuple[int, int] = (0, 0)
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
            f"into other_organisms, {c.repointed} pointed at an organism "
            "already held"
            for name, c in self.counts.items()
        ]
        out += [
            f"re-id {old} -> {new} {name!r}: {why}"
            for old, new, name, why in self.reids
        ]
        out.append(f"left dangling, not attested by BRENDA: {self.unattested}")
        out.append(f"bacteria table: {self.records_added} added")
        before, after = self.names_split
        out.append(
            f"names under more than one id: {before} before, {after} after"
        )
        out += [f"no LPSN record: {name!r}" for name in self.no_lpsn]
        return out


def _count(
    counts: Tally,
    fixes: Mapping[int, Fix],
    unit: Unit,
    table: Mapping[int, set[str]],
) -> None:
    for old in dangling(unit) & fixes.keys():
        _, name, kind = fixes[old]
        if held_id(unit, name, kind, table) is not None:
            counts.repointed += 1
        elif kind == BAC:
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

    ids = _choose_ids(
        [
            (unit, got)
            for doc_id, got in found.items()
            for unit in (docs[doc_id], *rows.get(doc_id, []))
        ],
        survey,
        table,
        report,
    )
    updates = {}
    counts = report.counts.setdefault(docdb_path.name, Tally())
    for doc_id, got in found.items():
        doc_fixes = _unit_fixes(docs[doc_id], got, ids, table)
        result = restore_unit(docs[doc_id], doc_fixes, table)
        if result is not None:
            _count(counts, doc_fixes, docs[doc_id], table)
            updates[doc_id] = result

    def rewrite_row(counts: Tally) -> Rewrite:
        def rewrite(row: Mapping[str, str]) -> dict[str, Any] | None:
            got = found.get(by_created.get(row["created"], ""), {})
            unit = parse_row(row)
            row_fixes = _unit_fixes(unit, got, ids, table)
            _count(counts, row_fixes, unit, table)
            return restore_unit(unit, row_fixes, table)

        return rewrite

    for path in splits.values():
        counts = report.counts.setdefault(path.name, Tally())
        patch_split(path, rewrite_row(counts), None)

    added: dict[str, dict[str, Any]] = {}
    for ident, name, kind in sorted(
        {(i, n, c) for (_, n, c), i in ids.items()}
    ):
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
        lambda path, dst: patch_split(path, rewrite_row(Tally()), dst),
    )
    return report


def _unit_fixes(
    unit: Unit,
    got: Mapping[int, Found],
    ids: Mapping[OrgKey, int],
    table: Mapping[int, set[str]],
) -> dict[int, Fix]:
    """The fixes `restore_unit` gets for `unit`, from its document's `got`.

    A key `unit` needs an id for and `ids` has none yet is left out, so it
    stays dangling.
    """
    return {
        # `restore_unit` ignores the id slot for an organism `unit` already
        # holds; `key` fills it when no id was chosen.
        key: (ids.get((key, name, kind), key), name, kind)
        for key, (name, kind) in got.items()
        if (key, name, kind) in ids
        or held_id(unit, name, kind, table) is not None
    }


def _choose_ids(
    units: list[tuple[Unit, Mapping[int, Found]]],
    survey: Survey,
    table: Mapping[int, set[str]],
    report: RepairReport,
) -> dict[OrgKey, int]:
    """One id per restored organism and class, recording each re-id.

    Organisms are chosen in sorted order, and a candidate is checked in each
    unit needing it together with the ids already chosen (`_unit_fixes`).
    """
    in_class: dict[tuple[int, str], set[str]] = {}
    for key, name, kind in survey.entries:
        in_class.setdefault((key, kind), set()).add(name)
    keys_of: dict[tuple[str, str], set[int]] = {}
    members: dict[tuple[str, str], dict[int, None]] = {}
    for i, (unit, got) in enumerate(units):
        for key, (name, kind) in got.items():
            if held_id(unit, name, kind, table) is None:
                keys_of.setdefault((name, kind), set()).add(key)
                members.setdefault((name, kind), {})[i] = None
    fresh = fresh_ids(survey, table)
    ids: dict[OrgKey, int] = {}
    for (name, kind), keys in sorted(keys_of.items()):
        held = {k for k, n, c in survey.entries if (n, c) == (name, kind)}
        if kind == BAC:
            held |= {k for k, names in table.items() if name in names}
        group = [units[i] for i in members[name, kind]]
        rejected: dict[int, str] = {}
        for new in dict.fromkeys([*sorted(held), *sorted(keys)]):
            trial = ids | {(key, name, kind): new for key in keys}
            if in_class.get((new, kind), {name}) != {name}:
                rejected[new] = f"id also names another organism in {kind}"
            elif kind == BAC and new in table and name not in table[new]:
                rejected[new] = "id is a different bacteria record"
            elif not _keeps_others(group, trial, table):
                rejected[new] = (
                    "id would change what another relation position names"
                )
            else:
                break
        else:
            new = next(fresh)
        in_class[new, kind] = {name}
        for key in sorted(keys):
            ids[key, name, kind] = new
            if key != new:
                why = rejected.get(key, "the organism already has an id")
                report.reids.append((key, new, name, why))
    report.names_split = (
        _names_split(survey.entries),
        _names_split(
            survey.entries | {(i, n, c) for (_, n, c), i in ids.items()}
        ),
    )
    return ids


def _names_split(entries: Set[tuple[int, str, str]]) -> int:
    """How many `(name, class)` pairs `entries` gives more than one id."""
    return sum(n > 1 for n in Counter((n, c) for _, n, c in entries).values())


def _keeps_others(
    units: list[tuple[Unit, Mapping[int, Found]]],
    ids: Mapping[OrgKey, int],
    table: Mapping[int, set[str]],
) -> bool:
    """Whether every unit restores under `ids` without changing an entity."""
    try:
        for unit, got in units:
            restore_unit(unit, _unit_fixes(unit, got, ids, table), table)
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
