#!/usr/bin/env python
"""Build the BRENDA-organism -> NCBI-taxid table the linking score reads.

Run once, on a machine that has the NCBI dump; the table it writes is a few
hundred kilobytes of `entity_id -> taxid` that `d3text.identifier_bridge`
reads with no resource and no network anywhere. That split is the point:
scoring the linker must not depend on a 176 MB dump nobody's CI has.

`ncbitax` is a pinned dependency (see `pyproject.toml`); no `PYTHONPATH`
workaround is needed to run this script anymore. Run it as::

    pdm run python scripts/build_organism_taxid_bridge.py \\
        brenda_references/src/brenda_references/data/documents.json \\
        data/organism_taxids.tsv \\
        brenda_references/src/brenda_references/data/training_data.csv \\
        brenda_references/src/brenda_references/data/validation_data.csv \\
        brenda_references/src/brenda_references/data/test_data.csv

The names are resolved against NCBI's own synonym lists, which are what makes
the resulting gold independent of BRENDA's: a name is normalized and looked up
in an index built from the taxonomy dump.

**Bacteria and other organisms are resolved by different indexes, and that is
not tidiness.** `ncbitax.resolve_tax_id` consults three indexes all built with
`division_id == 0`, so it cannot resolve a plant, a fungus or a vertebrate at
all — which is the entire population BRENDA's `other_organisms` holds. This
script therefore builds `all_division_name_index`, the same normalized
name -> taxid mapping over every division, and caches it beside ncbitax's own
pickles.

**A bacterium is paired by identifier first and by name only after.** The
`strains` table carries StrainInfo's cached `taxon`, which holds an LPSN
identifier beside an NCBI taxid, so a bacterium's `lpsn_id` reaches a taxid
with no string comparison anywhere. That is a correctness argument before it
is a coverage one: BRENDA's synonyms are binomials even where the entity is a
subspecies, so resolving `Bacillus subtilis subtilis` by name lands on the
species. Where no LPSN pairing exists the names are resolved, `resolve_tax_id`
first and the all-division index only where it is mute — so a name the
bacteria division already answered keeps that answer.

BRENDA's other-organism IDs live nowhere but the corpus: each document carries
an inline `id -> name` column, which is why the splits are arguments here. An
organism BRENDA records as `Agaricus sp.` has no taxid by definition, and
those are most of what does not resolve.
"""

import argparse
import collections
import functools
import pathlib
import sys
from collections.abc import Iterable, Mapping
from typing import Any

from taxonomy.ncbitax import ncbitax

from d3text import corpus
from d3text.identifier_bridge import NCBI_TAXID, BridgeRow, write_bridge
from d3text.schema import BRENDA_SCHEMA
from d3text.surface_forms import (
    load_entity_tables,
    pooled_other_organism_names,
)
from d3text.taxonomy import merged_taxids

BACTERIA = "bacteria"
OTHER_ORGANISMS = "other_organisms"
STRAINS = "strains"

LPSN_JOIN = "lpsn_id"
"""Source of a row paired through StrainInfo's cached LPSN -> NCBI taxon."""

ALL_DIVISIONS = "_all_divisions"
"""Suffix marking a row the bacteria-division indexes were mute on."""

BATCH_SIZE = 512

NAME_CLASSES = frozenset(
    {"scientific name", "synonym", "equivalent name", "common name"}
)
"""The name classes indexed, which are `bacterial_name_index`'s own.

`authority` and `type material` are left out for the reason it leaves them
out: they name a publication and a deposited culture, not the organism.
"""

SCIENTIFIC = "scientific name"

INDEX_CACHE = ncbitax.DATA_DIR / "all_division_name_index.pickle"


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="build_organism_taxid_bridge.py",
        description=(
            "Pair every BRENDA bacterium and other organism with an NCBI "
            "taxid, offline, and write the table the linking evaluation "
            "reads."
        ),
    )
    parser.add_argument(
        "documents", help="TinyDB dump carrying the `bacteria` table"
    )
    parser.add_argument("output", help="bridge table to write")
    parser.add_argument(
        "corpora",
        nargs="+",
        help=(
            "split CSVs carrying the inline other-organism names; pass all "
            "three, since an ID named in none of them is unreachable"
        ),
    )

    return parser.parse_args()


def require_resources() -> None:
    """Stop before the first lookup if the NCBI dump is out of reach.

    Without this the failure is a `FileNotFoundError` deep inside a lookup,
    which reads as a broken install rather than as a missing resource.
    """
    if not ncbitax.NAMES_PARQUET_PATH.exists():
        sys.exit(
            f"{ncbitax.__file__} resolves its resources under "
            f"{ncbitax.DATA_DIR}, which holds none. ncbitax auto-downloads "
            "the taxonomy dump on first lookup unless NCBITAX_AUTO_DOWNLOAD=0 "
            "is set; unset it, or place the dump there yourself."
        )


@functools.cache
def all_division_name_index() -> ncbitax.NameIndex:
    """Normalized NCBI name -> `(name, taxid)`, over every division.

    A name two taxa share is **dropped** rather than resolved to whichever row
    the dump lists last: the genus `Oenanthe` is a bird and a plant, and a
    bridge row picked by row order is a gold nobody can check. A scientific
    name beats the synonyms it collides with, since NCBI keeps the former
    unique per taxon and disambiguates homonyms in `unique_name`.

    Cached beside ncbitax's own indexes, keyed on the dump's mtime, because
    building it reads all 4.4 million names, and held for the process because
    both organism halves consult it.
    """
    cached = ncbitax.get_index(INDEX_CACHE)
    if cached:
        return cached

    names = ncbitax.names()
    selected = names[names["name_class"].isin(NAME_CLASSES)]
    normalized = [
        ncbitax.normalize(ncbitax.remove_citations(name))
        for name in selected["name_txt"]
    ]

    preferred: dict[str, tuple[str, int]] = {}
    fallback: dict[str, tuple[str, int]] = {}
    contested: dict[bool, set[str]] = {True: set(), False: set()}
    rows = zip(
        normalized,
        selected["name_txt"],
        selected["tax_id"],
        selected["name_class"],
    )
    for key, name, tax_id, name_class in rows:
        if not key:
            continue
        scientific = name_class == SCIENTIFIC
        table = preferred if scientific else fallback
        held = table.get(key)
        if held is None:
            table[key] = (name, int(tax_id))
        elif held[1] != int(tax_id):
            contested[scientific].add(key)

    index = {
        key: entry
        for key, entry in fallback.items()
        if key not in preferred and key not in contested[False]
    }
    index |= {
        key: entry
        for key, entry in preferred.items()
        if key not in contested[True]
    }

    ncbitax.save_index(index=index, path=INDEX_CACHE)
    return index


def lpsn_taxids(
    strains: Mapping[str, Any], merged: Mapping[int, int]
) -> tuple[dict[int, int], int]:
    """LPSN identifier -> NCBI taxid, from the strains' cached taxa.

    Retired taxids are forwarded before the pairing is checked, so two
    strains recording one taxon under an old and a current identifier agree
    rather than contest each other. An identifier still naming two taxa is
    dropped, for the reason `inline_name_row` drops one; the second return
    value counts how many were dropped that way.
    """
    found: dict[int, set[int]] = collections.defaultdict(set)
    for strain in strains.values():
        taxon = strain.get("taxon") or {}
        lpsn, taxid = taxon.get("lpsn"), taxon.get("ncbi")
        if lpsn is None or taxid is None:
            continue
        found[int(lpsn)].add(merged.get(int(taxid), int(taxid)))

    resolved = {
        lpsn: next(iter(taxids))
        for lpsn, taxids in found.items()
        if len(taxids) == 1
    }
    return resolved, len(found) - len(resolved)


def index_taxid(index: ncbitax.NameIndex, name: str) -> int | None:
    """`name`'s taxid in an all-division index, or None if it holds none."""
    found = index.get(ncbitax.normalize(name.strip()))
    return None if found is None else found[1]


def taxid_row(
    entity_id: str,
    record: Mapping[str, Any],
    index: ncbitax.NameIndex,
    taxids_by_lpsn: Mapping[int, int],
) -> BridgeRow | None:
    """`record`'s taxid: its LPSN identifier first, then its names.

    An entity BRENDA spells as a trinomial carries binomial synonyms, so a
    name lookup answers with the species where the entity is the subspecies;
    the identifier join compares no strings and does not.
    """
    lpsn = record.get("lpsn_id")
    if lpsn is not None:
        taxid = taxids_by_lpsn.get(int(lpsn))
        if taxid is not None:
            return BridgeRow(entity_id, str(taxid), LPSN_JOIN)

    organism = (record.get("organism") or "").strip()
    synonyms = [
        name
        for synonym in record.get("synonyms") or []
        if (name := (synonym or "").strip())
    ]
    resolvers = (
        (ncbitax.resolve_tax_id, ""),
        (functools.partial(index_taxid, index), ALL_DIVISIONS),
    )
    for resolve, suffix in resolvers:
        if organism:
            taxid = resolve(organism)
            if taxid is not None:
                return BridgeRow(entity_id, str(taxid), f"organism{suffix}")
        for name in synonyms:
            taxid = resolve(name)
            if taxid is not None:
                return BridgeRow(entity_id, str(taxid), f"synonym{suffix}")

    return None


def inline_name_row(
    entity_id: str, names: Iterable[str], index: ncbitax.NameIndex
) -> BridgeRow | None:
    """The taxid every name this entity is called in the corpus agrees on.

    Two names resolving to two taxa make the entity's identity the thing in
    doubt, so it is dropped rather than paired with one of them.
    """
    taxids = {
        taxid
        for name in names
        if (taxid := index_taxid(index, name)) is not None
    }
    if len(taxids) != 1:
        return None
    return BridgeRow(entity_id, str(taxids.pop()), "inline_name")


def bacteria_rows(
    tables: Mapping[str, Any], prefix: str
) -> tuple[list[BridgeRow], int, int]:
    """Bridge rows for the dump's `bacteria` table, its size, and how many
    LPSN ids the join dropped as contested."""
    table = tables.get(BACTERIA, {})
    taxids_by_lpsn, contested = lpsn_taxids(
        tables.get(STRAINS, {}), merged_taxids()
    )
    index = all_division_name_index()
    rows = [
        row
        for entity_id, record in table.items()
        if (
            row := taxid_row(
                f"{prefix}{entity_id}", record, index, taxids_by_lpsn
            )
        )
        is not None
    ]
    return rows, len(table), contested


def other_organism_rows(
    corpora: Iterable[str], prefix: str
) -> tuple[list[BridgeRow], int]:
    """Bridge rows for the corpus's other organisms, and how many there are.

    The names are taken verbatim, not through `other_organism_forms`: that
    extractor adds the genus abbreviations the linker needs to match running
    text, and NCBI listing `S. argus` against some other taxon would make an
    entity whose binomial resolves cleanly look contested.
    """
    forms = pooled_other_organism_names(
        column
        for path in corpora
        for column in corpus.other_organism_names(
            pathlib.Path(path), BATCH_SIZE
        )
    )
    index = all_division_name_index()
    rows = [
        row
        for entity_id, names in forms.items()
        if (row := inline_name_row(f"{prefix}{entity_id}", names, index))
        is not None
    ]
    return rows, len(forms)


def report(
    rows: list[BridgeRow], population: int, plural: str, singular: str
) -> None:
    """One line saying how much of a population the table reaches."""
    taxids = collections.Counter(row.external_id for row in rows)
    sole = sum(1 for count in taxids.values() if count == 1)
    sources = collections.Counter(row.source for row in rows)
    paired = ", ".join(
        f"{count} by {source}" for source, count in sorted(sources.items())
    )
    print(
        f"{len(rows)} of {population} {plural} paired with a taxid "
        f"({len(rows) / population:.1%}); {len(taxids)} distinct taxids, "
        f"{sole} of them naming exactly one {singular}. Paired {paired}."
    )


def main() -> None:
    args = read_args()
    require_resources()

    prefixes = {
        entity_type.name: entity_type.prefix
        for entity_type in BRENDA_SCHEMA.entity_types
    }
    tables = load_entity_tables(args.documents)
    bacteria, curated, contested_lpsn = bacteria_rows(
        tables, prefixes[BACTERIA]
    )
    others, named = other_organism_rows(args.corpora, prefixes[OTHER_ORGANISMS])

    written = write_bridge(args.output, NCBI_TAXID, bacteria + others)
    report(bacteria, curated, "bacteria", "bacterium")
    print(
        f"{contested_lpsn} LPSN ids named more than one forwarded taxid "
        "and were dropped from the join."
    )
    report(others, named, "other organisms", "other organism")

    shared = {row.external_id for row in bacteria} & {
        row.external_id for row in others
    }
    print(
        f"{len(shared)} taxids are carried by both a bacterium and an other "
        f"organism, and are gold for neither on their own. "
        f"Wrote {written} rows to {pathlib.Path(args.output)}."
    )


if __name__ == "__main__":
    main()
