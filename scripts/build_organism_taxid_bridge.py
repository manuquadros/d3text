#!/usr/bin/env python
"""Build the BRENDA-organism -> NCBI-taxid table the linking score reads.

Run once, on a machine that has the NCBI dump; the table it writes is a few
hundred kilobytes of `entity_id -> taxid` that `d3text.identifier_bridge`
reads with no resource and no network anywhere. That split is the point:
scoring the linker must not depend on a 176 MB dump nobody's CI has.

**`taxonomy.ncbitax` is installed but inert.** Its `ROOTDIR` is
`Path(__file__).parent` four times over, which addresses `resources/` in a
source checkout and lands on `.../lib/python3.12` under the installed wheel —
and the wheel packages `src/taxonomy` alone, so the resources never ship.
Resolution therefore dies on a missing `taxdump.tar.gz` rather than on a
missing name. Put the checkout ahead of the installed package and its
resources come with it::

    PYTHONPATH=/path/to/ncbitax/src \\
        python scripts/build_organism_taxid_bridge.py \\
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
pickles. The bacteria half keeps calling `resolve_tax_id`, so its rows are the
ones already measured rather than a second resolver's answer to the same
question.

BRENDA's other-organism IDs live nowhere but the corpus: each document carries
an inline `id -> name` column, which is why the splits are arguments here. An
organism BRENDA records as `Agaricus sp.` has no taxid by definition, and
those are most of what does not resolve.
"""

import argparse
import collections
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

BACTERIA = "bacteria"
OTHER_ORGANISMS = "other_organisms"

BATCH_SIZE = 512

NAME_CLASSES = frozenset(
    {"scientific name", "synonym", "equivalent name", "common name"}
)
"""The name classes indexed, which are `bacterial_name_index`'s own.

`authority` and `type material` are left out for the reason it leaves them
out: they name a publication and a deposited culture, not the organism.
"""

SCIENTIFIC = "scientific name"

INDEX_CACHE = ncbitax.ROOTDIR / "resources/all_division_name_index.pickle"


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

    Without this the failure is a `FileNotFoundError` on a path inside the
    venv, which reads as a broken install rather than as a missing
    `PYTHONPATH`.
    """
    if not ncbitax.NAMES_PARQUET_PATH.exists():
        sys.exit(
            f"{ncbitax.__file__} resolves its resources under "
            f"{ncbitax.ROOTDIR}, which holds none. Re-run with the ncbitax "
            "checkout ahead of the installed package: "
            "PYTHONPATH=/path/to/ncbitax/src"
        )


def all_division_name_index() -> ncbitax.NameIndex:
    """Normalized NCBI name -> `(name, taxid)`, over every division.

    A name two taxa share is **dropped** rather than resolved to whichever row
    the dump lists last: the genus `Oenanthe` is a bird and a plant, and a
    bridge row picked by row order is a gold nobody can check. A scientific
    name beats the synonyms it collides with, since NCBI keeps the former
    unique per taxon and disambiguates homonyms in `unique_name`.

    Cached beside ncbitax's own indexes, keyed on the dump's mtime, because
    building it reads all 4.4 million names.
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


def taxid_row(entity_id: str, record: Mapping[str, Any]) -> BridgeRow | None:
    """`record`'s taxid, from its organism name or failing that a synonym."""
    organism = (record.get("organism") or "").strip()
    if organism:
        taxid = ncbitax.resolve_tax_id(organism)
        if taxid is not None:
            return BridgeRow(entity_id, str(taxid), "organism")

    for synonym in record.get("synonyms") or []:
        name = (synonym or "").strip()
        if not name:
            continue
        taxid = ncbitax.resolve_tax_id(name)
        if taxid is not None:
            return BridgeRow(entity_id, str(taxid), "synonym")

    return None


def inline_name_row(
    entity_id: str, names: Iterable[str], index: ncbitax.NameIndex
) -> BridgeRow | None:
    """The taxid every name this entity is called in the corpus agrees on.

    Two names resolving to two taxa make the entity's identity the thing in
    doubt, so it is dropped rather than paired with one of them.
    """
    taxids = {
        found[1]
        for name in names
        if (found := index.get(ncbitax.normalize(name.strip()))) is not None
    }
    if len(taxids) != 1:
        return None
    return BridgeRow(entity_id, str(taxids.pop()), "inline_name")


def bacteria_rows(documents: str, prefix: str) -> tuple[list[BridgeRow], int]:
    """Bridge rows for the dump's `bacteria` table, and its size."""
    table = load_entity_tables(documents).get(BACTERIA, {})
    rows = [
        row
        for entity_id, record in table.items()
        if (row := taxid_row(f"{prefix}{entity_id}", record)) is not None
    ]
    return rows, len(table)


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
    print(
        f"{len(rows)} of {population} {plural} paired with a taxid "
        f"({len(rows) / population:.1%}); {len(taxids)} distinct taxids, "
        f"{sole} of them naming exactly one {singular}."
    )


def main() -> None:
    args = read_args()
    require_resources()

    prefixes = {
        entity_type.name: entity_type.prefix
        for entity_type in BRENDA_SCHEMA.entity_types
    }
    bacteria, curated = bacteria_rows(args.documents, prefixes[BACTERIA])
    others, named = other_organism_rows(args.corpora, prefixes[OTHER_ORGANISMS])

    written = write_bridge(args.output, NCBI_TAXID, bacteria + others)
    report(bacteria, curated, "bacteria", "bacterium")
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
