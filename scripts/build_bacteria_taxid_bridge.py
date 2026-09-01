#!/usr/bin/env python
"""Build the BRENDA-bacterium -> NCBI-taxid table the linking score reads.

Run once, on a machine that has the NCBI dump; the table it writes is a
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
        python scripts/build_bacteria_taxid_bridge.py \\
        brenda_references/src/brenda_references/data/documents.json \\
        data/bacteria_taxids.tsv

The names are resolved against NCBI's own synonym lists, which are what makes
the resulting gold independent of BRENDA's: `resolve_tax_id` normalises a name
and looks it up in three prebuilt indexes (species, then strain, then genus)
built from the taxonomy dump. BRENDA's `organism` is tried first and its LPSN
synonyms only as a fallback, so the pairing rests on the curated name wherever
one resolves.

An organism BRENDA records as `Bacillus sp.` has no taxid by definition, and
those are most of what does not resolve.
"""

import argparse
import collections
import pathlib
import sys
from collections.abc import Mapping
from typing import Any

from taxonomy.ncbitax import ncbitax

from d3text.identifier_bridge import NCBI_TAXID, BridgeRow, write_bridge
from d3text.schema import BRENDA_SCHEMA
from d3text.surface_forms import load_entity_tables

BACTERIA = "bacteria"


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="build_bacteria_taxid_bridge.py",
        description=(
            "Pair every BRENDA bacterium with an NCBI taxid, offline, and "
            "write the table the linking evaluation reads."
        ),
    )
    parser.add_argument(
        "documents", help="TinyDB dump carrying the `bacteria` table"
    )
    parser.add_argument("output", help="bridge table to write")

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


def main() -> None:
    args = read_args()
    require_resources()

    prefix = next(
        entity_type.prefix
        for entity_type in BRENDA_SCHEMA.entity_types
        if entity_type.name == BACTERIA
    )
    table = load_entity_tables(args.documents).get(BACTERIA, {})
    rows = [
        row
        for entity_id, record in table.items()
        if (row := taxid_row(f"{prefix}{entity_id}", record)) is not None
    ]

    written = write_bridge(args.output, NCBI_TAXID, rows)
    taxids = collections.Counter(row.external_id for row in rows)
    sole = sum(1 for count in taxids.values() if count == 1)
    print(
        f"{written} of {len(table)} bacteria paired with a taxid "
        f"({written / len(table):.1%}); {len(taxids)} distinct taxids, "
        f"{sole} of them naming exactly one bacterium. "
        f"Wrote {pathlib.Path(args.output)}."
    )


if __name__ == "__main__":
    main()
