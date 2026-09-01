#!/usr/bin/env python
"""Build the BRENDA-enzyme -> EC-number table the linking score reads.

The cheapest of the bridges and the strongest: `ec_class` is a curated column
on every BRENDA enzyme, so this is a pure identifier join with no name
comparison anywhere. Run it once and commit the table::

    python scripts/build_enzyme_ec_bridge.py \\
        brenda_references/src/brenda_references/data/documents.json \\
        data/enzyme_ec_numbers.tsv

Because the join is exact, `sole_entity` filters nothing here: every EC number
names one enzyme. The subset the enzyme evaluation scores is therefore chosen
entirely by the external nomenclature, not by this table.
"""

import argparse
import collections
import pathlib
import re

from d3text.identifier_bridge import EC_NUMBER, BridgeRow, write_bridge
from d3text.schema import BRENDA_SCHEMA
from d3text.surface_forms import load_entity_tables

ENZYMES = "enzymes"

PUBLISHED = re.compile(r"\d+\.\d+\.\d+\.\d+")
"""A four-level EC number as the nomenclature publishes it.

BRENDA also carries preliminary classes (`1.1.1.B20`), which no outside
nomenclature holds, so a span can never resolve to one.
"""


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="build_enzyme_ec_bridge.py",
        description=(
            "Pair every BRENDA enzyme with its EC number and write the table "
            "the linking evaluation reads."
        ),
    )
    parser.add_argument(
        "documents", help="TinyDB dump carrying the `enzymes` table"
    )
    parser.add_argument("output", help="bridge table to write")

    return parser.parse_args()


def enzyme_rows(documents: str, prefix: str) -> tuple[list[BridgeRow], int]:
    """Bridge rows for the dump's `enzymes` table, and its size."""
    table = load_entity_tables(documents).get(ENZYMES, {})
    rows = [
        BridgeRow(f"{prefix}{entity_id}", ec_class, "ec_class")
        for entity_id, record in table.items()
        if (ec_class := (record.get("ec_class") or "").strip())
    ]
    return rows, len(table)


def main() -> None:
    args = read_args()

    prefixes = {
        entity_type.name: entity_type.prefix
        for entity_type in BRENDA_SCHEMA.entity_types
    }
    rows, curated = enzyme_rows(args.documents, prefixes[ENZYMES])
    written = write_bridge(args.output, EC_NUMBER, rows)

    numbers = collections.Counter(row.external_id for row in rows)
    sole = sum(1 for count in numbers.values() if count == 1)
    published = sum(
        1 for number in numbers if PUBLISHED.fullmatch(number) is not None
    )
    print(
        f"{len(rows)} of {curated} enzymes paired with an EC number "
        f"({len(rows) / curated:.1%}); {len(numbers)} distinct EC numbers, "
        f"{sole} of them naming exactly one enzyme, {published} of them "
        f"published four-level classes an outside nomenclature can hold. "
        f"Wrote {written} rows to {pathlib.Path(args.output)}."
    )


if __name__ == "__main__":
    main()
