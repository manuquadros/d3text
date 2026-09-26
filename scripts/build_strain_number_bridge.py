#!/usr/bin/env python
"""Build the BRENDA-strain -> culture-number table the linking score reads.

A pure identifier join: a strain's `cultures` numbers and any `designations`
that parse as an accession, no name compared. Usage is in
docs/how-to/evaluate-linking.md, why the collection list is closed and a
strain is multivalued in docs/explanation/evaluation.md.
"""

import argparse
import collections
import pathlib

from d3text.datasets.culture_numbers import parse
from d3text.identifier_bridge import STRAIN_NUMBER, BridgeRow, write_bridge
from d3text.schema import BRENDA_SCHEMA
from d3text.surface_forms import load_entity_tables

STRAINS = "strains"

CULTURE_NUMBER = "culture_number"
"""Source of a row paired through a strain's deposit in a collection."""

DESIGNATION = "designation"
"""Source of a row paired through a strain's `designations` field.

BRENDA records some deposits here instead of in `cultures`, and sometimes the
same deposit in both, on two different strain records; a designation that
parses as an accession joins the bridge exactly like a `cultures` one, under
the same canonical key, rather than being reachable only by the surface-form
index that also indexes `designations`.
"""


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="build_strain_number_bridge.py",
        description=(
            "Pair every BRENDA strain with the culture-collection accessions "
            "it is deposited under and write the table the linking "
            "evaluation reads."
        ),
    )
    parser.add_argument(
        "documents", help="TinyDB dump carrying the `strains` table"
    )
    parser.add_argument("output", help="bridge table to write")

    return parser.parse_args()


def strain_rows(
    documents: str, prefix: str
) -> tuple[list[BridgeRow], int, int]:
    """Bridge rows for the dump's `strains` table, its size, and its deposits.

    The third number is how many culture numbers the table holds at all, which
    is the denominator the grammar's coverage is only readable against.
    """
    table = load_entity_tables(documents).get(STRAINS, {})
    deposits = 0
    rows: dict[tuple[str, str], BridgeRow] = {}
    for entity_id, record in table.items():
        for culture in record.get("cultures") or []:
            number = culture.get("strain_number") or ""
            if not number:
                continue
            deposits += 1
            accession = parse(number)
            if accession is not None:
                key = (f"{prefix}{entity_id}", accession.canonical)
                rows.setdefault(key, BridgeRow(*key, CULTURE_NUMBER))
        for designation in record.get("designations") or []:
            accession = parse(designation)
            if accession is not None:
                key = (f"{prefix}{entity_id}", accession.canonical)
                rows.setdefault(key, BridgeRow(*key, DESIGNATION))
    return (
        sorted(rows.values(), key=lambda row: (row.entity_id, row.external_id)),
        len(table),
        deposits,
    )


def main() -> None:
    args = read_args()

    prefixes = {
        entity_type.name: entity_type.prefix
        for entity_type in BRENDA_SCHEMA.entity_types
    }
    rows, curated, deposits = strain_rows(args.documents, prefixes[STRAINS])
    written = write_bridge(args.output, STRAIN_NUMBER, rows)

    accessions = collections.Counter(row.external_id for row in rows)
    sole = sum(1 for count in accessions.values() if count == 1)
    strains = len({row.entity_id for row in rows})
    collections_named = len({row.external_id.split(" ", 1)[0] for row in rows})
    print(
        f"{strains} of {curated} strains carry a culture-collection "
        f"accession ({strains / curated:.1%}), over {len(rows)} deposits of "
        f"the {deposits} the table records; {len(accessions)} distinct "
        f"accessions from {collections_named} collections, {sole} of them "
        f"naming exactly one strain. "
        f"Wrote {written} rows to {pathlib.Path(args.output)}."
    )


if __name__ == "__main__":
    main()
