#!/usr/bin/env python
"""Build the BRENDA-strain -> culture-number table the linking score reads.

A pure identifier join, like the enzyme bridge and unlike the organism one:
`cultures[].strain_number` is StrainInfo's cached record of where a strain is
deposited, so no name is compared anywhere. Run it once and commit the table::

    python scripts/build_strain_number_bridge.py \\
        brenda_references/src/brenda_references/data/documents.json \\
        data/strain_numbers.tsv

A strain is admitted **only** on collection-number form: a designation like
`P-24` or `K-12` names a strain in a paper, not a deposit, and the whole point
of this namespace is that the identifier is issued by somebody outside BRENDA.
A number the grammar reads only in part — `CCUG 12534 C`, `IMI 034912ii` — is
dropped rather than truncated, since the part that parses names a different
deposit.

One strain therefore contributes several rows, one per collection it is held
in, which is why `strain_number` is a multivalued namespace. That is the
opposite direction from `sole_entity`'s concern and does not touch it: the
judged subset is still the accessions exactly one strain carries.
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
    rows: set[BridgeRow] = set()
    for entity_id, record in table.items():
        for culture in record.get("cultures") or []:
            number = culture.get("strain_number") or ""
            if not number:
                continue
            deposits += 1
            accession = parse(number)
            if accession is not None:
                rows.add(
                    BridgeRow(
                        f"{prefix}{entity_id}",
                        accession.canonical,
                        CULTURE_NUMBER,
                    )
                )
    return (
        sorted(rows, key=lambda row: (row.entity_id, row.external_id)),
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
