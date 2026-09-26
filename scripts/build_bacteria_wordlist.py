#!/usr/bin/env python
"""Regenerate `data/bacteria.txt` from NCBI's bacterial species and genera.

NCBI mints a tax_id for unidentified isolates, so the raw ncbitax index is
dominated by placeholder names like `Bacillus sp. MN07-04`, which
`is_isolate_placeholder` drops. It is not exhaustive: a name like
`Gram-positive bacterium ADG_ECW` carries no digit and no `sp.` and still
slips through.
"""

import pathlib

from taxonomy.ncbitax import ncbitax as nt

OUTPUT = (
    pathlib.Path(__file__).resolve().parent.parent / "data" / "bacteria.txt"
)


def is_isolate_placeholder(name: str) -> bool:
    """True for a name that reads as an isolate code, not a binomial."""

    if any(ch.isdigit() for ch in name):
        return True

    tokens = name.split()
    # "Candidatus" is a real nomenclature prefix, not part of the binomial
    # itself -- without stripping it, "Candidatus Foo sp. bar123" reads as
    # a 4-token name and the `sp.` check below misses it at index 1.
    if tokens[:1] == ["Candidatus"]:
        tokens = tokens[1:]
    if len(tokens) == 2:
        # "<Family/genus> bacterium" is NCBI's other placeholder shape for
        # an unclassified isolate; no real binomial ends in the literal
        # word "bacterium".
        return tokens[1].lower() == "bacterium"
    if len(tokens) > 2:
        return tokens[1].rstrip(".").lower() == "sp"
    return False


def curated_names() -> set[str]:
    species = nt.bacterial_name_index("species")
    genus = nt.bacterial_name_index("genus")
    candidates = {v[0] for v in species.values()} | {
        v[0] for v in genus.values()
    }
    return {name for name in candidates if not is_isolate_placeholder(name)}


def main() -> None:
    names = sorted(curated_names())
    OUTPUT.write_text("\n".join(names) + "\n")
    print(f"wrote {len(names)} names to {OUTPUT}")


if __name__ == "__main__":
    main()
