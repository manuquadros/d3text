"""Forwarding for NCBI taxonomy identifiers NCBI has since retired.

A taxid cached from an outside source can predate the taxonomy dump on disk.
Reading such a taxid without forwarding it through ``merged.dmp`` first risks
reading one taxon as two, or gold nothing current can match; any caller of a
raw NCBI taxid should go through `merged_taxids` rather than reinvent it.
"""

from taxonomy.ncbitax import ncbitax


def merged_taxids() -> dict[int, int]:
    """Every taxid NCBI has retired, mapped to the one it was merged into.

    :return: retired taxid -> current taxid, read from ``ncbitax``'s
        ``merged.dmp`` table.
    """
    table = ncbitax.load_df("merged")
    return {
        int(old): int(new)
        for old, new in zip(table["old_tax_id"], table["new_tax_id"])
    }
