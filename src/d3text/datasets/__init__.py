"""Dataset adapters: a `Schema`, and the loader that indexes its splits.

One module per corpus. Importing this package pulls in the BRENDA data layer,
which writes `lpsn.log` into the cwd at import time, so import it where the
dataset is actually wanted; `d3text.schema` itself stays a leaf.
"""

from d3text.datasets.brenda import BRENDA_SCHEMA, brenda_dataset

__all__ = ["BRENDA_SCHEMA", "brenda_dataset"]
