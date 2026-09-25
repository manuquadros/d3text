"""Dataset adapters: a `Schema`, and the loader that indexes its splits.

One module per corpus. `BRENDA_SCHEMA` and `brenda_dataset` resolve lazily
(PEP 562): importing this package, or any of its other submodules, does not
pull in the BRENDA data layer (`brenda_references` and its dependencies).
Only accessing one of those two names does.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from d3text.datasets.brenda import BRENDA_SCHEMA, brenda_dataset

__all__ = ["BRENDA_SCHEMA", "brenda_dataset"]


def __getattr__(name: str):
    if name in __all__:
        from d3text.datasets import brenda

        return getattr(brenda, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
