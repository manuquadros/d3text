"""The model classes, and the modules they are assembled from.

The three classes resolve lazily (PEP 562): importing a leaf of this package
— `d3text.models.config`, which reads a TOML file and needs nothing else —
no longer executes `base` and the stack behind it (transformers, lmdb,
sklearn, `d3text.utils`) just to re-export names the importer never asked
for. Only naming one of the three does.
"""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Read by the type checker, never executed: the names keep their real
    # types at every call site, which a bare `__getattr__` would flatten to
    # `Any`.
    from d3text.models.entity_linking import (
        BrendaClassificationModel as BrendaClassificationModel,
    )
    from d3text.models.ete import ETEBrendaModel as ETEBrendaModel
    from d3text.models.ner import (
        NERClassificationModel as NERClassificationModel,
    )

_LAZY_MODULES = {
    "BrendaClassificationModel": "entity_linking",
    "ETEBrendaModel": "ete",
    "NERClassificationModel": "ner",
}

__all__ = list(_LAZY_MODULES)


def __getattr__(name: str) -> object:
    """Import the module owning `name` on first access."""
    module = _LAZY_MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    return getattr(importlib.import_module(f".{module}", __name__), name)


def __dir__() -> list[str]:
    """List the lazy names too, which the default `dir()` would not see."""
    return sorted([*globals(), *__all__])
