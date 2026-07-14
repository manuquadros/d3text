"""Parses config.toml and makes the settings available to the package"""

import tomllib
from importlib import resources
from typing import Any

PKGROOT = resources.files("brenda_references")
CONFIG = PKGROOT / "config.toml"

# Every key names a machine-local resource that only the data-collection entry
# points touch (the TinyDB document store, the BRENDA source dumps, the SQL
# connection). The training/validation/test splits are read from CSVs packaged
# with this distribution and need none of them, so importing must not require
# the file: it is gitignored, which means a fresh checkout — and CI — has none.
# Consumers that do need a key ask for it through `require`.
config: dict[str, Any] = {}

if CONFIG.is_file():
    with CONFIG.open(mode="rb") as cf:
        config = tomllib.load(cf)

    config["documents"] = PKGROOT / config["documents"]

    for resource in config["sources"]:
        config["sources"][resource] = PKGROOT / config["sources"][resource]


def require(key: str) -> Any:
    """Return `config[key]`, or explain which file is missing."""
    try:
        return config[key]
    except KeyError:
        raise KeyError(
            f"'{key}' is not configured: {CONFIG} does not exist or does not "
            f"define it. Copy config.toml.example to {CONFIG} and fill it in."
        ) from None
