"""Parses config.toml and makes the settings available to the package"""

import tomllib
from importlib import resources

PKGROOT = resources.files("brenda_references")
CONFIG = PKGROOT / "config.toml"

with CONFIG.open(mode="rb") as cf:
    config = tomllib.load(cf)

config["documents"] = PKGROOT / config["documents"]

for resource in config["sources"]:
    config["sources"][resource] = PKGROOT / config["sources"][resource]
