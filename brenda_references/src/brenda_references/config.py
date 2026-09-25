"""Parses config.toml and makes the settings available to the package"""

import tomllib
from importlib import resources

PKGROOT = resources.files("brenda_references")
CONFIG = PKGROOT / "config.toml"

with CONFIG.open(mode="rb") as cf:
    config = tomllib.load(cf)

# `config["documents"]` and `config["datasets"]` are deliberately left as
# bare file names: those blobs are fetched to `data_paths.DATA_DIR`, which is
# the package directory only in an editable checkout that already holds
# them. `data_paths` resolves them; a script that opens `config["documents"]`
# directly bypasses it and resolves the bare name against the working
# directory instead.

for resource in config["sources"]:
    config["sources"][resource] = PKGROOT / config["sources"][resource]
