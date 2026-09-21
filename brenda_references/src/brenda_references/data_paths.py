"""Where the BRENDA data blobs live, independent of how this package installed.

The blobs are fetched from the Hugging Face Hub and never built into the
wheel, so their location cannot be package-relative. Under a non-editable
install a package-relative path resolves inside ``site-packages``, while the
downloader writes into the checkout it was run from, and the two never meet:
the fetch appears to succeed and every later read raises
``FileNotFoundError``.
"""

from __future__ import annotations

import os
import pathlib
from importlib import resources

from .config import config

#: The digest manifest, which *does* ship with the package: it is a few
#: hundred bytes and pins the Hub revision the recorded model numbers were
#: produced from.
MANIFEST = pathlib.Path(
    str(resources.files("brenda_references") / "data" / "SHA256SUMS")
)

_LEGACY_DIR = MANIFEST.parent


def _default_dir() -> pathlib.Path:
    """Return the user data directory the blobs are downloaded into."""
    base = os.environ.get("XDG_DATA_HOME")
    root = pathlib.Path(base) if base else pathlib.Path.home() / ".local/share"

    return root / "brenda-references"


def resolve_data_dir() -> pathlib.Path:
    """Return the directory the data blobs are read from and written to.

    Resolution touches no network and cannot fail; the directory it names
    need not exist yet.

    :return: ``BRENDA_DATA_DIR`` if set; otherwise the package's own
        ``data/`` directory when an earlier editable checkout already filled
        it, so that such a checkout keeps reading the copy it has; otherwise
        the user data directory.
    """
    override = os.environ.get("BRENDA_DATA_DIR")
    if override:
        return pathlib.Path(override).expanduser()

    # `documents.json` is the one blob nothing in this repository can
    # rebuild, so its presence is what distinguishes a filled editable
    # checkout from a package directory that merely carries the manifest.
    if (_LEGACY_DIR / "documents.json").is_file():
        return _LEGACY_DIR

    return _default_dir()


DATA_DIR = resolve_data_dir()


def split_path(split: str) -> pathlib.Path:
    """Where one split's CSV sits.

    :param split: the split name, as `config.toml`'s `datasets.splits` keys
        it -- `training`, `validation` or `test`.
    :return: the file, which need not exist yet.
    :raises KeyError: if `split` is not a configured split, naming the ones
        that are; an unconfigured name would otherwise reach the reader as a
        path that simply does not exist.
    """
    splits = config["datasets"]["splits"]
    if split not in splits:
        msg = f"{split!r} is not a split; configured: {', '.join(splits)}"
        raise KeyError(msg)
    return DATA_DIR / splits[split]


def noise_pool_path(pool: str) -> pathlib.Path:
    """Where one noise pool sits.

    :param pool: the pool name, as `config.toml`'s `datasets.noise_pools`
        keys it -- `psycholinguistics` or `enzyme_negative`.
    :return: the file, which need not exist yet.
    :raises KeyError: if `pool` is not a configured pool, naming the ones
        that are.
    """
    pools = config["datasets"]["noise_pools"]
    if pool not in pools:
        msg = f"{pool!r} is not a noise pool; configured: {', '.join(pools)}"
        raise KeyError(msg)
    return DATA_DIR / pools[pool]


def documents_path() -> pathlib.Path:
    """Where BRENDA's TinyDB dump of entity tables sits.

    Unlike the split and pool files, this blob is resolved package-relative
    (`config["documents"]`, set up in `config.py`), not against `DATA_DIR` --
    it is not one `BRENDA_DATA_DIR` relocates.

    :return: the file, which need not exist yet.
    """
    return config["documents"]


def corpus_files() -> tuple[pathlib.Path, ...]:
    """Every file a precompute command has to read, splits and pools alike.

    The pools are part of this list because `load_split` appends a block of
    each to every split, so a store built from the three CSVs alone holds
    none of those documents: the encodings reader drops each from its batch
    and the tagger masks it out of the loss.

    :return: the split files, then the noise pools, in configuration order.
    """
    return tuple(
        DATA_DIR / name
        for table in ("splits", "noise_pools")
        for name in config["datasets"][table].values()
    )
