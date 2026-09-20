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
