"""Guards on the ad-hoc scripts that read and write the data files."""

import hashlib
import pathlib

import pytest

from brenda_references import data_paths as package
from scripts import pull_data


def test_file_digest_matches_a_plain_sha256(tmp_path: pathlib.Path) -> None:
    """Pins the digest itself, not just that some string comes back."""
    target = tmp_path / "blob.bin"
    target.write_bytes(b"some bytes to hash")

    assert (
        pull_data.file_digest(target)
        == hashlib.sha256(b"some bytes to hash").hexdigest()
    )


def test_every_script_agrees_with_the_package_on_the_data_dir() -> None:
    """One home for the path, or a writer misses every reader.

    `pull_data.py` used to derive its own path from `__file__`, which is
    the checkout, while the loader derived it from the installed package,
    which under a non-editable install is `site-packages`. The two
    disagreed silently: the fetch reported success and every later read of
    a blob still raised `FileNotFoundError`. The same disagreement would
    send a regenerated split somewhere nothing loads.
    """
    assert pull_data.DATA_DIR == package.DATA_DIR


@pytest.mark.integration
def test_data_dir_holds_the_splits() -> None:
    assert (package.DATA_DIR / "training_data.csv").is_file()
