"""Guards on the ad-hoc scripts that read and write the data files."""

import pytest

from brenda_references import data_paths as package
from scripts import augment_training_data, generate_dataset, pull_data


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
    assert generate_dataset.DATA_DIR == package.DATA_DIR
    # `augment_training_data` names no directory of its own: it asks
    # `split_path` for each file, so there is nothing left to disagree.
    assert augment_training_data.split_path is package.split_path


@pytest.mark.integration
def test_data_dir_holds_the_splits() -> None:
    assert (package.DATA_DIR / "training_data.csv").is_file()
