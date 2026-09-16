"""Guards on the ad-hoc scripts that write the packaged data files."""

from importlib import resources

import pytest

from brenda_references import brenda_references as package
from scripts import generate_dataset


def test_generate_dataset_writes_where_the_package_reads() -> None:
    assert generate_dataset.DATA_DIR == package.DATA_DIR


@pytest.mark.integration
def test_generate_dataset_data_dir_holds_the_splits() -> None:
    with resources.as_file(generate_dataset.DATA_DIR) as data_dir:
        assert (data_dir / "training_data.csv").is_file()
