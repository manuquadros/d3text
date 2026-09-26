import pathlib

import pytest
from d3text.datasets.brenda import BRENDA_SCHEMA, brenda_dataset
from d3text.models.config import encodings_path

# Precomputed encodings ship with no checkout, so skip where the file
# `brenda_dataset` opens is absent. Resolved at import, before `conftest`'s
# autouse fixture hides the machine's config.
try:
    _ENCODINGS_PATH: pathlib.Path | None = encodings_path(
        "michiyasunaga/BioLinkBERT-base"
    )
except LookupError:
    _ENCODINGS_PATH = None


@pytest.mark.integration
@pytest.mark.skipif(
    _ENCODINGS_PATH is None or not _ENCODINGS_PATH.exists(),
    reason=(
        f"needs precomputed encodings at {_ENCODINGS_PATH} "
        "(run precompute-encodings); local/self-hosted only"
    ),
)
def test_all_entity_classes_in_splits():
    assert _ENCODINGS_PATH is not None
    dataset = brenda_dataset(schema=BRENDA_SCHEMA, encodings=_ENCODINGS_PATH)

    # For each split in the dataset, check that all entity classes appear at
    # least once. The class targets are built in schema order, which is the
    # order `class_map` carries, so column `i` is the `i`-th class name.
    for split_name, split_dataset in dataset.data.items():
        found_per_class = {cls: False for cls in dataset.class_map}

        # Iterate over samples until all classes have been found.
        for i in range(len(split_dataset)):
            try:
                sample = split_dataset[i]
            except KeyError:
                # pmid present in the split frame but absent from the encodings
                # HDF5; a coverage check simply skips it.
                continue
            # sample["classes"] is a multi-hot array over the class columns
            for column, cls in enumerate(found_per_class):
                if sample["classes"][column] == 1:
                    found_per_class[cls] = True
            if all(found_per_class.values()):
                print(f"{split_name} OK")
                break
        assert all(
            found_per_class.values()
        ), f"Split '{split_name}' missing some entity classes: {found_per_class}"
