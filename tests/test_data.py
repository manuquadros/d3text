import pytest
from d3text.data.data import DATA_DIR
from d3text.datasets.brenda import BRENDA_SCHEMA, brenda_dataset
from d3text.models.config import encodings

# Neither the ~300 MB BRENDA corpus nor a precomputed encodings HDF5 ships in
# the repo (the latter is produced by `precompute-encodings`), so this test can
# only run where the full pipeline has been set up locally / on a self-hosted
# runner. Guard on the encodings file — the thing `brenda_dataset` opens — so a
# fresh checkout and hosted CI skip cleanly instead of erroring.
_ENCODINGS = encodings["michiyasunaga/BioLinkBERT-base"]
_ENCODINGS_PATH = DATA_DIR / _ENCODINGS


@pytest.mark.integration
@pytest.mark.skipif(
    not _ENCODINGS_PATH.exists(),
    reason=(
        f"needs precomputed encodings at {_ENCODINGS_PATH} "
        "(run precompute-encodings); local/self-hosted only"
    ),
)
def test_all_entity_classes_in_splits():
    dataset = brenda_dataset(schema=BRENDA_SCHEMA, encodings=_ENCODINGS)
    entity_index = dataset.entity_index

    # For each split in the dataset, check that all entity classes appear at least once.
    for split_name, split_dataset in dataset.data.items():
        # Create a flag for each entity class
        found_per_class = {cls: False for cls in dataset.class_map.keys()}

        # Iterate over samples until all classes have been found.
        for i in range(len(split_dataset)):
            try:
                sample = split_dataset[i]
            except KeyError:
                # pmid present in the split frame but absent from the encodings
                # HDF5 (see BUG-03); a coverage check simply skips it.
                continue
            # sample["entities"] is a multi-hot array
            for cls, possible_entities in dataset.class_map.items():
                if found_per_class[cls]:
                    continue  # already found this class
                for ent in possible_entities:
                    idx = entity_index.get(ent)
                    # Check if the entity flag is on.
                    if idx is not None and sample["entities"][idx] == 1:
                        found_per_class[cls] = True
                        break
            if all(found_per_class.values()):
                print(f"{split_name} OK")
                break
        assert all(
            found_per_class.values()
        ), f"Split '{split_name}' missing some entity classes: {found_per_class}"
