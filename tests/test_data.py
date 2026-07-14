import pytest
from d3text.datasets.brenda import BRENDA_SCHEMA, brenda_dataset

# Documents are fetched in batches rather than one by one: the list path is
# also the only one that tolerates a pmid present in the DataFrame but absent
# from the HDF5 (it skips the row, where the int path raises).
BATCH = 64


@pytest.mark.data
def test_all_entity_classes_in_splits(brenda_encodings):
    """Every entity class must be represented in every split.

    A class absent from a split means the split cannot supervise it: the head
    would be trained, validated or tested against a column that is always zero.
    """
    dataset = brenda_dataset(BRENDA_SCHEMA, encodings=str(brenda_encodings))
    entity_index = dataset.entity_index

    for split_name, split in dataset.data.items():
        found = dict.fromkeys(dataset.class_map, False)

        for start in range(0, len(split), BATCH):
            batch = split[list(range(start, min(start + BATCH, len(split))))]
            for sample in batch:
                for cls, possible_entities in dataset.class_map.items():
                    if found[cls]:
                        continue
                    for entity in possible_entities:
                        index = entity_index.get(entity)
                        if index is not None and sample["entities"][index] == 1:
                            found[cls] = True
                            break
            if all(found.values()):
                break

        missing = [cls for cls, seen in found.items() if not seen]
        assert not missing, (
            f"split {split_name!r} has no document for entity "
            f"class(es): {missing}"
        )
