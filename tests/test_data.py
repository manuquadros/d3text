from d3text.data import brenda_dataset


def test_all_entity_classes_in_splits():
    dataset = brenda_dataset()
    entity_index = dataset.entity_index

    # For each split in the dataset, check that all entity classes appear at least once.
    for split_name, split_dataset in dataset.data.items():
        # Create a flag for each entity class
        found_per_class = {cls: False for cls in dataset.class_map.keys()}

        # Iterate over samples until all classes have been found.
        for sample in split_dataset:
            # sample["entities"] is a multi-hot numpy array
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
        assert all(found_per_class.values()), (
            f"Split '{split_name}' missing some entity classes: {found_per_class}"
        )
