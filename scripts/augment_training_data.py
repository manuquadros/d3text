"""Add articles not sampled by generate_dataset to the training data."""

from importlib import resources

import pandas as pd
from brenda_references.docdb import BrendaDocDB

DATA_DIR = resources.files("brenda_references") / "data"

if __name__ == "__main__":
    pubmed_ids: set[int] = set()
    for dataset_path in (
        "training_data.csv",
        "test_data.csv",
        "validation_data.csv",
    ):
        with resources.as_file(DATA_DIR / dataset_path) as csv:
            dataset = pd.read_csv(csv)
            pubmed_ids |= set(dataset["pubmed_id"])

    print("Loading articles...")
    with BrendaDocDB() as docdb:
        data = docdb.fulltext_articles()
        data = tuple(
            doc
            for doc in data
            if int(doc["pubmed_id"]) not in pubmed_ids
            and (doc["strains"] or not doc["bacteria"])
        )

    new_data = pd.DataFrame(data)
    with resources.as_file(DATA_DIR / "training_data.csv") as train_path:
        backup = train_path.with_suffix(".bak")
        sampled_data = pd.read_csv(train_path, index_col=0)
        sampled_data.to_csv(backup)

        if not new_data.columns.equals(sampled_data.columns):
            msg = (
                f"{train_path} and the augmentd data don't have"
                " the same columns."
            )
            raise ValueError(msg)

        pd.concat((new_data, sampled_data), axis=0, ignore_index=True).to_csv(
            train_path
        )

        print(
            f"Augmented data saved to {train_path}. Backup saved to {backup}."
        )
