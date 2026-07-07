"""Module providing functions for sampling references from the dataset."""

import math
from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd
from gme.gme import GreedyMaximumEntropySampler

pd.options.mode.copy_on_write = True


def relation_records(doc: Mapping[str, Any]) -> list[dict[str, str]]:
    """Build relation records from a document."""
    pmid = doc["pubmed_id"]
    records = []

    if "relations" not in doc:
        return []

    for predicate, argpairs in doc["relations"].items():
        for args in argpairs:
            subj = str(args["subject"])
            obj = str(args["object"])

            if predicate == "HasSpecies":
                subj_prefix = "str_"
                obj_prefix = "bac_"
            else:
                obj_prefix = "enz_"
                if subj in doc["bacteria"]:
                    subj_prefix = "bac_"
                elif subj in doc["strains"]:
                    subj_prefix = "str_"
                else:
                    subj_prefix = "oos_"

            records.append(
                {
                    "pubmed_id": pmid,
                    "predicate": predicate,
                    "subject": subj_prefix + subj,
                    "object": obj_prefix + obj,
                }
            )

    return records


def build_sampling_df(docs: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Build DataFrame where each row is a relation found in the database."""
    rows = (
        relation_record
        for doc in docs
        for relation_record in relation_records(doc)
        if doc["pubmed_id"]
    )

    return pd.DataFrame(rows).astype(
        dtype={
            "pubmed_id": "int32",
            "predicate": "category",
            "subject": "string",
            "object": "string",
        }
    )


class GMESampler:
    def __init__(
        self,
        data: Iterable[Mapping[str, Any]],
        item_column: str = "pubmed_id",
        on_columns: list[str] | None = None,
    ) -> None:
        """Initialize sampler.

        :param data: Iterable containing records to sample from
        """
        self.on_columns = on_columns or ["subject", "object"]
        self.item_column = item_column
        self._sampler = GreedyMaximumEntropySampler(
            selector="dutopia", binarised=False
        )
        self._data = data

        self._sampling_df = build_sampling_df(self._data)

    def sample(
        self,
        n: int,
        approx: int = 0,
    ) -> pd.DataFrame:
        """Sample `N` items from the dataset, without replacement."""
        sample = self._sampler.sample(
            data=self._sampling_df,
            N=min(n, len(self._data)),
            item_column=self.item_column,
            on_columns=self.on_columns,
            approx=approx,
        )

        # Update the sampling_df so there is no overlap between splits.
        self._sampling_df = self._sampling_df[
            ~self._sampling_df[self.item_column].isin(sample[self.item_column])
        ]

        return sample

    def dataset_splits(
        self,
        training: float = 0.7,
        validation: float = 0.15,
    ) -> dict[str, pd.DataFrame]:
        """Split dataset into training, validation and test, using GME sampling.

        :param data: Iterable with records to sample from
        :param training: The ratio of training samples to dataset size
        :param validation: The ration of validation samples to dataset size

        :return: Dictionary mapping dataset split to DataFrames containing
            `pubmed_id` and entropy values for each entity category
        """

        def get_sample(size: int) -> pd.DataFrame:
            """Retrieve a sample with the required `size`.

            The number of samples to take from the dataset is estimated
            to guarantee that the best document in the sample is in the top-20
            documents of the whole dataset, with 90% confidence.
            """
            approx = round(
                math.log(1 - 0.9) / math.log(1 - 20 / len(self._data))
            )
            return self.sample(n=size, approx=approx)

        test_ratio = 1.0 - training - validation
        val_size = round(len(self._data) * validation)
        test_size = round(len(self._data) * test_ratio)
        train_size = len(self._data) - val_size - test_size

        train = get_sample(size=train_size)
        val = get_sample(size=val_size)
        test = get_sample(size=test_size)

        dfs = {
            "validation": val,
            "test": test,
            "training": train,
        }

        for split, dataset in dfs.items():
            last_row = dataset.iloc[-1]
            print(f"{split}\n {last_row['subject']}, {last_row['object']}")

        return dfs
