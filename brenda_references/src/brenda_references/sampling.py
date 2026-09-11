"""Module providing functions for sampling references from the dataset."""

import logging
import math
from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd
from gme.gme import GreedyMaximumEntropySampler

pd.options.mode.copy_on_write = True

logger = logging.getLogger(__name__)


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
        """Sample `n` items from the pool, without replacement.

        A zero-size or drained-pool draw never reaches gme, which cannot
        index an empty pool and types an empty draw as all floats.

        :param n: how many items to draw; fewer come back if the pool runs
            out first.
        :param approx: forwarded to `GreedyMaximumEntropySampler.sample`.
        :return: the drawn `item_column` values, typed like the pool's, each
            with the per-column entropies reached once it was added.
        """
        item_dtype = self._sampling_df[self.item_column].dtype

        if n <= 0 or self._sampling_df.empty:
            if n > 0:
                logger.warning(
                    "%d items requested from an exhausted pool; drawing none",
                    n,
                )
            return pd.DataFrame(
                {self.item_column: pd.Series(dtype=item_dtype)}
                | {
                    column: pd.Series(dtype="float64")
                    for column in self.on_columns
                }
            )

        sample = self._sampler.sample(
            data=self._sampling_df,
            N=min(n, len(self._data)),
            item_column=self.item_column,
            on_columns=self.on_columns,
            approx=approx,
        ).astype({self.item_column: item_dtype})

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
        """Split `data` into training, validation and test by GME sampling.

        :param data: the records to sample from.
        :param training: the ratio of training samples to dataset size.
        :param validation: the ratio of validation samples to dataset size.
        :return: split name -> a frame of `pubmed_id` and per-category
            entropies, empty for a split whose size rounds to 0 or that
            finds the pool already drained.
        :raises ValueError: if `training` or `validation` is outside
            `[0, 1]`, or their sum exceeds 1 (which would make the test
            share, and therefore its sample size, negative).
        """
        if not 0 <= training <= 1:
            raise ValueError(f"training must be within [0, 1], got {training}")
        if not 0 <= validation <= 1:
            raise ValueError(
                f"validation must be within [0, 1], got {validation}"
            )
        if training + validation > 1:
            raise ValueError(
                "training + validation must not exceed 1, got "
                f"{training + validation}"
            )

        def get_sample(size: int) -> pd.DataFrame:
            """Retrieve a sample with the required `size`.

            The number drawn is estimated so that the best document in the
            sample is in the whole dataset's top 20, with 90% confidence.
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
            if dataset.empty:
                continue
            last_row = dataset.iloc[-1]
            print(f"{split}\n {last_row['subject']}, {last_row['object']}")

        return dfs
