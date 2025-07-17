import collections
import dataclasses
import functools
import math
import os
import pathlib
import random
import re
from collections.abc import Iterable, Iterator, Mapping, Sized
from typing import Any

import datasets
import h5py
import hdf5plugin  # noqa: F401
import loggers
import numpy
import pandas as pd
import sklearn
import torch
import xmlparser
from brenda_references import brenda_references
from jaxtyping import UInt8
from ordered_set import OrderedSet
from torch import Tensor
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    RandomSampler,
    Sampler,
)

os.environ["TOKENIZERS_PARALLELISM"] = "true"
DATA_DIR = pathlib.Path(__file__).parent.parent.parent.parent / "data"

g = torch.manual_seed(42)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


@dataclasses.dataclass
class DatasetConfig:
    data: datasets.DatasetDict | DataLoader | Dataset | dict[str, Dataset]


@dataclasses.dataclass
class SequenceLabellingDataset(DatasetConfig):
    classes: list[str]
    null_index: int
    class_weights: torch.Tensor
    max_length: int


@dataclasses.dataclass
class EntityRelationDataset(DatasetConfig):
    entity_index: dict[str, int]
    class_map: dict[str, set[str]]


class LengthLimitedRandomSampler(RandomSampler):
    """Random Sampler that only retrieved documents under a maximum length."""

    def __init__(
        self,
        data_source: Sized,
        replacement: bool = False,
        num_samples: int | None = None,
        max_length: int = 1000,
    ) -> None:
        """Initialize LengthLimitedRandomSampler.

        :param data_source: Data to sample from
        :param replacement: Sample with replacement flag
        :param num_samples: number of samples to draw.
            Default is len(data_source)
        :param max_length: maximum length of document in terms of number of
            512-token sized sequences"""
        super().__init__(
            data_source=data_source,
            replacement=replacement,
            num_samples=num_samples,
        )
        self.max_length = max_length

    def __iter__(self) -> Iterator[list[int]]:
        for ix in super().__iter__():
            if (
                self.data_source[ix]["sequence"]["input_ids"].shape[0]
                < self.max_length
            ):
                yield ix


def get_batch_loader(
    dataset: Dataset, batch_size: int, sampler: Sampler | None = None
) -> DataLoader:
    if sampler is None:
        sampler = RandomSampler(
            data_source=dataset, replacement=False, generator=g
        )

    sampler = BatchSampler(
        sampler=sampler,
        batch_size=batch_size,
        drop_last=False,
    )
    return DataLoader(
        dataset=dataset,
        sampler=sampler,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )


def get_loader(
    dataset_config: DatasetConfig,
    split: str,
    batch_size: int,
    sampler: torch.utils.data.sampler.Sampler | None = None,
) -> DatasetConfig:
    if isinstance(dataset_config.data, DataLoader):
        return dataset_config
    else:
        data_split = dataset_config.data[split]

    return dataclasses.replace(
        dataset_config,
        data=torch.utils.data.DataLoader(
            dataset=data_split,
            batch_size=batch_size,
            sampler=sampler,
            pin_memory=True,
        ),
    )


class BrendaDataset(Dataset):
    """Class defining a dataset split for and end-to-end relational model.

    Items are returned in the following format:
    {
        "sequence": BatchEncoding
                    | Float[Tensor, "chunk token embedding"],
        "relations": list[Relation]
        "entities": UInt8[Tensor, " indexes"]
    }
    """

    def __init__(
        self,
        df: pd.DataFrame,
        embeddings: os.PathLike | None = None,
        encodings: os.PathLike | None = None,
    ):
        self.data = df[
            [
                "pubmed_id",
                "relations",
                "entities",
            ]
        ]
        self.h5df = embeddings or encodings
        self.logger = loggers.logger(filename="brenda_dataset.log")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int | list[int]):
        """Return the requested idx.

        The tokenized sequences are returned batched into their respective
        documents.
        """
        if isinstance(idx, list):
            return self._getitems(idx)

        row = self.data.iloc[idx]

        with h5py.File(self.h5df, "r") as f:
            group = f[str(row["pubmed_id"])]
            if hasattr(group, "keys"):
                sequence = {key: group[key][()] for key in group.keys()}
            else:
                sequence = group[()]

        return {
            "id": self.data.iloc[idx]["pubmed_id"],
            "sequence": sequence,
            "entities": row["entities"],
            "relations": row["relations"],
        }

    def _getitems(self, idx: list[int]) -> list[dict[str, Any]]:
        seqdict = {}
        with h5py.File(self.h5df, "r") as f:
            for ix in idx:
                pubmed_id = str(self.data.iloc[ix]["pubmed_id"])
                group = f[pubmed_id]
                try:
                    if hasattr(group, "keys"):
                        seqdict[ix] = {
                            key: group[key][()] for key in group.keys()
                        }
                    else:
                        seqdict[ix] = group[()]
                except TypeError:
                    msg = f"No data for pmid {pubmed_id} from {self.h5df}"
                    self.logger.error(msg)

        return [
            {
                "id": self.data.iloc[ix]["pubmed_id"],
                "sequence": seqdict[ix],
                "doc_id": torch.tensor(
                    [doc_id] * seqdict[ix]["input_ids"].shape[0],
                    dtype=torch.uint8,
                ),
                "entities": self.data.iloc[ix]["entities"],
                "relations": self.data.iloc[ix]["relations"],
            }
            for doc_id, ix in enumerate(idx)
            if ix in seqdict
            if seqdict[ix]
        ]


def flatten_entity_indices(
    class_index: dict[str, dict[int, int]],
) -> dict[int, int]:
    """Create global label index across all entity types."""
    offset = 0
    global_index = {}
    for cl, index in class_index.items():
        for k, v in index.items():
            global_index[k] = v + offset
        offset += len(index)
    return global_index


def index_tensor(
    values: Iterable[str],
    index: Mapping[str, int],
) -> UInt8[Tensor, " indices"]:
    """Encode `values` according to `index`.

    The values in the series are assumed to correspond to keys of the index.

    :param values: The Iterable to be encoded
    :param index: Mapping from values to indices of the encoding vector.
    """
    # Keep only known indices
    known_indices = [index[x] for x in values if x in index]

    nclasses = max(index.values()) + 1
    output = torch.zeros(nclasses, dtype=torch.uint8)

    if known_indices:
        output.scatter_(0, torch.tensor(known_indices), 1)

    return output


def multi_hot_encode_series(
    series: pd.Series,
    index: Mapping[str, int],
) -> pd.Series:
    """Encode `series` according to `index`.

    The values in the series are assumed to correspond to keys of the index.

    :param series: The Series to be encoded.
    :param index: Mapping from values to indices of the encoding vector.
    :return: Pandas series with values converted to numpy ndarrays.
    """
    return series.apply(
        lambda values: index_tensor(values=values, index=index).numpy()
    )


def multi_hot_encode_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
    indices: Mapping[str, Mapping[int, int]],
):
    for col in columns:
        df[col] = multi_hot_encode_series(
            series=df[col],
            index=indices[col],
        )

    return df


def brenda_dataset(
    limit: int = 0,
    encodings: str = "prajjwal1_bert_mini-zstd-22-encodings.hdf5",
) -> EntityRelationDataset:
    """Preprocess and return BRENDA dataset splits"""
    val = brenda_references.validation_data(noise=450)
    train = brenda_references.training_data(noise=450)
    test = brenda_references.test_data(noise=50)

    entity_cols: list[str] = [
        "strains",
        "bacteria",
        "other_organisms",
        "enzymes",
    ]

    entities: dict[str, set[str]] = {
        col: set(
            col[:3] + str(entid)
            for entid in functools.reduce(lambda a, b: a + b, train[col])
        )
        for col in entity_cols
    }

    if limit:
        val = val.truncate(after=limit - 1)
        train = train.truncate(after=limit - 1)
        test = test.truncate(after=limit - 1)

    all_entities = OrderedSet.union(*entities.values())
    entity_index: dict[str, int] = dict(
        zip(all_entities, range(len(all_entities)))
    )

    def preprocess(df: pd.DataFrame):
        df["entities"] = multi_hot_encode_series(
            series=df["entities"], index=entity_index
        )
        df["fulltext"] = df["fulltext"].apply(xmlparser.remove_tags)

        # TODO: add classes column, with a multi_hot tensor, specifying whether
        # each class appears in the document

        # TODO: add classes column, with a multi_hot tensor, specifying whether
        # each class appears in the document

        return df

    encodings_path = pathlib.Path(DATA_DIR / encodings)
    return EntityRelationDataset(
        data={
            "train": BrendaDataset(preprocess(train), encodings=encodings_path),
            "val": BrendaDataset(preprocess(val), encodings=encodings_path),
            "test": BrendaDataset(preprocess(test), encodings=encodings_path),
        },
        entity_index=entity_index,
        class_map=entities,
    )


def get_class_weights(dataset: datasets.DatasetDict) -> torch.Tensor:
    """
    Compute a vector of class weights, as a function of their frequency
    """

    print("Getting class weights")
    counter: collections.Counter = collections.Counter()

    for split in dataset:
        for sample in dataset[split]:
            counter += collections.Counter(sample["nerc_tags"])

    total = counter.total()
    counter = collections.Counter(
        {
            idx: freq if freq > 100 else counter.most_common(1)[0][1]
            for idx, freq in counter.items()
        }
    )

    weights = sorted(
        (
            (idx, (1 / math.log(frequency)) * (total / len(counter)))
            for idx, frequency in counter.items()
        )
    )
    weights = sklearn.preprocessing.minmax_scale(
        [weight[1] for weight in weights]
    )
    weights = torch.nn.functional.softmax(torch.Tensor(weights), dim=-1) + 1

    return weights


def keep_only(keep: list[str], sample: dict, oos: bool) -> dict:
    """
    Return a new `sample` with only the labels specified in `keep`.
    """
    keep_regex = re.compile(
        rf"[BI]-({'|'.join(keep)})" r"|(?<![^\/])O+(?![^\/])"
    )
    # keep_regex will match any label contained in `keep` plus any sequence
    # of the letter O, as long as it is not part of another label.

    sample["nerc_tags"] = [
        "/".join(
            map(
                lambda label: (
                    label if keep_regex.match(label) else replace(label, oos)
                ),
                tag.split("/"),
            )
        )
        for tag in sample["nerc_tags"]
    ]

    return sample


def replace(label: str, oos: bool) -> str:
    if oos:
        return label[:2] + "OOS"
    else:
        return "O"
