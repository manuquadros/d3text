import collections
import dataclasses
import functools
import math
import os
import pathlib
import random
from collections.abc import Iterable, Iterator, Mapping, Sized
from numbers import Real
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
from jaxtyping import Float, UInt8
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
    class_matrix: Float[Tensor, "entities classes"]


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
        self.data = df[["pubmed_id", "relations", "entities", "classes"]]
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
            "classes": row["classes"],
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
                "classes": self.data.iloc[ix]["classes"],
            }
            for doc_id, ix in enumerate(idx)
            if ix in seqdict
            if seqdict[ix]
        ]


def compute_frequencies(dataset: BrendaDataset, column: str) -> torch.Tensor:
    """Compute marginal frequency of each label in a column of the training dataset."""
    data = dataset.data[column]

    all_labels = torch.stack(
        [
            torch.tensor(e, dtype=torch.float32)
            if not torch.is_tensor(e)
            else e.float()
            for e in data
        ]
    )

    freq = all_labels.mean(dim=0)
    return freq.clamp(min=1e-5, max=1 - 1e-5)


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


def brenda_dataset(
    encodings: str,
    limit: int = 0,
) -> EntityRelationDataset:
    """Preprocess and return BRENDA dataset splits"""
    train = brenda_references.training_data(noise=450, limit=limit)
    val = brenda_references.validation_data(noise=100)
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
    entity_to_class = {
        entity: cl for cl, ents in entities.items() for entity in ents
    }

    all_entities: OrderedSet[str] = OrderedSet.union(*entities.values())
    entity_index: dict[str, int] = dict(
        zip(all_entities, range(len(all_entities)))
    )

    class_matrix = numpy.zeros(
        shape=(len(all_entities), len(entity_cols)), dtype=numpy.float32
    )
    for entity_id, class_name in entity_to_class.items():
        ent_idx = entity_index[entity_id]
        class_idx = entity_cols.index(class_name)
        class_matrix[ent_idx, class_idx] = 1.0

    def preprocess(df: pd.DataFrame):
        df["entities"] = multi_hot_encode_series(
            series=df["entities"], index=entity_index
        )

        # computing class labels directly from entity_cols so we still count
        # classes for UNK entites in validation and evaluation
        cls_array = numpy.stack(
            [
                numpy.array(
                    [1 if len(row[col]) > 0 else 0 for col in entity_cols],
                    dtype=numpy.float32,
                )
                for _, row in df.iterrows()
            ]
        )
        df["relations"] = df["relations"].apply(_filter_relations)
        df["classes"] = pd.Series(list(cls_array))
        df["fulltext"] = df["fulltext"].apply(xmlparser.remove_tags)

        # TODO: add classes column, with a multi_hot tensor, specifying whether
        # each class appears in the document

        # TODO: add classes column, with a multi_hot tensor, specifying whether
        # each class appears in the document

        return df

    def _filter_relations(
        rels: list[dict[tuple[str, str], Iterable[Real]]],
    ) -> list[dict[tuple[str, str], Iterable[Real]]]:
        filtered = [
            {
                pair: rel
                for pair, rel in d.items()
                if all(argument in all_entities for argument in pair)
            }
            for d in rels
        ]

        if not filtered or not filtered[0]:
            # Prevent lists containing empty dicts
            return []
        return filtered

    encodings_path = pathlib.Path(DATA_DIR / encodings)
    return EntityRelationDataset(
        data={
            "train": BrendaDataset(preprocess(train), encodings=encodings_path),
            "val": BrendaDataset(preprocess(val), encodings=encodings_path),
            "test": BrendaDataset(preprocess(test), encodings=encodings_path),
        },
        entity_index=entity_index,
        class_map=entities,
        class_matrix=torch.tensor(class_matrix),
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
