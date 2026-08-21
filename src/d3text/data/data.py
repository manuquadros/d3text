import collections
import dataclasses
import functools
import logging
import math
import os
import pathlib
import random
from collections.abc import Iterable, Iterator, Mapping, Sized
from typing import Any, cast

import datasets
import h5py
import hdf5plugin  # noqa: F401
import numpy

try:
    import loggers  # type: ignore[import-not-found]
except ModuleNotFoundError:  # `loggers` is an optional external helper
    loggers = None
import pandas as pd
import sklearn
import torch
from jaxtyping import Float, UInt8
from torch import Tensor
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    RandomSampler,
    Sampler,
)

DATA_DIR = pathlib.Path(__file__).parent.parent.parent.parent / "data"

# The samplers below draw from torch's global generator, which
# `runtime.configure()` seeds at start-up. Naming that generator here rather
# than seeding it (`torch.manual_seed` returns this very object) keeps a
# library import from resetting the RNG of whoever imported us.
g = torch.default_generator


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


@dataclasses.dataclass
class DatasetConfig:
    # Split name -> split. The only producer, `brenda_dataset`, always builds
    # the three BrendaDataset splits, and every consumer indexes by split name
    # (`dataset.data["train"]`); a wider union would not be indexable.
    data: dict[str, "BrendaDataset"]


@dataclasses.dataclass
class EntityRelationDataset(DatasetConfig):
    entity_index: dict[str, int]
    class_map: dict[str, set[str]]
    class_matrix: Float[Tensor, "entities classes"]


class LengthLimitedRandomSampler(RandomSampler):
    """Random Sampler that only retrieved documents under a maximum length."""

    def __init__(
        self,
        data_source: "BrendaDataset",
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
        # Taken once, here, rather than per index per epoch: the filter needs
        # one number per row, but `dataset[ix]` opens the HDF5 file and reads
        # the whole document to get it, so an epoch used to read the corpus
        # twice over.
        self.lengths = data_source.sequence_lengths

    def __iter__(self) -> Iterator[int]:
        for ix in super().__iter__():
            # An index missing from the mapping is a pmid absent from the HDF5
            # file: the lookup raises KeyError here, just as indexing the
            # dataset did.
            if self.lengths[ix] < self.max_length:
                yield ix


def get_batch_loader(
    dataset: Dataset, batch_size: int, sampler: Sampler | None = None
) -> DataLoader:
    if sampler is None:
        sampler = RandomSampler(
            data_source=cast(Sized, dataset), replacement=False, generator=g
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
        if loggers is not None:
            self.logger = loggers.logger(filename="brenda_dataset.log")
        else:
            self.logger = logging.getLogger("brenda_dataset")

    def __len__(self):
        return len(self.data)

    @functools.cached_property
    def sequence_lengths(self) -> dict[int, int]:
        """Row position -> the number of sequences stored for that document.

        Read from the HDF5 metadata in a single pass, so a length-filtering
        sampler never has to materialise a document to learn its length.
        Computed on first access rather than in `__init__` because almost no
        run asks: every run builds all three splits, and only a
        `LengthLimitedRandomSampler` needs the lengths.

        A row whose pmid is absent from the file — or stored without
        `input_ids` — is absent from the mapping, mirroring the skip in
        `_getitems`.
        """
        lengths: dict[int, int] = {}
        with h5py.File(self.h5df, "r") as f:
            for ix, pubmed_id in enumerate(self.data["pubmed_id"]):
                group = f.get(str(pubmed_id))
                if isinstance(group, h5py.Group) and "input_ids" in group:
                    lengths[ix] = group["input_ids"].shape[0]
                else:
                    msg = f"No data for pmid {pubmed_id} from {self.h5df}"
                    self.logger.error(msg)

        return lengths

    def __getitem__(self, idx: int | list[int]):
        """Return the requested idx.

        The tokenized sequences are returned batched into their respective
        documents. A single int yields one document dict; both index types go
        through `_getitems`, so they return the identical schema (including
        `doc_id`) and share the missing-pmid guard.
        """
        if isinstance(idx, list):
            return self._getitems(idx)

        items = self._getitems([idx])
        if not items:
            raise KeyError(
                f"No data for pmid {self.data.iloc[idx]['pubmed_id']} "
                f"in {self.h5df}"
            )
        return items[0]

    def _getitems(self, idx: list[int]) -> list[dict[str, Any]]:
        seqdict = {}
        with h5py.File(self.h5df, "r") as f:
            for ix in idx:
                pubmed_id = str(self.data.iloc[ix]["pubmed_id"])
                try:
                    group = f[pubmed_id]
                    if hasattr(group, "keys"):
                        seqdict[ix] = {
                            key: group[key][()] for key in group.keys()
                        }
                    else:
                        seqdict[ix] = group[()]
                except (KeyError, TypeError):
                    # KeyError: pmid in the DataFrame but absent from the HDF5
                    # file; TypeError: empty/scalar group. Skip either — the
                    # `if ix in seqdict` filter below drops the row.
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
    encodings: str | os.PathLike[str],
    limit: int = 0,
) -> EntityRelationDataset:
    """The BRENDA dataset splits, indexed under `BRENDA_SCHEMA`.

    Transitional shim over `d3text.datasets.brenda.brenda_dataset`, which takes
    the schema explicitly; it keeps `train`, `tune` and `evaluate` working
    while they are moved onto a `Schema` of their own (SCHEMA-03).

    Imported inside the call rather than at module scope because the adapter
    imports *this* module for `BrendaDataset` and the encoding helpers.
    """
    from d3text.datasets.brenda import BRENDA_SCHEMA
    from d3text.datasets.brenda import brenda_dataset as schema_driven

    return schema_driven(schema=BRENDA_SCHEMA, encodings=encodings, limit=limit)


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
    scaled = sklearn.preprocessing.minmax_scale(
        [weight[1] for weight in weights]
    )
    return torch.nn.functional.softmax(torch.Tensor(scaled), dim=-1) + 1
