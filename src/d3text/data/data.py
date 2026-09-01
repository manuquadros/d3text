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

from d3text import encodings_store, utils

# The batch contract itself. `d3text.models` never imports this module, so the
# edge does not close a cycle; a `TYPE_CHECKING` import would, since beartype
# resolves the annotation at call time and cannot see a name that is not there.
from d3text.models.model_types import BatchItem

logger = logging.getLogger(__name__)

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
    # Split name -> split. The only producer, `d3text.datasets.brenda.
    # brenda_dataset`, always builds the three BrendaDataset splits, and every
    # consumer indexes by split name (`dataset.data["train"]`); a wider union
    # would not be indexable.
    data: dict[str, "BrendaDataset"]


@dataclasses.dataclass
class EntityRelationDataset(DatasetConfig):
    entity_index: dict[str, int]
    class_map: dict[str, set[str]]
    class_matrix: Float[Tensor, "entities classes"]


class LengthLimitedRandomSampler(RandomSampler):
    """Random sampler restricted to documents under a maximum length."""

    def __init__(
        self,
        data_source: "BrendaDataset",
        replacement: bool = False,
        num_samples: int | None = None,
        max_length: int = 1000,
    ) -> None:
        """Restrict sampling to documents of at most `max_length` sequences.

        :param data_source: the dataset to sample from.
        :param replacement: whether to sample with replacement.
        :param num_samples: how many to draw; the dataset's size by default.
        :param max_length: longest document to admit, in 512-token sequences.
        """
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
            # A pmid in the split frame but absent from the encodings file has
            # no length. Skip it, as `__getitems__` and `TokenBudgetBatchSampler`
            # skip it: the dataset cannot serve the document either way, so
            # there is nothing to gain from ending the run over it.
            if ix not in self.lengths:
                continue
            if self.lengths[ix] < self.max_length:
                yield ix


class TokenBudgetBatchSampler(Sampler[list[int]]):
    """Batch by padded chunk count instead of by document count.

    Peak VRAM is linear in a batch's *padded* token count and a batch pads to
    its longest document, so a fixed document count makes the peak a lottery
    over which documents the sampler drew. A document longer than `budget` on
    its own is yielded alone rather than dropped or truncated. No `__len__`:
    the batch count depends on the order the inner sampler draws.
    """

    def __init__(
        self,
        sampler: Sampler[int] | Iterable[int],
        lengths: Mapping[int, int],
        budget: int,
    ) -> None:
        """Batch `sampler`'s indices under a padded-token `budget`.

        :param sampler: draws the document indices, in the order to batch them.
            Typed as torch's own `BatchSampler` types it, so a bare iterable
            must be admitted explicitly.
        :param lengths: index -> the document's chunk count. It need not cover
            every index the sampler draws: one it omits is a document the
            dataset cannot serve, and is skipped rather than batched.
        :param budget: the largest `documents * longest` a batch may reach.
        """
        if budget < 1:
            raise ValueError(f"budget must be positive, got {budget}")
        self.sampler = sampler
        self.lengths = lengths
        self.budget = budget

    def __iter__(self) -> Iterator[list[int]]:
        batch: list[int] = []
        longest = 0
        for index in self.sampler:
            # A pmid in the split frame but absent from the encodings file has
            # no length. Skip it, as `__getitems__` skips it: charging it a
            # fabricated length would only reserve budget for a document that
            # is then dropped out of the batch.
            if index not in self.lengths:
                continue
            length = self.lengths[index]
            padded = max(longest, length)
            if batch and (len(batch) + 1) * padded > self.budget:
                yield batch
                batch, longest, padded = [], 0, length
            batch.append(index)
            longest = padded
        if batch:
            yield batch


def collate_documents(batch: list[dict[str, Any]]) -> list[BatchItem]:
    """Turn the rows a dataset yields into the batch the models consume.

    A batch *is* a list of documents, with no batch dimension anywhere: two
    documents hold different numbers of 512-token chunks, so their `sequence`
    tensors do not stack, yet `default_collate` adds a phantom leading
    singleton regardless. A field the row does not carry is passed over rather
    than invented.

    :param batch: the rows to collate.
    :return: one `BatchItem` per document.
    """
    return [
        cast(
            BatchItem,
            {
                key: convert(doc[key])
                for key, convert in (
                    ("id", torch.as_tensor),
                    ("doc_id", _identity),
                    ("sequence", _tensor_values),
                    ("entities", torch.as_tensor),
                    ("classes", torch.as_tensor),
                    ("relations", _tensor_relations),
                )
                if key in doc
            },
        )
        for doc in batch
    ]


def _identity(value: Any) -> Any:
    return value


def _tensor_values(sequence: Mapping[str, Any]) -> dict[str, Tensor]:
    return {key: torch.as_tensor(value) for key, value in sequence.items()}


def _tensor_relations(relations: Any) -> list[dict[tuple[str, str], Tensor]]:
    """The document's relation dicts, labels as tensors.

    A document the corpus holds no relations for carries a null cell rather
    than an empty list, and that is no relations rather than a malformed one.
    """
    if not isinstance(relations, Iterable) or isinstance(relations, str):
        return []
    return [
        {args: torch.as_tensor(label) for args, label in pairs.items()}
        for pairs in relations
    ]


def get_batch_loader(
    dataset: Dataset,
    batch_size: int,
    sampler: Sampler | None = None,
    max_chunks: int | None = None,
) -> DataLoader:
    """A loader over `dataset`, batched by document count or by chunk budget.

    :param dataset: the split to load.
    :param batch_size: documents per batch. Ignored when `max_chunks` is set.
    :param sampler: draws document indices; a `RandomSampler` by default.
    :param max_chunks: switches to `TokenBudgetBatchSampler` with this budget,
        which bounds peak VRAM instead of batch size, and requires a dataset
        exposing `sequence_lengths`. `0` and `None` both keep the fixed
        document count, since `ModelConfig` carries the off state as `0` (TOML
        has no null) while the parameter itself is naturally optional.
    :return: the loader.
    """
    if sampler is None:
        sampler = RandomSampler(
            data_source=cast(Sized, dataset), replacement=False, generator=g
        )

    if max_chunks:
        sampler = TokenBudgetBatchSampler(
            sampler=sampler,
            lengths=cast("BrendaDataset", dataset).sequence_lengths,
            budget=max_chunks,
        )
    else:
        sampler = BatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
        )
    return DataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        collate_fn=collate_documents,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )


class BrendaDataset(Dataset):
    """One split of the corpus, indexed for an end-to-end relational model.

    An item carries its tokenized sequences batched into their document, its
    relations and its multi-hot entity vector.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        encodings: os.PathLike | None = None,
        base_model: str | None = None,
    ):
        self.h5df = encodings
        self._h5_handle: h5py.File | None = None
        self._h5_pid: int | None = None
        if loggers is not None:
            self.logger = loggers.logger(filename="brenda_dataset.log")
        else:
            self.logger = logging.getLogger("brenda_dataset")
        self._check_encodings_provenance(base_model)
        self.data = self._drop_empty_documents(
            df[["pubmed_id", "relations", "entities", "classes"]]
        )

    def _check_encodings_provenance(self, base_model: str | None) -> None:
        """Refuse an encodings file this run cannot read as it was written.

        `None` is a caller with no base model to check against, and then
        nothing is checked. An unstamped file is warned about once and read
        anyway, on the same continuity argument `d3text.checkpoint.load` makes.
        The stamped `max_length` is deliberately not compared: windows are
        stitched off the attention mask, so a shorter window still reconstructs
        each document token-for-token.

        :param base_model: the model this run will feed the ids to.
        :raises ValueError: if the store records another base model — the ids
            come from another vocabulary, which is a confident wrong answer
            rather than a shape error — or another stride than
            `aggregate_embeddings` will merge its windows under.
        """
        if base_model is None or self.h5df is None:
            return
        if not os.path.exists(self.h5df):
            return

        with h5py.File(self.h5df, "r") as f:
            recorded = encodings_store.read_provenance(f)

        if recorded is None:
            self.logger.warning(
                "%s does not record which model or stride tokenized it, so "
                "its ids cannot be attributed to %s and its windows are "
                "merged at the assumed stride of %d; reading it anyway.",
                self.h5df,
                base_model,
                utils.WINDOW_STRIDE,
            )
            return

        if recorded.base_model != base_model:
            msg = (
                f"{self.h5df} was tokenized by {recorded.base_model} and "
                f"this run's base model is {base_model}. Their input ids "
                f"come from different vocabularies, so the embedding layer "
                f"would read every id under the wrong one; rebuild the "
                f"encodings with `precompute-encodings`."
            )
            raise ValueError(msg)

        if recorded.stride != utils.WINDOW_STRIDE:
            msg = (
                f"{self.h5df} was tokenized with a stride of "
                f"{recorded.stride} and this run merges its windows at "
                f"{utils.WINDOW_STRIDE}. Every seam would be stitched at the "
                f"wrong offset — tokens duplicated or dropped once per "
                f"window, with the row count and every shape still "
                f"plausible; rebuild the encodings with "
                f"`precompute-encodings`."
            )
            raise ValueError(msg)

    def _drop_empty_documents(self, data: pd.DataFrame) -> pd.DataFrame:
        """`data` without the rows whose encoding carries no token.

        A document whose text was whitespace tokenizes to `[CLS]` and `[SEP]`
        alone, both of which the aggregation slices away, leaving a document of
        zero tokens the poolings variously mis-score, NaN on, or refuse.
        Dropped here rather than in `__getitems__`, which would leave
        `evaluate`'s `batch_size=1` loader yielding an empty batch. A row whose
        pmid the file does not hold is left in place, as is every row when
        there is no file to read.
        """
        if self.h5df is None or not os.path.exists(self.h5df):
            return data

        empty: set[int] = set()
        with h5py.File(self.h5df, "r") as f:
            for ix, pubmed_id in enumerate(data["pubmed_id"]):
                group = f.get(str(pubmed_id))
                if not isinstance(group, h5py.Group):
                    continue
                mask = group.get("attention_mask")
                if not isinstance(mask, h5py.Dataset) or mask.shape[0] != 1:
                    continue
                if int(numpy.asarray(mask[0]).sum()) <= 2:
                    empty.add(ix)
                    self.logger.warning(
                        "%s encodes to no token of its own in %s; "
                        "dropping it from the split",
                        pubmed_id,
                        self.h5df,
                    )

        if not empty:
            return data
        return data.iloc[[ix for ix in range(len(data)) if ix not in empty]]

    def __len__(self):
        return len(self.data)

    @property
    def _h5(self) -> h5py.File:
        """This process's own read handle on the encodings file.

        Keyed on the pid rather than installed by a `worker_init_fn`, which a
        loader with `num_workers=0` never runs: a handle inherited across a
        fork shares the parent's file offset and yields wrong bytes instead of
        raising. Not opened `swmr=True`, since nothing writes the file while a
        run reads it and SWMR reads are only legal on a file the writer created
        for them.
        """
        pid = os.getpid()
        if self._h5_pid != pid:
            # Inherited from a parent process: dropped unread, never shared.
            self._h5_handle = None
        if self._h5_handle is None:
            self._h5_handle = h5py.File(self.h5df, "r")
            self._h5_pid = pid
        return self._h5_handle

    def close(self) -> None:
        """Release this process's handle. The next access reopens it."""
        if self._h5_handle is not None:
            self._h5_handle.close()
            self._h5_handle = None

    def __getstate__(self) -> dict[str, Any]:
        # `h5py.File` is unpicklable, and `DataLoader` pickles the dataset to
        # reach a worker under the `spawn` start method — so a dataset that
        # had already been read from would make `num_workers > 0` unusable.
        return {**self.__dict__, "_h5_handle": None, "_h5_pid": None}

    @functools.cached_property
    def sequence_lengths(self) -> dict[int, int]:
        """Row position -> the number of sequences stored for that document.

        Read from the HDF5 metadata in one pass, so a length-filtering sampler
        never materialises a document to learn its length, and computed on
        first access because almost no run asks. A row whose pmid is absent
        from the file, or stored without `input_ids`, is absent here too.
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
        """The requested document or documents.

        Both index types go through `__getitems__`, so they return the
        identical schema (including `doc_id`) and share the missing-pmid guard.

        :param idx: one row position, or several.
        :return: one document dict, or a list of them.
        """
        if isinstance(idx, list):
            return self.__getitems__(idx)

        items = self.__getitems__([idx])
        if not items:
            raise KeyError(
                f"No data for pmid {self.data.iloc[idx]['pubmed_id']} "
                f"in {self.h5df}"
            )
        return items[0]

    def __getitems__(self, idx: list[int]) -> list[dict[str, Any]]:
        """Read several documents in one pass over the HDF5 file.

        Torch's map-dataset fetcher calls this when the loader batches. A pmid
        the file does not hold is dropped and the batch comes back short rather
        than failing.

        :param idx: the row positions to read.
        :return: one dict per document the file holds.
        """
        seqdict = {}
        f = self._h5
        for ix in idx:
            pubmed_id = str(self.data.iloc[ix]["pubmed_id"])
            try:
                group = f[pubmed_id]
                if hasattr(group, "keys"):
                    seqdict[ix] = {key: group[key][()] for key in group.keys()}
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
    """Marginal frequency of each label in a column of the dataset.

    Summed one row at a time rather than stacked, which would hold the whole
    column in float32 to produce a result one row wide. The values are bitwise
    those of the stacked mean: the column is multi-hot, so every sum is a small
    integer exact in float32 below 2**24 and independent of summation order.

    :param dataset: the split to count over.
    :param column: the multi-hot column to count.
    :return: one frequency per label.
    """
    data = dataset.data[column]

    total: Tensor | None = None
    for e in data:
        if torch.is_tensor(e):
            row = e.float()
        else:
            row = torch.tensor(e, dtype=torch.float32)

        if total is None:
            # Not `total = row`: `Tensor.float()` returns *self* for a float32
            # tensor, so accumulating into it would rewrite the frame's labels.
            total = torch.zeros_like(row)
        elif total.shape != row.shape:
            raise ValueError(
                f"Ragged label column {column!r}: {tuple(row.shape)} "
                f"after {tuple(total.shape)}"
            )
        total += row

    if total is None:
        raise ValueError(f"Cannot compute frequencies over empty {column!r}")

    freq = total / len(data)
    return freq.clamp(min=1e-5, max=1 - 1e-5)


def index_tensor(
    values: Iterable[str],
    index: Mapping[str, int],
) -> UInt8[Tensor, " indices"]:
    """Encode `values` according to `index`.

    :param values: the values to encode, assumed to be keys of `index`.
    :param index: value -> its position in the encoding vector.
    :return: the multi-hot vector.
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

    :param series: the values to encode, assumed to be keys of `index`.
    :param index: value -> its position in the encoding vector.
    :return: the series, each value replaced by its multi-hot array.
    """
    return series.apply(
        lambda values: index_tensor(values=values, index=index).numpy()
    )


def get_class_weights(dataset: datasets.DatasetDict) -> torch.Tensor:
    """Class weights as a function of each class's frequency.

    :param dataset: the splits to count over.
    :return: one weight per class.
    """

    logger.info("Getting class weights")
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
