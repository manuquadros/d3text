"""The corpus-agnostic half of the data layer.

The HDF5-backed split, the samplers and the multi-hot encoders live here; which
columns a corpus has, and how its IDs are indexed, is the business of a dataset
adapter under `d3text.datasets` — `datasets.brenda` builds the splits below
against a `Schema`.
"""

import collections
import dataclasses
import logging
import math
import os
import pathlib
import random
from collections.abc import Iterable, Iterator, Mapping, Sequence, Sized
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

from d3text.vocabulary import Vocabulary

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
    # Split name -> split. A dataset adapter always builds all three
    # BrendaDataset splits, and every consumer indexes by split name
    # (`dataset.data["train"]`); a wider union would not be indexable.
    data: dict[str, "BrendaDataset"]


@dataclasses.dataclass
class EntityRelationDataset(DatasetConfig):
    # The extraction target the adapter read this corpus against. It fixes the
    # column order of both heads, so the model takes its class and relation
    # names from here rather than from the corpus-shaped fields below — those
    # are derived from the very same schema, but nothing about their types says
    # so, and a mapping's key order is not a contract.
    schema: Schema

    # The three indices the model cannot re-derive and must be handed: which
    # column of the entity head an entity ID occupies, which entities make up
    # each class, and the one-hot class of each indexed entity. A dataset
    # adapter derives all three from its `Schema`, so their orders agree.
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
            # A pmid in the split frame but absent from the encodings file has
            # no length. Skip it, as `_getitems` and `TokenBudgetBatchSampler`
            # skip it: the dataset cannot serve the document either way, so
            # there is nothing to gain from ending the run over it.
            if ix not in self.lengths:
                continue
            if self.lengths[ix] < self.max_length:
                yield ix


class TokenBudgetBatchSampler(Sampler[list[int]]):
    """Batch by padded chunk count instead of by document count.

    Peak VRAM in a training step is linear in a batch's **padded** token count
    — measured at ~0.05 GiB per 1000 tokens for the entity head — and a batch
    pads to its longest document. `BatchSampler` fixes the document count
    instead, so with documents spanning 6 to 182 chunks the peak is a lottery
    over which ones the sampler happened to draw: a run trains for a while and
    then dies on an unlucky batch.

    This closes a batch when `(documents + 1) * longest` would exceed `budget`,
    which is the padded size the batch will actually allocate, not the sum of
    its documents' lengths. Batch size therefore varies: many short documents
    ride together, and a long one travels with few or no companions.

    A document longer than `budget` on its own is yielded **alone** rather than
    dropped or truncated — the least destructive reading, and the only one that
    trains on the same corpus as before. It can still exceed the budget; the
    budget bounds batches, and cannot bound a single document.

    No `__len__`: the number of batches depends on the order the inner sampler
    draws, which is not known until the epoch runs. Nothing in the pipeline
    asks a loader for its length; the training bars go through
    `d3text.progress.batch_progress`, which totals the split's documents
    instead of its batches for exactly this reason.
    """

    def __init__(
        self,
        sampler: Sampler[int] | Iterable[int],
        lengths: Mapping[int, int],
        budget: int,
    ) -> None:
        """
        :param sampler: draws the document indices, in the order to batch them.
            Typed as torch's own `BatchSampler` types it — only iteration is
            used, and `beartype_this_package` enforces the annotation at run
            time, so a bare iterable must be admitted explicitly.
        :param lengths: index -> the document's chunk count, as
            `BrendaDataset.sequence_lengths` provides it. It need not cover
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
            # no length. Skip it, as `_getitems` skips it: charging it a
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


def get_batch_loader(
    dataset: Dataset,
    batch_size: int,
    sampler: Sampler | None = None,
    max_chunks: int | None = None,
) -> DataLoader:
    """A loader over `dataset`, batched by document count or by chunk budget.

    :param batch_size: documents per batch. Ignored when `max_chunks` is set.
    :param sampler: draws document indices; a `RandomSampler` by default.
    :param max_chunks: switches to `TokenBudgetBatchSampler` with this budget,
        which bounds peak VRAM instead of batch size. Requires a dataset
        exposing `sequence_lengths`. `0` or `None` keeps the fixed document
        count — both, because `ModelConfig` carries the off state as `0` (TOML
        has no null) while the parameter itself is naturally optional.
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
        self.h5df = embeddings or encodings
        self._h5_handle: h5py.File | None = None
        self._h5_pid: int | None = None
        if loggers is not None:
            self.logger = loggers.logger(filename="brenda_dataset.log")
        else:
            self.logger = logging.getLogger("brenda_dataset")
        self.data = self._drop_empty_documents(
            df[["pubmed_id", "relations", "entities", "classes"]]
        )

    def _drop_empty_documents(self, data: pd.DataFrame) -> pd.DataFrame:
        """`data` without the rows whose encoding carries no token.

        A document whose text was whitespace tokenizes to one window holding
        `[CLS]` and `[SEP]` and nothing else, and `aggregate_embeddings` slices
        both away — so the model is handed a document of zero tokens, which the
        supported poolings variously score as a confident negative, turn into
        `NaN`, or refuse. Such a row is dropped from the split here, before any
        sampler can draw it: dropping it in `_getitems` instead would leave
        `evaluate`'s `batch_size=1` loader yielding an empty batch.

        The encodings already on disk hold such a document, so this reads the
        file rather than trusting the reader that wrote it. Only a one-window
        document can be empty — a second window exists only because the first
        one filled up — so all but a handful of rows cost a shape lookup and no
        read at all.

        A row whose pmid is absent from the file is left in place: it is
        `_getitems`' to skip, exactly as before. So is every row when there is
        no file to read at all — a split built for its labels alone indexes
        fine without one, and a run that means to fetch documents raises on the
        same missing path at its first batch.
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

        Cached rather than reopened per fetch, and keyed on the pid rather
        than installed by a `DataLoader`'s `worker_init_fn`: a loader with
        `num_workers=0` never runs one, and an HDF5 handle inherited across a
        fork shares the parent's file offset, so reading through it yields
        wrong bytes instead of raising.

        Not opened with `swmr=True`: nothing writes the file while a run reads
        it (`precompute-encodings` finishes first), and SWMR reads are only
        legal on a file the writer created for them.
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
    """Compute marginal frequency of each label in a column of the training dataset.

    The rows are summed one at a time rather than stacked into an
    ``[n_documents, n_labels]`` tensor, which holds the whole column in float32
    (``4 * n_documents * n_labels`` bytes) to produce a result one row wide.

    The returned values are bitwise those of the stacked mean: the column is
    multi-hot, so every column sum is a small integer, exact in float32 at any
    document count below 2**24 and therefore independent of summation order,
    and the final ``/ len(data)`` is the same division ``Tensor.mean`` applies
    (``* (1 / n)`` is *not* — it disagrees in the last place for most n).
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
    limit: int | None = None,
    vocabulary: Vocabulary | None = None,
    split_names: Sequence[str] = ("train", "val", "test"),
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

    return schema_driven(
        schema=BRENDA_SCHEMA,
        encodings=encodings,
        limit=limit,
        vocabulary=vocabulary,
        split_names=split_names,
    )


def get_class_weights(dataset: datasets.DatasetDict) -> torch.Tensor:
    """
    Compute a vector of class weights, as a function of their frequency
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
