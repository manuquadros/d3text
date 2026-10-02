import dataclasses
import logging
import os
import pathlib
import random
from collections.abc import Iterable, Iterator, Mapping, Sized
from typing import Any, cast

import numpy

try:
    import loggers  # type: ignore[import-not-found]
except ModuleNotFoundError:  # `loggers` is an optional external helper
    loggers = None
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    RandomSampler,
    Sampler,
)

from d3text import encodings_store, token_labels, utils

# The batch contract itself. `d3text.models` never imports this module, so the
# edge does not close a cycle; a `TYPE_CHECKING` import would, since beartype
# resolves the annotation at call time and cannot see a name that is not there.
from d3text.constraints import FREQUENCY_CLAMP_EPS, NonNegative, Positive
from d3text.models.model_types import BatchItem

logger = logging.getLogger(__name__)

DATA_DIR = pathlib.Path(__file__).parent.parent.parent.parent / "data"


# Torch's global generator, which `runtime.configure()` seeds; naming it
# rather than seeding it here keeps an import from resetting the caller's RNG.
g = torch.default_generator


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


@dataclasses.dataclass
class EntityRelationDataset:
    # Split name -> split, as `brenda_dataset` builds it; every consumer
    # indexes by split name, so a wider union would not be indexable.
    data: dict[str, "BrendaDataset"]
    class_map: dict[str, set[str]]


class LengthLimitedRandomSampler(RandomSampler):
    """Random sampler restricted to documents under a maximum length."""

    def __init__(
        self,
        data_source: "BrendaDataset",
        replacement: bool = False,
        num_samples: int | None = None,
        max_length: Positive = 1000,
    ) -> None:
        """Restrict sampling to documents shorter than `max_length` sequences.

        :param data_source: the dataset to sample from.
        :param replacement: whether to sample with replacement.
        :param num_samples: how many to draw; the dataset's size by default.
        :param max_length: first document length to exclude, in windows
            of `utils.WINDOW_LENGTH` tokens.
        """
        super().__init__(
            data_source=data_source,
            replacement=replacement,
            num_samples=num_samples,
        )
        self.max_length = max_length
        # Taken once rather than per index: `dataset[ix]` would read the
        # whole document from the store just to learn its length.
        self.lengths = data_source.sequence_lengths

    def __iter__(self) -> Iterator[int]:
        for ix in super().__iter__():
            # A pmid absent from the encodings store has no length; skip it,
            # as `__getitems__` does, since the dataset cannot serve it.
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
        budget: Positive,
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
            # A pmid absent from the encodings store has no length; skip it,
            # as `__getitems__` does, rather than reserve budget for it.
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
    documents hold different numbers of `utils.WINDOW_LENGTH`-token chunks,
    so their `sequence` tensors do not stack, yet `default_collate` adds a
    phantom leading singleton regardless. A field the row does not carry is
    passed over rather than invented.

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
    batch_size: Positive,
    sampler: Sampler | None = None,
    max_chunks: NonNegative | None = None,
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
        # No `pin_memory`: every field is re-stacked into a fresh pageable
        # tensor downstream before the H2D copy, so a pin here is wasted.
        worker_init_fn=seed_worker,
        generator=g,
    )


class BrendaDataset(Dataset):
    """One split of the corpus, indexed for an end-to-end relational model.

    An item carries its tokenized sequences batched into their document, its
    relations and its multi-hot class vector.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        encodings: os.PathLike | None = None,
        base_model: str | None = None,
        tokenizer: token_labels.TokenizerStamp | None = None,
    ):
        self.encodings = encodings
        self._store_handle: encodings_store.EncodingsStore | None = None
        self._store_pid: int | None = None
        self._sequence_lengths: dict[int, int] | None = None
        if loggers is not None:
            self.logger = loggers.logger(filename="brenda_dataset.log")
        else:
            self.logger = logging.getLogger("brenda_dataset")
        self._check_encodings_provenance(base_model, tokenizer)
        columns = ["pubmed_id", "relations", "classes"]
        # `source` is optional: only `load_split` tags it, and without it the
        # whole-source check has nothing to group by.
        if "source" in df.columns:
            columns.append("source")
        self.data = self._drop_empty_documents(df[columns])
        # `.iloc[ix]` builds a fresh Series per call; these columns keep
        # `self.data`'s row order, so position `ix` still matches `.iloc`.
        self._pubmed_ids = self.data["pubmed_id"].to_numpy()
        self._relations = self.data["relations"].to_list()
        self._classes = self.data["classes"].to_list()

    def _check_encodings_provenance(
        self,
        base_model: str | None,
        tokenizer: token_labels.TokenizerStamp | None = None,
    ) -> None:
        """Refuse an encodings store this run cannot read as it was written.

        `None` is a caller with no base model to check against, and then
        nothing is checked; a missing store is left for the dataset's own
        reads to complain about. `encodings_store.check_provenance` makes
        the comparison itself, so another caller can run the same check
        without building a dataset around it. Given the label store's
        `tokenizer` stamp, the two stores' tokenizers are compared too, since
        each store's own name check passes when the tokenizer behind one name
        changed between their builds; an encodings store recording no digest
        is read with a warning.

        :param base_model: the model this run will feed the ids to.
        :param tokenizer: the label store's tokenizer stamp, or None for a
            run reading no label store.
        :raises ValueError: if the store records no provenance, another base
            model or another stride than `aggregate_embeddings` will merge
            its windows under; if `tokenizer` names another base model; or
            if it records another tokenizer digest than the store does.
        """
        if base_model is None or self.encodings is None:
            return
        if not os.path.exists(self.encodings):
            return

        recorded = encodings_store.check_provenance(
            self.encodings, base_model, utils.WINDOW_STRIDE
        )
        if tokenizer is None:
            return

        if tokenizer.base_model != base_model:
            msg = (
                f"The label store was tokenized by {tokenizer.base_model} "
                f"and this run's base model, which {self.encodings} was "
                f"tokenized by, is {base_model}. Their ids come from "
                f"different vocabularies; regenerate the label store with "
                f"`precompute-token-labels`."
            )
            raise ValueError(msg)

        if recorded.tokenizer_digest is None:
            self.logger.warning(
                "%s records no tokenizer digest, so whether its ids share the "
                "label store's vocabulary (tokenizer %s) is unverified; "
                "rebuilding it with `precompute-encodings` records one",
                self.encodings,
                tokenizer.digest[:12],
            )
        elif recorded.tokenizer_digest != tokenizer.digest:
            msg = (
                f"{self.encodings} and the label store were both tokenized "
                f"by {base_model}, but under tokenizers "
                f"{recorded.tokenizer_digest[:12]} and "
                f"{tokenizer.digest[:12]}: the tokenizer behind that name "
                f"changed between the two builds, so their ids come from "
                f"different vocabularies. Rebuild the stale store with "
                f"`precompute-encodings` or `precompute-token-labels`."
            )
            raise ValueError(msg)

    def _drop_empty_documents(self, data: pd.DataFrame) -> pd.DataFrame:
        """`data` without the rows whose encoding carries no token.

        A whitespace-only document encodes to `[CLS]` `[SEP]` alone, which
        aggregation slices away. Rows whose pmid the store lacks stay in place.
        The same walk fills `sequence_lengths` and the missing-row set
        `_refuse_if_a_source_is_wholly_missing` checks; why, and why the store
        is opened here rather than through `_store`, is in the data
        explanation.
        """
        if self.encodings is None or not os.path.exists(self.encodings):
            return data

        empty: set[int] = set()
        missing: set[int] = set()
        lengths: dict[int, int] = {}
        with encodings_store.EncodingsStore(self.encodings) as store:
            for ix, pubmed_id in enumerate(data["pubmed_id"]):
                windows = store.windows(str(pubmed_id))
                if windows is None:
                    missing.add(ix)
                    self.logger.error(
                        "No data for pmid %s from %s",
                        pubmed_id,
                        self.encodings,
                    )
                    continue

                if windows == 1:
                    encoding = store.get(str(pubmed_id))
                    if (
                        encoding is not None
                        and int(encoding["attention_mask"][0].sum()) <= 2
                    ):
                        empty.add(ix)
                        self.logger.warning(
                            "%s encodes to no token of its own in %s; "
                            "dropping it from the split",
                            pubmed_id,
                            self.encodings,
                        )
                        continue

                lengths[ix] = windows

        if "source" in data.columns:
            self._refuse_if_a_source_is_wholly_missing(data["source"], missing)

        survivors = [ix for ix in range(len(data)) if ix not in empty]
        self._sequence_lengths = {
            new_ix: lengths[old_ix]
            for new_ix, old_ix in enumerate(survivors)
            if old_ix in lengths
        }

        if not empty:
            return data
        return data.iloc[survivors]

    def _refuse_if_a_source_is_wholly_missing(
        self, sources: pd.Series, missing: set[int]
    ) -> None:
        """Refuse construction when every row of a configured source is gone.

        Scattered missing documents are left to `__getitems__`' per-row
        skip; a whole source missing is a corpus file the store was never
        built over, which no rate threshold catches once `limit` shrinks the
        source to a handful of rows.

        :param sources: `data`'s `source` column, positional — its row order
            matches `missing`'s positions.
        :param missing: row positions whose pmid the store holds no document
            for.
        :raises ValueError: naming every source none of whose rows the store
            held data for.
        """
        totals = sources.value_counts()
        gone = sources.iloc[sorted(missing)].value_counts()
        wholly_missing = sorted(
            source
            for source, total in totals.items()
            if gone.get(source, 0) == total
        )
        if wholly_missing:
            msg = (
                f"{self.encodings} holds no data for any row of source(s) "
                f"{wholly_missing}: it was never built over that corpus "
                "file. Rebuild the encodings with `precompute-encodings`."
            )
            raise ValueError(msg)

    def __len__(self):
        return len(self.data)

    @property
    def _store(self) -> encodings_store.EncodingsStore:
        """This process's own read handle on the encodings store.

        Keyed on the pid rather than set by a `worker_init_fn`, which a
        loader with `num_workers=0` never runs.
        """
        pid = os.getpid()
        if self._store_pid != pid:
            # Belongs to a parent process, whose LMDB environment must not
            # be used across the fork; the store reopens its own.
            self._store_handle = None
        if self._store_handle is None:
            if self.encodings is None:
                msg = "this dataset was built without an encodings store"
                raise KeyError(msg)
            self._store_handle = encodings_store.EncodingsStore(self.encodings)
            self._store_pid = pid
        return self._store_handle

    def close(self) -> None:
        """Release this process's handle. The next access reopens it."""
        if self._store_handle is not None:
            self._store_handle.close()
            self._store_handle = None

    def __getstate__(self) -> dict[str, Any]:
        # An LMDB environment is unpicklable, and `DataLoader` pickles the
        # dataset to reach a worker under the `spawn` start method — so a
        # dataset that had already been read from would make
        # `num_workers > 0` unusable.
        return {**self.__dict__, "_store_handle": None, "_store_pid": None}

    @property
    def sequence_lengths(self) -> dict[int, int]:
        """Row position -> the number of sequences stored for that document.

        Filled by `_drop_empty_documents`; read lazily only when `__init__`
        had no store to read. A row whose pmid the store lacks is absent here
        too.
        """
        if self._sequence_lengths is None:
            lengths: dict[int, int] = {}
            store = self._store
            for ix, pubmed_id in enumerate(self.data["pubmed_id"]):
                windows = store.windows(str(pubmed_id))
                if windows is not None:
                    lengths[ix] = windows
                else:
                    msg = f"No data for pmid {pubmed_id} from {self.encodings}"
                    self.logger.error(msg)
            self._sequence_lengths = lengths

        return self._sequence_lengths

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
                f"in {self.encodings}"
            )
        return items[0]

    def __getitems__(self, idx: list[int]) -> list[dict[str, Any]]:
        """Read several documents through one handle on the encodings store.

        Torch's map-dataset fetcher calls this when the loader batches. A pmid
        the store does not hold is dropped and the batch comes back short
        rather than failing.

        :param idx: the row positions to read.
        :return: one dict per document the store holds.
        """
        seqdict: dict[int, encodings_store.Encoding] = {}
        store = self._store
        for ix in idx:
            pubmed_id = str(self._pubmed_ids[ix])
            encoding = store.get(pubmed_id)
            if encoding is None:
                msg = f"No data for pmid {pubmed_id} from {self.encodings}"
                self.logger.error(msg)
                continue
            seqdict[ix] = encoding

        survivors = [ix for ix in idx if ix in seqdict]

        return [
            {
                "id": self._pubmed_ids[ix],
                "sequence": seqdict[ix],
                # Not uint8: `TokenBudgetBatchSampler` caps a batch's chunks,
                # not its documents, so a position can pass 255.
                "doc_id": torch.tensor(
                    [doc_id] * seqdict[ix]["input_ids"].shape[0],
                    dtype=torch.int64,
                ),
                "relations": self._relations[ix],
                "classes": self._classes[ix],
            }
            for doc_id, ix in enumerate(survivors)
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
    return freq.clamp(min=FREQUENCY_CLAMP_EPS, max=1 - FREQUENCY_CLAMP_EPS)
