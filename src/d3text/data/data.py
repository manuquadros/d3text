import collections
import dataclasses
import functools
import math
import re
from collections.abc import Iterable, Iterator, Mapping, Sized
from typing import Any

import datasets
import numpy
import pandas as pd
import sklearn
import torch
import transformers
from brenda_references import brenda_references
from d3text import utils
from jaxtyping import UInt8
from torch import Tensor
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler
from transformers import PreTrainedTokenizer


@dataclasses.dataclass
class DatasetConfig:
    data: datasets.DatasetDict | DataLoader | Dataset | dict[str, Dataset]
    tokenizer: transformers.PreTrainedTokenizerBase


@dataclasses.dataclass
class SequenceLabellingDataset(DatasetConfig):
    classes: list[str]
    null_index: int
    class_weights: torch.Tensor
    max_length: int


@dataclasses.dataclass
class EntityRelationDataset(DatasetConfig):
    entity_index: dict[str, dict[int | str, int]]
    class_map: dict[str, set[int]]


tokenizer = transformers.AutoTokenizer.from_pretrained(
    "michiyasunaga/BioLinkBERT-base"
)


class LengthLimitedRandomSampler(RandomSampler):
    """Random Sampler that only retrieved documents under a maximum length."""

    def __init__(
        self,
        data_source: Sized,
        replacement: bool = False,
        num_samples: int | None = None,
        max_length: int = 157,
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


def get_batch_loader(dataset: Dataset, batch_size: int) -> DataLoader:
    sampler = BatchSampler(
        sampler=LengthLimitedRandomSampler(dataset),
        batch_size=batch_size,
        drop_last=False,
    )
    return DataLoader(dataset=dataset, sampler=sampler)


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
        ),
    )


class BrendaDataset(Dataset):
    """Class defining a dataset split for and end-to-end relational model.

    Items are returned in the following format:
    {
        "encodings": BatchEncoding,
        "relations": list[Relation]
        "bacteria" : UInt8[Tensor, " indexes"]
        "enzymes" : UInt8[Tensor, " indexes"]
        "other_organisms" : UInt8[Tensor, " indexes"]
        "strains" : UInt8[Tensor, " indexes"]
    }
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ):
        self.data = df[
            [
                "pubmed_id",
                "abstract",
                "fulltext",
                "enzymes",
                "bacteria",
                "strains",
                "other_organisms",
                "relations",
                "entities",
            ]
        ]
        self.data["text"] = (
            self.data["abstract"] + self.data["fulltext"]
        ).astype("string")
        self.entcols = ("enzymes", "bacteria", "other_organisms", "strains")

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
        text = row["text"]
        sequences = utils.split_and_tokenize(tokenizer=tokenizer, inputs=text)

        return {
            "sequence": sequences,
            "entities": row["entities"],
            "relations": row["relations"],
        }

    def _getitems(self, idx: list[int]) -> list[dict[str, Any]]:
        rows = self.data.iloc[idx]
        text = rows["text"].tolist()
        sequences = utils.split_and_tokenize(tokenizer=tokenizer, inputs=text)

        docs = {}
        for idx, doc_idx in enumerate(sequences["overflow_to_sample_mapping"]):
            doc = docs.setdefault(doc_idx.item(), {})
            for key in ("input_ids", "attention_mask", "offset_mapping"):
                doc.setdefault(key, []).append(sequences[key][idx])

        return [
            {
                "sequence": docs[key],
                "entities": rows.iloc[key]["entities"],
                "relations": rows.iloc[key]["relations"],
            }
            for key in docs.keys()
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
    index: Mapping[int, int],
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


def brenda_dataset(limit: int | None = None) -> EntityRelationDataset:
    """Preprocess and return BRENDA dataset splits"""
    val = brenda_references.validation_data(noise=25, limit=limit)
    train = brenda_references.training_data(noise=50, limit=limit)
    test = brenda_references.test_data(noise=25, limit=limit)

    entity_cols = ["bacteria", "enzymes", "strains", "other_organisms"]

    entities = {
        col: set(
            col[:1] + str(entid)
            for entid in functools.reduce(lambda a, b: a + b, train[col])
        )
        for col in entity_cols
    }
    all_entities = set.union(*entities.values())
    entity_index = dict(zip(all_entities, range(len(all_entities))))

    def merge_entcols(row: pd.Series):
        ents = (
            entcol[:1] + str(ent)
            for entcol in entity_cols
            for ent in row[entcol]
        )
        return list(ents)

    def preprocess(df: pd.DataFrame):
        df["entities"] = df.apply(merge_entcols, axis=1)
        df["entities"] = multi_hot_encode_series(
            series=df["entities"], index=entity_index
        )

        return df

    return EntityRelationDataset(
        data={
            "train": BrendaDataset(preprocess(train), tokenizer=tokenizer),
            "val": BrendaDataset(preprocess(val), tokenizer=tokenizer),
            "test": BrendaDataset(preprocess(test), tokenizer=tokenizer),
        },
        entity_index=entity_index,
        class_map=entities,
        tokenizer=tokenizer,
    )


def preprocess_dataset(
    dataset: datasets.DatasetDict,
    tokenizer: transformers.PreTrainedTokenizerBase = tokenizer,
    validation_split: bool = False,
    test_split: bool = True,
) -> DatasetConfig:
    """
    Load dataset and tokenize it, keeping track of NERC tags.
    """

    dataset = dataset.map(
        lambda sample: utils.tokenize_and_align(
            sample, max_length=512, tokenizer=tokenizer
        ),
        remove_columns="tokens",
    )

    label_encoder = sklearn.preprocessing.LabelEncoder()
    label_encoder.fit(
        [
            label
            for split in dataset.values()
            for sample in split
            for label in sample["nerc_tags"]
        ]
        + ["#"]
    )

    dataset = dataset.map(
        lambda sample: {
            "nerc_tags": label_encoder.transform(sample["nerc_tags"])
        }
    )

    if not test_split:
        dataset["train"] = datasets.concatenate_datasets(
            (dataset["train"], dataset["test"])
        )
        del dataset["test"]

    if not validation_split:
        dataset["train"] = datasets.concatenate_datasets(
            (dataset["train"], dataset["validation"])
        )
        del dataset["validation"]

    return DatasetConfig(
        data=dataset.with_format("torch"),
        tokenizer=tokenizer,
        classes=label_encoder.classes_,
        null_index=numpy.where(label_encoder.classes_ == "#")[0][0],
        class_weights=get_class_weights(dataset),
        max_length=512,
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
