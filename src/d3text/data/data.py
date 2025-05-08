import collections
import dataclasses
import math
import re
from collections.abc import Iterable, Generator, Mapping

import datasets
import numpy
import pandas as pd
import sklearn
import torch
import transformers
import utils
from brenda_references import brenda_references
from jaxtyping import UInt8, UInt64
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


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


class EntityRelationDataset(DatasetConfig):
    entity_index: dict[int | str, int]
    entity_class_map: dict[int | str, str]


tokenizer = transformers.AutoTokenizer.from_pretrained(
    "michiyasunaga/BioLinkBERT-base"
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
        ),
    )


class BrendaDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.data = df[
            [
                "pmc_id",
                "abstract",
                "full-text",
                "enzymes",
                "bacteria",
                "strains",
                "other_organisms",
                "relations",
            ]
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]


def index_tensor(
    values: Iterable[int] | UInt64[Tensor, "..."],
    index: Mapping[int, int],
    unk_index: int | None = None,
) -> UInt8[Tensor, "indices"]:
    """Encode `values` according to `index`.

    The values in the series are assumed to correspond to keys of the index.

    :param values: The Iterable to be encoded
    :param index: Mapping from values to indices of the encoding vector.
    :param unk_key: `index` key for unknown entities
    :raises: KeyError, when a value in the series does not exist on the index.
    """
    # check if unk_index would shadow an existing index
    if unk_index in index.values():
        msg = f"unk_index param, {unk_index}, should not be in the index"
        raise ValueError(msg)

    nclasses = max((*index.values(), unk_index or 0)) + 1

    if not isinstance(values, Tensor):
        values = torch.tensor(values, dtype=torch.uint64)

    if unk_index is not None:
        indexing = lambda x: index.get(x, unk_index)
    else:
        indexing = lambda x: index[x]

    indices = torch.tensor(numpy.vectorize(indexing)(values))
    zeros = torch.zeros(
        torch.Size((*indices.shape[:-1], nclasses)), dtype=torch.uint8
    )

    return zeros.scatter(dim=-1, index=indices, value=1)


def multi_hot_encode_series(
    series: pd.Series,
    index: Mapping[int | str, int],
    known_values: pd.Series | None = None,
    unknown_value: str = "unk",
):
    """Encode `series` according to `index`.

    The values in the series are assumed to correspond to keys of the index.

    :param series: The Series to be encoded.
    :param index: Mapping from values to indices of the encoding vector.
    :param known_values: If provided, the values that are not present in
        `known_values` are marked as unknown.
    :raises: KeyError, when a value in the series does not exist on the index.
    """
    pass


def brenda_dataset() -> EntityRelationDataset:
    """Preprocess and return BRENDA dataset splits"""
    val = brenda_references.validation_data()
    train = brenda_references.training_data()
    test = brenda_references.test_data()

    entity_cols = ("bacteria", "enzymes", "strains", "other_organisms")
    entity_classes = tuple(map(lambda col: col.rstrip("s"), entity_cols))
    entities_dfs: Generator[pd.DataFrame] = (
        pd.concat(
            objs=(
                pd.Series(df[col], name="entity"),
                pd.Series([col] * len(df[col]), name="class"),
            ),
            axis=1,
        )
        for df in (val, train, test)
        for col in entity_classes
    )
    unknowns = pd.DataFrame(
        {
            "entity": map(lambda entclass: "unk_" + entclass, entity_classes),
            "class": entity_classes,
        }
    )

    entities = pd.concat(entities_dfs, axis=0)
    entities = pd.concat((entities, unknowns), axis=0)

    entity_to_class = dict(zip(entities["entity"], entities["class"]))
    entity_index = dict(zip(entities["entity"], range(len(entities))))

    val = multi_hot_encode_columns(
        df=val, columns=entity_classes, index=entity_index, training_data=train
    )

    return EntityRelationDataset(
        data={
            "train": BrendaDataset(train),
            "val": BrendaDataset(val),
            "test": BrendaDataset(test),
        },
        entity_index=entity_index,
        entity_class_map=entity_to_class,
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
