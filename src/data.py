import collections
import dataclasses
import math
import re

import datasets
import numpy
import s800classed
import sklearn
import torch
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils


@dataclasses.dataclass
class DatasetConfig:
    data: datasets.DatasetDict | DataLoader
    tokenizer: transformers.PreTrainedTokenizerBase
    classes: list[str]
    null_index: int
    class_weights: torch.Tensor
    max_length: int


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


def species800(upsample: bool = True) -> datasets.Dataset:
    dataset = s800classed.load()
    if upsample:
        dataset["train"] = utils.upsample(dataset["train"], "Strain")

    return dataset


def only_species_and_strains800(upsample: bool = True) -> datasets.Dataset:
    dataset = species800(upsample=upsample)
    dataset = dataset.map(
        lambda sample: keep_only(["Bacteria", "Strain"], sample, oos=True)
    )

    return dataset


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
                lambda label: label
                if keep_regex.match(label)
                else replace(label, oos),
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
