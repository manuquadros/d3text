import collections
import dataclasses
import re

import datasets
import numpy
import s800classed
import sklearn
import torch
import transformers
from torch.utils.data import DataLoader

from entities import utils


@dataclasses.dataclass
class Dataset:
    train: DataLoader
    validation: DataLoader
    test: DataLoader
    tokenizer: transformers.PreTrainedTokenizerBase
    classes: list[str]
    null_index: int
    class_weights: list[int]


tokenizer = transformers.AutoTokenizer.from_pretrained("michiyasunaga/BioLinkBERT-base")


def load_dataset(
    dataset: datasets.DatasetDict,
    tokenizer: transformers.PreTrainedTokenizerBase = tokenizer,
) -> Dataset:
    """
    Load dataset and tokenize it, keeping track of NERC tags.
    """

    # We need to know the maximum length of a tokenized sequence for padding.
    max_length = max(
        len(sample["input_ids"])
        for split in dataset.map(
            lambda s: tokenizer(s["tokens"], is_split_into_words=True)
        ).values()
        for sample in split
    )

    dataset = dataset.map(
        lambda sample: utils.tokenize_and_align(sample, max_length, tokenizer),
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
        lambda sample: {"nerc_tags": label_encoder.transform(sample["nerc_tags"])}
    )

    return Dataset(
        train=dataset["train"].with_format("torch"),
        validation=dataset["validation"].with_format("torch"),
        test=dataset["test"].with_format("torch"),
        tokenizer=tokenizer,
        classes=label_encoder.classes_,
        null_index=numpy.where(label_encoder.classes_ == "#")[0][0],
        class_weights=get_class_weights(dataset),
    )


def get_class_weights(dataset: datasets.DatasetDict) -> torch.Tensor:
    """
    Compute a vector of class weights, as a function of their frequency
    """

    print("Getting class weights")
    counter: collections.Counter = collections.Counter()
    from tqdm import tqdm

    for split in dataset:
        for sample in tqdm(dataset[split]):
            counter += collections.Counter(sample["nerc_tags"])

    counter[0] = 0
    weights = sorted(((idx, frequency) for idx, frequency in counter.items()))
    weights = sklearn.preprocessing.robust_scale([weight[1] for weight in weights])

    return torch.Tensor(weights)


def species800(upsample: bool = True) -> datasets.Dataset:
    dataset = s800classed.load()
    if upsample:
        dataset["train"] = utils.upsample(dataset["train"], "Strain")

    return dataset


def only_species_and_strains800(upsample: bool = True) -> datasets.Dataset:
    dataset = species800(upsample=upsample)
    dataset = dataset.map(
        lambda sample: keep_only(["Species", "Strain"], sample, oos=True)
    )

    return dataset


def keep_only(keep: list[str], sample: dict, oos: bool) -> dict:
    """
    Return a new `sample` with only the labels specified in `keep`.
    """
    keep_regex = re.compile(rf"[BI]-({'|'.join(keep)})" r"|(?<![^\/])O+(?![^\/])")
    # keep_regex will match any label contained in `keep` plus any sequence
    # of the letter O, as long as it is not part of another label.

    sample["nerc_tags"] = [
        "/".join(
            map(
                lambda label: label if keep_regex.match(label) else replace(label, oos),
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
