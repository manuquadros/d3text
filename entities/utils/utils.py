import collections
import csv
import dataclasses
import functools
import os
import typing
from collections.abc import Iterable, Sequence

import datasets
import torch
import transformers
from jaxtyping import Int
from torch import Tensor


@dataclasses.dataclass
class ModelConfig:
    optimizer: str = "adam"
    lr: float = 0.0003
    lr_scheduler: str = ""
    dropout: float = 0
    hidden_layers: int = 1
    hidden_size: int = 32
    normalization: str = "layer"
    batch_size: int = 32
    num_epochs: int = 100
    patience: int = 5
    base_model: str = "michiyasunaga/BioLinkBERT-base"
    num_labels: int = 0


class Pointer(typing.NamedTuple):
    token: str
    prediction: str
    gold_label: typing.Optional[str] = None


class Token(typing.NamedTuple):
    string: str
    offset: tuple[int, int]
    prediction: str
    gold_label: typing.Optional[str] = None


def merge_tokens(
    tokens: Iterable, predictions: Iterable, gold_labels: Iterable | None = None
) -> dict[str, list[str]]:
    """
    Merge the BPE tokens in `tokens` and combine the tags accordingly.

    The function will remove [CLS], [SEP] and [PAD] tokens.
    """
    merged_tokens: list[str] = []
    merged_labels: list[str] = []
    tokens = iter(tokens)
    predictions = iter(predictions)

    if gold_labels is not None:
        merged_gold: list[str] = []
        gold_labels = iter(gold_labels)

    pointers = (
        Pointer(*tup)
        for tup in zip(
            *(it for it in (tokens, predictions, gold_labels) if it is not None)
        )
    )

    pointer = next(pointers)

    while pointer.token != "[SEP]":
        if pointer.token == "[CLS]":
            pointer = next(pointers)

        if pointer.token.startswith("##"):
            merged_tokens[-1] += pointer.token[2:]
        else:
            merged_tokens.append(pointer.token)
            merged_labels.append(pointer.prediction)
            if pointer.gold_label is not None:
                merged_gold.append(pointer.gold_label)

        pointer = next(pointers)

    result = {
        "tokens": merged_tokens,
        "predicted": merged_labels,
    }

    if gold_labels is not None:
        result["gold_labels"] = merged_gold

    return result


def tokenize_and_align(
    sample: dict[str, list[str]],
    max_length: int,
    tokenizer: transformers.PreTrainedTokenizerBase,
) -> dict[str, list[str]]:
    sequence = tokenizer(
        sample["tokens"],
        is_split_into_words=True,
        padding="max_length",
        max_length=max_length,
        truncation=True,
    )

    labels = []
    for idx in sequence.word_ids():
        if idx is None:
            labels.append("#")
        else:
            labels.append(sample["nerc_tags"][idx])

    return {"sequence": sequence, "nerc_tags": labels}


def pad_offsets(
    offsets: Int[Tensor, "x 2"], length: int
) -> Int[Tensor, "length 2"]:
    return torch.cat(
        [offsets, torch.Tensor([0, 0]).repeat(length - len(offsets), 1)]
    )


def upsample(data: datasets.Dataset, label: str) -> datasets.Dataset:
    print("Upsampling...")
    if not label.startswith("B-"):
        label = "B-" + label

    counter = functools.reduce(
        lambda a, b: a + b, (entity_counter(seq) for seq in data["nerc_tags"])
    )

    target = counter.most_common(1)[0][1] - counter[label]
    label_seqs = data.filter(lambda s: label in s["nerc_tags"])
    label_seqs = label_seqs.filter(
        lambda s: entity_counter(s["nerc_tags"]).most_common(1)[0][0] == label
    )

    new_samples: list[datasets.Dataset] = []

    while len(new_samples) < target:
        new_samples.append(label_seqs.shuffle().take(1))

    print(f"Adding {len(new_samples)} samples.")

    return datasets.concatenate_datasets(new_samples + [data])


def entity_counter(sequence: list[str]) -> collections.Counter:
    return collections.Counter(
        label for label in sequence if label.startswith("B")
    )


def log_config(filename: str, config: ModelConfig, **metrics) -> None:
    config_dict = dataclasses.asdict(config)
    for metric, value in metrics.items():
        config_dict[metric] = value

    newfile = not os.path.exists(filename) or os.stat(filename).st_size == 0

    with open("models.csv", "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=config_dict.keys())
        if newfile:
            writer.writeheader()
        writer.writerow(config_dict)
