import collections
import csv
import itertools
import os
from collections.abc import Iterable, Iterator
from functools import reduce
from typing import NamedTuple, Optional

import datasets
import torch
import transformers
from datamodel import Response
from jaxtyping import Float, Int
from pydantic import (BaseModel, EmailStr, NonNegativeFloat, NonNegativeInt,
                      PositiveFloat, PositiveInt)
from torch import Tensor
from transformers import BatchEncoding, PreTrainedTokenizer


class Token(NamedTuple):
    string: str
    offset: tuple[int, int]
    prediction: str
    gold_label: Optional[str] = None


class ModelConfig(BaseModel):
    classes: list[str] = []
    optimizer: str = "adam"
    lr: PositiveFloat = 0.0003
    lr_scheduler: str = ""
    dropout: NonNegativeFloat = 0
    hidden_layers: NonNegativeInt = 1
    hidden_size: NonNegativeInt = 32
    normalization: str = "layer"
    batch_size: PositiveInt = 32
    num_epochs: PositiveInt = 100
    patience: NonNegativeInt = 5
    base_model: str = "michiyasunaga/BioLinkBERT-base"


class Pointer(NamedTuple):
    token: str
    prediction: str
    gold_label: Optional[str] = None


def merge_tokens(
    tokens: Iterable[str],
    predictions: Iterable[str],
    gold_labels: Iterable[str] | None = None,
) -> dict[str, list[str]]:
    """
    Merge the BPE tokens in `tokens` and combine the labels accordingly.

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

    counter = reduce(
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


def split_and_tokenize(
    tokenizer: PreTrainedTokenizer,
    inputs: str | list[str],
    stride: int = 50,
) -> BatchEncoding:
    if isinstance(inputs, str):
        inputs = [inputs]

    return tokenizer(
        inputs,
        padding=True,
        return_offsets_mapping=True,
        return_token_type_ids=False,
        max_length=512,
        truncation=True,
        stride=stride,
        return_overflowing_tokens=True,
    )


def midhash(token: str) -> str:
    if token[:2] == "##":
        return "##"
    else:
        return ""


def tokenize_cased(
    original: str,
    tokenizer: PreTrainedTokenizer,
    stride: int = 50,
    clssep: bool = False,
) -> Iterator[list[str]]:
    tokenized = split_and_tokenize(tokenizer, original, stride)

    for sequence, offsets in zip(
        tokenized["input_ids"], tokenized["offset_mapping"]
    ):
        sequence = tokenizer.convert_ids_to_tokens(sequence)
        yield (
            ["[CLS]"]
            + [
                midhash(token) + original[offset[0] : offset[1]]
                for token, offset in zip(sequence, offsets)
                if token not in ("[CLS]", "[SEP]", "[PAD]")
            ]
            + ["[SEP]"]
        )


def merge_predictions(
    preds: Iterable[list[Token]],
    sample_mapping: Int[Tensor, " splits"],
    stride: int,
) -> list[list[Token]]:
    mapping = iter(sample_mapping)
    result = []

    for _, group in itertools.groupby(preds, lambda _: next(mapping)):
        result.append(list(reduce(lambda u, v: u + v[stride:], group)))

    return result


def merge_off_tokens(tokens: Iterable[Token]) -> list[Token]:
    """
    Merge the BPE tokens in `tokens` and combine their labels accordingly.

    The function will remove [CLS], [SEP] and [PAD] tokens.
    """
    merged_tokens: list[Token] = []

    for token in tokens:
        if token.string not in ("[SEP]", "[CLS]"):
            if token.string.startswith("##"):
                merged_tokens[-1] = token_merge(merged_tokens[-1], token)
            else:
                merged_tokens.append(token)

    return merged_tokens


def token_merge(a: Token, b: Token) -> Token:
    text = a.string + b.string[2:]
    offset = (
        (a.offset[0], b.offset[1])
        if a.offset is not None and b.offset is not None
        else None
    )
    return Token(text, offset, a.prediction, a.gold_label)


def safe_concat(string: Optional[str], suffix: Optional[str]) -> str | None:
    match (string, suffix):
        case (None, None):
            return None
        case (s, None) | (None, s):
            return s
        case (s, t):
            return s + t
