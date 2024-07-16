import collections
import csv
import dataclasses
import functools
import os
from collections.abc import Iterable, Iterator

import datasets
import transformers


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
    patience: int = 6


def merge_tokens(
    tokens: Iterable, tags: Iterator, true_tags: Iterable
) -> dict[str, list[str]]:
    """
    Merge the BPE tokens in `tokens` and combine the tags accordingly.

    The function will remove [CLS], [SEP] and [PAD] tokens.
    """
    merged_tokens: list[str] = []
    merged_labels: list[str] = []
    merged_true: list[str] = []
    tokens = iter(tokens)
    tags = iter(tags)
    true_tags = iter(true_tags)

    cur_token, cur_pred, cur_true = next(tokens), next(tags), next(true_tags)

    while cur_token != "[SEP]":
        if cur_token == "[CLS]":
            cur_token, cur_pred, cur_true = next(tokens), next(tags), next(true_tags)

        if cur_token.startswith("##"):
            merged_tokens[-1] += cur_token[2:]
        else:
            merged_tokens.append(cur_token)
            merged_labels.append(cur_pred)
            merged_true.append(cur_true)

        cur_token, cur_pred, cur_true = (
            next(tokens),
            next(tags),
            next(true_tags),
        )

    return {"tokens": merged_tokens, "predicted": merged_labels, "true": merged_true}


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
    )

    labels = []
    for idx in sequence.word_ids():
        if idx is None:
            labels.append("#")
        else:
            labels.append(sample["nerc_tags"][idx])

    return {"sequence": sequence, "nerc_tags": labels}


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
    return collections.Counter(label for label in sequence if label.startswith("B"))


def log_model(filename, config, val_loss) -> None:
    config = dataclasses.asdict(config)
    config["val_loss"] = val_loss
    newfile = not os.path.exists(filename) or os.stat(filename).st_size == 0

    with open("models.csv", "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=config.keys())
        if newfile:
            writer.writeheader()
        writer.writerow(config)
