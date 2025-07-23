import collections
import csv
import math
import os
import pdb
from collections.abc import Iterable, Iterator
from functools import reduce
from itertools import chain, dropwhile, groupby, islice, tee
from pprint import pprint
from typing import Any, NamedTuple, Optional

import datasets
import torch
import transformers
from jaxtyping import Float, Integer, Num
from pydantic import BaseModel
from torch import Tensor
from transformers import BatchEncoding, PreTrainedTokenizerFast


class Token(NamedTuple):
    string: str
    offset: tuple[int, int]
    prediction: str
    gold_label: str | None = None
    prob: float | None = None


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
    tokenizer: transformers.PreTrainedTokenizerFast,
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
    offsets: Integer[Tensor, "x 2"], length: int
) -> Integer[Tensor, "length 2"]:
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


def log_config(filename: str, config: BaseModel, **metrics) -> None:
    config_dict = config.model_dump()
    for metric, value in metrics.items():
        config_dict[metric] = value

    newfile = not os.path.exists(filename) or os.stat(filename).st_size == 0

    with open(filename, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=config_dict.keys())
        if newfile:
            writer.writeheader()
        writer.writerow(config_dict)


def split_and_tokenize(
    tokenizer: PreTrainedTokenizerFast,
    inputs: str | list[str],
    max_length: int = 512,
    stride: int = 20,
) -> BatchEncoding:
    """Tokenize `inputs`, splitting them into segments of `max_length`

    :param tokenizer: Tokenizer
    :param inputs: Inputs
    :param max_length: Maximum length of the sequences into which `inputs` are
        to be split
    :param stride: Length of the overlap between the sequences into which
        `inputs` are to be split
    """
    if isinstance(inputs, str):
        inputs = [inputs]

    return tokenizer(
        inputs,
        padding="max_length",
        return_offsets_mapping=True,
        return_token_type_ids=False,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        stride=stride,
        return_overflowing_tokens=True,
    )


def aggregate_embeddings(
    embeddings: Num[Tensor, "sequence token embedding"],
    attention_mask: Integer[Tensor, "sequence token"],
    stride: int = 20,
) -> Num[Tensor, "token embedding"]:
    r"""Aggregate sequence embeddings along the token dimension.

    Within the overlap between two regions, assign token at position n to the
    first sequence if $n < \frac{\mathrm{stride}}{2}$, where $n = 0$ for the
    first token in the overlap region. When $n \ge \frac{\mathrm{stride}}{2}$,
    assign the token to the following sequence, if there is one. In this way,
    we select the embeddings containing the most balanced context (considering
    preceding and following tokens) for each token.

    :param embeddings: sequences to be aggregated.
    :param stride: size of the overlap between adjacent sequences, in tokens.
    """
    output_tensors = []
    end = -math.ceil(stride / 2)
    start = math.floor(stride / 2)

    for emb, mask in zip(embeddings, attention_mask):
        emb = emb[mask.bool()][1:-1]
        if not output_tensors:
            output_tensors.append(emb[:end])
        else:
            output_tensors.append(emb[start:end])

    output_tensors.append(emb[end:])

    return torch.concat(output_tensors)


def embed_document(
    doc: str,
    tokenizer: transformers.PreTrainedTokenizerFast,
    model: transformers.BertModel,
    stride: int = 20,
) -> Float[Tensor, "tokens features"]:
    """Compute token embeddings for `doc`."""
    encoding = split_and_tokenize(
        tokenizer=tokenizer, inputs=doc, stride=stride
    )
    input_ids = encoding["input_ids"].to(model.device)
    attention_mask = encoding["attention_mask"].to(model.device)
    with torch.no_grad():
        embedding = model(input_ids, attention_mask).last_hidden_state

    return aggregate_embeddings(
        embeddings=embedding, attention_mask=attention_mask
    )


def midhash(token: str) -> str:
    if token[:2] == "##":
        return "##"
    else:
        return ""


def tokenize_cased(
    original: str,
    tokenizer: PreTrainedTokenizerFast,
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


def strip_sequence(sequence: Iterable[Token]) -> Iterator[Token]:
    return (
        token
        for token in sequence
        if token.string not in ("[CLS]", "[SEP]", "[PAD]")
    )


def merge_predictions(
    preds: Iterable[Iterable[Token]],
    sample_mapping: Integer[Tensor, " splits"],
    stride: int,
) -> Iterator[list[Token]]:
    """Merge predictions for different segments of a large sequence.

    `stride`: for strings uv, stride is the number of suffix characters at the end of u
              that are repeated at the beginning of v.
    """
    mapping = iter(sample_mapping)

    for _, group in groupby(preds, lambda _: next(mapping)):
        # We add one to stride when indexing the continuation because of the [CLS]
        # character.
        yield list(
            reduce(
                lambda u, v: chain(
                    strip_sequence(u), islice(strip_sequence(v), stride, None)
                ),
                group,
            )
        )


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
    space = " " * (b.offset[0] - a.offset[1])
    text = a.string + space + "".join(dropwhile(lambda c: c == "#", b.string))
    offset = (
        (a.offset[0], b.offset[1])
        if a.offset is not None and b.offset is not None
        else None
    )
    return Token(text, offset, a.prediction, a.gold_label, a.prob)


def safe_concat(string: str | None, suffix: str | None) -> str | None:
    if isinstance(string, str) and isinstance(suffix, str):
        return string + suffix

    match (string, suffix):
        case (s, None) | (None, s):
            return s
        case _:
            return None


def concat(s: str, t: str, sep: str = "") -> str:
    if s and t:
        return s + sep + t
    else:
        return s + t


def debug() -> None:
    pdb.set_trace()


def debug_iter(it: Iterator) -> Iterator[Any]:
    copy, dummy = tee(it)
    print("\n", "-" * 20)
    pprint(list(dummy))

    return copy


def repr_sequence(sequence: Iterable[Token]) -> str:
    output: str
    last: int

    for token in sequence:
        try:
            output += " " * (token.offset[0] - last) + token.string
            last = token.offset[1]
        except NameError:
            output = token.string
            last = token.offset[1]

    return output
