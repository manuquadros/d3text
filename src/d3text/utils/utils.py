import csv
import logging
import math
import os
import typing
from collections.abc import Iterable, Iterator
from functools import reduce
from itertools import chain, dropwhile, groupby, islice
from typing import NamedTuple, Optional

import torch
import transformers
from jaxtyping import Float, Integer, Num
from pydantic import BaseModel
from torch import Tensor
from transformers import BatchEncoding, PreTrainedTokenizerFast
from d3text.constraints import NonNegative, Positive
from d3text.runtime import select_amp_dtype

logger = logging.getLogger(__name__)


class Token(NamedTuple):
    """A token, its span in the source text, and what was predicted for it.

    `candidate_labels` is filled only when several wordlists matched a span
    equally well, and `prediction` is then `"AMBIGUOUS"`: such a span is
    evidence for no type, and a consumer building targets must drop it. An
    unambiguous match leaves the set empty, so no label is stored twice.
    """

    string: str
    offset: tuple[int, int]
    prediction: str
    gold_label: str | None = None
    prob: float | None = None
    candidate_labels: frozenset[str] = frozenset()


class Pointer(NamedTuple):
    token: str
    prediction: str
    gold_label: Optional[str] = None


def merge_tokens(
    tokens: Iterable[str],
    predictions: Iterable[str],
    gold_labels: Iterable[str] | None = None,
) -> dict[str, list[str]]:
    """Merge the BPE tokens in `tokens` and combine the labels accordingly.

    :param tokens: the BPE tokens, `[CLS]`, `[SEP]` and `[PAD]` included.
    :param predictions: one label per token.
    :param gold_labels: one gold label per token, if any.
    :return: the merged tokens and labels, special tokens removed.
    """
    merged_tokens: list[str] = []
    merged_labels: list[str] = []
    merged_gold: list[str] = []
    tokens = iter(tokens)
    predictions = iter(predictions)

    if gold_labels is not None:
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
    max_length: Positive,
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


def log_config(filename: str, config: BaseModel, **metrics) -> None:
    """Append `config` and `metrics` as one row of `filename`'s results CSV.

    When the file already has a header, the row is written under that
    header's column order rather than the current keys' order, so a column
    that was reordered since the file was started still lands under the
    right name.

    :param filename: path to the results CSV; given a header if missing or
        empty.
    :param config: config whose fields become columns.
    :param metrics: extra columns to log alongside `config`'s fields.
    :raises ValueError: if the file already has a header whose column set
        differs from the set of columns being written.
    """
    config_dict = config.model_dump()
    for metric, value in metrics.items():
        config_dict[metric] = value

    newfile = not os.path.exists(filename) or os.stat(filename).st_size == 0

    header: list[str] | None = None
    if not newfile:
        with open(filename, newline="") as csvfile:
            header = next(csv.reader(csvfile), None)
        if header is not None:
            existing, current = set(header), set(config_dict)
            if existing != current:
                missing = sorted(existing - current)
                extra = sorted(current - existing)
                msg = (
                    f"{filename}'s header does not match the columns "
                    f"being written (missing: {missing}, extra: {extra})."
                )
                raise ValueError(msg)

    fieldnames = header if header is not None else list(config_dict.keys())
    with open(filename, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if newfile:
            writer.writeheader()
        writer.writerow(config_dict)


def load_fast_tokenizer(base_model: str) -> PreTrainedTokenizerFast:
    """Load `base_model`'s tokenizer, requiring a fast one.

    `split_and_tokenize` and `embed_document` both depend on fast-only
    features, so a slow tokenizer is rejected where the base model is named
    rather than deeper in the pipeline.

    :param base_model: the checkpoint whose tokenizer to load.
    :return: the fast tokenizer.
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model)
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        msg = (
            f"{base_model} resolves to a slow tokenizer "
            f"({type(tokenizer).__name__}); a fast tokenizer is required."
        )
        raise TypeError(msg)
    return tokenizer


# The window geometry the whole pipeline assumes; the tokenization and
# `aggregate_embeddings` must agree on it to the token, and a store's recorded
# stride is compared against this one name rather than a copy of it.
WINDOW_LENGTH = 512
WINDOW_STRIDE = 20


def split_and_tokenize(
    tokenizer: PreTrainedTokenizerFast,
    inputs: str | list[str],
    max_length: Positive = WINDOW_LENGTH,
    stride: NonNegative = WINDOW_STRIDE,
    return_offsets_mapping: bool = True,
) -> BatchEncoding:
    """Tokenize `inputs`, splitting them into overlapping windows.

    :param tokenizer: the tokenizer to use.
    :param inputs: the text to tokenize.
    :param max_length: tokens per window.
    :param stride: tokens of overlap between adjacent windows.
    :param return_offsets_mapping: whether to compute the char-span offset
        mapping. Defaults to `True`; a caller that never reads the offsets
        should pass `False` to skip computing them.
    :return: the encoding, one row per window.
    """
    if isinstance(inputs, str):
        inputs = [inputs]

    return tokenizer(
        inputs,
        padding="max_length",
        return_offsets_mapping=return_offsets_mapping,
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
    stride: NonNegative = WINDOW_STRIDE,
) -> Num[Tensor, "token embedding"]:
    """Aggregate sequence embeddings along the token dimension.

    An overlap's first `stride / 2` tokens come from the earlier window, the
    rest from the later one, for the most balanced context. Window lengths
    are a host-side sum over the right-padded mask, so the kept positions
    are computed on the host and gathered in one `index_select`. On a CUDA
    device the index is copied from pinned memory without blocking the host;
    a device mask is rejected rather than moved, which would bring a sync
    back.

    :param embeddings: the windows to aggregate.
    :param attention_mask: which positions carry a real token, as a CPU
        tensor.
    :param stride: tokens of overlap between adjacent windows.
    :return: one row per token of the document.
    :raises ValueError: if `attention_mask` is not a CPU tensor, or is not
        right-padded (a contiguous run of 1s followed by 0s in every row).
    """
    if attention_mask.device.type != "cpu":
        msg = (
            "aggregate_embeddings requires a CPU attention_mask; got "
            f"{attention_mask.device}"
        )
        raise ValueError(msg)

    lengths = attention_mask.sum(dim=-1)
    positions = torch.arange(attention_mask.shape[-1])
    right_padded = (positions.unsqueeze(0) < lengths.unsqueeze(-1)).to(
        attention_mask.dtype
    )
    if not torch.equal(attention_mask, right_padded):
        msg = "aggregate_embeddings assumes a right-padded attention_mask"
        raise ValueError(msg)

    end = -math.ceil(stride / 2)
    start = math.floor(stride / 2)
    token = embeddings.shape[1]
    kept: list[int] = []

    # Positions are sliced as ranges, which follow the list-slice rules, and
    # gathered once: a slice per window would give backward one allocation
    # and copy per window.
    windows = [
        range(w * token, w * token + n)[1:-1]
        for w, n in enumerate(lengths.tolist())
    ]
    for window, flat in enumerate(windows):
        kept.extend(flat[:end] if window == 0 else flat[start:end])

    kept.extend(windows[-1][end:])

    # A non_blocking copy from pageable memory can still block the host
    # behind queued GPU work; from pinned memory it does not. Freeing the
    # pinned index right away is safe: the copy records an event on its
    # block, which is not reused until the event completes
    # (torch/include/ATen/cuda/CachingHostAllocator.h:18-21, 29-30).
    # Pinning needs a CUDA context, so CPU inputs skip it.
    index = torch.tensor(kept, dtype=torch.long)
    if embeddings.device.type == "cuda":
        index = index.pin_memory()
    index = index.to(embeddings.device, non_blocking=True)
    return embeddings.flatten(0, 1).index_select(0, index)


def embed_document(
    doc: str,
    tokenizer: transformers.PreTrainedTokenizerFast,
    model: transformers.BertModel,
    stride: NonNegative = WINDOW_STRIDE,
    batch_size: Positive = 50,
    max_len: Positive = WINDOW_LENGTH,
) -> Float[Tensor, "tokens features"]:
    """Compute token embeddings for `doc`.

    :param doc: the document text.
    :param tokenizer: the tokenizer the windows are cut with.
    :param model: the frozen base model.
    :param stride: tokens of overlap between adjacent windows.
    :param batch_size: windows per forward pass.
    :param max_len: tokens per window.
    :return: one embedding row per token of the document.
    """
    encoding = split_and_tokenize(
        tokenizer=tokenizer,
        inputs=doc,
        stride=stride,
        max_length=max_len,
        return_offsets_mapping=False,
    )

    input_ids_all = typing.cast(torch.Tensor, encoding["input_ids"])
    attention_mask_all = typing.cast(torch.Tensor, encoding["attention_mask"])

    seg_embeds_cpu = []
    N = input_ids_all.size(0)

    with torch.inference_mode():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            ids = input_ids_all[start:end].to(model.device, non_blocking=True)
            mask = attention_mask_all[start:end].to(
                model.device, non_blocking=True
            )

            # Same dtype source as the training forward, so one machine runs
            # one precision on both paths; see the data explanation for why
            # a stored run still does not reproduce a live one.
            with torch.amp.autocast(
                device_type=model.device.type,
                dtype=select_amp_dtype(model.device.type),
            ):
                embedding = model(ids, mask).last_hidden_state

            seg_embeds_cpu.append(embedding.detach().cpu())

            del embedding, ids, mask

    embeddings_cpu = torch.cat(seg_embeds_cpu, dim=0)
    del seg_embeds_cpu

    return aggregate_embeddings(
        embeddings=embeddings_cpu, attention_mask=attention_mask_all
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
    stride: NonNegative,
) -> Iterator[list[Token]]:
    """Merge predictions for different segments of a large sequence.

    :param preds: one sequence of tokens per segment.
    :param sample_mapping: which document each segment came from.
    :param stride: tokens of overlap between adjacent segments.
    :return: one merged token sequence per document.
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
    """Merge the BPE tokens in `tokens` and combine their labels.

    :param tokens: the tokens to merge, special tokens included.
    :return: the merged tokens, `[CLS]`, `[SEP]` and `[PAD]` removed.
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
    offset = (a.offset[0], b.offset[1])
    return Token(
        text, offset, a.prediction, a.gold_label, a.prob, a.candidate_labels
    )


def concat(s: str, t: str, sep: str = "") -> str:
    if s and t:
        return s + sep + t
    else:
        return s + t


def repr_sequence(sequence: Iterable[Token]) -> str:
    output = ""
    last: int | None = None

    for token in sequence:
        if last is not None:
            output += " " * (token.offset[0] - last)
        output += token.string
        last = token.offset[1]

    return output
