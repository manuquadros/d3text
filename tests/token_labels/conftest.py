"""Shared fixtures and helpers for the token-label test package.

The tokenizer is built in-process from an inline vocabulary, because the
assertions are about *offsets* and stubbing those would test the stub. Each
word becomes one token per character, which gives exact control over which
tokens cover which characters. Extra multi-character pieces are how a token
that *straddles* two mentions is built, using a `°`: `BertPreTokenizer` keeps
it inside a pre-token while `form_words` reads it as a separator.
"""

import functools
import string

import numpy
import pytest
from d3text import surface_forms, token_labels
from d3text.utils import split_and_tokenize
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

_SPECIALS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]")

_FORMS = {
    "enz1": ["cholesterol oxidase", "CAMP"],
    "enz2": ["catalase"],
    "bac3": ["Streptomyces griseocarneus"],
    "bac4": ["Streptomyces"],
}

_ENZYME = token_labels.BRENDA_LABELS.code_of("enz1")
_BACTERIUM = token_labels.BRENDA_LABELS.code_of("bac3")

_STAMP = token_labels.IndexStamp.from_index(
    surface_forms.build_index(_FORMS), sources=("split.csv",)
)


@functools.cache
def _tokenizer(extra: tuple[str, ...] = ()) -> PreTrainedTokenizerFast:
    vocabulary = {token: index for index, token in enumerate(_SPECIALS)}
    for character in string.ascii_letters + string.digits:
        vocabulary.setdefault(character, len(vocabulary))
        vocabulary.setdefault("##" + character, len(vocabulary))
    for piece in extra:
        vocabulary.setdefault(piece, len(vocabulary))

    backend = Tokenizer(models.WordPiece(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[
            ("[CLS]", vocabulary["[CLS]"]),
            ("[SEP]", vocabulary["[SEP]"]),
        ],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )


@pytest.fixture(scope="module")
def index() -> surface_forms.SurfaceFormIndex:
    return surface_forms.build_index(_FORMS)


def _encode(
    text: str,
    max_length: int = 512,
    stride: int = 20,
    extra: tuple[str, ...] = (),
):
    return split_and_tokenize(
        _tokenizer(extra), text, max_length=max_length, stride=stride
    )


def _labels_over(
    encoding, labels: numpy.ndarray, start: int, end: int
) -> set[int]:
    """Every target given to a token overlapping characters `[start, end)`."""
    offsets = numpy.asarray(encoding["offset_mapping"]).reshape(-1, 2)
    flat = labels.reshape(-1)
    return {
        int(label)
        for (low, high), label in zip(offsets.tolist(), flat.tolist())
        if high > low and low < end and high > start
    }


def _rows(spans: numpy.ndarray) -> list[tuple[int, int, int, int]]:
    return [tuple(int(value) for value in row) for row in spans]


def _empty_labels() -> token_labels.DocumentLabels:
    return token_labels.DocumentLabels(
        codes=numpy.zeros(4, dtype=numpy.int8),
        spans=numpy.zeros((0, token_labels.SPAN_COLUMNS), dtype=numpy.int32),
        text_length=4,
    )
