"""Three-way distant-supervision targets over a real tokenizer's offsets.

The tokenizer is built in-process from an inline vocabulary — no download, no
network — because the assertions are about *offsets*, and stubbing those would
test the stub. Its vocabulary is every ASCII letter and digit as both a word
start and a continuation, so each word becomes one token per character. That is
not what BioLinkBERT does, but it gives the tests exact control over which
tokens cover which characters, which is the whole subject here.
"""

import functools
import pathlib
import string

import h5py
import numpy
import pytest
from d3text import corpus, surface_forms, token_labels
from d3text.utils import split_and_tokenize
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

_TESTDB = (
    pathlib.Path(__file__).resolve().parent.parent
    / "brenda_references"
    / "tests"
    / "test_files"
    / "testdb.json"
)

_SPECIALS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]")

_FORMS = {
    "enz1": ["cholesterol oxidase", "COD"],
    "enz2": ["catalase"],
    "bac3": ["Streptomyces griseocarneus"],
    "bac4": ["Streptomyces"],
}


@functools.cache
def _tokenizer() -> PreTrainedTokenizerFast:
    vocabulary = {token: index for index, token in enumerate(_SPECIALS)}
    for character in string.ascii_letters + string.digits:
        vocabulary.setdefault(character, len(vocabulary))
        vocabulary.setdefault("##" + character, len(vocabulary))

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


def _encode(text: str, max_length: int = 512, stride: int = 20):
    return split_and_tokenize(
        _tokenizer(), text, max_length=max_length, stride=stride
    )


def _labels_over(
    encoding, labels: numpy.ndarray, start: int, end: int
) -> set[int]:
    """Every target given to a token overlapping the characters `[start, end)`."""
    offsets = numpy.asarray(encoding["offset_mapping"]).reshape(-1, 2)
    flat = labels.reshape(-1)
    return {
        int(label)
        for (low, high), label in zip(offsets.tolist(), flat.tolist())
        if high > low and low < end and high > start
    }


def test_a_gold_mention_is_positive(index) -> None:
    text = "catalase and cholesterol oxidase"
    encoding = _encode(text)

    labels = token_labels.document_token_labels(
        text, index, {"enz2"}, encoding["offset_mapping"]
    )

    assert _labels_over(encoding, labels, 0, len("catalase")) == {
        token_labels.POSITIVE
    }


def test_another_entitys_mention_is_ignored_rather_than_negative(
    index,
) -> None:
    """The whole point of the third target.

    `cholesterol oxidase` is an entity BRENDA knows and this document was not
    annotated with. Calling it negative teaches the tagger that a curated
    enzyme name is not an enzyme name, which is BRENDA's notion of *salience*
    rather than of entity-hood.
    """
    text = "catalase and cholesterol oxidase"
    start = text.index("cholesterol")
    encoding = _encode(text)

    labels = token_labels.document_token_labels(
        text, index, {"enz2"}, encoding["offset_mapping"]
    )

    assert _labels_over(encoding, labels, start, len(text)) == {
        token_labels.IGNORE_INDEX
    }


def test_text_matching_nothing_is_negative(index) -> None:
    text = "catalase and cholesterol oxidase"
    start = text.index("and")
    encoding = _encode(text)

    labels = token_labels.document_token_labels(
        text, index, {"enz2"}, encoding["offset_mapping"]
    )

    assert _labels_over(encoding, labels, start, start + 3) == {
        token_labels.NEGATIVE
    }


def test_the_three_targets_partition_one_document(index) -> None:
    """All three land in the same document, which is the ticket's assertion."""
    text = "catalase and cholesterol oxidase"
    encoding = _encode(text)

    labels = token_labels.document_token_labels(
        text, index, {"enz2"}, encoding["offset_mapping"]
    )

    assert set(numpy.unique(labels).tolist()) == {
        token_labels.NEGATIVE,
        token_labels.POSITIVE,
        token_labels.IGNORE_INDEX,
    }


def test_special_and_padding_tokens_are_ignored(index) -> None:
    """A `[PAD]` in the divisor is the dilution bug one level down."""
    text = "catalase"
    encoding = _encode(text, max_length=32, stride=4)
    labels = token_labels.document_token_labels(
        text, index, {"enz2"}, encoding["offset_mapping"]
    )

    offsets = numpy.asarray(encoding["offset_mapping"])
    empty = offsets[..., 1] <= offsets[..., 0]

    assert empty.any()
    assert (labels[empty] == token_labels.IGNORE_INDEX).all()


def test_the_targets_have_the_encodings_geometry(index) -> None:
    """One target per stored `input_id`, window for window."""
    text = "catalase and cholesterol oxidase " * 20
    encoding = _encode(text, max_length=64, stride=8)

    labels = token_labels.document_token_labels(
        text, index, {"enz2"}, encoding["offset_mapping"]
    )

    assert labels.shape == tuple(encoding["input_ids"].shape)
    assert labels.dtype == numpy.int8


def test_a_mention_in_the_window_overlap_is_labelled_in_both_windows(
    index,
) -> None:
    """Deduped per document, not per sequence.

    The windows overlap by `stride` tokens, so a mention near a boundary is
    tokenized twice. Matching once per *window* would have to decide which copy
    is the real one; matching once per document and projecting onto every
    window makes the two copies agree by construction.
    """
    text = "aa bb cc dd ee catalase ff gg hh ii jj kk ll mm nn oo"
    start = text.index("catalase")
    encoding = _encode(text, max_length=16, stride=4)

    labels = token_labels.document_token_labels(
        text, index, {"enz2"}, encoding["offset_mapping"]
    )

    offsets = numpy.asarray(encoding["offset_mapping"])
    covering = (
        (offsets[..., 1] > offsets[..., 0])
        & (offsets[..., 0] < start + len("catalase"))
        & (offsets[..., 1] > start)
    )

    assert covering.any(axis=1).sum() >= 2, "the overlap is not exercised"
    assert (labels[covering] == token_labels.POSITIVE).all()


def test_the_longest_surface_form_wins(index) -> None:
    """`Streptomyces griseocarneus` is one bacterium, not a genus plus one."""
    text = "Streptomyces griseocarneus grows"

    mentions = token_labels.find_mentions(text, index)

    assert [
        (text[mention.start : mention.end], sorted(mention.entity_ids))
        for mention in mentions
    ] == [("Streptomyces griseocarneus", ["bac3"])]


def test_words_far_apart_are_not_one_mention(index) -> None:
    """The separator is not compared, so something has to bound it."""
    text = "Streptomyces, an unrelated clause, griseocarneus"

    mentions = token_labels.find_mentions(text, index)

    assert [text[m.start : m.end] for m in mentions] == ["Streptomyces"]


def test_a_symbol_form_does_not_fire_on_the_folded_word(index) -> None:
    """`COD` names the enzyme; `cod` is a fish."""
    assert token_labels.find_mentions("the cod was fresh", index) == []
    assert len(token_labels.find_mentions("COD activity", index)) == 1


def test_a_brenda_document_gets_positives_where_its_gold_entity_is_named() -> (
    None
):
    """End to end over tracked BRENDA data and a real offset mapping.

    Document 287675 is annotated with `enz34567`, cholesterol oxidase, and its
    abstract names it. The text is built the way the encodings were —
    `corpus.document_text`, not `encode_split`'s `fulltext` column — because
    offsets taken against any other string do not address the stored
    `input_ids`.
    """
    tables = surface_forms.load_entity_tables(_TESTDB)
    documents = tables["documents"]
    index = surface_forms.build_index(
        surface_forms.brenda_surface_forms(
            tables,
            (
                document.get("other_organisms") or {}
                for document in documents.values()
            ),
        )
    )
    document = documents["287675"]
    text = corpus.document_text(
        document.get("abstract"), document.get("fulltext")
    )
    start = text.index("cholesterol oxidase")
    encoding = _encode(text)

    labels = token_labels.document_token_labels(
        text, index, {"enz34567"}, encoding["offset_mapping"]
    )

    assert _labels_over(
        encoding, labels, start, start + len("cholesterol oxidase")
    ) == {token_labels.POSITIVE}


def test_the_same_span_is_ignored_for_a_document_that_lacks_it() -> None:
    """Same text, same index, a different gold set — and the target changes.

    Which is the mechanism in one assertion: `positive` and `ignore` are not
    properties of the string, they are properties of the string *in this
    document*. Nothing in the text of 287675 changes when it is labelled for a
    document annotated with photosystem I instead; what changes is that
    nothing may be asserted about its enzyme mentions.
    """
    tables = surface_forms.load_entity_tables(_TESTDB)
    documents = tables["documents"]
    index = surface_forms.build_index(
        surface_forms.brenda_surface_forms(
            tables,
            (
                document.get("other_organisms") or {}
                for document in documents.values()
            ),
        )
    )
    document = documents["287675"]
    text = corpus.document_text(
        document.get("abstract"), document.get("fulltext")
    )
    start = text.index("cholesterol oxidase")
    encoding = _encode(text)

    labels = token_labels.document_token_labels(
        text, index, {"enz64878"}, encoding["offset_mapping"]
    )

    assert _labels_over(
        encoding, labels, start, start + len("cholesterol oxidase")
    ) == {token_labels.IGNORE_INDEX}


def test_the_label_store_round_trips(tmp_path, index) -> None:
    """The targets live beside the encodings, keyed by pubmed id."""
    text = "catalase and cholesterol oxidase"
    encoding = _encode(text)
    labels = token_labels.document_token_labels(
        text, index, {"enz2"}, encoding["offset_mapping"]
    )
    path = tmp_path / "labels.hdf5"

    with h5py.File(path, "w-", libver="latest") as store:
        token_labels.store_token_labels(store, "10822008", labels)

    with h5py.File(path, "r") as store:
        assert numpy.array_equal(
            token_labels.load_token_labels(store, "10822008"), labels
        )
        with pytest.raises(KeyError):
            token_labels.load_token_labels(store, "99999999")


def test_an_offset_mapping_of_the_wrong_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="size-2 axis"):
        token_labels.project_onto_tokens(
            numpy.zeros(4, dtype=numpy.int8), numpy.zeros((2, 3))
        )
