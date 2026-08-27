"""Typed distant-supervision targets over a real tokenizer's offsets.

The tokenizer is built in-process from an inline vocabulary — no download, no
network — because the assertions are about *offsets*, and stubbing those would
test the stub. Its vocabulary is every ASCII letter and digit as both a word
start and a continuation, so each word becomes one token per character. That is
not what BioLinkBERT does, but it gives the tests exact control over which
tokens cover which characters, which is the whole subject here.

`_tokenizer` also takes extra multi-character pieces, which is how a token that
*straddles* two mentions is built. It takes a `°` to do it: `BertPreTokenizer`
splits on whitespace and punctuation, and a degree sign is neither, so it
survives inside a pre-token — while `form_words` reads it as a separator and
puts a mention boundary there. One WordPiece can then cover the last character
of one mention and the first of the next, which is the only way a subword
reaches two types at once.
"""

import functools
import pathlib
import string

import h5py
import numpy
import pytest
from d3text import corpus, surface_forms, token_labels
from d3text.schema import BRENDA_SCHEMA
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

_ENZYME = token_labels.BRENDA_LABELS.code_of("enz1")
_BACTERIUM = token_labels.BRENDA_LABELS.code_of("bac3")


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
    """Every target given to a token overlapping the characters `[start, end)`."""
    offsets = numpy.asarray(encoding["offset_mapping"]).reshape(-1, 2)
    flat = labels.reshape(-1)
    return {
        int(label)
        for (low, high), label in zip(offsets.tolist(), flat.tolist())
        if high > low and low < end and high > start
    }


def test_a_gold_mention_carries_its_entity_type(index) -> None:
    text = "catalase and cholesterol oxidase"
    encoding = _encode(text)

    labels = token_labels.document_token_labels(
        text, index, {"enz2"}, encoding["offset_mapping"]
    ).codes

    assert _labels_over(encoding, labels, 0, len("catalase")) == {_ENZYME}


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
    ).codes

    assert _labels_over(encoding, labels, start, len(text)) == {
        token_labels.IGNORE_INDEX
    }


def test_text_matching_nothing_is_negative(index) -> None:
    text = "catalase and cholesterol oxidase"
    start = text.index("and")
    encoding = _encode(text)

    labels = token_labels.document_token_labels(
        text, index, {"enz2"}, encoding["offset_mapping"]
    ).codes

    assert _labels_over(encoding, labels, start, start + 3) == {
        token_labels.NEGATIVE
    }


def test_the_three_targets_partition_one_document(index) -> None:
    """All three land in the same document, which is the ticket's assertion."""
    text = "catalase and cholesterol oxidase"
    encoding = _encode(text)

    labels = token_labels.document_token_labels(
        text, index, {"enz2"}, encoding["offset_mapping"]
    ).codes

    assert set(numpy.unique(labels).tolist()) == {
        token_labels.NEGATIVE,
        _ENZYME,
        token_labels.IGNORE_INDEX,
    }


def test_special_and_padding_tokens_are_ignored(index) -> None:
    """A `[PAD]` in the divisor is the dilution bug one level down."""
    text = "catalase"
    encoding = _encode(text, max_length=32, stride=4)
    labels = token_labels.document_token_labels(
        text, index, {"enz2"}, encoding["offset_mapping"]
    ).codes

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
    ).codes

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
    ).codes

    offsets = numpy.asarray(encoding["offset_mapping"])
    covering = (
        (offsets[..., 1] > offsets[..., 0])
        & (offsets[..., 0] < start + len("catalase"))
        & (offsets[..., 1] > start)
    )

    assert covering.any(axis=1).sum() >= 2, "the overlap is not exercised"
    assert (labels[covering] == _ENZYME).all()


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


def test_a_brenda_document_is_typed_where_its_gold_entity_is_named() -> None:
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
    ).codes

    assert _labels_over(
        encoding, labels, start, start + len("cholesterol oxidase")
    ) == {_ENZYME}


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
    ).codes

    assert _labels_over(
        encoding, labels, start, start + len("cholesterol oxidase")
    ) == {token_labels.IGNORE_INDEX}


# ---------------------------------------------------------------------------
# The label space: which integer means which entity type.
# ---------------------------------------------------------------------------


def test_every_declared_entity_type_has_its_own_code() -> None:
    """The label space covers the schema, one distinct code per type.

    Not asserted against a literal list, which would only restate the schema:
    what has to hold is that each declared type is reachable and that no two
    share a target, since a collision would train two types onto one column
    without failing anywhere.
    """
    space = token_labels.BRENDA_LABELS

    codes = [
        space.code_of(f"{entity_type.prefix}1")
        for entity_type in BRENDA_SCHEMA.entity_types
    ]

    assert len(set(codes)) == len(BRENDA_SCHEMA.entity_types)
    assert token_labels.OUTSIDE not in codes
    assert token_labels.IGNORE_INDEX not in codes


def test_the_codes_fit_the_stored_dtype() -> None:
    """`int8` has to hold every code and the ignore target at once."""
    stored = numpy.array(
        [*token_labels.BRENDA_LABELS.codes, token_labels.IGNORE_INDEX],
        dtype=numpy.int8,
    )

    assert stored.tolist() == [
        *token_labels.BRENDA_LABELS.codes,
        token_labels.IGNORE_INDEX,
    ]


def test_a_type_set_too_large_for_the_dtype_is_rejected() -> None:
    with pytest.raises(ValueError, match="do not fit"):
        token_labels.LabelSpace(
            types=tuple(f"type{n}" for n in range(200)),
            prefixes=tuple(f"t{n:03d}" for n in range(200)),
        )


def test_a_label_space_with_mismatched_columns_is_rejected() -> None:
    with pytest.raises(ValueError, match="ID prefixes"):
        token_labels.LabelSpace(types=("enzymes",), prefixes=("enz", "bac"))


@pytest.mark.parametrize(
    "entity_type", BRENDA_SCHEMA.entity_types, ids=lambda t: t.name
)
def test_a_mention_of_each_type_is_labelled_with_that_type(
    entity_type,
) -> None:
    """One document per namespace, each labelled with its own code.

    The binary predecessor gave all four the same target, so nothing here
    distinguished a strain designation from an enzyme name.
    """
    entity_id = f"{entity_type.prefix}7"
    index = surface_forms.build_index({entity_id: ["angstrom widget"]})
    text = "the angstrom widget again"
    start = text.index("angstrom")
    encoding = _encode(text)

    labels = token_labels.document_token_labels(
        text, index, {entity_id}, encoding["offset_mapping"]
    ).codes

    assert _labels_over(
        encoding, labels, start, start + len("angstrom widget")
    ) == {token_labels.BRENDA_LABELS.code_of(entity_id)}


def test_a_non_default_label_space_reaches_both_halves_of_the_labelling() -> (
    None
):
    """`document_token_labels`' `space` is forwarded, not decorative.

    Every production call takes the `BRENDA_LABELS` default, so dropping the
    forwarding to either `mention_spans` or `project_onto_tokens` would leave a
    documented parameter silently ignored. A space with a fifth type is what
    makes both drops visible: its prefix is unknown to `BRENDA_LABELS`, so the
    typing half cannot fall back silently, and its code is outside
    `BRENDA_LABELS.codes`, so the projection half cannot either — a default
    projection reads a code it does not know as `OUTSIDE`.
    """
    space = token_labels.LabelSpace(
        types=("alpha", "beta", "gamma", "delta", "gadgets"),
        prefixes=("aaa", "bbb", "ccc", "ddd", "gad"),
    )
    code = space.code_of("gad7")
    assert code not in token_labels.BRENDA_LABELS.codes
    index = surface_forms.build_index({"gad7": ["angstrom widget"]})
    text = "the angstrom widget again"
    start = text.index("angstrom")
    encoding = _encode(text)

    labels = token_labels.document_token_labels(
        text, index, {"gad7"}, encoding["offset_mapping"], space=space
    )

    assert _labels_over(
        encoding, labels.codes, start, start + len("angstrom widget")
    ) == {code}
    assert _rows(labels.spans) == [
        (start, start + len("angstrom widget"), code, 1)
    ]


def test_two_types_in_one_document_get_different_codes(index) -> None:
    text = "catalase from Streptomyces"
    encoding = _encode(text)

    labels = token_labels.document_token_labels(
        text, index, {"enz2", "bac4"}, encoding["offset_mapping"]
    ).codes

    assert _labels_over(encoding, labels, 0, len("catalase")) == {_ENZYME}
    assert _labels_over(
        encoding, labels, text.index("Streptomyces"), len(text)
    ) == {_BACTERIUM}
    assert _ENZYME != _BACTERIUM


# ---------------------------------------------------------------------------
# Resolving a token that has more than one candidate answer.
# ---------------------------------------------------------------------------


def test_a_form_naming_several_entities_of_one_type_keeps_that_type() -> None:
    """`AS-A` names four separate enzymes; the token is still an enzyme.

    Ambiguity about *which* entity is not ambiguity about the target, so this
    is the case that must not abstain — otherwise every acronym BRENDA shares
    between enzymes would be dropped from the supervision.
    """
    forms = {"enz11": ["angstrom widget"], "enz12": ["angstrom widget"]}
    index = surface_forms.build_index(forms)
    text = "the angstrom widget again"
    encoding = _encode(text)

    labels = token_labels.document_token_labels(
        text, index, {"enz11", "enz12"}, encoding["offset_mapping"]
    ).codes

    assert _labels_over(
        encoding, labels, text.index("angstrom"), text.index(" again")
    ) == {_ENZYME}


def test_a_form_naming_gold_entities_of_two_types_is_ignored() -> None:
    """A species nested in a strain designation names both, and one code is
    all a flat scheme has. Asserting either would teach the tagger that the
    other type is wrong here, so the loss does not read the token at all."""
    forms = {"bac11": ["angstrom widget"], "str12": ["angstrom widget"]}
    index = surface_forms.build_index(forms)
    text = "the angstrom widget again"
    encoding = _encode(text)

    labels = token_labels.document_token_labels(
        text, index, {"bac11", "str12"}, encoding["offset_mapping"]
    ).codes

    assert _labels_over(
        encoding, labels, text.index("angstrom"), text.index(" again")
    ) == {token_labels.IGNORE_INDEX}


def test_a_gold_entity_decides_the_type_over_a_non_gold_one() -> None:
    """The typed reading of "a positive beats an ignore".

    The same string names a bacterium this document was annotated with and an
    enzyme it was not. The non-gold candidate is exactly what `IGNORE_INDEX`
    exists not to assert, so it does not get to make the answer ambiguous.
    """
    forms = {"bac11": ["angstrom widget"], "enz12": ["angstrom widget"]}
    index = surface_forms.build_index(forms)
    text = "the angstrom widget again"
    encoding = _encode(text)

    labels = token_labels.document_token_labels(
        text, index, {"bac11"}, encoding["offset_mapping"]
    ).codes

    assert _labels_over(
        encoding, labels, text.index("angstrom"), text.index(" again")
    ) == {_BACTERIUM}


def _straddling(text: str, piece: str):
    """`text`'s encoding, with `piece` in the vocabulary.

    Asserts the straddle actually happened: without the piece in the vocabulary
    every token is one character wide and the case under test disappears
    silently.
    """
    encoding = _encode(text, extra=(piece,))
    offsets = numpy.asarray(encoding["offset_mapping"]).reshape(-1, 2)
    boundary = text.index("°")
    straddles = (offsets[:, 0] <= boundary) & (offsets[:, 1] > boundary + 1)
    assert straddles.any(), "no token straddles the mention boundary"
    return encoding, straddles


def test_a_token_straddling_two_types_is_ignored() -> None:
    """One subword, two mentions, two types — and no way to say both.

    The mentions are word-aligned, so a straddling token needs a character that
    `form_words` reads as a separator and `BertPreTokenizer` does not; `°` is
    one. Same resolution as the ambiguous form above, one level down.
    """
    forms = {"enz11": ["catalase"], "bac12": ["Streptomyces"]}
    index = surface_forms.build_index(forms)
    text = "catalase°Streptomyces"
    encoding, straddles = _straddling(text, "##e°S")

    labels = token_labels.document_token_labels(
        text, index, {"enz11", "bac12"}, encoding["offset_mapping"]
    ).codes

    assert set(labels.reshape(-1)[straddles].tolist()) == {
        token_labels.IGNORE_INDEX
    }


def test_a_token_straddling_a_type_and_plain_text_keeps_the_type() -> None:
    """The half that fell outside the mention must not win.

    A subword is not evidence that the mention it overlaps is absent, so a
    token covering one type and nothing else takes that type — the rule the
    binary version had, restated per type rather than for `positive`.
    """
    forms = {"enz11": ["catalase"]}
    index = surface_forms.build_index(forms)
    text = "catalase°Streptomyces"
    encoding, straddles = _straddling(text, "##e°S")

    labels = token_labels.document_token_labels(
        text, index, {"enz11"}, encoding["offset_mapping"]
    ).codes

    assert set(labels.reshape(-1)[straddles].tolist()) == {_ENZYME}


def test_a_token_straddling_a_type_and_an_ignored_mention_keeps_the_type() -> (
    None
):
    """A type beats an ignore on the same token, as a positive used to.

    The other half of the straddle is a curated entity this document was not
    annotated with, so its characters are ignored. Letting that win would let
    any neighbouring uncurated name delete a gold mention's supervision — the
    abstention spreading beyond the tokens it was meant to cover.
    """
    forms = {"enz11": ["catalase"], "enz12": ["Streptomyces"]}
    index = surface_forms.build_index(forms)
    text = "catalase°Streptomyces"
    encoding, straddles = _straddling(text, "##e°S")

    labels = token_labels.document_token_labels(
        text, index, {"enz11"}, encoding["offset_mapping"]
    ).codes

    assert set(labels.reshape(-1)[straddles].tolist()) == {_ENZYME}


def test_the_token_after_a_mention_stays_outside_under_real_subwords(
    index,
) -> None:
    """A mention followed immediately by punctuation, whole-word tokens.

    The one-character-per-token vocabulary the other tests use can never see an
    inclusive span end: the character after every mention there is whitespace
    or end-of-string, which no token covers. Under whole-word pieces the comma
    of `catalase,` gets its own token on the very character a `labels[start :
    end + 1]` painting would spill the enzyme type onto — silently, in the
    training targets, and in the direction that corrupts rather than abstains.
    """
    text = "catalase, and more"
    comma = text.index(",")
    encoding = _encode(text, extra=("catalase", ","))
    offsets = numpy.asarray(encoding["offset_mapping"]).reshape(-1, 2)
    flat_offsets = offsets.tolist()
    assert [comma, comma + 1] in flat_offsets, "the comma has no token"
    assert [0, comma] in flat_offsets, "the whole-word piece did not take"

    labels = token_labels.document_token_labels(
        text, index, {"enz2"}, encoding["offset_mapping"]
    ).codes

    assert _labels_over(encoding, labels, 0, comma) == {_ENZYME}
    assert _labels_over(encoding, labels, comma, comma + 1) == {
        token_labels.OUTSIDE
    }


def test_a_token_covering_two_types_directly_is_ignored() -> None:
    """The same rule stated against `project_onto_tokens` alone.

    No tokenizer in the way, so the arithmetic is the only thing under test:
    the first token covers one type, the second covers two.
    """
    characters = numpy.array([_ENZYME, _ENZYME, _BACTERIUM], dtype=numpy.int8)

    projected = token_labels.project_onto_tokens(characters, [[[0, 2], [1, 3]]])

    assert projected.reshape(-1).tolist() == [
        _ENZYME,
        token_labels.IGNORE_INDEX,
    ]


# ---------------------------------------------------------------------------
# The mention spans: the boundaries the per-token codes cannot carry.
# ---------------------------------------------------------------------------

_SPAN_FORMS = {
    "enz2": ["catalase"],
    "enz1": ["cholesterol oxidase"],
    "bac11": ["angstrom widget"],
    "str12": ["angstrom widget"],
}
_SPAN_TEXT = "catalase catalase and cholesterol oxidase in angstrom widget"
_SPAN_GOLD = frozenset({"enz2", "bac11", "str12"})


@pytest.fixture(scope="module")
def span_index() -> surface_forms.SurfaceFormIndex:
    """One document holding a gold type, an abstention of each kind, and O."""
    return surface_forms.build_index(_SPAN_FORMS)


def _rows(spans: numpy.ndarray) -> list[tuple[int, int, int, int]]:
    return [tuple(int(value) for value in row) for row in spans]


def _empty_labels() -> token_labels.DocumentLabels:
    return token_labels.DocumentLabels(
        codes=numpy.zeros(4, dtype=numpy.int8),
        spans=numpy.zeros((0, token_labels.SPAN_COLUMNS), dtype=numpy.int32),
        text_length=4,
    )


def test_two_mentions_split_by_a_space_are_one_code_run_and_two_spans(
    index,
) -> None:
    """The defect this record exists for, and the record answering it.

    `catalase catalase` is two mentions of one type with no token between them:
    a space produces none, so every code across both reads `enzyme` and a
    consumer of the codes alone cannot tell one mention from two. The spans
    can, because `find_mentions` never lost the boundary.
    """
    text = "catalase catalase"
    encoding = _encode(text)

    labels = token_labels.document_token_labels(
        text, index, {"enz2"}, encoding["offset_mapping"]
    )

    assert _labels_over(encoding, labels.codes, 0, len(text)) == {_ENZYME}
    assert _rows(labels.spans) == [(0, 8, _ENZYME, 1), (9, 17, _ENZYME, 1)]


def test_an_abstaining_mention_keeps_its_span_and_the_type_it_would_have(
    index,
) -> None:
    """`IGNORE_INDEX` says only "do not look", and the span says the rest.

    `cholesterol oxidase` is an enzyme this document was not annotated with, so
    its tokens abstain — but the mention is still located, and still known to
    be an enzyme name, which is the pair of facts a span objective or a
    weighted abstention needs and the flat code destroys.
    """
    text = "catalase and cholesterol oxidase"
    start = text.index("cholesterol")

    labels = token_labels.document_token_labels(
        text, index, {"enz2"}, _encode(text)["offset_mapping"]
    )

    assert (start, len(text), _ENZYME, 0) in _rows(labels.spans)


def test_a_mention_naming_gold_entities_of_two_types_records_no_type() -> None:
    """The other abstention, and it is not the same one.

    Here the candidates disagree about the type rather than about the
    annotation, so there is no type to record — and the two cases have to stay
    apart, since one names a type the loss may not assert and the other names
    none at all.
    """
    index = surface_forms.build_index(
        {"bac11": ["angstrom widget"], "str12": ["angstrom widget"]}
    )
    text = "the angstrom widget again"
    start = text.index("angstrom")

    labels = token_labels.document_token_labels(
        text,
        index,
        {"bac11", "str12"},
        _encode(text)["offset_mapping"],
    )

    assert _rows(labels.spans) == [
        (start, start + len("angstrom widget"), token_labels.OUTSIDE, 0)
    ]


def test_the_stored_spans_reconstruct_the_stored_codes(
    tmp_path, span_index
) -> None:
    """The invariant that makes storing both safe.

    Painting the spans back over the document and projecting them onto the same
    offsets has to reproduce the stored code array element for element,
    including the abstentions — a mention that matched no gold entity and one
    whose gold candidates disagreed both place `IGNORE_INDEX` over their
    characters, so a reconstruction that only painted the gold spans would come
    back `OUTSIDE` there. Windows and padding are exercised too: the encoding
    is deliberately narrow enough to overflow.
    """
    encoding = _encode(_SPAN_TEXT, max_length=32, stride=4)
    labels = token_labels.document_token_labels(
        _SPAN_TEXT, span_index, _SPAN_GOLD, encoding["offset_mapping"]
    )
    path = tmp_path / "labels.hdf5"

    with h5py.File(path, "w-", libver="latest") as store:
        token_labels.write_label_space(store)
        token_labels.store_token_labels(store, "10822008", labels)

    with h5py.File(path, "r") as store:
        stored = token_labels.load_token_labels(store, "10822008")

    rebuilt = token_labels.project_onto_tokens(
        token_labels.character_labels_from_spans(
            stored.text_length, stored.spans
        ),
        encoding["offset_mapping"],
    )

    assert stored.codes.shape[0] > 1, "the windowing is not exercised"
    assert set(numpy.unique(stored.codes).tolist()) == {
        token_labels.OUTSIDE,
        _ENZYME,
        token_labels.IGNORE_INDEX,
    }
    abstentions = {
        (row[token_labels.SPAN_TYPE], row[token_labels.SPAN_GOLD])
        for row in _rows(stored.spans)
        if not row[token_labels.SPAN_GOLD]
    }
    assert abstentions == {(_ENZYME, 0), (token_labels.OUTSIDE, 0)}, (
        "both kinds of abstention have to be in the reconstruction"
    )
    assert numpy.array_equal(rebuilt, stored.codes)


def test_a_document_is_stored_with_its_spans_or_not_at_all(
    tmp_path, span_index
) -> None:
    """Codes without spans must not be creatable, so the pair is one value."""
    encoding = _encode(_SPAN_TEXT)
    labels = token_labels.document_token_labels(
        _SPAN_TEXT, span_index, _SPAN_GOLD, encoding["offset_mapping"]
    )
    path = tmp_path / "labels.hdf5"

    with h5py.File(path, "w-", libver="latest") as store:
        token_labels.write_label_space(store)
        token_labels.store_token_labels(store, "10822008", labels)

    with h5py.File(path, "r") as store:
        assert set(store["10822008"]) == {"codes", "spans"}


def test_a_document_that_matched_nothing_stores_an_empty_span_table(
    tmp_path, index
) -> None:
    """No mentions is a legitimate document, and a filter needs a chunk."""
    text = "nothing here names anything"
    labels = token_labels.document_token_labels(
        text, index, set(), _encode(text)["offset_mapping"]
    )
    path = tmp_path / "labels.hdf5"

    with h5py.File(path, "w-", libver="latest") as store:
        token_labels.write_label_space(store)
        token_labels.store_token_labels(store, "10822008", labels)

    with h5py.File(path, "r") as store:
        stored = token_labels.load_token_labels(store, "10822008")

    assert stored.spans.shape == (0, token_labels.SPAN_COLUMNS)
    assert (stored.codes[stored.codes != token_labels.IGNORE_INDEX] == 0).all()


def test_the_painting_matches_a_hand_written_character_array() -> None:
    """`character_labels_from_spans`, against an answer worked out by hand.

    The round-trip test cannot see a wrong painting rule: it applies the same
    rule on the produce and the reconstruct side, so an inclusive span end
    stays self-consistent and every test passes while a comma inherits the
    enzyme type. Only an expected array written down with no projection in the
    loop pins the half-open `end`, the gold/ignore branch and the `OUTSIDE`
    fill at once.
    """
    spans = numpy.array(
        [[2, 5, _ENZYME, 1], [6, 9, _ENZYME, 0]], dtype=numpy.int32
    )

    labels = token_labels.character_labels_from_spans(12, spans)

    outside = token_labels.OUTSIDE
    ignored = token_labels.IGNORE_INDEX
    assert labels.tolist() == [
        outside,
        outside,
        _ENZYME,
        _ENZYME,
        _ENZYME,
        outside,
        ignored,
        ignored,
        ignored,
        outside,
        outside,
        outside,
    ]


def test_spans_of_the_wrong_width_are_rejected() -> None:
    with pytest.raises(ValueError, match="mention spans must be"):
        token_labels.character_labels_from_spans(
            4, numpy.zeros((2, 3), dtype=numpy.int32)
        )


# ---------------------------------------------------------------------------
# The store, and the meaning it has to carry with it.
# ---------------------------------------------------------------------------


def test_the_label_store_round_trips(tmp_path, index) -> None:
    """The targets live beside the encodings, keyed by pubmed id."""
    text = "catalase and cholesterol oxidase"
    encoding = _encode(text)
    labels = token_labels.document_token_labels(
        text, index, {"enz2"}, encoding["offset_mapping"]
    )
    path = tmp_path / "labels.hdf5"

    with h5py.File(path, "w-", libver="latest") as store:
        token_labels.write_label_space(store)
        token_labels.store_token_labels(store, "10822008", labels)

    with h5py.File(path, "r") as store:
        stored = token_labels.load_token_labels(store, "10822008")
        assert numpy.array_equal(stored.codes, labels.codes)
        assert numpy.array_equal(stored.spans, labels.spans)
        assert stored.text_length == len(text)
        with pytest.raises(KeyError):
            token_labels.load_token_labels(store, "99999999")


def test_the_store_records_what_its_codes_mean(tmp_path) -> None:
    """The artifact has to say which column is which type.

    Nothing in an array of small integers does, so a store written under one
    declaration order and read under another scores every type against another
    type's target — silently, because the shapes still agree. This is the
    lesson `d3text.checkpoint` records a vocabulary for, at one level down.
    """
    path = tmp_path / "labels.hdf5"

    with h5py.File(path, "w-", libver="latest") as store:
        token_labels.write_label_space(store)

    with h5py.File(path, "r") as store:
        recorded = token_labels.read_label_space(store)

    assert recorded == token_labels.BRENDA_LABELS
    assert recorded.types == BRENDA_SCHEMA.class_names


def test_a_store_written_under_another_order_reads_back_as_that_order(
    tmp_path,
) -> None:
    """The failure the recording exists to catch, made visible.

    Reversing the declaration keeps every width identical, so nothing about
    the arrays would object. What separates the two stores is this attribute
    and nothing else.
    """
    reversed_space = token_labels.LabelSpace(
        types=token_labels.BRENDA_LABELS.types[::-1],
        prefixes=token_labels.BRENDA_LABELS.prefixes[::-1],
    )
    path = tmp_path / "labels.hdf5"

    with h5py.File(path, "w-", libver="latest") as store:
        token_labels.write_label_space(store, reversed_space)

    with h5py.File(path, "r") as store:
        recorded = token_labels.read_label_space(store)

    assert recorded == reversed_space
    assert recorded != token_labels.BRENDA_LABELS


def test_reading_a_store_under_another_label_space_is_refused(
    tmp_path,
) -> None:
    """The read side of the recording, which is the side that gets it wrong.

    A permuted declaration order leaves every width and every dtype identical,
    so the codes come back looking perfectly ordinary while `enz3494`'s code
    now names a strain. Recording the order on the way in only helps if the
    way out compares it, and a reader that must remember to call
    `read_label_space` first is a reader that will one day not.
    """
    permuted = token_labels.LabelSpace(
        types=token_labels.BRENDA_LABELS.types[::-1],
        prefixes=token_labels.BRENDA_LABELS.prefixes[::-1],
    )
    path = tmp_path / "labels.hdf5"

    with h5py.File(path, "w-", libver="latest") as store:
        token_labels.write_label_space(store, permuted)
        token_labels.store_token_labels(store, "10822008", _empty_labels())

    with h5py.File(path, "r") as store:
        with pytest.raises(ValueError, match="records the label space"):
            token_labels.load_token_labels(store, "10822008")

        under_its_own_space = token_labels.load_token_labels(
            store, "10822008", permuted
        )

    assert numpy.array_equal(
        under_its_own_space.codes, _empty_labels().codes
    ), "a store read under the space it records still reads"


def test_targets_cannot_be_written_without_their_label_space(
    tmp_path, index
) -> None:
    """A store of unattributed codes cannot be repaired, only regenerated, so
    it must not be possible to start one."""
    path = tmp_path / "labels.hdf5"

    with h5py.File(path, "w-", libver="latest") as store:
        with pytest.raises(KeyError, match="records no label space"):
            token_labels.store_token_labels(store, "10822008", _empty_labels())


def test_a_store_that_records_no_label_space_is_refused(tmp_path) -> None:
    path = tmp_path / "labels.hdf5"

    with h5py.File(path, "w-", libver="latest") as store:
        pass

    with h5py.File(path, "r") as store:
        with pytest.raises(KeyError, match="records no label space"):
            token_labels.read_label_space(store)


def test_a_store_written_under_another_ignore_index_is_refused(
    tmp_path,
) -> None:
    """`IGNORE_INDEX` is torch's `ignore_index` and the targets are handed to
    the loss unchanged, so a store that spelled it differently would train on
    the tokens this scheme abstains from."""
    path = tmp_path / "labels.hdf5"

    with h5py.File(path, "w-", libver="latest") as store:
        token_labels.write_label_space(store)
        store.attrs["ignore_index"] = -1

    with h5py.File(path, "r") as store:
        with pytest.raises(ValueError, match="this build does not use"):
            token_labels.read_label_space(store)


def test_a_store_written_before_the_mention_spans_is_refused(
    tmp_path,
) -> None:
    """A format-1 store keys each document to a bare code array.

    It cannot be read as a format-2 document and cannot be completed without
    re-running the matcher, so it is refused outright rather than defaulted
    into — the same answer an unstamped store gets, for the same reason.
    """
    path = tmp_path / "labels.hdf5"

    with h5py.File(path, "w-", libver="latest") as store:
        token_labels.write_label_space(store)
        store.attrs["d3text_token_labels_format"] = 1

    with h5py.File(path, "r") as store:
        with pytest.raises(ValueError, match="format-1 label store"):
            token_labels.read_label_space(store)
        with pytest.raises(ValueError, match="regenerate it"):
            token_labels.load_token_labels(store, "10822008")


def test_an_offset_mapping_of_the_wrong_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="size-2 axis"):
        token_labels.project_onto_tokens(
            numpy.zeros(4, dtype=numpy.int8), numpy.zeros((2, 3))
        )
