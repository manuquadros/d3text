"""Gold-mention labelling and the window-overlap rules."""

import pathlib

import numpy
import pytest
from conftest import _ENZYME, _encode, _labels_over
from d3text import corpus, surface_forms, token_labels
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

_TESTDB = (
    pathlib.Path(__file__).resolve().parent.parent.parent
    / "brenda_references"
    / "tests"
    / "test_files"
    / "testdb.json"
)


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

    Calling a curated enzyme name this document was not annotated with a
    negative teaches BRENDA's notion of *salience* rather than of entity-hood.
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
    """One document holds all three targets at once.

    The target is a property of a token's string, not of the document: a
    form annotated here is positive, one annotated elsewhere is ignored, and
    unmatched text is negative, so a single document carries all three.
    """
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

    Matching once per window would have to decide which copy of a boundary
    mention is the real one; projecting one document-level match onto every
    window makes them agree by construction.
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


_CHARACTER_CODES = (
    token_labels.OUTSIDE,
    token_labels.IGNORE_INDEX,
    *token_labels.BRENDA_LABELS.codes,
)


def _offsets(length: int) -> st.SearchStrategy[list[list[tuple[int, int]]]]:
    """`[window, token, 2]` bounds into `length` characters, drawn freely.

    Abutting, nested, empty and reversed tokens all occur, and so does the
    `(0, 0)` of a special or padding token.
    """
    bound = st.integers(min_value=0, max_value=length)
    token = st.tuples(bound, bound)
    return st.integers(min_value=1, max_value=8).flatmap(
        lambda width: st.lists(
            st.lists(token, min_size=width, max_size=width),
            min_size=1,
            max_size=3,
        )
    )


def _code_by_hand(characters: list[int], start: int, end: int) -> int:
    """The code a token over `characters[start:end]` takes, read by hand."""
    if end <= start:
        return token_labels.IGNORE_INDEX
    spanned = set(characters[start:end])
    types = spanned - {token_labels.OUTSIDE, token_labels.IGNORE_INDEX}
    if len(types) == 1:
        return types.pop()
    if types or token_labels.IGNORE_INDEX in spanned:
        return token_labels.IGNORE_INDEX
    return token_labels.OUTSIDE


@given(
    case=st.lists(
        st.sampled_from(_CHARACTER_CODES), min_size=1, max_size=24
    ).flatmap(
        lambda characters: st.tuples(
            st.just(characters), _offsets(len(characters))
        )
    )
)
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_a_token_is_coded_from_the_characters_it_spans(case) -> None:
    """`project_onto_tokens` against a character-by-character reading.

    Pins the overlap rule the codes and the entity presence masks share, over
    layouts no tokenizer would produce as well as those it does.
    """
    characters, offsets = case

    projected = token_labels.project_onto_tokens(
        numpy.array(characters, dtype=numpy.int8), offsets
    )

    assert projected.dtype == numpy.int8
    assert projected.tolist() == [
        [_code_by_hand(characters, start, end) for start, end in window]
        for window in offsets
    ]


@given(
    case=st.integers(min_value=1, max_value=24).flatmap(
        lambda length: st.tuples(
            st.just(length),
            st.lists(
                st.tuples(
                    st.integers(min_value=0, max_value=length),
                    st.integers(min_value=0, max_value=length),
                ),
                max_size=4,
            ),
            _offsets(length),
        )
    )
)
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_an_entity_is_present_on_every_token_spanning_its_mention(
    case,
) -> None:
    """The presence mask against the same character-by-character reading.

    The codes' test above reads a token the same way, so the two cannot
    drift onto different overlap rules.
    """
    length, spans, offsets = case
    inside = [
        any(start <= position < end for start, end in spans)
        for position in range(length)
    ]

    present = token_labels._entity_token_presence(length, spans, offsets)

    assert present.dtype == numpy.int8
    assert present.tolist() == [
        [int(any(inside[start:end])) for start, end in window]
        for window in offsets
    ]


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


def _type_m_index() -> surface_forms.SurfaceFormIndex:
    """A strain designated `type M` beside *Magnaporthe oryzae*."""
    return surface_forms.build_index(
        surface_forms.brenda_surface_forms(
            {
                "strains": {
                    "1": {
                        "taxon": None,
                        "cultures": [],
                        "designations": ["type M"],
                    }
                },
                "bacteria": {
                    "2": {"organism": "Magnaporthe oryzae", "synonyms": []}
                },
            }
        )
    )


@pytest.mark.parametrize(
    "text",
    ["wild type M. oryzae", "wild-type M. oryzae", "wild type M.oryzae"],
)
def test_a_form_ending_on_a_genus_initial_leaves_it_to_the_binomial(
    text: str,
) -> None:
    """Longest match first, `type M` consumed the `M.` of `M. oryzae`, so the
    binomial was never found and `oryzae` was painted `OUTSIDE`: a trained
    negative on an organism name."""
    mentions = token_labels.find_mentions(text, _type_m_index())

    assert [
        (text[mention.start : mention.end], sorted(mention.entity_ids))
        for mention in mentions
    ] == [(text[text.index("M") :], ["bac2"])]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("the type M strain", [("type M", ["str1"])]),
        ("grown as type M. The mutant", [("type M", ["str1"])]),
        ("type M. mRNA levels rose", [("type M", ["str1"])]),
        (
            "type M, M. oryzae",
            [("type M", ["str1"]), ("M. oryzae", ["bac2"])],
        ),
    ],
)
def test_a_form_ending_on_a_capital_still_matches_off_a_binomial(
    text: str, expected: list[tuple[str, list[str]]]
) -> None:
    """A designation ending on a capital is not itself the fault — `37 Y` and
    `CCUG 42182 C` are real strains — so only a dot and an all-lowercase word
    after the capital may take it: a sentence break and `mRNA` are neither."""
    mentions = token_labels.find_mentions(text, _type_m_index())

    assert [
        (text[mention.start : mention.end], sorted(mention.entity_ids))
        for mention in mentions
    ] == expected


def test_a_deposit_number_is_not_split_at_its_thousands_separator() -> None:
    """`DSM 22` is a real culture number held by another strain.

    Reading the separator as a boundary offers that shorter window to the
    index, which resolves it confidently and labels the wrong strain.
    """
    index = surface_forms.build_index(
        {"str1": ["DSM 22"], "str2": ["DSM 22228"], "enz2": ["catalase"]}
    )
    text = "Orbus hercynius DSM 22,228 produces catalase."

    mentions = token_labels.find_mentions(text, index)

    assert [
        (mention.start, mention.end, sorted(mention.entity_ids))
        for mention in mentions
    ] == [(16, 26, ["str2"]), (36, 44, ["enz2"])]
    assert text[16:26] == "DSM 22,228"
    assert text[36:44] == "catalase"


def test_a_list_of_deposit_numbers_is_not_glued_into_one() -> None:
    """`ATCC 35984, 35983` is two deposits; joining them names neither."""
    index = surface_forms.build_index({"str1": ["ATCC 35984"]})
    text = "ATCC 35984, 35983 were compared"

    mentions = token_labels.find_mentions(text, index)

    assert [text[m.start : m.end] for m in mentions] == ["ATCC 35984"]


def test_a_deposit_number_written_without_its_space_is_labelled() -> None:
    """The sweep offers the index windows of whole words, and `ATCC14990` is
    one word where BRENDA's `ATCC 14990` is two — so the window the text
    yields never matched the key the strain was held under."""
    index = surface_forms.build_index(
        {"str1": ["ATCC 14990"], "enz2": ["catalase"]}
    )
    text = "Staphylococcus aureus ATCC14990 produces catalase."

    mentions = token_labels.find_mentions(text, index)

    assert [
        (text[mention.start : mention.end], sorted(mention.entity_ids))
        for mention in mentions
    ] == [("ATCC14990", ["str1"]), ("catalase", ["enz2"])]


def test_a_symbol_form_does_not_fire_on_the_folded_word(index) -> None:
    """`CAMP` names the enzyme; `camp` is a field with tents in it."""
    assert token_labels.find_mentions("the camp was quiet", index) == []
    assert len(token_labels.find_mentions("CAMP activity", index)) == 1


def test_a_near_miss_is_recorded_as_a_fuzzy_mention(index) -> None:
    """`catalases` is one edit from the registered `catalase`."""
    text = "catalases are active"

    mentions = token_labels.find_mentions(text, index)

    assert len(mentions) == 1
    assert mentions[0].fuzzy is True
    assert mentions[0].entity_ids == {"enz2"}


def test_a_fuzzy_hit_on_a_gold_entity_is_ignored_not_asserted(index) -> None:
    """A near-miss may abstain, never assert.

    `catalases` reaches only the entity this document *is* annotated with, so
    an exact matcher's "gold beats ignore" rule would read straight through to
    a positive. An uncalibrated cutoff cannot be trusted with that.
    """
    text = "catalases are active"
    encoding = _encode(text)

    labels = token_labels.document_token_labels(
        text, index, {"enz2"}, encoding["offset_mapping"]
    ).codes

    assert _labels_over(encoding, labels, 0, len("catalases")) == {
        token_labels.IGNORE_INDEX
    }


def test_a_fuzzy_hit_on_a_non_gold_entity_is_also_ignored(index) -> None:
    """Same mechanism, the other way: still `ignore`, never `negative`."""
    text = "catalases are active"
    encoding = _encode(text)

    labels = token_labels.document_token_labels(
        text, index, set(), encoding["offset_mapping"]
    ).codes

    assert _labels_over(encoding, labels, 0, len("catalases")) == {
        token_labels.IGNORE_INDEX
    }


def test_an_exact_hit_is_not_also_read_as_fuzzy(index) -> None:
    """A word the exact index already matched never reaches `fuzzy_ids`."""
    text = "catalase is active"

    mentions = token_labels.find_mentions(text, index)

    assert len(mentions) == 1
    assert mentions[0].fuzzy is False
    assert mentions[0].entity_ids == {"enz2"}


def test_ordinary_prose_around_a_variant_stays_negative(index) -> None:
    """The fuzzy layer must not turn common words into abstentions.

    Every word here is ordinary English but `oxidase`, which is a near-miss of
    a two-word form and so stays unmatched by design.
    """
    text = "the enzyme showed strong activity under these conditions"

    assert token_labels.find_mentions(text, index) == []


def test_a_brenda_document_is_typed_where_its_gold_entity_is_named() -> None:
    """End to end over tracked BRENDA data and a real offset mapping.

    The text is built the way the encodings were — `corpus.document_text`, not
    `encode_split`'s `fulltext` column — since offsets against any other string
    do not address the stored `input_ids`.
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

    Positive and ignore are not properties of the string, they are properties
    of the string *in this document*.
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
