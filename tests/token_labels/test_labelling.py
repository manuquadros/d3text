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
        token_labels.OUTSIDE
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
        token_labels.OUTSIDE,
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


def test_a_designation_after_a_bacterium_is_withheld_not_negative(
    index,
) -> None:
    """`RC-14` names no BRENDA entity, but it is not an ordinary negative
    either: withheld from the loss the same way a fuzzy near-miss is,
    because a pattern match carries no entity ID to be judged gold by."""
    text = "Streptomyces griseocarneus RC-14 was isolated"

    mentions = token_labels.find_mentions(text, index)

    assert [
        (text[mention.start : mention.end], sorted(mention.entity_ids))
        for mention in mentions
    ] == [("Streptomyces griseocarneus", ["bac3"]), ("RC-14", [])]


def test_a_designation_after_a_near_miss_bacterium_is_not_withheld(
    index,
) -> None:
    """Only an exact match opens the designation route: `bacterium_end` is
    set in the exact branch alone, so a fuzzy near-miss on a bacterium-only
    form must not withhold the token that follows it."""
    text = "Streptomycess RC-14 was isolated"

    mentions = token_labels.find_mentions(text, index)

    assert [
        (text[mention.start : mention.end], sorted(mention.entity_ids))
        for mention in mentions
    ] == [("Streptomycess", ["bac4"])]


def test_an_ordinary_word_after_a_bacterium_stays_outside(index) -> None:
    """A designation-shaped token is letters and a digit; a plain word or a
    bare number after the species name must not be swept up with it."""
    text = "Streptomyces griseocarneus cells were held at 30 degrees"

    mentions = token_labels.find_mentions(text, index)

    assert [text[m.start : m.end] for m in mentions] == [
        "Streptomyces griseocarneus"
    ]


def test_a_culture_collection_accession_is_withheld_not_negative(
    index,
) -> None:
    """`surface_forms.ACCESSION` already reads a deposit number; wiring it
    into the sweep keeps a bare `DSM 40738` -- no entity in this index's
    small vocabulary carries it -- from training as an ordinary negative."""
    text = "The strain DSM 40738 was cultured"

    mentions = token_labels.find_mentions(text, index)

    assert [
        (text[mention.start : mention.end], sorted(mention.entity_ids))
        for mention in mentions
    ] == [("DSM 40738", [])]


@pytest.mark.parametrize(
    ("text", "deposit"),
    [
        ("strain DSM 22,228 was cultured", "DSM 22,228"),
        ("strain DSM22,228 was cultured", "DSM22,228"),
        ("strain NRRL B-1,234 was cultured", "NRRL B-1,234"),
    ],
)
def test_an_unclaimed_accession_keeps_its_thousands_tail(
    index, text: str, deposit: str
) -> None:
    """`ACCESSION`'s lookahead accepts the `THOUSANDS` comma, so the regex
    alone reads only as far as `DSM 22` or `NRRL B-1` -- whatever spelling
    put the number's first digit outside its own `word_spans` word. The
    withheld span must cover the whole deposit number, comma included, or
    the tail trains as an ordinary negative."""
    mentions = token_labels.find_mentions(text, index)

    assert [
        (text[mention.start : mention.end], sorted(mention.entity_ids))
        for mention in mentions
    ] == [(deposit, [])]


def test_an_unclaimed_accession_does_not_widen_onto_a_quantity(index) -> None:
    """`AS` is a collection acronym, so `ACCESSION` reads `AS 1` out of
    `AS 1,000g` -- but the rest of the word is a gravities quantity, not a
    deposit suffix, and must not be swallowed into the withheld span."""
    text = "cells were spun at AS 1,000g for centrifugation"

    mentions = token_labels.find_mentions(text, index)

    assert [
        (text[mention.start : mention.end], sorted(mention.entity_ids))
        for mention in mentions
    ] == [("AS 1", [])]


def test_an_accession_wholly_inside_an_earlier_longer_mention_is_claimed() -> (
    None
):
    """`_unclaimed_accessions` now finds overlap with `bisect` rather than
    scanning every mention. A mention that starts well before the accession
    and, being long, ends after it must still claim the accession even when
    a shorter, closer-by-start mention in between does not -- a bisect that
    checks only the nearest preceding mention's end (rather than a prefix
    maximum) would miss it and wrongly call the accession unclaimed."""
    text = "Bacillus subtilis strain DSM 40738 was reported previously"
    accession_start = text.index("DSM 40738")
    accession_end = accession_start + len("DSM 40738")
    covering = token_labels.Mention(
        start=0, end=accession_end + 5, entity_ids=frozenset(), fuzzy=True
    )
    closer_but_short = token_labels.Mention(
        start=10, end=accession_start - 1, entity_ids=frozenset(), fuzzy=True
    )

    unclaimed = token_labels._unclaimed_accessions(
        text, [covering, closer_but_short]
    )

    assert unclaimed == []


def _naive_unclaimed_accessions(
    text: str, mentions: list[token_labels.Mention]
) -> list[tuple[int, int]]:
    """The linear `any(...)` overlap check `_unclaimed_accessions` used
    before it moved to `bisect`, kept here as the reference the bisect must
    still agree with byte-for-byte."""
    words = surface_forms.word_spans(text)
    accessions = [
        (match.start(), token_labels._accession_end(text, words, match.end()))
        for match in surface_forms.ACCESSION.finditer(text)
    ]
    return [
        (start, end)
        for start, end in accessions
        if not any(
            mention.start < end and start < mention.end for mention in mentions
        )
    ]


@given(
    spans=st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=60),
            st.integers(min_value=0, max_value=60),
        ).map(lambda pair: (min(pair), max(pair))),
        max_size=6,
    )
)
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_unclaimed_accessions_matches_the_naive_overlap_check(spans) -> None:
    """Property check: whatever mix of touching, nested and overlapping
    mention spans is thrown at it, the `bisect` overlap check must agree
    with the plain `any(...)` scan over the same mentions."""
    text = "strain DSM 40738 and also NRRL B-1,234 were both cultured"
    mentions = [
        token_labels.Mention(
            start=start, end=end, entity_ids=frozenset(), fuzzy=True
        )
        for start, end in sorted(spans)
    ]

    fast = [
        (mention.start, mention.end)
        for mention in token_labels._unclaimed_accessions(text, mentions)
    ]
    naive = _naive_unclaimed_accessions(text, mentions)

    assert fast == naive


def _naive_accession_end(
    text: str, words: list[tuple[str, int, int]], match_end: int
) -> int:
    """The linear scan `_accession_end` used before it moved to `bisect`,
    kept here as the reference the bisect must still agree with."""
    for word, word_start, word_end in words:
        if word_start >= match_end:
            break
        if word_end >= match_end:
            if surface_forms.is_quantity(text, word_start, word_end):
                return match_end
            return word_end
    return match_end


@pytest.mark.parametrize(
    "case",
    ["inside_a_word", "at_a_word_end", "past_the_last_word", "between_words"],
)
def test_accession_end_matches_the_naive_scan_at_a_boundary(case: str) -> None:
    """Pins `_accession_end`'s `bisect` to the old linear scan's result at
    each shape of `match_end` the loop's two branches distinguish: strictly
    inside a word, exactly at a word's end, past every word, and in the gap
    between two words. An off-by-one bisect index would only show up at one
    of these boundaries, not in the middle of a word."""
    text = "strain DSM 22,228T was deposited yesterday here"
    words = surface_forms.word_spans(text)
    match_end = {
        "inside_a_word": text.index("228") + 1,
        "at_a_word_end": text.index("228T") + len("228T"),
        "past_the_last_word": len(text) + 5,
        "between_words": text.index(" was") + 1,
    }[case]

    assert token_labels._accession_end(
        text, words, match_end
    ) == _naive_accession_end(text, words, match_end)


@given(match_end=st.integers(min_value=0, max_value=70))
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_accession_end_matches_the_naive_scan_everywhere(
    match_end: int,
) -> None:
    """Property check: whatever offset `ACCESSION` hands it as `match_end`,
    the `bisect` lookup must agree with the plain linear scan over the same
    words, including a quantity word (`AS 1,000g`) that must not widen."""
    text = "strain DSM 22,228T and AS 1,000g were both cultured"
    words = surface_forms.word_spans(text)

    assert token_labels._accession_end(
        text, words, match_end
    ) == _naive_accession_end(text, words, match_end)


def test_a_bare_strain_designation_trains_as_ignored(index) -> None:
    """The full pipeline: a designation with no entity ID cannot be gold for
    the document, so its tokens are `IGNORE_INDEX`, not `OUTSIDE`."""
    text = "Streptomyces griseocarneus RC-14 was isolated"
    start = text.index("RC-14")
    encoding = _encode(text)

    labels = token_labels.document_token_labels(
        text, index, {"bac3"}, encoding["offset_mapping"]
    ).codes

    assert _labels_over(encoding, labels, start, start + len("RC-14")) == {
        token_labels.IGNORE_INDEX
    }


def test_a_connector_word_between_species_and_designation_is_withheld() -> None:
    """`strain` interposed between the species and the designation must not
    break the chain -- `Enterobacter cloacae strain JWM6` is real running
    text, not `Enterobacter cloacae, JWM6`."""
    connector_index = surface_forms.build_index(
        {"bac9": ["Enterobacter cloacae"]}
    )
    text = "Enterobacter cloacae strain JWM6 was isolated"

    mentions = token_labels.find_mentions(text, connector_index)

    assert [
        (text[mention.start : mention.end], sorted(mention.entity_ids))
        for mention in mentions
    ] == [("Enterobacter cloacae", ["bac9"]), ("JWM6", [])]


@pytest.mark.parametrize("gap", ["  ", "\t"])
def test_a_wider_whitespace_gap_still_opens_the_designation_route(
    index, gap: str
) -> None:
    """A double space or a tab is as real a separator as the single space
    the check used to require exactly."""
    text = f"Streptomyces griseocarneus{gap}RC-14 was isolated"

    mentions = token_labels.find_mentions(text, index)

    assert [
        (text[mention.start : mention.end], sorted(mention.entity_ids))
        for mention in mentions
    ] == [("Streptomyces griseocarneus", ["bac3"]), ("RC-14", [])]


def test_a_newline_gap_does_not_open_the_designation_route(index) -> None:
    """`str.isspace` accepts a newline, but the gap check must not: a
    species sitting at a line's end must not withhold the next line's
    opening word."""
    text = "Streptomyces griseocarneus\n\nA1 Introduction"

    mentions = token_labels.find_mentions(text, index)

    assert [text[m.start : m.end] for m in mentions] == [
        "Streptomyces griseocarneus"
    ]


@pytest.mark.parametrize(
    ("connector", "designation"),
    [
        ("str.", "K-12"),
        ("sp.", "X12"),
        ("subsp.", "W23"),
    ],
)
def test_a_dotted_connector_still_opens_the_designation_route(
    index, connector: str, designation: str
) -> None:
    """`str`, `sp` and `subsp` are always written dotted in real text
    (`E. coli str. K-12`) -- the gap right after the connector must admit
    that one abbreviation dot, or these three connectors never fire at
    all."""
    text = f"Streptomyces griseocarneus {connector} {designation} was grown"

    mentions = token_labels.find_mentions(text, index)

    assert [
        (text[mention.start : mention.end], sorted(mention.entity_ids))
        for mention in mentions
    ] == [("Streptomyces griseocarneus", ["bac3"]), (designation, [])]


@pytest.mark.parametrize("connector", ["strain", "isolate"])
def test_a_connector_not_in_the_dotted_subset_leaves_its_dot_closed(
    index, connector: str
) -> None:
    """`strain` and `isolate` are full words, not abbreviations, so a dot
    right after one of them ends a sentence rather than opening the
    designation route -- unlike `str.`, `sp.` and `subsp.`, which are
    genuinely dotted abbreviations."""
    text = f"Streptomyces griseocarneus {connector}. A1 was next"

    mentions = token_labels.find_mentions(text, index)

    assert [text[m.start : m.end] for m in mentions] == [
        "Streptomyces griseocarneus"
    ]


def test_a_dotted_isolate_does_not_propagate_a_false_designation() -> None:
    """The rejected second attempt admitted the dot after every connector,
    so `Escherichia coli isolate. IL-6 levels rose. IL-6 was high` withheld
    both `IL-6` occurrences through propagation, once the first one was
    wrongly confirmed as a designation. With `isolate` outside the dotted
    subset, `IL-6` must stay OUTSIDE everywhere."""
    ecoli_index = surface_forms.build_index({"bac10": ["Escherichia coli"]})
    text = "Escherichia coli isolate. IL-6 levels rose. IL-6 was high"

    mentions = token_labels.find_mentions(text, ecoli_index)

    assert [text[m.start : m.end] for m in mentions] == ["Escherichia coli"]


def test_a_dot_right_after_a_bacterium_opens_nothing(index) -> None:
    """A bacterium match never carries a trailing dot of its own -- a
    sentence-ending period right after it must not be read as a
    connector's abbreviation dot either."""
    text = "Streptomyces griseocarneus. A1 was next"

    mentions = token_labels.find_mentions(text, index)

    assert [text[m.start : m.end] for m in mentions] == [
        "Streptomyces griseocarneus"
    ]


def test_two_connectors_in_a_row_do_not_chain_into_one_route(index) -> None:
    """A connector only ever follows a bacterium directly -- `Bacillus sp.
    strain X12` is two connector hops, and the second one must not open the
    route either, so `X12` stays unwithheld rather than the docs quietly
    implying chained connectors work."""
    text = "Streptomyces griseocarneus sp. strain X12 was grown"

    mentions = token_labels.find_mentions(text, index)

    assert [text[m.start : m.end] for m in mentions] == [
        "Streptomyces griseocarneus"
    ]


def test_a_designation_confirmed_once_is_withheld_everywhere_it_repeats(
    index,
) -> None:
    """A paper that drops the species after the first mention and calls the
    strain `RC-14` alone from then on still withholds every later spelling
    of the same text, once it was confirmed next to a bacterium."""
    text = (
        "Streptomyces griseocarneus RC-14 was isolated. "
        "RC-14 grew well, and RC-14 was sequenced."
    )

    mentions = token_labels.find_mentions(text, index)

    designations = [
        (text[mention.start : mention.end], sorted(mention.entity_ids))
        for mention in mentions
        if text[mention.start : mention.end] == "RC-14"
    ]
    assert designations == [("RC-14", []), ("RC-14", []), ("RC-14", [])]


def test_an_unconfirmed_designation_shaped_token_stays_outside(
    index,
) -> None:
    """`pUC19` and `IL-6` are designation-shaped but never sit next to a
    bacterium anywhere in this document, so propagation must not sweep them
    up on shape alone -- there is nothing here for them to have repeated."""
    text = "The plasmid pUC19 was used for cloning; IL-6 was measured too"

    assert token_labels.find_mentions(text, index) == []


def test_an_underscored_designation_is_withheld_after_a_bacterium(
    index,
) -> None:
    """`word_spans` splits `FORC_075` at the underscore into two words;
    `_DESIGNATION` must still match the whole thing."""
    text = "Streptomyces griseocarneus FORC_075 was sequenced"

    mentions = token_labels.find_mentions(text, index)

    assert [
        (text[mention.start : mention.end], sorted(mention.entity_ids))
        for mention in mentions
    ] == [("Streptomyces griseocarneus", ["bac3"]), ("FORC_075", [])]


def test_a_non_ascii_designation_is_withheld_after_a_bacterium(
    index,
) -> None:
    """`DH5α` ends on a non-ASCII letter that the old ASCII-only character
    classes left outside the match."""
    text = "Streptomyces griseocarneus DH5α was transformed"

    mentions = token_labels.find_mentions(text, index)

    assert [
        (text[mention.start : mention.end], sorted(mention.entity_ids))
        for mention in mentions
    ] == [("Streptomyces griseocarneus", ["bac3"]), ("DH5α", [])]


def test_a_bare_number_directly_after_a_bacterium_stays_outside(
    index,
) -> None:
    """A digits-only designation (`Bacillus subtilis 168`) is a known gap
    this fix leaves open: opening it risks reading an ordinary quantity
    (`E. coli 37 °C`) as a strain number instead. `_DESIGNATION` still
    requires a leading letter, so a bare number stays negative."""
    text = "Streptomyces griseocarneus 37 degrees was the growth temperature"

    mentions = token_labels.find_mentions(text, index)

    assert [text[mention.start : mention.end] for mention in mentions] == [
        "Streptomyces griseocarneus"
    ]


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


def test_the_ambiguous_flag_stays_off_outside_the_comma_shape(index) -> None:
    """`Mention.ambiguous` is only ever set by the comma-separator rule.

    None of these matches join their words with a comma, so the exact, the
    fuzzy and the space-joined multi-word branch must all still carry the
    field's default. See `test_a_comma_joined_match_is_ambiguous` and
    `test_a_non_comma_multi_word_match_is_not_ambiguous` for the branch that
    does set it.
    """
    texts = [
        "catalase is active",  # exact
        "catalases are active",  # fuzzy near-miss
        "cholesterol oxidase and Streptomyces griseocarneus",  # multi-word
    ]
    mentions = [
        mention
        for text in texts
        for mention in token_labels.find_mentions(text, index)
    ]

    assert {mention.fuzzy for mention in mentions} == {
        False,
        True,
    }, "the sample must exercise both the exact and the fuzzy branch"
    assert all(mention.ambiguous is False for mention in mentions)
    assert (
        token_labels.Mention(start=0, end=1, entity_ids=frozenset()).ambiguous
        is False
    )


def test_a_comma_joined_match_is_ambiguous() -> None:
    """The shape a BRENDA comma-joined name and a prose list share."""
    index = surface_forms.build_index({"enz1": ["FooA, FooB"]})
    text = "FooA, FooB was measured"

    mentions = token_labels.find_mentions(text, index)

    assert len(mentions) == 1
    assert mentions[0].ambiguous is True


@pytest.mark.parametrize("separator", [" ", "/", "-"])
def test_a_non_comma_multi_word_match_is_not_ambiguous(separator: str) -> None:
    """Only the comma shape is ambiguous; any other joiner is trusted as-is."""
    index = surface_forms.build_index({"enz1": ["FooA FooB"]})
    text = f"FooA{separator}FooB was measured"

    mentions = token_labels.find_mentions(text, index)

    assert len(mentions) == 1
    assert mentions[0].ambiguous is False


def test_a_single_word_match_is_never_ambiguous(index) -> None:
    """A single-word mention has no separator to be ambiguous about."""
    mentions = token_labels.find_mentions("catalase is active", index)

    assert len(mentions) == 1
    assert mentions[0].ambiguous is False


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
