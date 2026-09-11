"""The mention spans: the boundaries the per-token codes cannot carry."""

import dataclasses

import h5py
import numpy
import pytest
from conftest import (
    _BACTERIUM,
    _ENZYME,
    _STAMP,
    _empty_labels,
    _encode,
    _labels_over,
    _rows,
)
from d3text import surface_forms, token_labels

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


def test_two_mentions_split_by_a_space_are_one_code_run_and_two_spans(
    index,
) -> None:
    """The defect this record exists for, and the record answering it.

    A space produces no token, so the codes across two adjacent same-type
    mentions read as one run. The spans can still tell them apart.
    """
    text = "catalase catalase"
    encoding = _encode(text)

    labels = token_labels.document_token_labels(
        text, index, {"enz2"}, encoding["offset_mapping"]
    )

    assert _labels_over(encoding, labels.codes, 0, len(text)) == {_ENZYME}
    assert _rows(labels.spans) == [(0, 8, _ENZYME, 1), (9, 17, _ENZYME, 1)]


def test_mentioned_types_keeps_a_type_matched_only_by_a_non_gold_mention(
    index,
) -> None:
    """The building block a document-level abstention needs.

    The projected codes abstain and so cannot say an enzyme was mentioned at
    all; the spans keep the type regardless of the gold flag.
    """
    text = "cholesterol oxidase"
    labels = token_labels.document_token_labels(
        text, index, set(), _encode(text)["offset_mapping"]
    )

    assert token_labels.mentioned_types(labels.spans) == {_ENZYME}


def test_mentioned_types_min_chars_excludes_a_short_span_only() -> None:
    """A cutoff on span length, not on gold status: a long non-gold match
    still counts, a short one — gold or not — does not."""
    spans = numpy.array(
        [
            (0, 3, _ENZYME, 1),  # 3 chars: below an 8-char cutoff
            (10, 20, _BACTERIUM, 0),  # 10 chars: at/above it
        ],
        dtype=numpy.int32,
    )

    assert token_labels.mentioned_types(spans) == {_ENZYME, _BACTERIUM}
    assert token_labels.mentioned_types(spans, min_chars=8) == {_BACTERIUM}


def test_mentioned_types_min_chars_can_differ_per_type() -> None:
    """A per-type mapping gates each type independently, at the same spans.

    A uniform cutoff could not both exclude the 3-character span and include
    the 10-character one in the same document.
    """
    spans = numpy.array(
        [
            (0, 3, _ENZYME, 1),  # 3 chars
            (10, 20, _BACTERIUM, 0),  # 10 chars
        ],
        dtype=numpy.int32,
    )

    assert token_labels.mentioned_types(
        spans, min_chars={_ENZYME: 8, _BACTERIUM: 8}
    ) == {_BACTERIUM}
    # A type absent from the mapping falls back to no gate (0).
    assert token_labels.mentioned_types(spans, min_chars={_BACTERIUM: 8}) == {
        _ENZYME,
        _BACTERIUM,
    }
    # Raising bacteria's own cutoff above its span length excludes it, while
    # enzymes falls back to the mapping's implicit no-gate default and stays
    # in — the two types move independently under the one call.
    assert token_labels.mentioned_types(spans, min_chars={_BACTERIUM: 20}) == {
        _ENZYME
    }


def test_mentioned_types_of_no_mentions_is_empty() -> None:
    empty = numpy.empty((0, token_labels.SPAN_COLUMNS), dtype=numpy.int32)
    assert token_labels.mentioned_types(empty) == frozenset()


def test_mentioned_types_excludes_a_type_disagreement() -> None:
    """A mention `OUTSIDE`-coded for naming two gold types names neither."""
    index = surface_forms.build_index(
        {"bac11": ["angstrom widget"], "str12": ["angstrom widget"]}
    )
    text = "the angstrom widget again"

    labels = token_labels.document_token_labels(
        text, index, {"bac11", "str12"}, _encode(text)["offset_mapping"]
    )

    assert token_labels.mentioned_types(labels.spans) == set()


def test_an_abstaining_mention_keeps_its_span_and_the_type_it_would_have(
    index,
) -> None:
    """`IGNORE_INDEX` says only "do not look", and the span says the rest.

    The mention is still located and still known to be an enzyme name, which is
    the pair of facts the flat code destroys.
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
    annotation, so there is no type to record.
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

    Reconstruction has to reproduce the stored codes element for element,
    abstentions included — painting only the gold spans would come back
    `OUTSIDE` there. The encoding is deliberately narrow enough to overflow.
    """
    encoding = _encode(_SPAN_TEXT, max_length=32, stride=4)
    labels = token_labels.document_token_labels(
        _SPAN_TEXT, span_index, _SPAN_GOLD, encoding["offset_mapping"]
    )
    path = tmp_path / "labels.hdf5"

    with h5py.File(path, "w-", libver="latest") as store:
        token_labels.write_label_space(store, stamp=_STAMP)
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
    assert abstentions == {
        (_ENZYME, 0),
        (token_labels.OUTSIDE, 0),
    }, "both kinds of abstention have to be in the reconstruction"
    assert numpy.array_equal(rebuilt, stored.codes)


def test_the_codes_do_not_pin_the_document_length(index) -> None:
    """Why `text_length` is stored rather than inferred from the spans.

    Guessing it as the last mention's `end` builds a character array short by
    the whole tail; re-projecting the real offsets onto that short array is
    exactly the mismatch `project_onto_tokens` now refuses, rather than
    silently returning a plausible-looking array.
    """
    text = "catalase and cholesterol oxidase " + "z" * 300
    encoding = _encode(text)
    labels = token_labels.document_token_labels(
        text, index, {"enz2"}, encoding["offset_mapping"]
    )
    guess = int(labels.spans[:, token_labels.SPAN_END].max())
    offsets = numpy.asarray(encoding["offset_mapping"])

    assert (offsets[..., 1] > guess).any(), "no token runs past the guess"
    assert guess < labels.text_length
    assert token_labels.character_labels_from_spans(
        labels.text_length, labels.spans
    ).shape[0] == len(text)

    with pytest.raises(ValueError, match="past the"):
        token_labels.project_onto_tokens(
            token_labels.character_labels_from_spans(guess, labels.spans),
            encoding["offset_mapping"],
        )


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
        token_labels.write_label_space(store, stamp=_STAMP)
        token_labels.store_token_labels(store, "10822008", labels)

    with h5py.File(path, "r") as store:
        assert set(store["10822008"]) == {
            "codes",
            "spans",
            "entity_ids",
            "entity_masks",
            "candidate_counts",
            "candidate_ids",
            "anchors",
        }


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
        token_labels.write_label_space(store, stamp=_STAMP)
        token_labels.store_token_labels(store, "10822008", labels)

    with h5py.File(path, "r") as store:
        stored = token_labels.load_token_labels(store, "10822008")

    assert stored.spans.shape == (0, token_labels.SPAN_COLUMNS)
    assert (stored.codes[stored.codes != token_labels.IGNORE_INDEX] == 0).all()


def test_the_painting_matches_a_hand_written_character_array() -> None:
    """`character_labels_from_spans`, against an answer worked out by hand.

    The round-trip test applies the same rule on both sides, so an inclusive
    span end stays self-consistent while a comma inherits the enzyme type. Only
    an expected array written down with no projection in the loop pins it.
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


@pytest.mark.parametrize("shape", [(2, 3), (8,)], ids=["narrow", "flat"])
def test_document_labels_refuse_a_malformed_span_table(shape) -> None:
    """The dataclass, not only the painter, refuses a table it cannot read.

    The painter above is one consumer; the store writes the array as given,
    so a table that is not `[n, SPAN_COLUMNS]` has to be stopped where it is
    built or it lands on disk and fails only when read back.
    """
    with pytest.raises(ValueError, match="mention spans must be"):
        dataclasses.replace(
            _empty_labels(), spans=numpy.zeros(shape, dtype=numpy.int32)
        )


def test_document_labels_refuse_spans_without_their_candidates() -> None:
    """Mentions stored with no candidate IDs could be found and never linked,
    which is the store this layout replaced; the default is right only for a
    document that matched nothing."""
    with pytest.raises(ValueError, match="candidate sets for"):
        token_labels.DocumentLabels(
            codes=numpy.zeros((1, 8), dtype=numpy.int8),
            spans=numpy.array([[0, 4, _ENZYME, 0]], dtype=numpy.int32),
            text_length=4,
        )


def test_document_labels_refuse_an_anchor_on_a_fuzzy_mention() -> None:
    """A fuzzy mention names no entity to link to, so no token may lead a
    linker to it; an anchor there would have to be written by hand."""
    with pytest.raises(ValueError, match="names no exact mention"):
        token_labels.DocumentLabels(
            codes=numpy.zeros((1, 8), dtype=numpy.int8),
            spans=numpy.array([[0, 4, _ENZYME, 0]], dtype=numpy.int32),
            text_length=4,
            candidate_ids=(frozenset(),),
            anchors=numpy.array([[0, 0, 1, 3]], dtype=numpy.int32),
        )


def test_document_labels_refuse_a_negative_text_length() -> None:
    """A negative length paints as an empty document rather than failing."""
    with pytest.raises(ValueError, match="negative text length"):
        dataclasses.replace(_empty_labels(), text_length=-1)
