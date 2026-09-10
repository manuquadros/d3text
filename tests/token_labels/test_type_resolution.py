"""Resolving a token that has more than one candidate answer."""

import numpy
from conftest import _BACTERIUM, _ENZYME, _encode, _labels_over
from d3text import surface_forms, token_labels


def test_a_form_naming_several_entities_of_one_type_keeps_that_type() -> None:
    """`AS-A` names four separate enzymes; the token is still an enzyme.

    Ambiguity about *which* entity is not ambiguity about the target, so this
    is the case that must not abstain.
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

    The non-gold candidate is exactly what `IGNORE_INDEX` exists not to assert,
    so it does not get to make the answer ambiguous.
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

    Asserts the straddle actually happened: without the piece every token is
    one character wide and the case under test disappears silently.
    """
    encoding = _encode(text, extra=(piece,))
    offsets = numpy.asarray(encoding["offset_mapping"]).reshape(-1, 2)
    boundary = text.index("°")
    straddles = (offsets[:, 0] <= boundary) & (offsets[:, 1] > boundary + 1)
    assert straddles.any(), "no token straddles the mention boundary"
    return encoding, straddles


def test_a_token_straddling_two_types_is_ignored() -> None:
    """One subword, two mentions, two types — and no way to say both.

    Same resolution as an ambiguous form, one level down.
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

    A subword is not evidence that the mention it overlaps is absent.
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

    Letting the ignore win would let any neighbouring uncurated name delete a
    gold mention's supervision.
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

    The one-character-per-token vocabulary can never see an inclusive span end.
    Under whole-word pieces the comma of `catalase,` gets its own token on the
    very character a `labels[start : end + 1]` painting would spill onto.
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
