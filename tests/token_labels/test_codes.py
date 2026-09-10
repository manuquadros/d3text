"""The label space: which integer means which entity type."""

import numpy
import pytest
from conftest import _BACTERIUM, _ENZYME, _encode, _labels_over, _rows
from d3text import surface_forms, token_labels
from d3text.schema import BRENDA_SCHEMA


def test_every_declared_entity_type_has_its_own_code() -> None:
    """The label space covers the schema, one distinct code per type.

    Not asserted against a literal list, which would only restate the schema: a
    collision would train two types onto one column without failing anywhere.
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


def test_an_empty_label_space_is_rejected() -> None:
    """A space of no types passes every other check with nothing to check.

    Its column counts agree at zero, nothing is duplicated and zero codes fit
    any dtype, so without its own refusal it constructs and labels every
    token `OUTSIDE`.
    """
    with pytest.raises(ValueError, match="at least one type"):
        token_labels.LabelSpace(types=(), prefixes=())


def test_a_label_space_with_a_duplicate_type_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate type names"):
        token_labels.LabelSpace(
            types=("enzymes", "enzymes"), prefixes=("enz", "bac")
        )


def test_a_label_space_with_a_duplicate_id_prefix_is_rejected() -> None:
    """The refusal that stands between a schema edit and wrong numbers.

    `by_prefix` is a dict keyed on the prefixes, so a repeated one keeps a
    single entry: the dropped type never fails a shape check, and every ID
    carrying that prefix is coded as its neighbour instead.
    """
    with pytest.raises(ValueError, match="duplicate ID prefixes"):
        token_labels.LabelSpace(
            types=("enzymes", "bacteria"), prefixes=("enz", "enz")
        )


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

    A space with a fifth type makes both possible drops visible: its prefix is
    unknown to `BRENDA_LABELS` so the typing half cannot fall back silently,
    and its code is outside `BRENDA_LABELS.codes` so the projection half cannot
    either.
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
