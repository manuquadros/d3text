from xmlparser import (get_tags, insert_tags, non_tag_chars, remove_tags,
                       tokenize_xml)

tryptophan = "with the indole precursor <sc>l</sc>-tryptophan, we observed"
italic = "with the <italic>indole precursor l-tryptophan</italic>, we observed"
spaced_tag_string = (
    '<sec id="s4.12"><title>CE-ESI-TOF-MS target analysis.</title><p>'
)


def test_tryptophan() -> None:
    assert tokenize_xml(tryptophan) == [
        "with the indole precursor ",
        "<sc>",
        "l",
        "</sc>",
        "-tryptophan, we observed",
    ]


def test_leading_and_trailing_tags() -> None:
    assert tokenize_xml(spaced_tag_string) == [
        '<sec id="s4.12">',
        "<title>",
        "CE-ESI-TOF-MS target analysis.",
        "</title>",
        "<p>",
    ]


def test_non_tag_chars_iterator_works() -> None:
    assert list(non_tag_chars("precursor <sc>l</sc>-tryptophan")) == [
        "p",
        "r",
        "e",
        "c",
        "u",
        "r",
        "s",
        "o",
        "r",
        " ",
        "<sc>l",
        "</sc>-",
        "t",
        "r",
        "y",
        "p",
        "t",
        "o",
        "p",
        "h",
        "a",
        "n",
    ]


def test_remove_and_insert_tags_are_inverses() -> None:
    assert (
        insert_tags(get_tags(tryptophan), remove_tags(tryptophan)) == tryptophan
    )
    assert (
        insert_tags(get_tags(spaced_tag_string), remove_tags(spaced_tag_string))
        == spaced_tag_string
    )


def test_remove_and_insert_with_annotation_is_valid_html() -> None:
    annotated_tryptophan = (
        "with the indole precursor <ent>l</ent>-tryptophan, we observed"
    )
    expected_tryptophan = (
        "with the indole precursor "
        "<sc><ent>l</ent></sc>-tryptophan, we observed"
    )
    assert (
        insert_tags(get_tags(tryptophan), annotated_tryptophan)
        == expected_tryptophan
    )

    annotated_italic = (
        "with the indole precursor <ent>l-tryptophan</ent>, we observed"
    )
    expected_italic = (
        "with the <italic>indole precursor "
        "<ent>l-tryptophan</ent></italic>, we observed"
    )
    assert insert_tags(get_tags(italic), annotated_italic) == expected_italic
