from xmlparser import (chars, fromstring, merge_children, promote_spans,
                       reinsert_tags, remove_tags, tostring)

tryptophan = (
    "<div>with the indole precursor <sc>l</sc>-tryptophan, we observed</div>"
)
italic = "<div>with the <italic>indole precursor l-tryptophan</italic>, we observed</div>"
spaced_tag_string = (
    '<sec id="s4.12"><title>CE-ESI-TOF-MS target analysis.</title></sec>'
)
spanseq = (
    '<root><italic><span resource="#T3" typeof="ncbitaxon:Strain">P</span></italic>'
    '<span resource="#T3" typeof="ncbitaxon:Strain">2</span>'
    '<sub><span resource="#T3" typeof="ncbitaxon:Strain">1</span></sub></root>'
)
spanlifted = (
    '<root><span resource="#T3" typeof="ncbitaxon:Strain"><italic>P</italic></span>'
    '<span resource="#T3" typeof="ncbitaxon:Strain">2</span>'
    '<span resource="#T3" typeof="ncbitaxon:Strain"><sub>1</sub></span></root>'
)


def test_non_tag_chars_iterator_works() -> None:
    assert list(chars("precursor <sc>l</sc>-tryptophan")) == [
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
        "<sc>l</sc>",
        "-",
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


def test_remove_and_reinsert_tags_are_inverses() -> None:
    assert reinsert_tags(remove_tags(tryptophan), tryptophan) == tryptophan
    assert (
        reinsert_tags(remove_tags(spaced_tag_string), spaced_tag_string)
        == spaced_tag_string
    )


def test_remove_and_insert_with_annotation_is_valid_html() -> None:
    annotated_tryptophan = (
        "with the indole precursor "
        '<span typeof="entity">l</span>-tryptophan, we observed'
    )
    expected_tryptophan = (
        "<div>with the indole precursor "
        '<sc><span typeof="entity">l</span></sc>-tryptophan, we observed</div>'
    )
    assert (
        reinsert_tags(annotated_tryptophan, tryptophan) == expected_tryptophan
    )

    annotated_italic = (
        'with the indole precursor <span typeof="entity">'
        "l-tryptophan</span>, we observed"
    )
    expected_italic = (
        "<div>with the <italic>indole precursor "
        '<span typeof="entity">l-tryptophan</span></italic>, we observed</div>'
    )
    assert reinsert_tags(annotated_italic, italic) == expected_italic


def test_div_with_attribs():
    div = (
        '<div prefix="ncbitaxon: http://purl.obolibrary.org/obo/NCBITaxon_"> '
        "Crystallization and preliminary X-ray diffraction analysis of two N-terminal "
        "fragments of the DNA-cleavage domain of topoisomerase IV from <span"
        ' resource="#T1" typeof="ncbitaxon:Species">Staphylococcus aureus</span>'
    )
    assert (
        next(chars(div))
        == '<div prefix="ncbitaxon: http://purl.obolibrary.org/obo/NCBITaxon_"> '
    )


def test_spans_to_the_top():
    tree = fromstring(spanseq)
    assert tostring(promote_spans(tree), encoding="unicode") == spanlifted


def test_cousin_spans_should_be_merged_when_possible():
    tree = fromstring(spanlifted)
    assert (
        tostring(merge_children(tree), encoding="unicode")
        == '<root><span resource="#T3" typeof="ncbitaxon:Strain"><italic>P</italic>2<sub>1</sub></span></root>'
    )
