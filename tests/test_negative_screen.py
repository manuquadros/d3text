"""Screening a candidate document for enzyme mentions.

The measurement these support found the literal screen refuted by its own
control: rejecting a document on *any* exact enzyme match rejects most of the
psycholinguistics pool, which names no enzyme by construction, because the
index registered ubiquitous acronyms (`PCR`, `PBS`, `Yes`) as enzyme forms.
None of those three carries an ID any more, but `CAMP` is the same shape one
character longer and still does.

So the invariant worth pinning is the discrimination, not the plumbing: a
document whose only enzyme hit is an acronym has to be classified differently
by the two readings, and one whose only hit is a name like `RNA polymerase`
has to be rejected by both, whatever the index's own case policy makes of it.
Everything runs off small hand-built indexes, so no BRENDA data file and no
network is touched.
"""

import subprocess
import sys

import pytest
from d3text import negative_screen, surface_forms, token_labels

_FORMS = {
    "enz1": ["catalase"],
    "enz2": ["nitrilase"],
    # A second messenger BRENDA registers against an enzyme, and an ordinary
    # English word once its case is folded away.
    "enz3": ["CAMP"],
    # A multi-word name the index stores case-sensitively, and a form of bare
    # digits, which `find_mentions` registers under the words of a section
    # number.
    "enz5": ["RNA polymerase"],
    "enz6": ["5.3.2.1"],
    "bac1": ["Escherichia coli"],
}

# A short form is symbol-like, so it is indexed case-sensitively, and BRENDA
# registers it against an enzyme and a strain alike — the shape `ApaI` takes
# in the real index.
_AMBIGUOUS = {"enz4": ["ApaI"], "str9": ["ApaI"]}

_NEAR_MISS = "catalse"
"""A typo of `catalase`, scoring above `FUZZY_CUTOFF` and in no table."""


@pytest.fixture
def index() -> surface_forms.SurfaceFormIndex:
    return surface_forms.build_index(_FORMS)


def test_a_document_naming_no_enzyme_survives_both_screens(index) -> None:
    matches = negative_screen.matched_forms(
        "Escherichia coli was grown overnight.", index
    )

    assert matches == negative_screen.Matches()
    assert negative_screen.DESCRIPTIVE.accepts(matches)
    assert negative_screen.LITERAL.accepts(matches)


def test_a_descriptive_name_disqualifies_under_either_screen(index) -> None:
    matches = negative_screen.matched_forms("catalase activity rose", index)

    assert matches.descriptive == ("catalase",)
    assert not negative_screen.DESCRIPTIVE.accepts(matches)
    assert not negative_screen.LITERAL.accepts(matches)


@pytest.mark.parametrize(
    "form",
    [
        "RNA polymerase",
        "ATP synthase",
        "DNA ligase",
        "cytochrome P450 monooxygenase",
        "alcohol dehydrogenase",
        "catalase",
    ],
)
def test_a_name_of_more_than_one_word_is_descriptive(form) -> None:
    """Case is load-bearing for a single token and for nothing longer, so the
    index's own case policy calls four of these six symbols. Reusing it here
    binned most real enzyme names as acronyms."""
    assert negative_screen.is_descriptive(form)


@pytest.mark.parametrize(
    "form", ["PBS", "DLD", "Yes", "LasI", "NADH", "renin", "5.3.2.1"]
)
def test_a_short_or_capitalised_single_word_is_not(form) -> None:
    """The forms the index is least reliable on: one token, either too short
    to separate from a figure label or capitalised where prose is not."""
    assert not negative_screen.is_descriptive(form)


def test_a_multi_word_name_disqualifies_under_either_screen(index) -> None:
    """The regression the split exists to prevent. `RNA polymerase` carries
    an uppercase character after the first, so a screen that asked whether
    case mattered read it as an acronym and let it pass."""
    matches = negative_screen.matched_forms(
        "RNA polymerase was purified", index
    )

    assert matches.descriptive == ("RNA polymerase",)
    assert matches.symbolic == ()
    assert not negative_screen.DESCRIPTIVE.accepts(matches)
    assert not negative_screen.LITERAL.accepts(matches)


def test_a_bare_number_sequence_counts_as_a_symbol(index) -> None:
    """`find_mentions` splits a bare number sequence into its digits, so a
    form registered under nothing else answers to every section number and
    confidence interval of that shape. A form with no letter in it names
    nothing, and calling it descriptive would put those matches beyond every
    reading of the screen."""
    matches = negative_screen.matched_forms("see section 5.3.2.1 below", index)

    assert matches.symbolic == ("5.3.2.1",)
    assert negative_screen.DESCRIPTIVE.accepts(matches)
    assert not negative_screen.LITERAL.accepts(matches)


@pytest.mark.parametrize(
    "form",
    [
        "met: (1",
        "N = 35",
        "or ((5",
        "P =.37",
        "Ex = 1",
        "or 5",
        "PP = 1",
        "PP\N{MINUS SIGN}1",
        "IF=2",
        "M = 2",
        "h (SD",
        "CAS: 1",
        "SP 1",
        "SP 2",
    ],
)
def test_a_symbol_spaced_out_by_notation_is_still_a_symbol(form) -> None:
    """BRENDA registers `PP-1`, `SP-1`, `or-5` and their like, and the index
    keys a form by its words, so it finds them across a statistic's `PP = 1`,
    a survey item's `SP 1` and the `or 5` of "4 or 5". Joined, each is as short
    as the `PP1` the screen already reads as a symbol."""
    assert not negative_screen.is_descriptive(form)


def test_a_statistic_read_as_a_registered_symbol_is_no_name() -> None:
    """The same regression where the screen is used, against the form as
    BRENDA spells it: a document reporting a statistic must not be certified
    enzyme-bearing by it."""
    index = surface_forms.build_index({"enz7": ["PP-1"]})

    matches = negative_screen.matched_forms("gains were small (PP = 1)", index)

    assert matches.symbolic == ("PP = 1",)
    assert matches.descriptive == ()
    assert negative_screen.DESCRIPTIVE.accepts(matches)
    assert not negative_screen.LITERAL.accepts(matches)


@pytest.mark.parametrize(
    "form",
    [
        "EC 6.1.1.1",
        "DNase I",
        "UGT 1A1",
        "HIV-1 RT",
        "complex I",
        "BL21(DE3)",
        "NAD(P)H oxidase",
        "NADH:ubiquinone oxidoreductase",
        "pyruvate, orthophosphate dikinase",
        "catechol 2,3-dioxygenase",
        "Zea mays",
    ],
)
def test_a_punctuated_name_stays_descriptive(form) -> None:
    """Length is what separates these from the notation above, and punctuation
    is not: names carry numerals, colons, commas and parentheses too, and
    `EC 6.1.1.1` is the one spelling that names an enzyme unambiguously."""
    assert negative_screen.is_descriptive(form)


@pytest.mark.parametrize("form", ["E. coli", "E.coli", "T. ni"])
def test_an_abbreviated_binomial_is_a_name(form) -> None:
    """`E. coli` is as short as a symbol only because its genus is cut to an
    initial, so length alone would file it with `PP = 1`."""
    assert negative_screen.is_descriptive(form)


@pytest.mark.parametrize(
    "form",
    ["B. sp. A3", "S. ce56", "C. phi6", "C. aeh1", "A. sp. 1", "Mus sp."],
)
def test_an_organism_name_survives_a_third_token_or_digits(form) -> None:
    """A third token (`sp.` plus a strain number) or a non-alphabetic
    epithet used to fall outside the exemption: the old regex required the
    whole span to be exactly one genus initial and a lowercase-only
    epithet, nothing else."""
    assert negative_screen.is_descriptive(form)


def test_the_widened_exemption_still_excludes_a_bare_statistic() -> None:
    """`PP = 1` joins to symbol length too, but carries no genus, so
    widening the organism exemption must not let it through."""
    assert not negative_screen.is_descriptive("PP = 1")


@pytest.mark.parametrize("form", ["NADP-ME", "HMG-CoA", "GAPDH-S"])
def test_a_hyphenated_acronym_over_the_bar_is_descriptive(form) -> None:
    """A hyphenated acronym whose words joined overflow `SYMBOL_MAX_LENGTH`
    reads as descriptive despite looking symbolic, unlike a short one of the
    same shape (`PEP-CK`, `CPT-II`)."""
    assert negative_screen.is_descriptive(form)


@pytest.mark.parametrize("form", ["PEP-CK", "CPT-II"])
def test_a_hyphenated_acronym_within_the_bar_is_not(form) -> None:
    """Short enough joined, the same shape stays symbolic and so ignored by
    the descriptive default."""
    assert not negative_screen.is_descriptive(form)


def test_an_acronym_disqualifies_only_the_literal_screen(index) -> None:
    """The discrimination the measurement rests on. `CAMP` is a messenger
    BRENDA happens to register as an enzyme form, and a screen that rejects a
    document over it is measuring which papers avoid common acronyms."""
    matches = negative_screen.matched_forms("CAMP levels rose", index)

    assert matches.symbolic == ("CAMP",)
    assert matches.descriptive == ()
    assert negative_screen.DESCRIPTIVE.accepts(matches)
    assert not negative_screen.LITERAL.accepts(matches)


def test_capitalised_and_lowercase_catalase_classify_the_same(index) -> None:
    """`CATALASE` is found only because the index folds `catalase`'s case,
    so the two spellings must be read as the same kind of match rather than
    flipping class on casing alone, and a document naming the enzyme only
    in capitals must still be rejected by the default screen."""
    upper = negative_screen.matched_forms("CATALASE activity rose", index)
    lower = negative_screen.matched_forms("catalase activity rose", index)

    assert upper.descriptive == ("CATALASE",)
    assert lower.descriptive == ("catalase",)
    assert not negative_screen.DESCRIPTIVE.accepts(upper)
    assert not negative_screen.DESCRIPTIVE.accepts(lower)


def test_a_short_folded_form_written_as_a_statistic_stays_symbolic() -> None:
    """Neutralising casing must not bypass the joined-length rule: `hsp-70`
    and `cyt b5` are folded because their stripped length exceeds
    `SYMBOL_MAX_LENGTH`, but read across a statistic's `HSP = 70` they still
    join to symbol length and name no organism, so they must stay symbols
    even though their key is registered case-insensitively."""
    index = surface_forms.build_index({"enz8": ["cyt b5"], "enz9": ["hsp-70"]})

    matches = negative_screen.matched_forms(
        "gains were small, HSP = 70 in", index
    )

    assert matches.symbolic == ("HSP = 70",)
    assert matches.descriptive == ()
    assert negative_screen.DESCRIPTIVE.accepts(matches)


def test_a_near_miss_is_counted_apart_from_an_exact_match(index) -> None:
    """A fuzzy hit may withhold a type but never assert one, so neither screen
    rejects on it unless asked: it cannot establish that the document names an
    enzyme at all."""
    matches = negative_screen.matched_forms(f"{_NEAR_MISS} was assayed", index)

    assert matches.fuzzy == (_NEAR_MISS,)
    assert matches.descriptive == ()
    assert negative_screen.DESCRIPTIVE.accepts(matches)
    assert negative_screen.LITERAL.accepts(matches)
    assert not negative_screen.Screen(fuzzy_disqualifies=True).accepts(matches)


def test_an_ambiguous_mention_screens_the_same_way_a_fuzzy_one_does(
    index, monkeypatch
) -> None:
    """`Mention.ambiguous` has no producer yet, but the screen has to be
    ready for one: an ambiguous hit withholds a type exactly like a fuzzy
    one, so it must land in the same non-asserting bucket."""
    monkeypatch.setattr(
        negative_screen,
        "find_mentions",
        lambda text, index, max_gap: [
            token_labels.Mention(
                start=0, end=8, entity_ids=frozenset({"enz1"}), ambiguous=True
            )
        ],
    )

    matches = negative_screen.matched_forms("whatever", index)

    assert matches.fuzzy == ("whatever",)
    assert matches.descriptive == ()
    assert matches.symbolic == ()
    assert negative_screen.DESCRIPTIVE.accepts(matches)
    assert negative_screen.LITERAL.accepts(matches)
    assert not negative_screen.Screen(fuzzy_disqualifies=True).accepts(matches)


def test_a_mention_of_another_type_is_not_an_enzyme_match(index) -> None:
    """The screen is per type: an organism the index knows says nothing about
    whether the document names an enzyme."""
    organisms = negative_screen.matched_forms(
        "Escherichia coli", index, prefix="bac"
    )

    assert organisms.descriptive == ("Escherichia coli",)
    assert negative_screen.matched_forms("Escherichia coli", index) == (
        negative_screen.Matches()
    )


def test_an_ambiguous_form_counts_against_the_candidate() -> None:
    """A form the index cannot resolve to one type is still a form that could
    be an enzyme, which is what a document claiming to name none must not
    hold."""
    index = surface_forms.build_index(_AMBIGUOUS)

    matches = negative_screen.matched_forms("ApaI was purified", index)

    assert matches.symbolic == ("ApaI",)


def test_every_occurrence_of_a_form_is_counted(index) -> None:
    """The frequencies are the finding: one acronym named three times is what
    a bare match count hides."""
    matches = negative_screen.matched_forms(
        "CAMP, then catalase, then CAMP again, then CAMP", index
    )

    assert matches.symbolic == ("CAMP", "CAMP", "CAMP")
    assert matches.descriptive == ("catalase",)


def test_the_screen_does_not_import_the_data_layer(tmp_path) -> None:
    """Screening a corpus must not cost the whole BRENDA stack, which writes
    `lpsn.log` into the cwd at import. Checked in a subprocess: the suite as a
    whole imports `d3text.data`, so an in-process check would pass either
    way."""
    probe = (
        "import sys; import d3text.negative_screen; "
        "print(any(m.startswith(('d3text.data', 'brenda_references', "
        "'lpsn_interface')) for m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=True,
    )

    assert result.stdout.strip().endswith(
        "False"
    ), "d3text.negative_screen pulled in the BRENDA data layer"
