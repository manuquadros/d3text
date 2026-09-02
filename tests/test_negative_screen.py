"""Screening a candidate document for enzyme mentions, and reporting the yield.

The measurement these support found the literal screen refuted by its own
control: rejecting a document on *any* exact enzyme match rejects most of the
psycholinguistics pool, which names no enzyme by construction, because the
index registers ubiquitous acronyms (`PCR`, `PBS`, `Yes`) as enzyme forms.

So the invariant worth pinning is the discrimination, not the plumbing: a
document whose only enzyme hit is an acronym has to be classified differently
by the two readings, and one whose only hit is a name like `RNA polymerase`
has to be rejected by both, whatever the index's own case policy makes of it.
Everything runs off small hand-built indexes, so no BRENDA data file and no
network is touched.
"""

import json
import pathlib
import subprocess
import sys
from collections.abc import Iterable, Iterator

import pytest
from beartype import beartype
from beartype.roar import BeartypeCallHintReturnViolation
from d3text import negative_screen, surface_forms
from tqdm import tqdm

_FORMS = {
    "enz1": ["catalase"],
    "enz2": ["nitrilase"],
    # A buffer BRENDA registers against an enzyme, and one of the commonest
    # words in the microbiology sample.
    "enz3": ["PBS"],
    # A multi-word name the index stores case-sensitively, and the EC number
    # `find_mentions` registers under the words of a section number.
    "enz5": ["RNA polymerase"],
    "enz6": ["5.3.2.1"],
    "bac1": ["Escherichia coli"],
}

# A three-letter form is symbol-like, so it is indexed case-sensitively, and
# BRENDA registers it against an enzyme and a strain alike — the shape `Pel`
# takes in the real index.
_AMBIGUOUS = {"enz4": ["Pel"], "str9": ["Pel"]}

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
    """`find_mentions` splits `5.3.2.1` into the four words an EC number is
    keyed under, so section numbers and confidence intervals resolve to
    enzymes. A form with no letter in it names nothing, and calling it
    descriptive would put those matches beyond every reading of the screen."""
    matches = negative_screen.matched_forms("see section 5.3.2.1 below", index)

    assert matches.symbolic == ("5.3.2.1",)
    assert negative_screen.DESCRIPTIVE.accepts(matches)
    assert not negative_screen.LITERAL.accepts(matches)


def test_an_acronym_disqualifies_only_the_literal_screen(index) -> None:
    """The discrimination the measurement rests on. `PBS` is a buffer BRENDA
    happens to register as an enzyme form, and a screen that rejects a
    document over it is measuring which papers avoid common acronyms."""
    matches = negative_screen.matched_forms("cells washed in PBS", index)

    assert matches.symbolic == ("PBS",)
    assert matches.descriptive == ()
    assert negative_screen.DESCRIPTIVE.accepts(matches)
    assert not negative_screen.LITERAL.accepts(matches)


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

    matches = negative_screen.matched_forms("Pel was purified", index)

    assert matches.symbolic == ("Pel",)


def test_every_occurrence_of_a_form_is_counted(index) -> None:
    """The frequencies are the finding: one buffer named four times is what a
    bare match count hides."""
    matches = negative_screen.matched_forms(
        "PBS, then catalase, then PBS again, then PBS", index
    )

    assert matches.symbolic == ("PBS", "PBS", "PBS")
    assert matches.descriptive == ("catalase",)


def _pool(path: pathlib.Path, rows: list[dict[str, object]]) -> pathlib.Path:
    """A candidate pool in the noise pool's shape: ndjson with a `body`."""
    path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows), encoding="utf8"
    )
    return path


def _row(pubmed_id: str, body: str, journal: str = "Microorganisms") -> dict:
    return {
        "pubmed_id": pubmed_id,
        "abstract": "",
        "body": f"<jats:body><jats:p>{body}</jats:p></jats:body>",
        "journal": journal,
    }


def test_one_pass_tallies_every_screen_it_was_given(index, tmp_path) -> None:
    """The two readings disagree about the same matches, not about what the
    matches are, so re-running the expensive step per reading would be paying
    twice for the same table."""
    path = _pool(
        tmp_path / "pool.json",
        [
            _row("1", "Escherichia coli was grown overnight"),
            _row("2", "cells washed in PBS"),
            _row("3", "catalase and catalase again"),
        ],
    )

    descriptive, literal = negative_screen.survey_corpus(path, index)

    assert (descriptive.screen, literal.screen) == (
        negative_screen.DESCRIPTIVE,
        negative_screen.LITERAL,
    )
    assert descriptive.negatives == 2
    assert literal.negatives == 1


def test_a_document_naming_a_multi_word_enzyme_is_no_negative(
    index, tmp_path
) -> None:
    """The same regression at the level the screen is used at: a pool whose
    documents name `RNA polymerase` throughout must not come back certified
    enzyme-free."""
    path = _pool(
        tmp_path / "pool.json", [_row("1", "RNA polymerase was purified")]
    )

    (descriptive,) = negative_screen.survey_corpus(
        path, index, screens=(negative_screen.DESCRIPTIVE,)
    )

    assert descriptive.negatives == 0
    assert descriptive.descriptive_forms == {"RNA polymerase": 1}


def test_the_survey_reports_the_rate_and_the_match_mass(
    index, tmp_path
) -> None:
    """The rate alone cannot say whether what it counted was an enzyme; the
    form tables are what make an index false positive visible."""
    path = _pool(
        tmp_path / "pool.json",
        [
            _row("1", "Escherichia coli was grown overnight"),
            _row("2", "cells washed in PBS"),
            _row("3", "catalase and catalase again"),
            _row("4", "nitrilase was assayed"),
        ],
    )

    (descriptive,) = negative_screen.survey_corpus(
        path, index, screens=(negative_screen.DESCRIPTIVE,)
    )

    assert descriptive.documents == 4
    assert descriptive.negatives == 2
    assert descriptive.negative_rate == 0.5
    assert descriptive.match_counts == {0: 2, 1: 1, 2: 1}
    assert descriptive.descriptive_forms == {"catalase": 2, "nitrilase": 1}
    assert descriptive.symbolic_forms == {"PBS": 1}
    assert len(descriptive.lengths) == 4
    assert len(descriptive.negative_lengths) == descriptive.negatives


def test_the_survey_reads_the_noise_pool_shape(index, tmp_path) -> None:
    """`body` is the column the PMC dumps call the fulltext, and both halves
    arrive as JATS markup: a screen matching against the tags would find no
    enzyme in a document that names one."""
    path = _pool(
        tmp_path / "pool.json",
        [
            {
                "pubmed_id": "9",
                "abstract": "<jats:p>catalase</jats:p>",
                "body": "<jats:body><jats:p>nothing</jats:p></jats:body>",
            }
        ],
    )

    (descriptive,) = negative_screen.survey_corpus(
        path, index, screens=(negative_screen.DESCRIPTIVE,)
    )

    assert descriptive.documents == 1
    assert descriptive.descriptive_forms == {"catalase": 1}


def test_the_survey_characterises_survivors_against_the_sample(
    index, tmp_path
) -> None:
    """A journal that is a third of the sample and all of the survivors is the
    biased slice the pool must not be: readable only from both counts."""
    path = _pool(
        tmp_path / "pool.json",
        [
            _row("1", "a survey of communities", journal="Ecology"),
            _row("2", "catalase rose", journal="Microorganisms"),
            _row("3", "nitrilase rose", journal="Microorganisms"),
        ],
    )

    (descriptive,) = negative_screen.survey_corpus(
        path,
        index,
        screens=(negative_screen.DESCRIPTIVE,),
        metadata_columns=("journal",),
    )

    assert descriptive.screened_values == {
        "journal": {"Ecology": 1, "Microorganisms": 2}
    }
    assert descriptive.negative_values == {"journal": {"Ecology": 1}}


def test_a_metadata_column_the_pool_lacks_is_no_error(index, tmp_path) -> None:
    path = _pool(tmp_path / "pool.json", [_row("1", "nothing")])

    (descriptive,) = negative_screen.survey_corpus(
        path,
        index,
        screens=(negative_screen.DESCRIPTIVE,),
        metadata_columns=("mesh_headings",),
    )

    assert descriptive.screened_values == {}
    assert descriptive.negatives == 1


def test_the_limit_caps_the_pass(index, tmp_path) -> None:
    """The measurement is a sample, and the rows are only comparable when
    every pool contributes the same number of documents."""
    path = _pool(
        tmp_path / "pool.json",
        [_row(str(n), "catalase") for n in range(10)],
    )

    (descriptive,) = negative_screen.survey_corpus(
        path, index, screens=(negative_screen.DESCRIPTIVE,), limit=3
    )

    assert descriptive.documents == 3


def test_the_summary_names_the_screen_and_the_matches(index, tmp_path) -> None:
    path = _pool(
        tmp_path / "pool.json",
        [_row("1", "catalase and PBS"), _row("2", "nothing here")],
    )

    descriptive, literal = negative_screen.survey_corpus(path, index)

    assert "catalase (1)" in descriptive.summary()
    assert "PBS (1)" in descriptive.summary()
    assert "only a descriptive name rejects" in descriptive.summary()
    assert "every exact match rejects" in literal.summary()


def test_the_comparison_puts_the_controls_beside_the_candidates(
    index, tmp_path
) -> None:
    """The table that made the failure visible: a low yield reads as a hard
    corpus until the control beside it, negative by construction, is rejected
    at the same rate."""
    candidates = _pool(tmp_path / "candidates.json", [_row("1", "PBS")])
    control = _pool(tmp_path / "control.json", [_row("2", "nothing")])

    table = negative_screen.comparison(
        {
            path.name: negative_screen.survey_corpus(path, index)
            for path in (candidates, control)
        }
    )

    assert "descriptive" in table
    assert "literal" in table
    assert "candidates.json" in table
    assert "control.json" in table


def test_the_comparison_refuses_rows_screened_differently(
    index, tmp_path
) -> None:
    """One column, two screens, no way to tell from the table which is which."""
    path = _pool(tmp_path / "pool.json", [_row("1", "PBS")])

    with pytest.raises(ValueError, match="screened differently"):
        negative_screen.comparison(
            {
                "a": negative_screen.survey_corpus(
                    path, index, screens=(negative_screen.DESCRIPTIVE,)
                ),
                "b": negative_screen.survey_corpus(
                    path, index, screens=(negative_screen.LITERAL,)
                ),
            }
        )


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


def test_the_progress_wrapper_delivers_every_document() -> None:
    """The pass reads a one-shot stream behind a bar, so anything that samples
    the bar eats a document: `beartype` deep-checks a `Sized` return value by
    pulling one item off it, and a `tqdm` is `Sized`. Handing the bar back
    rather than yielding from it would therefore drop the first document of
    every pass, with nothing raised and no count to notice it by."""
    rows = [(str(n), f"document {n}") for n in range(5)]

    delivered = list(negative_screen._limited(iter(rows), len(rows), None))

    assert delivered == rows


def test_a_returned_progress_bar_loses_its_first_item() -> None:
    """The trap itself, recorded where the site that avoids it is tested.
    Exact rather than approximate because the loss is exactly one item and
    always the first, whatever the length of the stream."""

    @beartype
    def returned(rows: Iterator[int]) -> Iterable[int]:
        return tqdm(rows, disable=True)

    @beartype
    def yielded(rows: Iterator[int]) -> Iterable[int]:
        yield from tqdm(rows, disable=True)

    assert list(yielded(iter(range(5)))) == [0, 1, 2, 3, 4]
    assert list(returned(iter(range(5)))) == [1, 2, 3, 4]


def test_the_narrower_annotation_refuses_a_returned_progress_bar() -> None:
    """`Iterator` is not `Sized`, so it buys no deep check — and a `tqdm`
    carries no `__next__`, so returning one under it is rejected outright.
    That is the second guard on the pass: rewritten to return its bar, it
    raises here rather than quietly shipping one document fewer."""

    @beartype
    def returned(rows: Iterator[int]) -> Iterator[int]:
        return tqdm(rows, disable=True)

    with pytest.raises(BeartypeCallHintReturnViolation):
        returned(iter(range(5)))
