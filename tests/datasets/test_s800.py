"""S800's coordinate convention, pinned against the corpus's own surfaces.

The corpus writes `end` inclusive and this package's spans are half-open, so
the loader adds one. Nothing about that is visible in a score: read half-open,
every span loses its last character, matches a slightly different dictionary
entry, and the linking accuracy comes out a few points low with no error
anywhere. What makes the conversion checkable is that the annotation table
carries the surface text beside the offsets — so the corpus validates its own
convention, and these tests assert against the annotation rather than against
a number someone wrote down.
"""

import pathlib

import pytest
from d3text.datasets import s800

# `end` as the corpus writes it: the last character's index, not one past it.
# 'Escherichia coli' occupies characters 10..25 of the abstract below.
ANNOTATIONS = "\n".join(
    (
        "562\tspecies001:111\t10\t25\tEscherichia coli",
        "1423\tspecies001:111\t45\t61\tBacillus subtilis",
        "5833\tspecies002:222\t4\t24\tPlasmodium falciparum",
    )
)

TEXTS = {
    "species001": "Growth of Escherichia coli was compared with Bacillus "
    "subtilis in vitro.",
    "species002": "The Plasmodium falciparum genome.",
}


@pytest.fixture
def corpus_root(tmp_path: pathlib.Path) -> pathlib.Path:
    (tmp_path / s800.ABSTRACTS).mkdir()
    (tmp_path / s800.ANNOTATIONS).write_text(
        ANNOTATIONS + "\n", encoding="utf8"
    )
    for document, text in TEXTS.items():
        (tmp_path / s800.ABSTRACTS / f"{document}.txt").write_text(
            text, encoding="utf8"
        )
    return tmp_path


def test_every_loaded_span_addresses_its_own_surface_form(
    corpus_root: pathlib.Path,
) -> None:
    """The self-checking assertion: the corpus states what each span says, so
    a half-open reading of an inclusive `end` truncates every one of them."""
    corpus = s800.load_s800(corpus_root)

    for mention in corpus.mentions:
        text = corpus.texts[mention.document]
        assert text[mention.start : mention.end] == mention.surface


def test_the_inclusive_end_becomes_one_past_the_span(
    corpus_root: pathlib.Path,
) -> None:
    corpus = s800.load_s800(corpus_root)
    first = corpus.mentions[0]

    assert (first.start, first.end) == (10, 26)
    assert first.surface == "Escherichia coli"
    assert first.external_id == "562"


def test_a_half_open_reading_would_lose_the_last_character() -> None:
    """What the conversion is worth, stated once: without it every span comes
    back one character short and nothing raises."""
    text = TEXTS["species001"]

    assert text[10:25] == "Escherichia col"
    assert text[10:26] == "Escherichia coli"


def test_offsets_that_miss_their_surface_form_are_refused(
    corpus_root: pathlib.Path,
) -> None:
    """A corpus whose offsets do not address what it says they address is a
    different corpus, and scoring it would produce a plausible wrong number
    rather than a failure."""
    (corpus_root / s800.ANNOTATIONS).write_text(
        "562\tspecies001:111\t11\t25\tEscherichia coli\n", encoding="utf8"
    )

    with pytest.raises(ValueError, match="do not address"):
        s800.load_s800(corpus_root)


def test_the_document_id_splits_into_file_stem_and_pubmed_id(
    corpus_root: pathlib.Path,
) -> None:
    corpus = s800.load_s800(corpus_root)

    assert corpus.pubmed_ids == {"species001": "111", "species002": "222"}
    assert set(corpus.texts) == {"species001", "species002"}


def test_every_species_is_annotated_not_only_the_curated_ones(
    corpus_root: pathlib.Path,
) -> None:
    """The property no BRENDA-derived artifact has: the corpus annotates a
    non-bacterial species the same as a bacterial one, so the mentions
    outside BRENDA's curation stay countable instead of vanishing."""
    corpus = s800.load_s800(corpus_root)

    assert {mention.external_id for mention in corpus.mentions} == {
        "562",
        "1423",
        "5833",
    }


def test_a_malformed_row_is_refused(corpus_root: pathlib.Path) -> None:
    (corpus_root / s800.ANNOTATIONS).write_text(
        "562\tspecies001:111\t10\n", encoding="utf8"
    )

    with pytest.raises(ValueError, match="3 fields"):
        s800.load_s800(corpus_root)


@pytest.mark.integration
def test_the_whole_corpus_agrees_with_the_inclusive_reading() -> None:
    """The same check over the real 3,708 annotations, where the conversion
    is worth having: the inclusive reading is exact on every row and the
    half-open one matches none."""
    root = pathlib.Path.home() / "Downloads" / "Species-800"
    if not (root / s800.ANNOTATIONS).exists():
        pytest.skip(f"no Species-800 corpus at {root}")

    corpus = s800.load_s800(root)

    assert len(corpus.mentions) == 3708
    assert not [
        mention
        for mention in corpus.mentions
        if corpus.texts[mention.document][mention.start : mention.end - 1]
        == mention.surface
    ]
