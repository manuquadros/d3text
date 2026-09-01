"""enzymeNER's coordinate convention, and its three rows that miss it.

The offsets are half-open — the opposite of S800's — so the two corpora cannot
share a reader, and reading this one the other way shifts every span by a
character with nothing raising. What makes the convention checkable is the
same property S800 has: the annotation table carries the surface text beside
the offsets. Three of the real 2,274 rows agree with neither reading, so
refusing the corpus outright would deliver nothing and trusting it would score
three spans against the wrong text; they are dropped and counted instead, and
the count is on the loaded corpus so a corpus rotting further is visible.
"""

import pathlib
from dataclasses import replace

import pytest
from d3text.datasets import enzymener

SENTENCES = "\n".join(
    (
        "PMC1\tS01\tThe pellet was digested with proteinase K overnight.",
        "PMC1\tS02\tTaq polymerase was added to the reaction mixture.",
        "PMC2\tS01\tAssays of beta-galactosidase were run in triplicate.",
    )
)

# `end` as the corpus writes it: one past the last character.
ANNOTATIONS = "\n".join(
    (
        "PMC1\tS01\t29\t41\tproteinase K",
        "PMC1\tS02\t0\t14\tTaq polymerase",
        "PMC2\tS01\t10\t28\tbeta-galactosidase",
    )
)


def _corpus(root: pathlib.Path, annotations: str = ANNOTATIONS) -> pathlib.Path:
    """Both tables on disk, byte-order marks and all."""
    (root / enzymener.SENTENCES).write_text(
        "\ufeff" + SENTENCES + "\n", encoding="utf8"
    )
    (root / enzymener.ANNOTATIONS).write_text(
        "\ufeff" + annotations + "\n", encoding="utf8"
    )
    return root


@pytest.fixture
def corpus_root(tmp_path: pathlib.Path) -> pathlib.Path:
    return _corpus(tmp_path)


def test_every_loaded_span_addresses_its_own_surface_form(
    corpus_root: pathlib.Path,
) -> None:
    """The self-checking assertion: the corpus states what each span says."""
    corpus = enzymener.load_enzymener(corpus_root)

    assert len(corpus.mentions) == 3
    for mention in corpus.mentions:
        text = corpus.texts[mention.document]
        assert text[mention.start : mention.end] == mention.surface


def test_the_end_offset_is_read_as_written(
    corpus_root: pathlib.Path,
) -> None:
    """S800's loader adds one to `end` and this one must not: the same
    conversion here overruns every span by a character."""
    corpus = enzymener.load_enzymener(corpus_root)
    first = corpus.mentions[0]
    text = corpus.texts[first.document]

    assert (first.start, first.end) == (29, 41)
    assert text[29:41] == "proteinase K"
    assert text[29:42] == "proteinase K "


def test_the_corpus_names_no_identifier(corpus_root: pathlib.Path) -> None:
    """Unlike S800, the annotations are spans and nothing else — the gold
    identifier comes from a nomenclature the corpus never saw."""
    corpus = enzymener.load_enzymener(corpus_root)

    assert {mention.external_id for mention in corpus.mentions} == {None}


def test_the_span_is_keyed_by_sentence_not_by_article(
    corpus_root: pathlib.Path,
) -> None:
    """Offsets are into the sentence, so two sentences of one article are two
    coordinate systems and pooling them would displace every span."""
    corpus = enzymener.load_enzymener(corpus_root)

    assert set(corpus.texts) == {"PMC1:S01", "PMC1:S02", "PMC2:S01"}
    assert corpus.mentions[1].document == "PMC1:S02"
    assert enzymener.article_of(corpus.mentions[1].document) == "PMC1"
    assert set(corpus.articles.values()) == {"PMC1", "PMC2"}


# --------------------------------------------------------------------------- #
# The bad-row policy: drop a handful, refuse a convention error                #
# --------------------------------------------------------------------------- #
def test_a_row_that_misses_its_surface_form_is_dropped_and_counted(
    tmp_path: pathlib.Path,
) -> None:
    """The real corpus has three of these in one sentence, each shifted by
    two characters. Refusing the corpus over them would deliver no
    measurement at all; scoring them would score the wrong text — so they are
    dropped, and the count is kept where a report can state it."""
    root = _corpus(tmp_path, ANNOTATIONS + "\nPMC1\tS01\t31\t43\tproteinase K")

    corpus = enzymener.load_enzymener(root, misplaced_limit=0.5)

    assert len(corpus.mentions) == 3
    assert [mention.start for mention in corpus.misplaced] == [31]


def test_a_wholesale_offset_error_is_refused_not_dropped(
    tmp_path: pathlib.Path,
) -> None:
    """The property the drop must not cost: a corpus read under the wrong
    convention misses nearly every surface form, and quietly dropping those
    would report a coverage of nothing rather than an error."""
    inclusive = "\n".join(
        (
            "PMC1\tS01\t29\t40\tproteinase K",
            "PMC1\tS02\t0\t13\tTaq polymerase",
            "PMC2\tS01\t10\t27\tbeta-galactosidase",
        )
    )
    root = _corpus(tmp_path, inclusive)

    with pytest.raises(ValueError, match="do not address"):
        enzymener.load_enzymener(root)


def test_an_annotation_on_an_absent_sentence_is_refused(
    tmp_path: pathlib.Path,
) -> None:
    """A span with no text is not a misplaced span, it is a corpus whose two
    tables disagree about what it contains."""
    root = _corpus(tmp_path, ANNOTATIONS + "\nPMC9\tS01\t0\t6\tlipase")

    with pytest.raises(ValueError, match="not in GoldSet"):
        enzymener.load_enzymener(root)


def test_a_malformed_row_is_refused(tmp_path: pathlib.Path) -> None:
    root = _corpus(tmp_path, "PMC1\tS01\t29\n")

    with pytest.raises(ValueError, match="3 fields"):
        enzymener.load_enzymener(root)


def test_non_integer_offsets_are_refused(tmp_path: pathlib.Path) -> None:
    root = _corpus(tmp_path, "PMC1\tS01\tx\t41\tproteinase K\n")

    with pytest.raises(ValueError, match="non-integer offsets"):
        enzymener.load_enzymener(root)


@pytest.mark.integration
def test_the_whole_corpus_agrees_with_the_half_open_reading() -> None:
    """The same check over the real 2,274 annotations: half-open is exact on
    2,271 of them and the three that address neither reading are the ones the
    policy drops. Converting as S800 needs would misplace so much of the
    corpus that the loader refuses it, which is why the two cannot share a
    reader."""
    root = pathlib.Path.home() / "Downloads" / "enzymeNER"
    if not (root / enzymener.ANNOTATIONS).exists():
        pytest.skip(f"no enzymeNER corpus at {root}")

    corpus = enzymener.load_enzymener(root)
    inclusive = [
        replace(mention, end=mention.end + 1) for mention in corpus.mentions
    ]

    assert len(corpus.mentions) == 2271
    assert len(corpus.misplaced) == 3
    with pytest.raises(ValueError, match="do not address"):
        enzymener.split_misplaced(corpus.texts, inclusive)
