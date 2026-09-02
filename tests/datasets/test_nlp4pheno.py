"""NLP4Pheno's coordinate convention, pinned against the export's surfaces.

Its `end` is already one past the span where S800's is the last character, so
the conversion S800 needs is a one-character error here and nothing about it is
visible in a score: every span would gain a character, tokenize differently and
match a different dictionary entry. The export states what each span says, so
these tests assert against the annotation rather than against a number.
"""

import json
import pathlib

import pytest
from d3text.datasets import nlp4pheno

SENTENCE = "Staphylococcus aureus ATCC 6538 was grown on DSM medium."
OTHER = "Bacillus subtilis 168 is a laboratory strain."

TASKS = [
    {
        "id": 19135,
        "data": {"text": SENTENCE},
        "annotations": [
            {
                "result": [
                    {
                        "type": "labels",
                        "value": {
                            "start": 0,
                            "end": 31,
                            "text": "Staphylococcus aureus ATCC 6538",
                            "labels": ["STRAIN"],
                        },
                    },
                    {
                        "type": "labels",
                        "value": {
                            "start": 45,
                            "end": 55,
                            "text": "DSM medium",
                            "labels": ["MEDIUM"],
                        },
                    },
                    {
                        "type": "relation",
                        "from_id": "a",
                        "to_id": "b",
                        "labels": ["HasMedium"],
                    },
                ]
            }
        ],
    },
    {
        "id": 19136,
        "data": {"text": OTHER},
        "annotations": [
            {
                "result": [
                    {
                        "type": "labels",
                        "value": {
                            "start": 0,
                            "end": 21,
                            "text": "Bacillus subtilis 168",
                            "labels": ["STRAIN"],
                        },
                    }
                ]
            },
            {
                "result": [
                    {
                        "type": "labels",
                        "value": {
                            "start": 18,
                            "end": 21,
                            "text": "168",
                            "labels": ["STRAIN"],
                        },
                    }
                ]
            },
        ],
    },
]


@pytest.fixture
def export(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "project-10.json"
    path.write_text(json.dumps(TASKS), encoding="utf8")
    return path


def test_every_loaded_span_addresses_its_own_surface_form(
    export: pathlib.Path,
) -> None:
    """The self-checking assertion: shifting the offsets by one in either
    direction breaks every span, and only this comparison notices."""
    corpus = nlp4pheno.load_nlp4pheno(export)

    for spans in corpus.spans.values():
        for mention in spans:
            text = corpus.texts[mention.document]
            assert text[mention.start : mention.end] == mention.surface


def test_the_end_offset_is_read_as_one_past_the_span(
    export: pathlib.Path,
) -> None:
    """The whole of the S800 difference: `end` here already is what that
    corpus's `end + 1` has to be made into."""
    first = nlp4pheno.load_nlp4pheno(export).labelled(nlp4pheno.STRAIN)[0]

    assert (first.start, first.end) == (0, 31)
    assert first.surface == "Staphylococcus aureus ATCC 6538"
    assert SENTENCE[first.start : first.end] == first.surface


def test_spans_are_kept_apart_by_label(export: pathlib.Path) -> None:
    """Eight labels are annotated and one of them is scored, so a loader that
    pooled them would put media and diseases into the strain population."""
    corpus = nlp4pheno.load_nlp4pheno(export)

    assert len(corpus.labelled(nlp4pheno.STRAIN)) == 3
    assert [mention.surface for mention in corpus.labelled("MEDIUM")] == [
        "DSM medium"
    ]
    assert corpus.labelled("DISEASE") == ()


def test_the_corpus_names_no_identifiers(export: pathlib.Path) -> None:
    """Like enzymeNER and unlike S800: the annotators marked spans and
    assigned nothing, which is why the accession inside the span is what
    grounds it."""
    corpus = nlp4pheno.load_nlp4pheno(export)

    assert {
        mention.external_id for mention in corpus.labelled(nlp4pheno.STRAIN)
    } == {None}


def test_relations_are_counted_rather_than_read(export: pathlib.Path) -> None:
    """A silent zero would read as a corpus that annotates no relations, which
    is a different corpus from this one."""
    assert nlp4pheno.load_nlp4pheno(export).relations == 1


def test_every_annotation_of_a_task_is_read(export: pathlib.Path) -> None:
    """Two annotators marked one sentence differently. Keeping only the first
    would drop the disagreement here rather than at the scorer, which is the
    only place that can key it by span."""
    strains = nlp4pheno.load_nlp4pheno(export).labelled(nlp4pheno.STRAIN)

    assert [
        mention.surface for mention in strains if mention.document == "19136"
    ] == ["Bacillus subtilis 168", "168"]


def test_offsets_that_miss_their_surface_form_are_refused(
    tmp_path: pathlib.Path,
) -> None:
    """A shifted export is a different corpus, and reading it would lower a
    score with nothing anywhere disagreeing."""
    shifted = json.loads(json.dumps(TASKS))
    shifted[0]["annotations"][0]["result"][0]["value"]["end"] = 30
    path = tmp_path / "shifted.json"
    path.write_text(json.dumps(shifted), encoding="utf8")

    with pytest.raises(ValueError, match="do not address the spans"):
        nlp4pheno.load_nlp4pheno(path)


def test_a_task_with_no_text_is_refused(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "empty.json"
    path.write_text(json.dumps([{"id": 1, "data": {}}]), encoding="utf8")

    with pytest.raises(ValueError, match="carries no text"):
        nlp4pheno.load_nlp4pheno(path)


def test_a_file_that_is_not_a_task_list_is_refused(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "object.json"
    path.write_text(json.dumps({"tasks": TASKS}), encoding="utf8")

    with pytest.raises(ValueError, match="not the list of tasks"):
        nlp4pheno.load_nlp4pheno(path)
