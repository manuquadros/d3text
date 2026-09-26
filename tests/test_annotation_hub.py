import logging

import pytest

from d3text import annotation_hub
from d3text.identifier_bridge import NCBI_TAXID, BridgeRow, IdentifierBridge
from d3text.schema import BRENDA_SCHEMA

ABSTRACT = "<p>Escherichia coli makes an enzyme.</p>"
BODY = (
    "<sec><p>In the cytoplasm of every growing cell the <i>catalase</i> is "
    "abundant.</p></sec>"
)
# document_text: "Escherichia coli makes an enzyme.\nIn the cytoplasm of
# every growing cell the catalase is abundant." -- the body starts at 34.
RECORD: annotation_hub.PredictionRecord = {
    "document": "12345",
    "spans": [
        {
            "start": 0,
            "end": 16,
            "surface": "Escherichia coli",
            "entity_type": "bacteria",
            "entity_ids": ["bac1"],
        },
        {
            "start": 77,
            "end": 85,
            "surface": "catalase",
            "entity_type": "enzymes",
            "entity_ids": ["enz7"],
        },
        {
            "start": 12,
            "end": 37,
            "surface": "coli makes an enzyme.\nIn ",
            "entity_type": "bacteria",
            "entity_ids": ["bac1"],
        },
        {
            "start": 51,
            "end": 59,
            "surface": "of every",
            "entity_type": "bacteria",
            "entity_ids": None,
        },
    ],
    "relations": [
        {"predicate": "HasEnzyme", "arguments": [["enz7"], ["bac1"]]}
    ],
}


def _annotation() -> annotation_hub.Annotation:
    bridge = IdentifierBridge.from_rows(
        NCBI_TAXID, [BridgeRow("bac1", "562", "lpsn_id")]
    )
    return annotation_hub.annotation(
        RECORD,
        ABSTRACT,
        BODY,
        {"NCBITaxon": bridge},
        {"HasEnzyme": "ex:hasEnzyme"},
        BRENDA_SCHEMA,
    )


def test_a_known_document_maps_onto_the_hub_object_field_by_field() -> None:
    """The written object for a two-field document, pinned whole.

    Offsets index the joined text, so a body span only lands on the hub's
    text when it is rebased onto the body and its context is cut from the
    body alone; a change to text assembly or to the bridge moves these.
    """
    assert _annotation() == {
        "reference": {"pubmed_id": 12345},
        "entities": [
            {
                "entity_id": "brenda:enz7",
                "preferred_name": "catalase",
                "kind": "enzymes",
                "confirmed": False,
            }
        ],
        "pointers": [
            {
                "entity_id": "NCBITaxon:562",
                "offset": 0,
                "length": 16,
                "field": "abstract",
                "exact_text": "Escherichia coli",
                "prefix_text": "",
                "suffix_text": " makes an enzyme.",
            },
            {
                "entity_id": "brenda:enz7",
                "offset": 43,
                "length": 8,
                "field": "body",
                "exact_text": "catalase",
                "prefix_text": "plasm of every growing cell the ",
                "suffix_text": " is abundant.",
            },
        ],
        "relations": [
            {
                "predicate": "ex:hasEnzyme",
                "subject": "NCBITaxon:562",
                "object": "brenda:enz7",
            }
        ],
        "completed": False,
    }


def test_offsets_that_miss_the_surface_are_refused() -> None:
    """Offsets from another text assembly raise rather than point elsewhere."""
    record: annotation_hub.PredictionRecord = {
        **RECORD,
        "spans": [{**RECORD["spans"][1], "start": 76, "end": 84}],
    }
    with pytest.raises(ValueError, match="catalase"):
        annotation_hub.annotation(record, ABSTRACT, BODY, {}, {}, BRENDA_SCHEMA)


def test_spans_with_no_entity_id_are_counted_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A run without a linker, or one that leaves a span unlinked, says so.

    `entity_ids` is `None` when no linker ran and `[]` when the linker ran
    and found nothing; both drop the span from the pointers, but the two
    counts are reported apart so a whole run without a linker is
    distinguishable from a handful of genuinely unlinked spans.
    """
    record: annotation_hub.PredictionRecord = {
        **RECORD,
        "spans": [
            RECORD["spans"][3],  # entity_ids: None -- no linker ran
            {**RECORD["spans"][1], "entity_ids": []},  # unlinked
        ],
        "relations": None,
    }
    with caplog.at_level(logging.WARNING, logger=annotation_hub.__name__):
        result = annotation_hub.annotation(
            record, ABSTRACT, BODY, {}, {}, BRENDA_SCHEMA
        )
    assert result["pointers"] == []
    assert len(caplog.records) == 1
    assert "document 12345" in caplog.text
    assert "2 span(s)" in caplog.text
    assert "1 with no linker run" in caplog.text
    assert "1 unlinked by the linker" in caplog.text


def test_write_annotations_writes_one_line_per_article(tmp_path) -> None:
    path = tmp_path / "annotations.jsonl"
    assert annotation_hub.write_annotations(path, [_annotation()] * 2) == 2
    assert len(path.read_text(encoding="utf8").splitlines()) == 2
