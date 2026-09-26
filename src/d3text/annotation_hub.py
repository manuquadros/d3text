"""Predictions in the write shape annotation-hub accepts on `POST /save/`.

Maps one record `infer` wrote onto one annotation object, carrying what only
the predictions know. What the hub also requires and they cannot — the
posting account, the project, the reference's own key and its bibliographic
record — belongs to whatever posts the object, which reads them back from the
hub itself.
"""

import json
import logging
import os
import pathlib
from collections.abc import Iterable, Mapping
from typing import Literal, TypedDict

from d3text import corpus
from d3text.identifier_bridge import IdentifierBridge
from d3text.schema import Schema

logger = logging.getLogger(__name__)

CONTEXT = 32
"""Characters of prefix and suffix text the hub's client anchors with."""

UNCONFIRMED_PREFIX = "brenda"
"""CURIE prefix of an entity no bridge pairs with an outside identifier."""


class SpanRecord(TypedDict):
    """One span of an `infer` record."""

    start: int
    end: int
    surface: str
    entity_type: str
    entity_ids: list[str] | None


class RelationRecord(TypedDict):
    """One relation of an `infer` record; its arguments carry no role."""

    predicate: str
    arguments: list[list[str]]


class PredictionRecord(TypedDict):
    """One document's line of the file `infer` writes."""

    document: str
    spans: list[SpanRecord]
    relations: list[RelationRecord] | None


class Pointer(TypedDict):
    """A span, as the hub's `PointerIn` takes it, less the hub's own key."""

    entity_id: str
    offset: int
    length: int
    field: Literal["abstract", "body"]
    exact_text: str
    prefix_text: str
    suffix_text: str


class Entity(TypedDict):
    """An entity proposal, as the hub's `EntityAnnotation` takes it."""

    entity_id: str
    preferred_name: str
    kind: str
    confirmed: bool


class Relation(TypedDict):
    """A triple, as the hub's `RelationIn` takes it."""

    predicate: str
    subject: str
    object: str


class Annotation(TypedDict):
    """One article's annotation object."""

    reference: dict[str, int]
    entities: list[Entity]
    pointers: list[Pointer]
    relations: list[Relation]
    completed: bool


def curie(entity_id: str, bridges: Mapping[str, IdentifierBridge]) -> str:
    """The CURIE `entity_id` is written under.

    :param entity_id: a BRENDA entity id.
    :param bridges: bridge tables keyed by the CURIE prefix the hub's
        ontology gives their namespace, e.g. `"NCBITaxon"`.
    :return: the bridged CURIE, or one under `UNCONFIRMED_PREFIX` where no
        bridge pairs `entity_id` with exactly one identifier.
    """
    for prefix, bridge in bridges.items():
        external = bridge.external_id(entity_id)
        if external is not None:
            return f"{prefix}:{external}"
    return f"{UNCONFIRMED_PREFIX}:{entity_id}"


def annotation(
    record: PredictionRecord,
    abstract: str | float | None,
    fulltext: str | float | None,
    bridges: Mapping[str, IdentifierBridge],
    predicates: Mapping[str, str],
    schema: Schema,
) -> Annotation:
    """Map one document's predictions onto the hub's annotation object.

    A span with no entity id has nothing for a pointer to name and is left
    out, counted and logged, as is one crossing from the abstract into the
    body, which no single field holds. Only unconfirmed entities are
    listed: the hub already holds every bridged one, and listing it would
    overwrite the ontology's preferred name with the surface. A relation
    keeps the pairs whose entities both have a pointer in the document.

    :param record: the document's `infer` record.
    :param abstract: the corpus row's abstract cell.
    :param fulltext: the corpus row's body cell.
    :param bridges: see `curie`.
    :param predicates: each relation name mapped to the ontology property
        the target project defines for it.
    :param schema: the schema the relations were predicted under, which says
        which argument is the object.
    :return: the object, ready for `json.dumps`.
    :raises ValueError: if a span's offsets do not cut its surface out of the
        corpus text, so they index some other assembly of it.
    :raises KeyError: if a relation's name is not in `predicates`.
    """
    abstract_text, body_start, body_text = corpus.document_fields(
        abstract, fulltext
    )
    entities: dict[str, Entity] = {}
    pointers: list[Pointer] = []
    no_linker = 0
    unlinked = 0
    for span in record["spans"]:
        field: Literal["abstract", "body"]
        if span["end"] <= len(abstract_text):
            field, text, start = "abstract", abstract_text, span["start"]
        elif span["start"] >= body_start:
            field, text, start = "body", body_text, span["start"] - body_start
        else:
            logger.warning(
                "document %s: span %d-%d crosses from the abstract into the "
                "body and is not written",
                record["document"],
                span["start"],
                span["end"],
            )
            continue
        end = start + span["end"] - span["start"]
        if text[start:end] != span["surface"]:
            raise ValueError(
                f"document {record['document']}: offsets {span['start']}-"
                f"{span['end']} cut {text[start:end]!r} out of the corpus "
                f"text, not the predicted {span['surface']!r}"
            )
        if span["entity_ids"] is None:
            no_linker += 1
            continue
        if not span["entity_ids"]:
            unlinked += 1
            continue
        for entity_id in span["entity_ids"]:
            name = curie(entity_id, bridges)
            if name.startswith(f"{UNCONFIRMED_PREFIX}:"):
                entities.setdefault(
                    name,
                    {
                        "entity_id": name,
                        "preferred_name": span["surface"],
                        "kind": span["entity_type"],
                        "confirmed": False,
                    },
                )
            pointers.append(
                {
                    "entity_id": name,
                    "offset": start,
                    "length": end - start,
                    "field": field,
                    "exact_text": text[start:end],
                    "prefix_text": text[max(0, start - CONTEXT) : start],
                    "suffix_text": text[end : end + CONTEXT],
                }
            )

    if no_linker or unlinked:
        logger.warning(
            "document %s: %d span(s) not written for lacking an entity id "
            "(%d with no linker run, %d unlinked by the linker)",
            record["document"],
            no_linker + unlinked,
            no_linker,
            unlinked,
        )

    pointed = {pointer["entity_id"] for pointer in pointers}
    object_types = {
        relation_type.name: relation_type.object_type
        for relation_type in schema.relation_types
    }
    relations: list[Relation] = []
    for relation in record["relations"] or ():
        predicate = predicates[relation["predicate"]]
        first, second = relation["arguments"]
        for a in first:
            for b in second:
                pair = (a, b)
                if (
                    schema.type_of(a).name
                    == object_types[relation["predicate"]]
                ):
                    pair = (b, a)
                subject, object_ = (curie(i, bridges) for i in pair)
                if subject in pointed and object_ in pointed:
                    relations.append(
                        {
                            "predicate": predicate,
                            "subject": subject,
                            "object": object_,
                        }
                    )

    return {
        "reference": {"pubmed_id": int(record["document"])},
        "entities": list(entities.values()),
        "pointers": pointers,
        "relations": relations,
        "completed": False,
    }


def write_annotations(
    path: str | os.PathLike[str], annotations: Iterable[Annotation]
) -> int:
    """Write one annotation object per line and return how many landed.

    :param path: the file to write.
    :param annotations: the objects to write, one per article.
    :return: how many were written.
    """
    written = 0
    with pathlib.Path(path).open("w", encoding="utf8") as output:
        for item in annotations:
            output.write(json.dumps(item) + "\n")
            written += 1
    return written
