"""NLP4Pheno: microbiology sentences with hand-marked `STRAIN` spans.

A Label Studio export, one sentence per task, whose offsets are **already
half-open** — the opposite of S800, so the `+ 1` that corpus needs is a
one-character error here — and which repeats each span's surface, so the loader
checks its own convention against the corpus rather than trusting it. The spans
name no identifier: what grounds a strain span is the culture-collection
accession inside it. See the evaluation page of the documentation.
"""

import json
import os
import pathlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from d3text.identifier_bridge import ExternalMention

STRAIN = "STRAIN"
"""The label this project's strain evaluation reads; the export has eight."""

SPAN_RESULT = "labels"
"""`type` of a result marking a span, as against `relation`."""

_REPORTED = 5


@dataclass(frozen=True, slots=True)
class NLP4Pheno:
    """The corpus: its sentences, its spans by label, and its relation count.

    `spans` carries half-open offsets into the matching `texts` entry and every
    one of them addresses its own surface form. `relations` counts the results
    this loader does not read, so a corpus whose relations grew is not silently
    reported as one that has none.
    """

    texts: Mapping[str, str]
    spans: Mapping[str, tuple[ExternalMention, ...]]
    relations: int

    def labelled(self, label: str) -> tuple[ExternalMention, ...]:
        """The spans marked `label`, none of them carrying an identifier.

        :param label: the annotation label, e.g. `STRAIN`.
        :return: the spans, empty if the corpus marks none.
        """
        return self.spans.get(label, ())


def document_of(task: Mapping[str, Any]) -> str:
    """The key a task's spans address it by.

    :param task: one exported task.
    :return: its Label Studio id as a string.
    :raises ValueError: if the task carries no id.
    """
    identifier = task.get("id")
    if identifier is None:
        raise ValueError(f"a task carries no id: {sorted(task)}")
    return str(identifier)


def check_offsets(
    texts: Mapping[str, str], mentions: Sequence[ExternalMention]
) -> None:
    """Assert every span addresses its own surface form.

    Self-checking by construction: the export states what each span says, so
    the corpus validates the coordinate convention the loader reads it under.

    :param texts: each sentence's text, by document key.
    :param mentions: the spans to check.
    :raises ValueError: naming the first few disagreements.
    """
    wrong = [
        f"{mention.document}[{mention.start}:{mention.end}] is "
        f"{texts.get(mention.document, '')[mention.start : mention.end]!r}, "
        f"annotated as {mention.surface!r}"
        for mention in mentions
        if texts.get(mention.document, "")[mention.start : mention.end]
        != mention.surface
    ]
    if wrong:
        raise ValueError(
            f"{len(wrong)} of {len(mentions)} NLP4Pheno offsets do not "
            "address the spans they annotate, so the export on disk is not "
            "the one this loader reads half-open: "
            + "; ".join(wrong[:_REPORTED])
        )


def parse_tasks(tasks: Iterable[Mapping[str, Any]]) -> NLP4Pheno:
    """The exported tasks as texts and spans grouped by label.

    Every annotation of a task is read, not only the first: two annotators
    marking one sentence is a disagreement the scorer keys by span, not
    something a loader should resolve by dropping one of them.

    :param tasks: the export's tasks.
    :return: the sentences, their spans by label, and the relation count.
    :raises ValueError: on a task with no id or no text, on a span result
        missing an offset, or on an offset that misses its own surface form.
    """
    texts: dict[str, str] = {}
    spans: dict[str, list[ExternalMention]] = {}
    relations = 0

    for task in tasks:
        document = document_of(task)
        data = task.get("data") or {}
        text = data.get("text")
        if not isinstance(text, str):
            raise ValueError(f"task {document} carries no text under `data`")
        texts[document] = text

        for annotation in task.get("annotations") or []:
            for result in annotation.get("result") or []:
                if result.get("type") != SPAN_RESULT:
                    relations += 1
                    continue
                value = result.get("value") or {}
                missing = [
                    field
                    for field in ("start", "end", "text", "labels")
                    if field not in value
                ]
                if missing:
                    raise ValueError(
                        f"a span of task {document} declares no {missing}"
                    )
                for label in value["labels"]:
                    spans.setdefault(label, []).append(
                        ExternalMention(
                            document=document,
                            start=int(value["start"]),
                            end=int(value["end"]),
                            surface=value["text"],
                            external_id=None,
                        )
                    )

    check_offsets(texts, [span for found in spans.values() for span in found])
    return NLP4Pheno(
        texts=texts,
        spans={label: tuple(found) for label, found in spans.items()},
        relations=relations,
    )


def load_nlp4pheno(path: str | os.PathLike[str]) -> NLP4Pheno:
    """Read the Label Studio export at `path`.

    :param path: the exported JSON file.
    :return: the sentences, their spans by label, and the relation count.
    :raises FileNotFoundError: if the export is missing.
    :raises ValueError: if the export is not a list of tasks, or a task or a
        span in it is malformed.
    """
    loaded = json.loads(pathlib.Path(path).read_text(encoding="utf8"))
    if not isinstance(loaded, list):
        raise ValueError(
            f"{path} holds {type(loaded).__name__}, not the list of tasks a "
            "Label Studio export is"
        )
    return parse_tasks(loaded)


__all__ = [
    "SPAN_RESULT",
    "STRAIN",
    "NLP4Pheno",
    "check_offsets",
    "document_of",
    "load_nlp4pheno",
    "parse_tasks",
]
