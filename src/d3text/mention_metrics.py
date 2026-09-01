"""The three mention-level scores: detection, linking, and what is masked.

Detection and linking are scored separately because a detection miss is
unrecoverable while a false positive is cheap, and the ignore set — mentions
distant supervision refuses to label — is masked rather than counted, since
calling a hit on one a false positive would rebuild the distortion that target
exists to remove. Coordinate-agnostic: a mention is `(start, end, type)` in
whatever axis both sides share. See the evaluation page of the documentation.
"""

import collections.abc
import enum
from dataclasses import dataclass, field

import numpy
from numpy.typing import NDArray

from d3text.token_labels import (
    BRENDA_LABELS,
    IGNORE_INDEX,
    OUTSIDE,
    SPAN_END,
    SPAN_GOLD,
    SPAN_START,
    SPAN_TYPE,
    LabelSpace,
    Mention,
    mention_spans,
)


@dataclass(frozen=True, slots=True)
class PredictedMention:
    """One span the tagger proposed, with whatever the linker resolved.

    `entity_ids` empty is a NIL mention — a typed span the linker could not
    ground — not a mention that skipped linking.
    """

    start: int
    end: int
    type_code: int
    entity_ids: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class GoldMention:
    """One annotated mention, and whether the scores may read it.

    `assertable=False` marks the ignore set: a mention whose entity BRENDA did
    not link to this document, or whose gold candidates span two types.
    `entity_ids` empty on an assertable mention means it has no BRENDA entity,
    which is what makes NIL the correct link for it.
    """

    start: int
    end: int
    type_code: int
    entity_ids: frozenset[str] = frozenset()
    assertable: bool = True


@dataclass(frozen=True, slots=True)
class DetectionScores:
    """Span-level detection counts, and the rates over them.

    `ignored` is the masked column, kept beside the real counts because a score
    with a growing masked share means something different from the same score
    with none.
    """

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    ignored: int = 0

    def __add__(self, other: "DetectionScores") -> "DetectionScores":
        return DetectionScores(
            true_positives=self.true_positives + other.true_positives,
            false_positives=self.false_positives + other.false_positives,
            false_negatives=self.false_negatives + other.false_negatives,
            ignored=self.ignored + other.ignored,
        )

    @property
    def precision(self) -> float:
        judged = self.true_positives + self.false_positives
        return self.true_positives / judged if judged else 0.0

    @property
    def recall(self) -> float:
        annotated = self.true_positives + self.false_negatives
        return self.true_positives / annotated if annotated else 0.0

    @property
    def f1(self) -> float:
        precision, recall = self.precision, self.recall
        if not precision + recall:
            return 0.0
        return 2 * precision * recall / (precision + recall)


@dataclass(frozen=True, slots=True)
class LinkingScores:
    """Link outcomes over the correctly detected spans, and nothing else.

    A mention the detector missed never reaches the linker, so it appears in no
    column here. The NIL answer is split by what it met: `nil_correct` where
    the mention has no BRENDA entity, `nil_missed` where an ID existed and the
    linker declined it.
    """

    correct: int = 0
    wrong: int = 0
    nil_correct: int = 0
    nil_missed: int = 0

    def __add__(self, other: "LinkingScores") -> "LinkingScores":
        return LinkingScores(
            correct=self.correct + other.correct,
            wrong=self.wrong + other.wrong,
            nil_correct=self.nil_correct + other.nil_correct,
            nil_missed=self.nil_missed + other.nil_missed,
        )

    @property
    def total(self) -> int:
        return self.correct + self.wrong + self.nil_correct + self.nil_missed

    @property
    def accuracy(self) -> float:
        right = self.correct + self.nil_correct
        return right / self.total if self.total else 0.0


def gold_mentions(
    mentions: collections.abc.Iterable[Mention],
    gold_entity_ids: collections.abc.Set[str],
    space: LabelSpace = BRENDA_LABELS,
) -> list[GoldMention]:
    """Every dictionary mention of a document, as scorable gold.

    The type and the assertable flag are read off `token_labels.mention_spans`
    rather than re-derived, so the evaluation and the training targets cannot
    disagree about which mentions are gold.

    :param mentions: the document's mentions, as `find_mentions` returns them.
    :param gold_entity_ids: the entities this document is linked to.
    :param space: the label space the type codes are written in.
    :return: the gold mentions, each carrying its candidates narrowed to the
        gold set, since those are the only IDs a link can be scored against.
    """
    placed = list(mentions)
    rows = mention_spans(placed, gold_entity_ids, space)
    gold = frozenset(gold_entity_ids)
    return [
        GoldMention(
            start=int(row[SPAN_START]),
            end=int(row[SPAN_END]),
            type_code=int(row[SPAN_TYPE]),
            entity_ids=mention.entity_ids & gold,
            assertable=bool(row[SPAN_GOLD]),
        )
        for mention, row in zip(placed, rows)
    ]


def _overlaps(predicted: PredictedMention, gold: GoldMention) -> bool:
    return predicted.start < gold.end and gold.start < predicted.end


def detection_scores(
    predicted: collections.abc.Iterable[PredictedMention],
    gold: collections.abc.Iterable[GoldMention],
) -> DetectionScores:
    """TP / FP / FN over spans, with the ignore set masked.

    A prediction is a TP when its span and type equal an assertable gold
    mention's, an FP when it matches none and touches no ignored mention, and
    masked when it misses but overlaps the ignore set. FN counts the assertable
    mentions no prediction matched.

    :param predicted: the spans the tagger proposed.
    :param gold: the document's gold mentions.
    :return: the counts and the rates over them.
    """
    gold = list(gold)
    masked = [mention for mention in gold if not mention.assertable]
    gold_keys = {
        (mention.start, mention.end, mention.type_code)
        for mention in gold
        if mention.assertable
    }

    true_positives = false_positives = ignored = 0
    matched: set[tuple[int, int, int]] = set()
    for span in predicted:
        key = (span.start, span.end, span.type_code)
        if key in gold_keys:
            true_positives += 1
            matched.add(key)
        elif any(_overlaps(span, mention) for mention in masked):
            ignored += 1
        else:
            false_positives += 1

    return DetectionScores(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=len(gold_keys - matched),
        ignored=ignored,
    )


class LinkingRule(enum.Enum):
    """How a linker's answer set is judged against a mention's gold IDs.

    `INTERSECTION` is the rule the corpus forces and the default; `STRICT` is
    exact-match top-1, and is only meaningful where the gold is known to be
    one entity on evidence from outside this project.
    """

    INTERSECTION = "intersection"
    STRICT = "strict"


def _links(
    rule: LinkingRule,
    predicted_ids: frozenset[str],
    gold_ids: frozenset[str],
) -> bool:
    if rule is LinkingRule.STRICT:
        return predicted_ids == gold_ids
    return bool(predicted_ids & gold_ids)


def linking_scores(
    predicted: collections.abc.Iterable[PredictedMention],
    gold: collections.abc.Iterable[GoldMention],
    rule: LinkingRule = LinkingRule.INTERSECTION,
) -> LinkingScores:
    """Link outcomes, conditional on a correct detection.

    A non-empty answer is judged by `rule`; NIL is correct exactly when the
    mention has no BRENDA entity, whichever rule is in force.

    :param predicted: the spans the tagger proposed.
    :param gold: the document's gold mentions.
    :param rule: how an answer set is compared with the mention's gold IDs.
    :return: the link outcomes over the correctly detected spans.
    """
    by_key = {
        (mention.start, mention.end, mention.type_code): mention
        for mention in gold
        if mention.assertable
    }

    scores = LinkingScores()
    for span in predicted:
        mention = by_key.get((span.start, span.end, span.type_code))
        if mention is None:
            continue
        if span.entity_ids:
            hit = _links(rule, span.entity_ids, mention.entity_ids)
            scores += LinkingScores(correct=int(hit), wrong=int(not hit))
        else:
            missed = bool(mention.entity_ids)
            scores += LinkingScores(
                nil_correct=int(not missed), nil_missed=int(missed)
            )
    return scores


def ignore_firing(
    predicted: collections.abc.Iterable[PredictedMention],
    gold: collections.abc.Iterable[GoldMention],
) -> tuple[int, int]:
    """How much of the ignore set the tagger fired on: `(regions, fired)`.

    The diagnostic the masked scores cannot give: ignored mentions are
    known-plausible mentions deliberately excluded from training, so the firing
    rate measures generalization past the gold set with no hand annotation.

    :param predicted: the spans the tagger proposed.
    :param gold: the document's gold mentions.
    :return: the ignore regions, and how many were fired on.
    """
    spans = list(predicted)
    regions = fired = 0
    for mention in gold:
        if mention.assertable:
            continue
        regions += 1
        fired += int(any(_overlaps(span, mention) for span in spans))
    return regions, fired


def spans_from_codes(
    codes: NDArray[numpy.integer], outside: int = OUTSIDE
) -> list[tuple[int, int, int]]:
    """Maximal same-code runs of a flat target array, `outside` dropped.

    `IGNORE_INDEX` runs come back like any other code: whether a run is a
    mention or an ignore region is the caller's reading.

    :param codes: one code per position.
    :param outside: the code to drop.
    :return: `(start, end, code)` in the array's own axis, half-open.
    """
    flat = numpy.asarray(codes).reshape(-1)
    spans: list[tuple[int, int, int]] = []
    start = 0
    for position in range(1, flat.shape[0] + 1):
        if position == flat.shape[0] or flat[position] != flat[start]:
            code = int(flat[start])
            if code != outside:
                spans.append((start, position, code))
            start = position
    return spans


def token_gold_mentions(
    codes: NDArray[numpy.integer], ignore_index: int = IGNORE_INDEX
) -> list[GoldMention]:
    """A document's gold codes as mentions on the token axis.

    Entity IDs are not recoverable from codes, so linking cannot be scored in
    this geometry — only detection and the firing rate.

    :param codes: the stored per-token codes.
    :param ignore_index: the code marking the ignore set, carried as
        non-assertable mentions of no type.
    :return: the gold mentions.
    """
    return [
        GoldMention(
            start=start,
            end=end,
            type_code=code if code != ignore_index else OUTSIDE,
            assertable=code != ignore_index,
        )
        for start, end, code in spans_from_codes(codes)
    ]


def token_predicted_mentions(
    codes: NDArray[numpy.integer],
) -> list[PredictedMention]:
    """A tagger's argmax codes as proposed mentions on the token axis.

    :param codes: the tagger's per-token argmax.
    :return: the proposed mentions.
    """
    return [
        PredictedMention(start=start, end=end, type_code=code)
        for start, end, code in spans_from_codes(codes)
    ]


def _zero_scores() -> DetectionScores:
    return DetectionScores()


@dataclass
class DetectionAccumulator:
    """Detection scores summed over a split, one document at a time.

    Keeps the totals — overall, per entity type, and the ignore-set diagnostics
    — so the metric assembly is testable without a model anywhere near it.
    """

    space: LabelSpace
    scores: DetectionScores = field(default_factory=_zero_scores)
    by_type: dict[int, DetectionScores] = field(default_factory=dict)
    ignore_regions: int = 0
    ignore_fired: int = 0
    documents: int = 0
    missing_documents: int = 0

    def __post_init__(self) -> None:
        for code in self.space.codes:
            self.by_type.setdefault(code, DetectionScores())

    def add_document(
        self,
        predicted_codes: NDArray[numpy.integer],
        gold_codes: NDArray[numpy.integer],
    ) -> None:
        predicted = token_predicted_mentions(predicted_codes)
        gold = token_gold_mentions(gold_codes)

        self.scores += detection_scores(predicted, gold)
        for code in self.space.codes:
            # The ignore set is typeless, so it masks every type's column.
            self.by_type[code] += detection_scores(
                [span for span in predicted if span.type_code == code],
                [
                    mention
                    for mention in gold
                    if mention.type_code == code or not mention.assertable
                ],
            )

        regions, fired = ignore_firing(predicted, gold)
        self.ignore_regions += regions
        self.ignore_fired += fired
        self.documents += 1

    def metrics(self) -> dict[str, float]:
        """The accumulated scores, keyed the way `evaluate_model` logs them.

        :return: the metrics; the firing rate is omitted when the split held no
            ignore regions, since 0/0 is not a measurement and an absent key
            cannot be mistaken for a tagger that never fired.
        """
        metrics = {
            "test/detection_precision": self.scores.precision,
            "test/detection_recall": self.scores.recall,
            "test/detection_f1": self.scores.f1,
            "test/detection_true_positives": float(self.scores.true_positives),
            "test/detection_false_positives": float(
                self.scores.false_positives
            ),
            "test/detection_false_negatives": float(
                self.scores.false_negatives
            ),
            "test/detection_ignored_predictions": float(self.scores.ignored),
            "test/detection_ignore_regions": float(self.ignore_regions),
            "test/detection_documents": float(self.documents),
            "test/detection_documents_missing_labels": float(
                self.missing_documents
            ),
        }
        if self.ignore_regions:
            metrics["test/detection_ignore_firing_rate"] = (
                self.ignore_fired / self.ignore_regions
            )
        for code, scores in self.by_type.items():
            name = self.space.type_of(code)
            metrics[f"test/detection_{name}_precision"] = scores.precision
            metrics[f"test/detection_{name}_recall"] = scores.recall
            metrics[f"test/detection_{name}_f1"] = scores.f1
        return metrics


__all__ = [
    "DetectionAccumulator",
    "DetectionScores",
    "GoldMention",
    "LinkingRule",
    "LinkingScores",
    "PredictedMention",
    "detection_scores",
    "gold_mentions",
    "ignore_firing",
    "linking_scores",
    "spans_from_codes",
    "token_gold_mentions",
    "token_predicted_mentions",
]
