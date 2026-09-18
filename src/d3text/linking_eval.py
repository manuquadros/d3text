"""Scoring a `Linker` against gold identifiers BRENDA did not produce.

A mention is judged when the outside authority's identifier pairs with exactly
one BRENDA entity of the types asked for, never because the linker returned
one candidate — selecting on the linker's side would make its own answer the
gold. See the evaluation page of the documentation.
"""

import collections
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from d3text.identifier_bridge import ExternalMention, IdentifierBridge
from d3text.linking import Linker
from d3text.mention_metrics import (
    GoldMention,
    LinkingRule,
    LinkingScores,
    PredictedMention,
    linking_scores,
)
from d3text.token_labels import BRENDA_LABELS, LabelSpace

_CANDIDATE_BUCKETS = (1, 2, 3)
"""Candidate counts reported on their own; anything above joins `4+`."""


@dataclass(frozen=True, slots=True)
class LinkingReport:
    """What a linker scored, over how much, and how far it disambiguated.

    The three population counts partition the annotated mentions —
    `__post_init__` refuses a report where they do not — so the coverage can
    never drift from the score it qualifies. `outside_bridge` holds the
    mentions this evaluation deliberately does not judge: scoring them as NIL
    would charge the linker for the bridge's misses. Which mentions those are
    depends on `entity_types`, so a report over one type is not a slice of a
    report over several — every count is over the same annotated population.
    """

    namespace: str
    entity_types: tuple[str, ...]
    documents: int
    annotated: int
    judged: int
    outside_bridge: int
    ambiguous_gold: int
    strict: LinkingScores
    lenient: LinkingScores
    candidates: Mapping[int, int]
    corpus_digest: str = ""
    """SHA-256 of the gold annotation file this report was scored over, empty
    where the caller has none (a fabricated fixture, an in-memory corpus).
    Comparable to `LinkingBlock.index_digest`: the block's digest says
    whether the dictionary side of an evaluation matches a prior run, this
    one says whether the gold side does."""

    def __post_init__(self) -> None:
        counted = self.judged + self.outside_bridge + self.ambiguous_gold
        if counted != self.annotated:
            raise ValueError(
                f"{counted} mentions accounted for against {self.annotated} "
                "annotated: the coverage denominator does not match the "
                "populations it is made of"
            )

    @property
    def coverage(self) -> float:
        """Share of annotated mentions the scores are over.

        :return: the coverage, 0.0 when nothing was annotated.
        """
        return self.judged / self.annotated if self.annotated else 0.0

    def candidate_share(self) -> dict[str, float]:
        """Judged spans by candidate count, as shares.

        :return: shares keyed `nil`, `1`, `2`, `3`, `4+`, empty if none were
            judged.
        """
        if not self.judged:
            return {}
        shares = {
            str(size): self.candidates.get(size, 0) / self.judged
            for size in _CANDIDATE_BUCKETS
        }
        wide = sum(
            count
            for size, count in self.candidates.items()
            if size > _CANDIDATE_BUCKETS[-1]
        )
        return {
            "nil": self.candidates.get(0, 0) / self.judged,
            **shares,
            "4+": wide / self.judged,
        }

    def metrics(self) -> dict[str, float]:
        """The report keyed the way an evaluation pass logs it.

        Every accuracy is emitted beside its coverage and the counts it is
        taken over, so a chart of the accuracy alone still has the denominator
        one key away. The identifier namespace is part of every key because an
        evaluation reports one of these per authority and `score_linking`
        refuses to mix two in one report — and because the authority is what
        the accuracy is a claim about.

        :return: the metric keys and their values.
        """
        key = f"test/linking_{self.namespace}"
        metrics = {
            f"{key}_strict_accuracy": self.strict.accuracy,
            f"{key}_lenient_accuracy": self.lenient.accuracy,
            f"{key}_coverage": self.coverage,
            f"{key}_annotated": float(self.annotated),
            f"{key}_judged": float(self.judged),
            f"{key}_outside_bridge": float(self.outside_bridge),
            f"{key}_ambiguous_gold": float(self.ambiguous_gold),
            f"{key}_documents": float(self.documents),
            f"{key}_correct": float(self.strict.correct),
            f"{key}_wrong": float(self.strict.wrong),
            f"{key}_nil_correct": float(self.strict.nil_correct),
            f"{key}_nil_missed": float(self.strict.nil_missed),
        }
        for size in _CANDIDATE_BUCKETS:
            metrics[f"{key}_candidates_{size}"] = float(
                self.candidates.get(size, 0)
            )
        metrics[f"{key}_candidates_nil"] = float(self.candidates.get(0, 0))
        metrics[f"{key}_candidates_4_plus"] = float(
            sum(
                count
                for size, count in self.candidates.items()
                if size > _CANDIDATE_BUCKETS[-1]
            )
        )
        return metrics

    def summary(self) -> str:
        """The result as a paragraph, coverage stated beside every score.

        :return: one paragraph naming the accuracies and what they are over;
            names the gold corpus's digest when one was carried, so a run
            comparison can tell a moved download from a moved index.
        """
        shares = ", ".join(
            f"{bucket} -> {share:.1%}"
            for bucket, share in self.candidate_share().items()
        )
        digest = (
            f" Gold corpus {self.corpus_digest}." if self.corpus_digest else ""
        )
        return (
            f"{' + '.join(self.entity_types)} linking against "
            f"{self.namespace} gold, "
            f"{self.documents} documents: strict accuracy "
            f"{self.strict.accuracy:.3f} (lenient "
            f"{self.lenient.accuracy:.3f}) on the {self.coverage:.1%} of "
            f"{self.annotated} annotated mentions that pair with exactly one "
            f"entity ({self.judged} judged; {self.outside_bridge} outside the "
            f"bridge, {self.ambiguous_gold} pairing with several). "
            f"Candidates per judged span: {shares or 'none judged'}."
            f"{digest}"
        )


def _typed(
    entity_ids: Iterable[str], types_by_prefix: Mapping[str, str]
) -> dict[str, str]:
    """The entities of the wanted types, each with the type it belongs to."""
    typed: dict[str, str] = {}
    for entity_id in entity_ids:
        for prefix, entity_type in types_by_prefix.items():
            if entity_id.startswith(prefix):
                typed[entity_id] = entity_type
    return typed


def score_linking(
    mentions: Iterable[ExternalMention],
    bridge: IdentifierBridge,
    linker: Linker,
    entity_types: Sequence[str],
    namespace: str,
    space: LabelSpace = BRENDA_LABELS,
    corpus_digest: str = "",
) -> LinkingReport:
    """Score `linker` on the mentions `bridge` gives a single gold entity.

    Mentions are keyed by `(document, start, end)`; a span annotated with two
    different identifiers joins `ambiguous_gold`, since its gold is no more a
    single entity than a duplicated BRENDA row's is, and one the authority
    named no identifier for joins `outside_bridge`. The type the linker is
    asked for is the gold entity's own, so asking for two types at once judges
    a species curated under both as ambiguous rather than twice.

    :param mentions: the annotator's spans, with the identifier each was
        given, or None where the authority gave none.
    :param bridge: the table pairing those identifiers with BRENDA entities.
    :param linker: the linker under test.
    :param entity_types: the types the gold may be drawn from, e.g.
        `["bacteria"]`. The bridge is read restricted to them, so an
        identifier carried only by an entity of another type counts as outside
        it.
    :param namespace: The identifier namespace the gold is in. The bridge must
        record the same one — a taxid table scored as if it held EC numbers
        raises nothing on its own and produces a number.
    :param space: the label space naming the entity types.
    :param corpus_digest: a fingerprint of the gold annotation file scored,
        carried onto the report unchanged; empty where the caller has none.
    :return: the scores, the populations they are over, and the ambiguity.
    :raises ValueError: if `bridge` records another namespace, or `space`
        declares none of `entity_types`.
    """
    if bridge.namespace != namespace:
        raise ValueError(
            f"bridge records {bridge.namespace!r} identifiers, but the gold "
            f"mentions are {namespace!r}: the two name different things"
        )
    codes = dict(zip(space.types, space.codes))
    prefixes = dict(zip(space.types, space.prefixes))
    if not entity_types:
        raise ValueError(
            "no entity type was asked for, so nothing could be judged"
        )
    unknown = [name for name in entity_types if name not in codes]
    if unknown:
        raise ValueError(
            f"{unknown} is not an entity type of this label space; "
            f"known: {list(codes)}"
        )
    types_by_prefix = {prefixes[name]: name for name in entity_types}

    by_span: dict[tuple[str, int, int], list[ExternalMention]] = {}
    for mention in mentions:
        key = (mention.document, mention.start, mention.end)
        by_span.setdefault(key, []).append(mention)

    documents = {document for document, _, _ in by_span}
    outside_bridge = ambiguous_gold = 0
    candidates: collections.Counter[int] = collections.Counter()
    predicted: dict[str, list[PredictedMention]] = {}
    gold: dict[str, list[GoldMention]] = {}

    for (document, start, end), annotations in by_span.items():
        external_ids = {mention.external_id for mention in annotations}
        if len(external_ids) != 1:
            ambiguous_gold += 1
            continue
        external_id = next(iter(external_ids))
        if external_id is None:
            outside_bridge += 1
            continue
        entities = _typed(bridge.entity_ids(external_id), types_by_prefix)
        if not entities:
            outside_bridge += 1
            continue
        if len(entities) != 1:
            ambiguous_gold += 1
            continue

        entity_id, entity_type = next(iter(entities.items()))
        code = codes[entity_type]
        answer = linker.link(annotations[0].surface, entity_type)
        candidates[len(answer)] += 1
        gold.setdefault(document, []).append(
            GoldMention(
                start=start,
                end=end,
                type_code=code,
                entity_ids=frozenset({entity_id}),
            )
        )
        predicted.setdefault(document, []).append(
            PredictedMention(
                start=start, end=end, type_code=code, entity_ids=answer
            )
        )

    strict = LinkingScores()
    lenient = LinkingScores()
    for document, spans in predicted.items():
        strict += linking_scores(spans, gold[document], LinkingRule.STRICT)
        lenient += linking_scores(
            spans, gold[document], LinkingRule.INTERSECTION
        )

    return LinkingReport(
        namespace=namespace,
        entity_types=tuple(entity_types),
        documents=len(documents),
        annotated=len(by_span),
        judged=strict.total,
        outside_bridge=outside_bridge,
        ambiguous_gold=ambiguous_gold,
        strict=strict,
        lenient=lenient,
        candidates=dict(candidates),
        corpus_digest=corpus_digest,
    )


@dataclass(frozen=True, slots=True)
class TaggedSpan:
    """One span a tagger proposed on its own pass, with the text it covers.

    Unlike `ExternalMention`, a tagger's own span carries no identifier at
    all — only the type it was tagged with and the text `Linker.link` needs,
    which may differ from the gold annotation's own surface and offsets
    wherever detection landed a token or two off.
    """

    document: str
    start: int
    end: int
    surface: str
    entity_type: str


@dataclass(frozen=True, slots=True)
class PredictedLinkingScores:
    """Link outcomes over every judged mention, a detection miss included.

    Mirrors `mention_metrics.LinkingScores`'s four outcomes exactly, plus
    `missed_detection`: a judged mention no predicted span reached at all,
    charged into `total` (and so into `accuracy`'s denominator) rather than
    left for the caller to notice is missing, the way a plain `LinkingScores`
    total would. Kept as its own type rather than added to `LinkingScores`
    itself, since that dataclass's shape is pinned by other callers that
    score detection and linking separately on purpose.
    """

    correct: int = 0
    wrong: int = 0
    nil_correct: int = 0
    nil_missed: int = 0
    missed_detection: int = 0

    def __add__(
        self, other: "PredictedLinkingScores"
    ) -> "PredictedLinkingScores":
        return PredictedLinkingScores(
            correct=self.correct + other.correct,
            wrong=self.wrong + other.wrong,
            nil_correct=self.nil_correct + other.nil_correct,
            nil_missed=self.nil_missed + other.nil_missed,
            missed_detection=self.missed_detection + other.missed_detection,
        )

    @property
    def total(self) -> int:
        return (
            self.correct
            + self.wrong
            + self.nil_correct
            + self.nil_missed
            + self.missed_detection
        )

    @property
    def accuracy(self) -> float:
        right = self.correct + self.nil_correct
        return right / self.total if self.total else 0.0


@dataclass(frozen=True, slots=True)
class PredictedLinkingReport:
    """What a linker scored against a tagger's own spans, over one namespace.

    Partitions `annotated` into `judged`, `outside_bridge` and
    `ambiguous_gold` exactly as `LinkingReport` does — bridge quality is a
    property of the outside authority, not of the tagger under test, so it is
    carved out the same way. What differs is inside `judged`: `strict` and
    `lenient` both fold in every gold mention no predicted span overlapped,
    via `PredictedLinkingScores.missed_detection`, so a detection miss lowers
    the accuracy instead of vanishing from it.
    """

    namespace: str
    entity_types: tuple[str, ...]
    documents: int
    annotated: int
    judged: int
    outside_bridge: int
    ambiguous_gold: int
    strict: PredictedLinkingScores
    lenient: PredictedLinkingScores

    def __post_init__(self) -> None:
        counted = self.judged + self.outside_bridge + self.ambiguous_gold
        if counted != self.annotated:
            raise ValueError(
                f"{counted} mentions accounted for against {self.annotated} "
                "annotated: the coverage denominator does not match the "
                "populations it is made of"
            )

    @property
    def coverage(self) -> float:
        """Share of annotated mentions the scores are over.

        :return: the coverage, 0.0 when nothing was annotated.
        """
        return self.judged / self.annotated if self.annotated else 0.0

    def metrics(self) -> dict[str, float]:
        """The report keyed the way an evaluation pass logs it.

        Keyed `test/predicted_linking_<namespace>_*`, distinct from
        `LinkingReport.metrics`'s `test/linking_<namespace>_*`: the two grade
        different things over the same namespace — one the gold annotation's
        own offsets, this one a tagger's own detections — and one must not
        overwrite the other in the same run.

        :return: the metric keys and their values.
        """
        key = f"test/predicted_linking_{self.namespace}"
        return {
            f"{key}_strict_accuracy": self.strict.accuracy,
            f"{key}_lenient_accuracy": self.lenient.accuracy,
            f"{key}_coverage": self.coverage,
            f"{key}_annotated": float(self.annotated),
            f"{key}_judged": float(self.judged),
            f"{key}_outside_bridge": float(self.outside_bridge),
            f"{key}_ambiguous_gold": float(self.ambiguous_gold),
            f"{key}_documents": float(self.documents),
            f"{key}_correct": float(self.strict.correct),
            f"{key}_wrong": float(self.strict.wrong),
            f"{key}_nil_correct": float(self.strict.nil_correct),
            f"{key}_nil_missed": float(self.strict.nil_missed),
            f"{key}_missed_detection": float(self.strict.missed_detection),
        }

    def summary(self) -> str:
        """The result as a paragraph, coverage stated beside every score.

        :return: one paragraph naming the accuracies, what they are over, and
            how many judged mentions no predicted span of the matching type
            ever reached.
        """
        return (
            f"{' + '.join(self.entity_types)} linking against "
            f"{self.namespace} gold, resolved through a tagger's own "
            f"predicted spans, {self.documents} documents: strict accuracy "
            f"{self.strict.accuracy:.3f} (lenient "
            f"{self.lenient.accuracy:.3f}) on the {self.coverage:.1%} of "
            f"{self.annotated} annotated mentions that pair with exactly one "
            f"entity ({self.judged} judged, "
            f"{self.strict.missed_detection} never reached by a predicted "
            f"span; {self.outside_bridge} outside the bridge, "
            f"{self.ambiguous_gold} pairing with several)."
        )


def _matching_span(
    spans: Sequence[TaggedSpan], start: int, end: int, entity_type: str
) -> TaggedSpan | None:
    """The first of `spans`, typed `entity_type`, overlapping `[start, end)`."""
    for span in spans:
        if span.entity_type == entity_type and (
            span.start < end and start < span.end
        ):
            return span
    return None


def _hit(
    rule: LinkingRule, predicted_ids: frozenset[str], gold_ids: frozenset[str]
) -> bool:
    """Whether `predicted_ids` counts as right for `gold_ids` under `rule`."""
    if rule is LinkingRule.STRICT:
        return predicted_ids == gold_ids
    return bool(predicted_ids & gold_ids)


def score_predicted_linking(
    predicted: Iterable[TaggedSpan],
    gold: Iterable[ExternalMention],
    bridge: IdentifierBridge,
    linker: Linker,
    entity_types: Sequence[str],
    namespace: str,
    space: LabelSpace = BRENDA_LABELS,
) -> PredictedLinkingReport:
    """Score `linker` on a tagger's own spans, a detection miss charged too.

    `score_linking` builds its query from the gold annotation's own surface
    and offsets, so it measures the surface-form index alone — a
    `DictionaryLinker` has no learned parameters, so wherever a span matches
    gold exactly the linker is handed its own answer back. This instead takes
    the spans a tagger actually proposed: a gold mention some predicted span
    of the matching type overlaps is resolved through *that* span's own
    surface text, the answer joined back onto the gold mention's offsets to
    grade against its bridged entity; a gold mention no predicted span
    reaches at all — a stage-1 false negative — is charged as a missed
    linking opportunity (`PredictedLinkingScores.missed_detection`) instead
    of being left out of the score the way a bare detection miss would be.

    Mentions are keyed by `(document, start, end)` exactly as `score_linking`
    keys them, so the same bridge population rules apply: an identifier two
    entities share leaves the mention `ambiguous_gold`, and one the bridge
    pairs with no entity of the wanted types leaves it `outside_bridge`.

    :param predicted: the tagger's own proposed spans, each already typed.
    :param gold: the annotator's spans, with the identifier each was given,
        or None where the authority gave none.
    :param bridge: the table pairing those identifiers with BRENDA entities.
    :param linker: the linker under test.
    :param entity_types: the types the gold may be drawn from, e.g.
        `["bacteria"]`.
    :param namespace: the identifier namespace the gold is in. The bridge
        must record the same one.
    :param space: the label space naming the entity types.
    :return: the scores, the populations they are over, and the ambiguity.
    :raises ValueError: if `bridge` records another namespace, `space`
        declares none of `entity_types`, or no entity type was asked for.
    """
    if bridge.namespace != namespace:
        raise ValueError(
            f"bridge records {bridge.namespace!r} identifiers, but the gold "
            f"mentions are {namespace!r}: the two name different things"
        )
    codes = dict(zip(space.types, space.codes))
    prefixes = dict(zip(space.types, space.prefixes))
    if not entity_types:
        raise ValueError(
            "no entity type was asked for, so nothing could be judged"
        )
    unknown = [name for name in entity_types if name not in codes]
    if unknown:
        raise ValueError(
            f"{unknown} is not an entity type of this label space; "
            f"known: {list(codes)}"
        )
    types_by_prefix = {prefixes[name]: name for name in entity_types}

    by_document: dict[str, list[TaggedSpan]] = {}
    for span in predicted:
        by_document.setdefault(span.document, []).append(span)

    by_span: dict[tuple[str, int, int], list[ExternalMention]] = {}
    for mention in gold:
        key = (mention.document, mention.start, mention.end)
        by_span.setdefault(key, []).append(mention)

    documents = {document for document, _, _ in by_span}
    outside_bridge = ambiguous_gold = 0
    strict = PredictedLinkingScores()
    lenient = PredictedLinkingScores()

    for (document, start, end), annotations in by_span.items():
        external_ids = {mention.external_id for mention in annotations}
        if len(external_ids) != 1:
            ambiguous_gold += 1
            continue
        external_id = next(iter(external_ids))
        if external_id is None:
            outside_bridge += 1
            continue
        entities = _typed(bridge.entity_ids(external_id), types_by_prefix)
        if not entities:
            outside_bridge += 1
            continue
        if len(entities) != 1:
            ambiguous_gold += 1
            continue

        entity_id, entity_type = next(iter(entities.items()))
        gold_ids = frozenset({entity_id})

        match = _matching_span(
            by_document.get(document, []), start, end, entity_type
        )
        if match is None:
            strict += PredictedLinkingScores(missed_detection=1)
            lenient += PredictedLinkingScores(missed_detection=1)
            continue

        answer = linker.link(match.surface, entity_type)
        if not answer:
            strict += PredictedLinkingScores(nil_missed=1)
            lenient += PredictedLinkingScores(nil_missed=1)
            continue

        strict_hit = _hit(LinkingRule.STRICT, answer, gold_ids)
        lenient_hit = _hit(LinkingRule.INTERSECTION, answer, gold_ids)
        strict += PredictedLinkingScores(
            correct=int(strict_hit), wrong=int(not strict_hit)
        )
        lenient += PredictedLinkingScores(
            correct=int(lenient_hit), wrong=int(not lenient_hit)
        )

    return PredictedLinkingReport(
        namespace=namespace,
        entity_types=tuple(entity_types),
        documents=len(documents),
        annotated=len(by_span),
        judged=strict.total,
        outside_bridge=outside_bridge,
        ambiguous_gold=ambiguous_gold,
        strict=strict,
        lenient=lenient,
    )


__all__ = [
    "LinkingReport",
    "PredictedLinkingReport",
    "PredictedLinkingScores",
    "TaggedSpan",
    "score_linking",
    "score_predicted_linking",
]
