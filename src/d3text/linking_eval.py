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

        Every accuracy is emitted beside `test/linking_coverage` and the
        counts it is taken over, so a chart of the accuracy alone still has
        the denominator one key away.

        :return: the metric keys and their values.
        """
        metrics = {
            "test/linking_strict_accuracy": self.strict.accuracy,
            "test/linking_lenient_accuracy": self.lenient.accuracy,
            "test/linking_coverage": self.coverage,
            "test/linking_annotated": float(self.annotated),
            "test/linking_judged": float(self.judged),
            "test/linking_outside_bridge": float(self.outside_bridge),
            "test/linking_ambiguous_gold": float(self.ambiguous_gold),
            "test/linking_documents": float(self.documents),
            "test/linking_correct": float(self.strict.correct),
            "test/linking_wrong": float(self.strict.wrong),
            "test/linking_nil_correct": float(self.strict.nil_correct),
            "test/linking_nil_missed": float(self.strict.nil_missed),
        }
        for size in _CANDIDATE_BUCKETS:
            metrics[f"test/linking_candidates_{size}"] = float(
                self.candidates.get(size, 0)
            )
        metrics["test/linking_candidates_nil"] = float(
            self.candidates.get(0, 0)
        )
        metrics["test/linking_candidates_4_plus"] = float(
            sum(
                count
                for size, count in self.candidates.items()
                if size > _CANDIDATE_BUCKETS[-1]
            )
        )
        return metrics

    def summary(self) -> str:
        """The result as a paragraph, coverage stated beside every score.

        :return: one paragraph naming the accuracies and what they are over.
        """
        shares = ", ".join(
            f"{bucket} -> {share:.1%}"
            for bucket, share in self.candidate_share().items()
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
) -> LinkingReport:
    """Score `linker` on the mentions `bridge` gives a single gold entity.

    Mentions are keyed by `(document, start, end)`; a span annotated with two
    different identifiers joins `ambiguous_gold`, since its gold is no more a
    single entity than a duplicated BRENDA row's is. The type the linker is
    asked for is the gold entity's own, so asking for two types at once judges
    a species curated under both as ambiguous rather than twice.

    :param mentions: the annotator's spans, each carrying its identifier.
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
        entities = (
            _typed(bridge.entity_ids(next(iter(external_ids))), types_by_prefix)
            if len(external_ids) == 1
            else {}
        )
        if len(external_ids) == 1 and not entities:
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
    )


__all__ = ["LinkingReport", "score_linking"]
