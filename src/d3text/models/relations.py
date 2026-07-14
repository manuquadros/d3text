"""Relation extraction as a component: the head, the candidate pairs it scores,
and the alignment that its loss and its metrics both go through.

A model *has* a `RelationExtractor` rather than *being* one. `BrendaModel` holds
one when its config asks for relations and holds `None` when it does not, and
keeps one signature per method either way — where the end-to-end model used to
be a subclass that widened almost every method of the two-head model it
inherited from, and no caller could tell which arity it was about to get.

The candidate pairs are the ones the *entity* head proposed: token positions
whose entity prediction is confident and not UNK, merged during training with
the gold pairs. That is what makes the relation metrics delicate. Gold the
entity head never proposed leaves no row to be scored on, so a metric computed
over the scored rows alone measures relation classification *conditional on the
entity head having already found both arguments*, over a denominator the model
picks for itself. `unscored_gold` names that missing gold, and every relation
metric goes through it.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int64
from torch import Tensor
from torch.autograd.profiler import record_function

from d3text.schema import Schema

from .base import pool_logits
from .config import ModelConfig
from .heads import BiaffineRelationClassifier
from .model_types import IndexedRelation

__all__ = [
    "AlignedRelations",
    "RelationExtractor",
    "RelationPairs",
    "balanced_class_weights",
    "focal_cross_entropy",
]


class RelationPairs(NamedTuple):
    """Candidate argument pairs, and the relation head's logits for them.

    One row per proposal. The same ``(document, subject, object)`` triple can be
    proposed more than once — by two token positions, or by both the hard mask
    and the gold path — so these rows are *not* yet one per triple; `align` is
    what makes them so.
    """

    # ``sequence`` / ``arg_pred_i`` / ``arg_pred_j``: the document, and the two
    # arguments as columns of the entity index.
    meta: dict[str, Tensor]
    logits: Float[Tensor, "pairs relations"]


class AlignedRelations(NamedTuple):
    """`RelationPairs` pooled to one row per triple, each with its gold label.

    The rows are the pairs the entity head *proposed*: gold only ever relabels a
    row that already exists here, and gold that was never proposed has no row at
    all. See `RelationExtractor.unscored_gold`.
    """

    meta: dict[str, Tensor]
    logits: Float[Tensor, "pairs relations"]
    targets: Int64[Tensor, " pairs"]


def balanced_class_weights(
    targets: Int64[Tensor, " relation"], num_classes: int
) -> Float[Tensor, " classes"]:
    """Inverse-frequency class weights for one batch of relation targets.

    Candidate pairs are proposed per batch by the entity hard mask, so the
    `none` share is a property of the current entity head rather than of the
    corpus: there is no dataset frequency to precompute, and the weights have to
    be re-derived every batch.

    A class absent from `targets` would divide by zero. Its weight is never read
    — `cross_entropy` gathers weights by target value — so clamping the count is
    enough to keep the tensor finite.
    """
    counts = torch.bincount(targets, minlength=num_classes)
    return targets.numel() / (num_classes * counts.clamp(min=1))


def focal_cross_entropy(
    preds: Float[Tensor, "relation logits"],
    targets: Int64[Tensor, " relation"],
    gamma: float,
    label_smoothing: float = 0.0,
) -> Float[Tensor, ""]:
    """Cross-entropy with each element scaled by `(1 - p_t) ** gamma`.

    Suppresses the loss from pairs the model already scores confidently, which
    is most of what the hard mask proposes. Unlike a fixed class weight this
    tracks the entity head: as the mask sharpens and stops emitting junk pairs,
    the down-weighting relaxes on its own. `gamma == 0` is plain cross-entropy.

    Normalising by the modulation mass rather than by the row count is what
    makes that work. Under a plain `.mean()` an easy pair still divides the
    denominator, so proposing more of them shrinks the loss on the rare
    positives — the dilution this weighting exists to remove. Dividing by the
    mass instead keeps an easy pair out of *both* sides. The clamp guards the
    degenerate batch in which every pair is already scored confidently: the
    numerator vanishes with the mass, so the loss decays to zero instead of
    exploding.
    """
    elementwise = torch.nn.functional.cross_entropy(
        preds, targets, reduction="none", label_smoothing=label_smoothing
    )
    p_t = preds.softmax(dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1)
    modulation = (1 - p_t) ** gamma
    return (modulation * elementwise).sum() / modulation.sum().clamp(min=1.0)


class RelationExtractor(nn.Module):
    """The biaffine relation head, and the candidate pairs it is given to score.

    :param hidden_size: width of the token representations the model hands over.
    :param schema: names the relation labels *and* their column order. The model
        does not name them a second time: `Schema.validate` already enforces that
        `none` comes last, which is the order the corpus' one-hot label vectors
        are in and whose argmax the losses and metrics take as the target index.
    :param entity_to_index: the entity index, for resolving a gold relation's
        arguments to the columns the entity head scores them in.
    :param unk_index: the entity head's UNK column. A token predicted UNK is not
        an entity, so it proposes no pair.
    """

    classifier: BiaffineRelationClassifier

    def __init__(
        self,
        hidden_size: int,
        schema: Schema,
        entity_to_index: Mapping[str, int],
        unk_index: int,
        config: ModelConfig,
    ) -> None:
        super().__init__()

        self.labels = schema.relation_names
        self.none_index = schema.none_relation_index
        self.num_relations = len(self.labels)

        self.classifier = BiaffineRelationClassifier(
            hidden_size=hidden_size,
            num_relations=self.num_relations,
            separate_predicate_layer=config.separate_predicate_layer,
            biaffine_hidden_size=config.biaffine_hidden_size,
        )

        self.entity_to_index = entity_to_index
        self.unk_index = unk_index
        self.entropy_threshold = config.entity_entropy_threshold
        self.pooling = config.entity_logits_pooling
        self.ramp_epochs = config.ramp_epochs
        self.label_smoothing = config.relation_label_smoothing
        self.loss_weighting = config.relation_loss_weighting
        self.focal_gamma = config.relation_focal_gamma

    @property
    def device(self) -> torch.device:
        """Read off the head's own parameters, so the extractor needs no copy of
        the model's device to fall out of step with."""
        return self.classifier.bias.device

    def loss_weight(self, epoch: int, w0: float = 0.1) -> float:
        """The relation loss' weight at `epoch`, ramping linearly from `w0` to
        1.0 over `ramp_epochs` (which, at 0, means no ramp at all).

        The schedule holds the relation head back until the entity head proposes
        usable pairs to classify. It belongs to the relation objective and to
        nothing else: the entity and class losses train at full weight from the
        first epoch, and a model with no relation extractor has no ramp to reach
        for.
        """
        if not self.ramp_epochs:
            return 1.0
        t = min(1.0, epoch / float(self.ramp_epochs))
        return w0 + (1.0 - w0) * t

    @record_function("relation_forward")
    def forward(
        self,
        hidden: Float[Tensor, "document token features"],
        entity_logits: Float[Tensor, "document token entities"],
        unmasked_entity_logits: Float[Tensor, "document token entities"],
        attention_mask: Bool[Tensor, "document token"],
        gold_relations: Sequence[IndexedRelation] | None = None,
    ) -> RelationPairs | None:
        """Score every candidate pair this batch offers.

        :param entity_logits: the entity head's logits with padding already
            masked out — what the hard mask reads.
        :param unmasked_entity_logits: the same logits before masking, which is
            what the gold path's soft attention weights are taken over.
        :param gold_relations: during training, the batch's gold. Their argument
            pairs are scored whether or not the entity head proposed them, so
            the head sees a positive even while the mask is still junk.
        :return: one row per proposal, or `None` if the batch offers no pair at
            all.
        """
        proposed = self._propose(hidden, entity_logits)
        gold = (
            None
            if gold_relations is None
            else self._gold_pairs(
                hidden, unmasked_entity_logits, attention_mask, gold_relations
            )
        )
        return self._merge(proposed, gold)

    def _propose(
        self,
        hidden: Float[Tensor, "document token features"],
        entity_logits: Float[Tensor, "document token entities"],
    ) -> RelationPairs | None:
        """The pairs the entity head proposes: token positions whose entity
        prediction is confident (low entropy) and is not UNK.

        The threshold decides how many candidate pairs the relation head ever
        sees, and therefore how much gold it can never recover. Raising it
        proposes more pairs, most of them `none`.
        """
        probs: Float[Tensor, "document token entities"] = torch.softmax(
            entity_logits, dim=-1
        )
        entropy = -(probs * probs.clamp_min(1e-9).log()).sum(-1)
        max_indices = probs.argmax(dim=-1)

        hard_entity_mask: Bool[Tensor, "document token"] = (
            max_indices != self.unk_index
        ) & (entropy <= self.entropy_threshold)

        if not hard_entity_mask.any():
            return None

        positions: Int64[Tensor, "n_entities 2"] = hard_entity_mask.nonzero(
            as_tuple=False
        )
        if positions.numel() < 2:
            return None

        reprs = hidden[positions[:, 0], positions[:, 1]]
        return self._pairs_from_positions(positions, reprs, max_indices)

    def _pairs_from_positions(
        self,
        entity_positions: Int64[Tensor, "n_entities 2"],
        entity_reprs: Float[Tensor, "n_entities features"],
        max_indices: Int64[Tensor, "document token"],
    ) -> RelationPairs | None:
        """Every unordered pair of *distinct predicted entities* within a
        document, scored by the head.

        Token positions predicting the same entity are pooled into one
        representation first, so an entity mentioned twice proposes one argument,
        not two.
        """
        device = entity_positions.device
        doc_ids = entity_positions[:, 0]
        token_positions = entity_positions[:, 1]

        # Which entity the entity classifier assigned to each masked token: an
        # index into the entity head's columns.
        entity_preds: Int64[Tensor, " n_entities"] = max_indices[
            doc_ids, token_positions
        ]

        doc_batch = []
        arg_pred_i = []
        arg_pred_j = []
        reprs_i = []
        reprs_j = []

        for doc_id in torch.unique(doc_ids):
            indices = torch.where(doc_ids == doc_id)[0]
            if len(indices) < 2:
                continue

            local_preds = entity_preds[indices]
            local_reprs = entity_reprs[indices]
            unique_local_preds = torch.unique(local_preds)

            pooled_reprs = torch.stack(
                [
                    local_reprs[local_preds == pred].mean(dim=0)
                    for pred in unique_local_preds
                ]
            )

            pairs = torch.combinations(
                torch.arange(len(unique_local_preds), device=device), r=2
            )
            if len(pairs) == 0:
                continue

            i, j = pairs[:, 0], pairs[:, 1]
            doc_batch.append(
                torch.full((len(i),), doc_id, dtype=torch.long, device=device)
            )
            arg_pred_i.append(unique_local_preds[i])
            arg_pred_j.append(unique_local_preds[j])
            reprs_i.append(pooled_reprs[i])
            reprs_j.append(pooled_reprs[j])

        if not reprs_i:
            return None

        logits = self.classifier(
            torch.cat(reprs_i, dim=0), torch.cat(reprs_j, dim=0)
        )
        meta = {
            "sequence": torch.cat(doc_batch),
            "arg_pred_i": torch.cat(arg_pred_i),
            "arg_pred_j": torch.cat(arg_pred_j),
        }
        return RelationPairs(meta, logits)

    def _gold_pairs(
        self,
        hidden: Float[Tensor, "document token features"],
        unmasked_entity_logits: Float[Tensor, "document token entities"],
        attention_mask: Bool[Tensor, "document token"],
        gold_relations: Sequence[IndexedRelation],
    ) -> RelationPairs | None:
        """The gold relations' argument pairs, each argument represented by the
        tokens the entity head *attends* to for it.

        A soft, differentiable stand-in for the hard mask's pooled positions: the
        gold pair gets a row even when the mask proposed neither argument, which
        is the only way the head sees a positive early in training.
        """
        needed_by_doc: dict[int, set[int]] = {}
        for relation in gold_relations:
            subject, object_ = self._argument_columns(relation)
            if subject < 0 or object_ < 0:
                continue
            needed_by_doc.setdefault(int(relation.docix), set()).update(
                (subject, object_)
            )

        soft_repr_by_doc = {
            docix: {
                entity: self._soft_entity_repr(
                    doc_hidden=hidden[docix],
                    doc_ent_logits=unmasked_entity_logits[docix],
                    doc_mask=attention_mask[docix].to(torch.bool),
                    entity=entity,
                )
                for entity in entities
            }
            for docix, entities in needed_by_doc.items()
        }

        rows_doc, rows_i, rows_j, rep_i, rep_j = [], [], [], [], []
        for relation in gold_relations:
            docix = int(relation.docix)
            doc_reps = soft_repr_by_doc.get(docix)
            if not doc_reps:
                continue
            subject, object_ = self._argument_columns(relation)
            if subject in doc_reps and object_ in doc_reps:
                rows_doc.append(docix)
                rows_i.append(subject)
                rows_j.append(object_)
                rep_i.append(doc_reps[subject])
                rep_j.append(doc_reps[object_])

        if not rep_i:
            return None

        device = self.device
        logits = self.classifier(
            torch.stack(rep_i, dim=0), torch.stack(rep_j, dim=0)
        )
        meta = {
            "sequence": torch.tensor(rows_doc, device=device, dtype=torch.long),
            "arg_pred_i": torch.tensor(rows_i, device=device, dtype=torch.long),
            "arg_pred_j": torch.tensor(rows_j, device=device, dtype=torch.long),
        }
        return RelationPairs(meta, logits)

    def _soft_entity_repr(
        self,
        doc_hidden: Float[Tensor, "tokens features"],
        doc_ent_logits: Float[Tensor, "tokens entities"],
        doc_mask: Bool[Tensor, " tokens"],
        entity: int,
    ) -> Float[Tensor, " features"]:
        """`entity`'s representation: the document's tokens, weighted by how
        strongly the entity head scores each of them for it."""
        with torch.autocast(device_type=doc_hidden.device.type, enabled=False):
            scores = doc_ent_logits[:, entity].float()
            scores = scores.masked_fill(~doc_mask, float("-inf"))
            weights = torch.softmax(scores, dim=0)
            representation = (weights.unsqueeze(-1) * doc_hidden.float()).sum(
                dim=0
            )
        return representation.to(doc_hidden.dtype)

    def _argument_columns(self, relation: IndexedRelation) -> tuple[int, int]:
        """A gold relation's arguments as entity-head columns, `(-1, -1)` for an
        argument the entity index does not know."""
        return (
            int(self.entity_to_index.get(relation.subject, -1)),
            int(self.entity_to_index.get(relation.object, -1)),
        )

    @staticmethod
    def _merge(
        proposed: RelationPairs | None, gold: RelationPairs | None
    ) -> RelationPairs | None:
        """One row per triple, preferring gold's soft representation.

        A triple can be produced by both paths. Keeping both rows would let the
        aligner pool two rows for one triple, biasing its logits upward, so the
        overlapping hard-mask row is dropped in favour of the richer gold one.
        """
        if proposed is None or gold is None:
            return proposed or gold

        gold_keys = set(
            zip(
                gold.meta["sequence"].tolist(),
                gold.meta["arg_pred_i"].tolist(),
                gold.meta["arg_pred_j"].tolist(),
            )
        )
        keep = [
            row
            for row, key in enumerate(
                zip(
                    proposed.meta["sequence"].tolist(),
                    proposed.meta["arg_pred_i"].tolist(),
                    proposed.meta["arg_pred_j"].tolist(),
                )
            )
            if key not in gold_keys
        ]
        keep_idx = torch.tensor(
            keep, device=proposed.logits.device, dtype=torch.long
        )

        meta = {
            field: torch.cat([proposed.meta[field][keep_idx], gold.meta[field]])
            for field in ("sequence", "arg_pred_i", "arg_pred_j")
        }
        logits = torch.cat([proposed.logits[keep_idx], gold.logits], dim=0)
        return RelationPairs(meta, logits)

    def align(
        self,
        true_relations: Sequence[IndexedRelation],
        pairs: RelationPairs | None,
    ) -> AlignedRelations | None:
        """Pool `pairs` to one row per triple and give each its gold label.

        Both the loss and the metrics go through this, so they pool duplicates
        and assign targets identically. A triple with no gold is a `none`.
        """
        if pairs is None or pairs.logits.numel() == 0:
            return None

        def as_list(x: Tensor) -> list[int]:
            return x.detach().cpu().tolist()

        rows = list(
            zip(
                as_list(pairs.meta["sequence"]),
                as_list(pairs.meta["arg_pred_i"]),
                as_list(pairs.meta["arg_pred_j"]),
            )
        )
        if len(rows) != pairs.logits.size(0):
            raise ValueError(
                f"rel_meta describes {len(rows)} pairs but the head scored "
                f"{pairs.logits.size(0)}"
            )
        if not rows:
            return None

        groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for row, key in enumerate(rows):
            groups[key].append(row)

        gold_by_key: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for relation in true_relations:
            subject, object_ = self._argument_columns(relation)
            if subject < 0 or object_ < 0:
                continue  # gold naming an entity outside the index
            gold_by_key[(int(relation.docix), subject, object_)].append(
                int(relation.label)
            )

        device = pairs.logits.device
        pooled_logits = []
        targets = []
        keys = []

        for key, row_indices in groups.items():
            pooled_logits.append(
                pool_logits(pairs.logits[row_indices], self.pooling, dim=0)
            )
            targets.append(self._target_for(gold_by_key.get(key)))
            keys.append(key)

        meta = {
            field: torch.tensor(
                [key[position] for key in keys],
                dtype=torch.long,
                device=device,
            )
            for position, field in enumerate(
                ("sequence", "arg_pred_i", "arg_pred_j")
            )
        }
        return AlignedRelations(
            meta=meta,
            logits=torch.stack(pooled_logits, dim=0).to(device),
            targets=torch.tensor(targets, dtype=torch.long, device=device),
        )

    def _target_for(self, labels: list[int] | None) -> int:
        """The gold label of one triple: a positive one if the corpus gives any,
        else `none`."""
        if not labels:
            return int(self.none_index)
        positives = [label for label in labels if label != self.none_index]
        return positives[0] if positives else labels[0]

    def unscored_gold(
        self,
        true_relations: Sequence[IndexedRelation],
        scored_meta: Mapping[str, Tensor] | None,
    ) -> tuple[list[int], list[int]]:
        """Gold relations that no scored row can account for.

        `align` builds its rows out of the *candidate* pairs the entity head
        proposed, and gold only ever relabels a row that already exists. Gold
        whose triple was never proposed therefore leaves no row at all, and so
        cannot show up in any metric computed over those rows: it is not a false
        negative, it is absent. A caller computing metrics must add these back as
        misses — see `none_predictions`.

        Deliberately not folded into `align`: the loss consumes that, and these
        relations carry no logits to backpropagate.

        :param scored_meta: the meta of the rows actually scored — the pooled
            meta `align` returned, or `None` when it returned nothing.
        :return: the labels of the missed gold, as
            ``(not_proposed, out_of_vocabulary)``. A relation is out of
            vocabulary when either argument is absent from `entity_to_index`,
            which no relation head can fix; the rest were simply never proposed.
        """
        scored: set[tuple[int, int, int]] = set()
        if scored_meta:
            scored = set(
                zip(
                    scored_meta["sequence"].tolist(),
                    scored_meta["arg_pred_i"].tolist(),
                    scored_meta["arg_pred_j"].tolist(),
                )
            )

        not_proposed: list[int] = []
        out_of_vocabulary: list[int] = []

        for relation in true_relations:
            subject, object_ = self._argument_columns(relation)
            if subject < 0 or object_ < 0:
                out_of_vocabulary.append(int(relation.label))
            elif (int(relation.docix), subject, object_) not in scored:
                not_proposed.append(int(relation.label))

        return not_proposed, out_of_vocabulary

    def none_predictions(
        self, missed: Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Score each missed gold relation as the `none` prediction the model
        effectively made by never proposing it."""
        return (
            np.asarray(missed, dtype=int),
            np.full(len(missed), int(self.none_index), dtype=int),
        )

    @record_function("compute_relation_loss")
    def loss(
        self,
        true_relations: Sequence[IndexedRelation],
        pairs: RelationPairs | None,
    ) -> Float[Tensor, ""]:
        """Cross-entropy over the aligned candidate pairs, weighted as
        `ModelConfig.relation_loss_weighting` asks.

        Because the hard mask proposes the candidates, the `none` share of the
        pairs tracks the *current* entity head rather than any corpus frequency,
        which is why `balanced` re-derives its weights every batch.
        """
        aligned = self.align(true_relations, pairs)
        if aligned is None:
            return torch.tensor(0.0, device=self.device)

        if self.loss_weighting == "focal":
            return focal_cross_entropy(
                aligned.logits,
                aligned.targets,
                gamma=self.focal_gamma,
                label_smoothing=self.label_smoothing,
            )

        weight = (
            balanced_class_weights(aligned.targets, self.num_relations)
            if self.loss_weighting == "balanced"
            else None
        )
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=weight,
            reduction="mean",
            label_smoothing=self.label_smoothing,
        )
        return loss_fn(aligned.logits, aligned.targets)
