"""`ETEBrendaModel` — entity ID + class detection + relation extraction."""

import logging
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from d3text import tracking
from d3text.progress import batch_progress
from d3text.schema import Schema
from jaxtyping import Bool, Float, Int64
from sklearn.metrics import (
    classification_report,
    f1_score,
    label_ranking_average_precision_score,
)
from torch import Tensor
from torch.autograd.profiler import record_function
from torch.utils.data import DataLoader

from . import base
from .base import (
    Model,
    Step,
    balanced_class_weights,
    coverage_metrics,
    focal_cross_entropy,
    relation_metrics,
    support_metrics,
)
from .config import ModelConfig
from .entity_linking import BrendaClassificationModel
from .heads import BiaffineRelationClassifier
from .model_types import (
    BatchItem,
    BatchLogits,
    BatchLosses,
    GroundTruth,
    IndexedRelation,
)
from .token_supervision import TokenLabelReader

logger = logging.getLogger(__name__)


class ETEBrendaModel(Model):
    """Entity ID + class detection + relation extraction.

    Composes a `BrendaClassificationModel` for the entity and class machinery
    rather than subclassing it, so both return the same typed containers
    instead of widening them. `__getattr__` reaches through to that model for
    what this class does not declare, and is read-only by construction: a value
    that must reach it on a write needs its own property.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        two_head = self.__dict__.get("_modules", {}).get("two_head")
        if two_head is None:
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            )
        return getattr(two_head, name)

    @property
    def _token_labels(self) -> TokenLabelReader | None:
        return self.two_head._token_labels

    @_token_labels.setter
    def _token_labels(self, reader: TokenLabelReader | None) -> None:
        self.two_head._token_labels = reader

    def __init__(
        self,
        schema: Schema,
        class_matrix: Float[Tensor, "entity class"],
        entity_index: dict[str, int],
        config: ModelConfig | None = None,
        entity_freqs: Float[Tensor, " entities"] | None = None,
        class_freqs: Float[Tensor, " classes"] | None = None,
        device: str | None = None,
    ) -> None:
        config = config if config is not None else ModelConfig()
        super().__init__(config, device=device)

        self.schema = schema
        self.two_head = BrendaClassificationModel(
            schema=schema,
            class_matrix=class_matrix,
            entity_index=entity_index,
            config=config,
            entity_freqs=entity_freqs,
            class_freqs=class_freqs,
            device=device,
        )

        self.relations = self.schema.relation_names
        self.relations_none_index = self.schema.none_relation_index
        self.num_relations = len(self.relations)
        self.relation_classifier = BiaffineRelationClassifier(
            hidden_size=self.two_head.hidden_block_output_size,
            num_relations=len(self.relations),
            separate_predicate_layer=self.config.separate_predicate_layer,
            biaff_hidden_size=self.config.biaffine_hidden_size,
        )

        self.relation_label_smoothing = self.config.relation_label_smoothing
        self.relation_loss_weighting = self.config.relation_loss_weighting
        self.relation_focal_gamma = self.config.relation_focal_gamma

    def relation_loss_weight(self, epoch: int, w0: float = 0.1) -> float:
        """The relation loss' weight at `epoch`, ramping `w0` to 1.0.

        The ramp runs over `ramp_epochs`, which at 0 means no ramp at all. It
        holds the relation head back until the entity head proposes usable
        pairs; no other objective in this package rides a schedule.

        :param epoch: the epoch about to run.
        :param w0: the weight at epoch 0.
        :return: the multiplier for this epoch.
        """
        if not self.ramp_epochs:
            return 1.0
        t = min(1.0, epoch / float(self.ramp_epochs))
        return w0 + (1.0 - w0) * t

    def compute_losses(
        self,
        batch: Sequence[BatchItem],
        step: Step,
        epoch: int,
    ) -> dict[str, Tensor]:
        """This batch's entity, class, relation and (optional) token losses.

        The relation term is scaled by the ramp here, before `run_epoch` sees
        it, so the generic accumulation stays oblivious to the schedule.
        Validation totals are scored under the ramp's final weight, since early
        stopping reads them as one series across epochs; only the training
        gradient follows it.

        :param batch: the batch to run.
        :param step: whether this is a training or a validation pass.
        :param epoch: the epoch number, which sets the ramp weight.
        :return: one loss per objective, `token` present only with a label
            store.
        """
        w_rel = (
            self.relation_loss_weight(epoch)
            if step == Step.TRAINING
            else self.relation_loss_weight(self.ramp_epochs)
        )

        batch_losses = self.compute_batch_losses(batch)
        assert batch_losses.relation is not None  # this model always scores one

        losses = {
            "entity": batch_losses.entity,
            "class": batch_losses.class_,
            "relation": batch_losses.relation * w_rel,
        }
        token_loss = batch_losses.token
        if token_loss is not None:
            # Unramped: the token targets are supervision available from
            # epoch 0, like the entity BCE, not a late-phase objective.
            losses["token"] = token_loss

        return losses

    def epoch_loss_weights(self, epoch: int) -> dict[str, float]:
        """Only the relation loss is scheduled.

        :param epoch: the epoch about to run.
        :return: every objective's multiplier, the rest at the full weight they
            train under so each has a curve.
        """
        weights = {
            "entity": 1.0,
            "class": 1.0,
            "relation": self.relation_loss_weight(epoch),
        }
        if getattr(self, "token_tagger", None) is not None:
            weights["token"] = 1.0
        return weights

    def ground_truth(
        self,
        batch: Sequence[BatchItem],
    ) -> GroundTruth:
        """The gold entities, classes and relations of the batch's documents.

        :param batch: the batch to read.
        :return: the targets, `relations` always a (possibly empty) list here.
        """
        parent = self.two_head.ground_truth(batch)
        entity_targets, class_targets = parent.entities, parent.classes

        relation_targets = []
        for docix, doc in enumerate(batch):
            # A document carries a *list* of pair-dicts, and every one of them
            # is gold: reading only the first silently trained the relation
            # head on a subset of its own labels.
            for doc_relations in doc.get("relations", []):
                for args, label in doc_relations.items():
                    relation_targets.append(
                        IndexedRelation(
                            docix=docix,
                            subject=args[0],
                            object=args[1],
                            label=label.argmax(),
                        )
                    )

        return GroundTruth(entity_targets, class_targets, relation_targets)

    def _gold_relation_key(
        self, relation: IndexedRelation
    ) -> tuple[int, int, int] | None:
        """`(doc, column, column)` for a gold relation, columns ascending.

        Candidate pairs arrive with ascending columns from
        `torch.combinations`, while gold arguments arrive lexicographic on the
        entity-ID strings, so every pair whose string order reverses its column
        order could never match. Sorting loses no direction: the string sort
        already discarded argument order, and the label is directional by
        argument *type*.

        :return: the key, or None when either argument is missing from
            `entity_to_index`.
        """
        try:
            i = int(self.entity_to_index[relation.subject])
            j = int(self.entity_to_index[relation.object])
        except KeyError:
            return None
        if i > j:
            i, j = j, i
        return int(relation.docix), i, j

    def _unindexed_gold_relation_key(
        self, relation: IndexedRelation
    ) -> tuple[int, str, str]:
        """`(doc, argument, argument)` for a gold relation, arguments sorted.

        What identifies gold that `_gold_relation_key` cannot key: an argument
        outside `entity_to_index` has no column, so the entity-ID strings are
        the only identity left. Sorted here rather than taken on trust from the
        corpus, for the reason that function orders its columns.
        """
        first, second = sorted((relation.subject, relation.object))
        return int(relation.docix), first, second

    def _missed_gold_label(self, labels: Sequence[int]) -> int:
        """The single label a repeated missed gold triple is counted under.

        The aligner prefers a non-none label, and a miss counted as `none`
        would leave the typed metrics — which exclude `none` — rather than
        count against the model.
        """
        none_idx = int(self.relations_none_index)
        return next((lbl for lbl in labels if lbl != none_idx), labels[0])

    def align_relation_predictions(
        self,
        true_relations: Sequence[IndexedRelation],
        rel_meta: dict[str, Tensor],
        rel_logits: Float[Tensor, "relation logits"] | None,
    ) -> (
        tuple[
            dict[str, Tensor],
            Float[Tensor, "relation logits"],
            Int64[Tensor, " relation"],
        ]
        | None
    ):
        if rel_logits is None or rel_logits.numel() == 0:
            return None

        device = rel_logits.device
        seq, subj, obj = (
            rel_meta[key].detach().to(device=device, dtype=torch.long)
            for key in ("sequence", "arg_pred_i", "arg_pred_j")
        )

        n_rows = rel_logits.size(0)
        assert (
            seq.numel() == n_rows
            and subj.numel() == n_rows
            and obj.numel() == n_rows
        ), "rel_meta fields must align with rel_logits rows"

        none_idx = int(self.relations_none_index)

        # The gold side is Python data — a Sequence of NamedTuples keyed by
        # entity *strings* — so its lookup is built host-side, as before. Only
        # the join against the candidate triples runs on the device.
        gold_by_key: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for tr in true_relations:
            key = self._gold_relation_key(tr)
            if key is None:
                continue  # gold refers to entity not mapped in this doc/batch
            gold_by_key[key].append(int(tr.label))

        gold_triples: list[tuple[int, int, int]] = []
        gold_labels: list[int] = []
        for key, labels in gold_by_key.items():
            # If multiple labels exist, prefer any non-none; else first.
            # (Adjust policy if your schema allows multi-label relations.)
            gold_triples.append(key)
            gold_labels.append(
                next((lbl for lbl in labels if lbl != none_idx), labels[0])
            )
        gold_index = torch.tensor(
            gold_triples, dtype=torch.long, device=device
        ).reshape(-1, 3)

        # Pack (sequence, subject, object) into one int64 so that grouping is a
        # single `torch.unique` and the gold join a single `searchsorted`.
        # The radices are read off the data instead of being fixed bit widths:
        # the argument indices are argmaxes over the whole entity vocabulary,
        # whose size is a property of the dataset, not of this function. Their
        # product is bounded by batch x |entities|^2 and stays far inside int64.
        def _radix(candidate: Tensor, gold: Tensor) -> Tensor:
            """One past the largest index either side of the join uses."""
            highest = candidate.max()
            if gold.numel():  # shape metadata, not a device read
                highest = torch.maximum(highest, gold.max())
            return highest + 1

        radix_i = _radix(subj, gold_index[:, 1])
        radix_j = _radix(obj, gold_index[:, 2])

        def _pack(s: Tensor, i: Tensor, j: Tensor) -> Tensor:
            return (s * radix_i + i) * radix_j + j

        keys = _pack(seq, subj, obj)
        unique_keys, inverse, counts = torch.unique(
            keys, return_inverse=True, return_counts=True
        )
        n_groups = int(unique_keys.numel())

        pooled_logits = self._pool_logits_segments(
            rel_logits, inverse, n_groups, counts
        )

        # One scratch slot past the groups absorbs gold triples that no
        # candidate pair proposed; masking them out instead would need a
        # boolean index, whose data-dependent shape is itself a device sync.
        targets = torch.full(
            (n_groups + 1,), none_idx, dtype=torch.long, device=device
        )
        if gold_labels:
            gold_keys = _pack(
                gold_index[:, 0], gold_index[:, 1], gold_index[:, 2]
            )
            slot = torch.searchsorted(unique_keys, gold_keys).clamp(
                max=n_groups - 1
            )
            slot = torch.where(unique_keys[slot] == gold_keys, slot, n_groups)
            targets = targets.scatter(
                0,
                slot,
                torch.tensor(gold_labels, dtype=torch.long, device=device),
            )
        targets = targets[:n_groups]

        # `torch.unique` returns its groups sorted; restore the first-appearance
        # order the row loop produced, so the returned rows keep the order every
        # caller has seen so far.
        first_row = torch.full(
            (n_groups,), n_rows, dtype=torch.long, device=device
        ).scatter_reduce(
            0, inverse, torch.arange(n_rows, device=device), reduce="amin"
        )
        order = first_row.argsort()
        ordered_keys = unique_keys[order]

        pooled_meta = {
            "sequence": ordered_keys // (radix_i * radix_j),
            "arg_pred_i": (ordered_keys // radix_j) % radix_i,
            "arg_pred_j": ordered_keys % radix_j,
        }

        return pooled_meta, pooled_logits[order], targets[order]

    def unscored_gold_relations(
        self,
        true_relations: Sequence[IndexedRelation],
        scored_meta: dict[str, Tensor] | None,
    ) -> tuple[list[int], list[int]]:
        """Gold relations that no scored row can account for.

        The aligner builds its rows out of the candidate pairs the entity head
        proposed, so gold that was never proposed leaves no row and cannot
        appear in any metric over those rows. A caller computing metrics must
        add these back as misses. Kept out of the aligner because the loss path
        consumes that function and these relations carry no logits to
        backpropagate.

        :param true_relations: the document's gold relations.
        :param scored_meta: the meta of the rows actually scored, or None when
            the aligner returned nothing.
        :return: the labels of the missed gold, as `(not_proposed,
            out_of_vocabulary)`. Out of vocabulary means an argument no entity
            column exists for, which no relation head can fix. A gold triple
            repeated across a document's pair-dicts yields one entry in
            either list.
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

        missed_by_key: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        out_of_vocabulary_by_key: dict[tuple[int, str, str], list[int]] = (
            defaultdict(list)
        )

        for relation in true_relations:
            key = self._gold_relation_key(relation)
            if key is None:
                out_of_vocabulary_by_key[
                    self._unindexed_gold_relation_key(relation)
                ].append(int(relation.label))
                continue

            if key not in scored:
                missed_by_key[key].append(int(relation.label))

        return (
            [
                self._missed_gold_label(labels)
                for labels in missed_by_key.values()
            ],
            [
                self._missed_gold_label(labels)
                for labels in out_of_vocabulary_by_key.values()
            ],
        )

    def _missed_gold_predictions(
        self, missed: Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Score each missed gold relation as a `none` prediction."""
        return (
            np.asarray(missed, dtype=int),
            np.full(len(missed), int(self.relations_none_index), dtype=int),
        )

    @record_function("compute_relation_loss")
    def compute_relation_loss(
        self,
        true_relations: Sequence[IndexedRelation],
        rel_meta: dict[str, Tensor],
        rel_logits: Float[Tensor, "relation logits"] | None,
    ) -> Float[Tensor, ""]:
        aligned_rel_preds = self.align_relation_predictions(
            true_relations=true_relations,
            rel_meta=rel_meta,
            rel_logits=rel_logits,
        )
        if aligned_rel_preds is None:
            return torch.tensor(0.0, device=self.device)

        _, preds, targets = aligned_rel_preds

        if self.relation_loss_weighting == "focal":
            return focal_cross_entropy(
                preds,
                targets,
                gamma=self.relation_focal_gamma,
                label_smoothing=self.relation_label_smoothing,
            )

        weight = (
            balanced_class_weights(targets, self.num_relations)
            if self.relation_loss_weighting == "balanced"
            else None
        )
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=weight,
            reduction="mean",
            label_smoothing=self.relation_label_smoothing,
        )
        return loss_fn(preds, targets)

    def get_batch_logits(
        self,
        batch: Sequence[BatchItem],
        gold_relations: list[IndexedRelation] | None = None,
    ) -> BatchLogits:
        token_embeddings, token_att_mask = self.get_token_embeddings(batch)

        return self(
            token_embeddings,
            token_att_mask,
            gold_relations=gold_relations,
        )

    def compute_batch_losses(self, batch: Sequence[BatchItem]) -> BatchLosses:
        """This batch's losses, one field per objective.

        :param batch: the batch to run.
        :return: the entity, class, relation and token losses.
        """
        ent_true, class_true, rel_true = self.ground_truth(batch)
        rel_true = rel_true or []
        token_embeddings, token_att_mask = self.get_token_embeddings(batch)
        entity_logits, class_logits, relation_index_logits = self(
            token_embeddings,
            token_att_mask,
            gold_relations=rel_true,
        )

        ent_loss, class_loss = self.compute_entity_loss(
            predictions=(entity_logits, class_logits),
            targets=(ent_true, class_true),
            class_abstain=self.class_negative_abstain_mask(batch, class_true),
        )

        if relation_index_logits is not None:
            rel_index, rel_logits = relation_index_logits
        else:
            rel_index, rel_logits = ({}, None)

        relation_loss = self.compute_relation_loss(
            true_relations=rel_true,
            rel_meta=rel_index,
            rel_logits=rel_logits,
        )

        return BatchLosses(
            entity=ent_loss,
            class_=class_loss,
            relation=relation_loss,
            token=self.compute_token_loss(
                batch, token_embeddings, token_att_mask
            ),
        )

    def compute_batch_true_x_pred(
        self, batch: Sequence[BatchItem]
    ) -> dict[str, dict[str, np.ndarray]]:
        """Gold and predicted arrays for each task the model tackles.

        :param batch: the batch to score.
        :return: task name -> its `y_true` and `y_pred` arrays.
        """
        entity_logits: Float[Tensor, "sequence entities"]
        class_logits: Float[Tensor, "sequence classes"]
        relation_index_logits: (
            tuple[dict[str, Tensor], Float[Tensor, "pairs relations"]] | None
        )
        entity_logits, class_logits, relation_index_logits = (
            self.get_batch_logits(batch)
        )

        entity_truth: Float[Tensor, "batch entities"]
        class_truth: Float[Tensor, "batch classes"]
        entity_truth, class_truth, rel_truth_optional = self.ground_truth(batch)
        rel_truth: list[IndexedRelation] = rel_truth_optional or []
        relations_true = np.array([], dtype=int)
        relations_pred = np.array([], dtype=int)

        if rel_truth:
            aligned_rel_preds = None
            if relation_index_logits:
                rel_meta: dict[str, Tensor]
                rel_logits: Float[Tensor, "pairs relations"]
                rel_meta, rel_logits = relation_index_logits
                aligned_rel_preds = self.align_relation_predictions(
                    true_relations=rel_truth,
                    rel_meta=rel_meta,
                    rel_logits=rel_logits,
                )

            scored_meta = None
            if aligned_rel_preds is not None:
                scored_meta, preds, targets = aligned_rel_preds
                relations_true = (
                    targets.numpy(force=True).reshape(-1).astype(int)
                )
                relations_pred = preds.numpy(force=True)
                relations_pred = (
                    relations_pred.argmax(axis=-1).reshape(-1).astype(int)
                )

            # Gold the entity head never proposed has no row to be scored on,
            # so without this it would vanish from the metrics rather than
            # count against them.
            not_proposed, out_of_vocabulary = self.unscored_gold_relations(
                rel_truth, scored_meta
            )
            missed_true, missed_pred = self._missed_gold_predictions(
                not_proposed + out_of_vocabulary
            )
            relations_true = np.concatenate([relations_true, missed_true])
            relations_pred = np.concatenate([relations_pred, missed_pred])

        if relations_true.shape != relations_pred.shape:
            logger.warning(
                "relations_true %s != relations_pred %s",
                relations_true.shape,
                relations_pred.shape,
            )

        return {
            "entities": {
                "true": entity_truth.numpy(force=True),  # no squeeze
                "pred": torch.sigmoid(entity_logits.float())
                .round()
                .numpy(force=True),
            },
            "classes": {
                "true": class_truth.numpy(force=True),
                "pred": torch.sigmoid(class_logits.float())
                .round()
                .numpy(force=True),
            },
            "relations": {
                "true": np.asarray(relations_true).reshape(-1),
                "pred": np.asarray(relations_pred).reshape(-1),
            },
        }

    def _compute_relations_vectorized(
        self,
        entity_positions: Int64[Tensor, "n_entities 2"],
        entity_reprs: Float[Tensor, "n_entities features"],
        max_indices: Int64[Tensor, "document token"],
    ) -> tuple[dict[str, Tensor], Float[Tensor, "n_pairs relations"]] | None:
        """Relation logits for all valid entity pairs.

        :return: the pairs' index tensors (`doc`, `arg_pred_i`, `arg_pred_j`)
            and their relation logits, or None when no pair was proposed.
        """
        device = self.device
        doc_ids = entity_positions[:, 0]
        token_positions = entity_positions[:, 1]

        # `entity_preds` is a vector of integers indexing self.entities, hence
        # indicating to which entity the token was assigned by the entity
        # classifier.
        entity_preds: Int64[Tensor, " entities"] = max_indices[
            doc_ids, token_positions
        ]

        # Precompute indices and prepare output buffers
        unique_doc_ids = torch.unique(doc_ids)
        doc_batch = []
        arg_pred_i = []
        arg_pred_j = []
        reprs_i = []
        reprs_j = []

        for doc_id in unique_doc_ids:
            indices = torch.where(doc_ids == doc_id)[0]

            if len(indices) < 2:
                continue

            local_pos = token_positions[indices]
            local_preds = entity_preds[indices]
            unique_local_preds = torch.unique(local_preds)
            local_reprs = entity_reprs[indices]

            grouped_entity_positions = [
                local_pos[local_preds == pred] for pred in unique_local_preds
            ]
            pooled_reprs = torch.stack(
                [
                    local_reprs[local_preds == pred].mean(dim=0)
                    for pred in unique_local_preds
                ]
            )

            pairs = torch.combinations(
                torch.arange(len(grouped_entity_positions), device=device),
                r=2,
            )

            if len(pairs) == 0:
                continue

            i, j = pairs[:, 0], pairs[:, 1]
            pred_i = unique_local_preds[i]
            pred_j = unique_local_preds[j]

            n_pairs = len(i)
            doc_batch.append(
                torch.full((n_pairs,), doc_id, dtype=torch.long, device=device)
            )
            arg_pred_i.append(pred_i)
            arg_pred_j.append(pred_j)
            reprs_i.append(pooled_reprs[i])
            reprs_j.append(pooled_reprs[j])

        if reprs_i:
            all_repr_i = torch.cat(reprs_i, dim=0)
            all_repr_j = torch.cat(reprs_j, dim=0)
            logits = self.relation_classifier(all_repr_i, all_repr_j)

            meta = {
                "sequence": torch.cat(doc_batch),
                "arg_pred_i": torch.cat(arg_pred_i),
                "arg_pred_j": torch.cat(arg_pred_j),
            }
        else:
            return None

        return meta, logits

    @record_function("forward")
    def forward(
        self,
        embeddings: Float[Tensor, "document token embedding"],
        attention_mask: Bool[Tensor, "document token"],
        gold_relations: list[IndexedRelation] | None = None,
    ) -> BatchLogits:
        """Entity, class and relation logits for one batch.

        :param embeddings: the batch's token embeddings.
        :param attention_mask: which positions carry a real token.
        :param gold_relations: gold arguments to add soft candidate pairs for,
            on a training pass.
        :return: the pooled logits, `relations` carrying which sequence and
            which pair of entity-index columns each scored row belongs to,
            beside its logits.
        """

        def _soft_entity_repr(
            doc_hidden: Float[Tensor, "tokens hidden_size"],
            doc_ent_logits: Float[Tensor, "tokens entities"],
            doc_mask: Bool[Tensor, " tokens"],
            ent_id: int,
        ) -> Float[Tensor, " hidden_size"]:
            with torch.autocast(device_type=self.device, enabled=False):
                scores = doc_ent_logits[:, ent_id].float()  # [T]
                scores = scores.masked_fill(~doc_mask, float("-inf"))
                w = torch.softmax(scores, dim=0)  # [T]
                rep = (w.unsqueeze(-1) * doc_hidden.float()).sum(dim=0)  # [H]
            return rep.to(doc_hidden.dtype)

        device = self.device
        with self.autocast_context():
            hidden_output: Float[Tensor, "document token features"] = (
                self.hidden(embeddings)
            )
            unmasked_entity_logits, unmasked_class_logits = self.classifier(
                hidden_output
            )
            token_mask = attention_mask.unsqueeze(-1)
            neg_inf = self._neg_inf
            entity_logits = torch.where(
                token_mask, unmasked_entity_logits, neg_inf
            )
            class_logits = torch.where(
                token_mask, unmasked_class_logits, neg_inf
            )

            # Nothing differentiable leaves this block: it yields a bool mask
            # and int64 indices, and the relation head's gradient reaches
            # `hidden_output` by *indexing* it with them, never through the
            # probabilities. Recorded by autograd, the four intermediates below
            # are each a full [document, token, entity] tensor — 864 MB apiece
            # at a p99-length batch — held for a backward that never reads
            # them. The arithmetic is unchanged, so the mask is bit-identical.
            # Sliced along the token dim for the same reason `pool_token_dim`
            # is: `torch.softmax` over the whole tensor, its clamp, its log and
            # the product are four more [document, token, entity] tensors, and
            # even freed immediately they set the peak of the whole step. Every
            # row of a softmax over the last dim is independent of every other,
            # so slicing changes no value — the mask is bitwise what the
            # unsliced expression gave.
            with torch.no_grad():
                entropies = []
                predictions = []
                chunk = base.pool_chunk_tokens(
                    entity_logits.shape[0], entity_logits.shape[2]
                )
                for start in range(0, entity_logits.shape[1], chunk):
                    entity_probs: Float[Tensor, "document token ent_probs"] = (
                        torch.softmax(
                            entity_logits[:, start : start + chunk],
                            dim=-1,
                        )
                    )
                    entropies.append(
                        -(
                            entity_probs * (entity_probs.clamp_min(1e-9)).log()
                        ).sum(-1)
                    )
                    predictions.append(entity_probs.argmax(dim=-1))
                    del entity_probs

                entropy = torch.cat(entropies, dim=1)
                max_indices = torch.cat(predictions, dim=1)
                del entropies, predictions

                hard_entity_mask: Bool[Tensor, "document token"]
                hard_entity_mask = (max_indices != self.unk_index) & (
                    entropy <= self.entity_threshold
                )
                del entropy

            rel_meta_logits = None
            if hard_entity_mask.any():
                # Select the predicted entity representations
                entity_positions: Int64[Tensor, "doc token"] = (
                    hard_entity_mask.nonzero(as_tuple=False)
                )
                if entity_positions.numel() >= 2:
                    entity_reprs = hidden_output[
                        entity_positions[:, 0],  # batch
                        entity_positions[:, 1],  # token
                    ]
                    rel_meta_logits = self._compute_relations_vectorized(
                        entity_positions, entity_reprs, max_indices
                    )

            gold_meta_logits = None
            if gold_relations is not None:
                batch, tokens, hidden_size = hidden_output.shape
                needed_by_doc: dict[int, set[int]] = {}
                for tr in gold_relations:
                    key = self._gold_relation_key(tr)
                    if key is None:
                        continue
                    docix, subj, obj = key
                    needed_by_doc.setdefault(docix, set()).update((subj, obj))

                soft_repr_by_doc = {}
                for docix, ent_ids in needed_by_doc.items():
                    doc_hidden = hidden_output[docix]
                    doc_logits = unmasked_entity_logits[docix]
                    doc_mask = attention_mask[docix].to(torch.bool)
                    reps = {
                        eid: _soft_entity_repr(
                            doc_hidden=doc_hidden,
                            doc_ent_logits=doc_logits,
                            doc_mask=doc_mask,
                            ent_id=eid,
                        )
                        for eid in ent_ids
                    }
                    soft_repr_by_doc[docix] = reps

                rows_doc, rows_i, rows_j, rep_i, rep_j = [], [], [], [], []
                seen_gold_keys: set[tuple[int, int, int]] = set()
                for tr in gold_relations:
                    key = self._gold_relation_key(tr)
                    if key is None:
                        continue
                    doc_ix, subj, obj = key
                    doc_reps = soft_repr_by_doc.get(doc_ix)
                    if not doc_reps:
                        continue
                    if subj in doc_reps and obj in doc_reps:
                        if key in seen_gold_keys:
                            continue
                        seen_gold_keys.add(key)
                        rows_doc.append(doc_ix)
                        rows_i.append(subj)
                        rows_j.append(obj)
                        rep_i.append(doc_reps[subj])
                        rep_j.append(doc_reps[obj])

                if rep_i:
                    rep_i_t = torch.stack(rep_i, dim=0)
                    rep_j_t = torch.stack(rep_j, dim=0)
                    logits = self.relation_classifier(rep_i_t, rep_j_t)
                    gold_meta_logits = (
                        {
                            "sequence": torch.tensor(
                                rows_doc, device=device, dtype=torch.long
                            ),
                            "arg_pred_i": torch.tensor(
                                rows_i, device=device, dtype=torch.long
                            ),
                            "arg_pred_j": torch.tensor(
                                rows_j, device=device, dtype=torch.long
                            ),
                        },
                        logits,
                    )

            # ---- Merge hard-pair logits (if any) with gold-pair logits (if any)
            #
            # A (doc, subj, obj) triple can be produced by both the hard-entity
            # mask and the gold path. Keep at most one row per triple: prefer
            # the gold soft representation (richer signal) and drop the
            # overlapping hard-mask row. This stops the downstream aligner from
            # logsumexp-pooling two rows for the same triple, which would bias
            # its logits upward.
            merged = None
            if rel_meta_logits and gold_meta_logits:
                (m1, l1), (m2, l2) = rel_meta_logits, gold_meta_logits
                gold_keys = set(
                    zip(
                        m2["sequence"].tolist(),
                        m2["arg_pred_i"].tolist(),
                        m2["arg_pred_j"].tolist(),
                    )
                )
                hard_keep = [
                    r
                    for r, k in enumerate(
                        zip(
                            m1["sequence"].tolist(),
                            m1["arg_pred_i"].tolist(),
                            m1["arg_pred_j"].tolist(),
                        )
                    )
                    if k not in gold_keys
                ]
                keep_idx = torch.tensor(
                    hard_keep, device=device, dtype=torch.long
                )
                merged_meta = {
                    "sequence": torch.cat(
                        [m1["sequence"][keep_idx], m2["sequence"]]
                    ),
                    "arg_pred_i": torch.cat(
                        [m1["arg_pred_i"][keep_idx], m2["arg_pred_i"]]
                    ),
                    "arg_pred_j": torch.cat(
                        [m1["arg_pred_j"][keep_idx], m2["arg_pred_j"]]
                    ),
                }
                merged_logits = torch.cat([l1[keep_idx], l2], dim=0)
                merged = (merged_meta, merged_logits)
            else:
                merged = rel_meta_logits or gold_meta_logits

            return BatchLogits(
                self._pool_logits(entity_logits, mask=attention_mask),
                self._pool_logits(class_logits, mask=attention_mask),
                merged,
            )

    def evaluate_model(
        self,
        test_data: DataLoader,
        tau_ids: float = 0.5,
        tau_cls: float = 0.5,
        topk_ids: int | None = None,
    ) -> dict[str, float]:
        """Evaluate the end-to-end model from document-level pooled logits.

        Returns what it prints and logs the same dict to the active tracking
        run.

        :param test_data: the split to score.
        :param tau_ids: threshold binarizing the entity logits.
        :param tau_cls: threshold binarizing the class logits.
        :param topk_ids: also keep this many top entity IDs per document.
        :return: the scores; a dict carrying nothing but the coverage counts
            means the split produced no samples at all.
        """
        self.eval()
        metrics: dict[str, float] = {}
        all_id_logits, all_id_true = [], []
        all_cls_logits, all_cls_true = [], []
        all_rel_logits, all_rel_true = [], []  # we'll argmax rel later
        detection = self._detection_accumulator()
        gold_relations = 0
        missed_not_proposed: list[int] = []
        missed_out_of_vocabulary: list[int] = []

        with torch.no_grad():
            # do NOT autocast around metric collection; keep numerics simple
            for batch in batch_progress(
                test_data, desc="Evaluating", position=0, leave=True
            ):
                # 1) pooled doc-level logits
                # shapes: [B, num_ids], [B, num_classes],
                # (meta, [N_pairs, R]) or None
                if detection is None:
                    id_logits_doc, cls_logits_doc, rel_meta_logits = (
                        self.get_batch_logits(batch)
                    )
                else:
                    # One embedding fetch serves the pooled heads and the
                    # tagger; `get_batch_logits` would hide it.
                    embeddings, token_mask = self.get_token_embeddings(batch)
                    id_logits_doc, cls_logits_doc, rel_meta_logits = self(
                        embeddings,
                        token_mask,
                    )
                    self.score_token_detection(
                        batch, embeddings, token_mask, detection
                    )

                # 2) document-level multi-hot targets
                id_true_doc, cls_true_doc, rel_true_list_optional = (
                    self.ground_truth(batch)
                )  # id_true_doc: [B,num_ids], cls_true_doc: [B,num_classes]
                rel_true_list: list[IndexedRelation] = (
                    rel_true_list_optional or []
                )

                # logits narrowed to the columns the targets carry
                all_id_logits.append(
                    self.drop_unk(id_logits_doc).detach().float().cpu()
                )
                all_id_true.append(id_true_doc.detach().to(torch.int64).cpu())

                all_cls_logits.append(
                    self.drop_oos(cls_logits_doc).detach().float().cpu()
                )
                all_cls_true.append(cls_true_doc.detach().to(torch.int64).cpu())

                # 3) relations: reuse the training-time aligner so eval and
                #    training pool duplicates and assign targets identically
                #    (one row per (doc, subj, obj) triple).
                aligned = None
                if rel_meta_logits is not None:
                    rel_meta, rel_logits = rel_meta_logits  # [N_pairs,R]
                    aligned = self.align_relation_predictions(
                        true_relations=rel_true_list,
                        rel_meta=rel_meta,
                        rel_logits=rel_logits,
                    )

                scored_meta = None
                if aligned is not None:
                    scored_meta, rel_logits_aligned, rel_targets = aligned
                    all_rel_logits.append(rel_logits_aligned.detach().cpu())
                    all_rel_true.append(rel_targets.detach().cpu())

                # The scored rows are the pairs the entity head proposed, so
                # gold it missed leaves no row and would otherwise never be
                # counted against the model -- the metric would be conditioned
                # on the entity head having already found both arguments.
                gold_relations += len(rel_true_list)
                not_proposed, out_of_vocabulary = self.unscored_gold_relations(
                    rel_true_list, scored_meta
                )
                missed_not_proposed.extend(not_proposed)
                missed_out_of_vocabulary.extend(out_of_vocabulary)

        # ----- stack
        if not all_id_logits:
            logger.warning("No samples found.")
            metrics.update(coverage_metrics(test_data, 0))
            tracking.log_metrics(metrics)
            return metrics

        id_logits = torch.cat(all_id_logits, dim=0).numpy()
        id_true = torch.cat(all_id_true, dim=0).numpy().astype(int)
        cls_logits = torch.cat(all_cls_logits, dim=0).numpy()
        cls_true = torch.cat(all_cls_true, dim=0).numpy().astype(int)

        # ---- IDs: probs -> binarize (threshold + optional top-K)
        id_probs = 1.0 / (1.0 + np.exp(-id_logits))
        id_pred = (id_probs >= tau_ids).astype(int)
        if topk_ids is not None and topk_ids > 0:
            # ensure at least top-K positives per doc (in addition to threshold)
            topk_idx = np.argpartition(
                -id_probs, kth=min(topk_ids, id_probs.shape[1] - 1), axis=1
            )[:, :topk_ids]
            rows = np.arange(id_probs.shape[0])[:, None]
            id_pred[rows, topk_idx] = 1

        # ---- CLASSES: probs -> binarize
        cls_probs = 1.0 / (1.0 + np.exp(-cls_logits))
        cls_pred = (cls_probs >= tau_cls).astype(int)

        # ---- sanity counts
        metrics.update(coverage_metrics(test_data, id_true.shape[0]))
        metrics.update(
            support_metrics(
                {"entity": (id_true, id_pred), "class": (cls_true, cls_pred)}
            )
        )
        logger.info(
            "\n[Entities] gold positives: %d | predicted positives: %d"
            " | classes with any preds: %d",
            int(id_true.sum()),
            int(id_pred.sum()),
            int((id_pred.sum(axis=0) > 0).sum()),
        )
        logger.info(
            "[Classes ] gold positives: %d | predicted positives: %d",
            int(cls_true.sum()),
            int(cls_pred.sum()),
        )
        scored_pairs = sum(int(true.numel()) for true in all_rel_true)
        logger.info(
            "[Relations] gold: %d | candidate pairs scored: %d "
            "| missed, never proposed: %d "
            "| missed, entity out of vocabulary: %d",
            gold_relations,
            scored_pairs,
            len(missed_not_proposed),
            len(missed_out_of_vocabulary),
        )

        # ======= METRICS =======

        # Entities (6k+ labels): prefer micro-F1 + LRAP; macro over frequent labels only
        logger.info("\n=== Entity ID metrics (multilabel, document-level) ===")
        try:
            metrics["test/entity_micro_f1"] = f1_score(
                id_true, id_pred, average="micro", zero_division=0
            )
            logger.info("micro-F1: %s", metrics["test/entity_micro_f1"])
        except ValueError:
            logger.info("micro-F1: (no positive labels or predictions) 0.0")

        try:
            metrics["test/entity_lrap"] = label_ranking_average_precision_score(
                id_true, id_probs
            )
            logger.info("LRAP: %s", metrics["test/entity_lrap"])
        except ValueError:
            logger.info("LRAP: undefined (no positives)")

        # macro-F1 over frequent labels
        support = id_true.sum(axis=0)
        keep = np.where(support >= 10)[0]  # tweak threshold as you like
        if keep.size > 0:
            metrics["test/entity_macro_f1_support10"] = f1_score(
                id_true[:, keep],
                id_pred[:, keep],
                average="macro",
                zero_division=0,
            )
            logger.info(
                "macro-F1 (support>=10): %s",
                metrics["test/entity_macro_f1_support10"],
            )
        else:
            logger.info(
                "macro-F1 (support>=10): n/a (no labels meet support threshold)"
            )

        logger.info(
            "\n=== Entity CLASS metrics (multilabel, document-level) ==="
        )
        metrics["test/class_micro_f1"] = f1_score(
            cls_true, cls_pred, average="micro", zero_division=0
        )
        logger.info("micro-F1: %s", metrics["test/class_micro_f1"])
        class_report = classification_report(
            y_true=cls_true,
            y_pred=cls_pred,
            target_names=self.known_classes,
            zero_division=0,
        )
        logger.info(class_report)
        tracking.log_text(str(class_report), "test/class_report.txt")

        # Relations: the candidate pairs, plus every gold relation that never
        # became one, scored as the `none` prediction the model effectively made
        # by not proposing it.
        missed_true, missed_pred = self._missed_gold_predictions(
            missed_not_proposed + missed_out_of_vocabulary
        )
        if all_rel_logits:
            rel_logits_np = torch.cat(all_rel_logits, dim=0).numpy()
            rel_true = torch.cat(all_rel_true, dim=0).numpy().astype(int)
            rel_pred = rel_logits_np.argmax(axis=1)
        else:
            rel_true = np.array([], dtype=int)
            rel_pred = np.array([], dtype=int)

        rel_true = np.concatenate([rel_true, missed_true])
        rel_pred = np.concatenate([rel_pred, missed_pred])

        if rel_true.size:
            logger.info(
                "\n=== Relation metrics (multiclass over candidate pairs "
                "and missed gold) ==="
            )
            labels = np.arange(len(self.relations))
            metrics.update(
                relation_metrics(
                    true=rel_true,
                    pred=rel_pred,
                    labels=labels,
                    none_index=int(self.relations_none_index),
                )
            )
            relation_report = classification_report(
                y_true=rel_true,
                y_pred=rel_pred,
                labels=labels,
                target_names=list(self.relations),
                zero_division=0,
            )
            logger.info(relation_report)
            tracking.log_text(str(relation_report), "test/relation_report.txt")
        else:
            logger.info("\n(No relation pairs produced on this split.)")

        if detection is not None:
            detection_metrics = detection.metrics()
            metrics.update(detection_metrics)
            logger.info("\n=== Detection metrics (span-level) ===")
            logger.info(
                "precision: %s recall: %s f1: %s",
                detection_metrics["test/detection_precision"],
                detection_metrics["test/detection_recall"],
                detection_metrics["test/detection_f1"],
            )

        tracking.log_metrics(metrics)

        return metrics
