"""`ETEBrendaModel` — entity ID + class detection + relation extraction.

Split out of what used to be `models.py`.
"""

import logging
from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import torch
from d3text import tracking
from d3text.progress import batch_progress
from d3text.training.update import BatchUpdate
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
    Step,
    balanced_class_weights,
    coverage_metrics,
    focal_cross_entropy,
    relation_metrics,
    support_metrics,
)
from .entity_linking import BrendaClassificationModel
from .heads import BiaffineRelationClassifier
from .model_types import BatchedLogits, BatchItem, IndexedRelation

logger = logging.getLogger(__name__)


class ETEBrendaModel(
    BrendaClassificationModel,
):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.relations = ("HasEnzyme", "HasSpecies", "none")
        self.relations_none_index = self.relations.index("none")
        self.num_relations = len(self.relations)
        self.relation_classifier = BiaffineRelationClassifier(
            hidden_size=self.hidden_block_output_size,
            num_relations=len(self.relations),
            separate_predicate_layer=self.config.separate_predicate_layer,
            biaff_hidden_size=self.config.biaffine_hidden_size,
        )

        self.relation_label_smoothing = self.config.relation_label_smoothing
        self.relation_loss_weighting = self.config.relation_loss_weighting
        self.relation_focal_gamma = self.config.relation_focal_gamma

    def run_epoch(
        self,
        data: DataLoader,
        step: Step,
        epoch: int,
        update: BatchUpdate,
    ) -> tuple[dict[str, float], int]:
        """Process all batches, computing loss and printing diagnostics.

        :param epoch: epoch number
        :param train_data: DataLoader for the training data
        :returns: combined loss for epoch
        """
        epoch_ent_loss = 0.0
        epoch_class_loss = 0.0
        epoch_rel_loss = 0.0
        epoch_token_loss = 0.0
        token_batches = 0
        n_batches = 0
        # Validation totals feed the trainer's early-stopping comparison,
        # which reads them as one series across epochs — so they are scored
        # under the ramp's final (t = 1) weights, the objective the run is
        # ramping toward. Only the training gradient follows the schedule.
        if step == Step.TRAINING:
            w_ent, w_rel = self.get_loss_weights(epoch)
        else:
            w_ent, w_rel = self.get_loss_weights(self.ramp_epochs)

        for batch in batch_progress(data):
            if step == Step.TRAINING:
                update.zero_grad()

            if n_batches == 0:
                logger.info(
                    "Epoch %d: w_ent=%.3f, w_rel=%.3f", epoch, w_ent, w_rel
                )

            # `*rest` absorbs the tagger term, which only a model with a
            # configured label store emits; without one the shape — and every
            # number below — is exactly what it was.
            ent_loss, class_loss, rel_loss, *rest = self.compute_batch_losses(
                batch
            )
            token_loss = rest[0] if rest else None

            ent_loss_scaled = ent_loss * w_ent
            class_loss_scaled = class_loss * w_ent
            rel_loss_scaled = rel_loss * w_rel
            scaled = [ent_loss_scaled, class_loss_scaled, rel_loss_scaled]
            if token_loss is not None:
                # Unramped: the token targets are supervision available from
                # epoch 0, like the entity BCE, not a late-phase objective.
                scaled.append(token_loss)

            if step == Step.TRAINING:
                update(*scaled)

            epoch_ent_loss += ent_loss_scaled.detach().cpu().item()
            epoch_class_loss += class_loss_scaled.detach().cpu().item()
            epoch_rel_loss += rel_loss_scaled.detach().cpu().item()
            if token_loss is not None:
                epoch_token_loss += token_loss.detach().cpu().item()
                token_batches += 1
            n_batches += 1

            del (
                rel_loss_scaled,
                ent_loss_scaled,
                class_loss_scaled,
                rel_loss,
                ent_loss,
                class_loss,
                token_loss,
                scaled,
            )

        losses = {
            "entity": epoch_ent_loss,
            "class": epoch_class_loss,
            "relation": epoch_rel_loss,
        }
        if token_batches:
            losses["token"] = epoch_token_loss

        return losses, n_batches

    def epoch_loss_weights(self, epoch: int) -> dict[str, float]:
        # `run_epoch` scales the class loss by the *entity* weight here; the
        # pair's second element is the relation ramp, not a class weight.
        w_ent, w_rel = self.get_loss_weights(epoch)
        weights = {"entity": w_ent, "class": w_ent, "relation": w_rel}
        if getattr(self, "token_tagger", None) is not None:
            weights["token"] = 1.0
        return weights

    def ground_truth(
        self,
        batch: Sequence[BatchItem],
    ) -> tuple[
        Float[Tensor, "batch entities"],
        Float[Tensor, "batch classes"],
        list[IndexedRelation],
    ]:
        """Get ground truth for each document in the batch

        :param: Batch of documents.
        :return: Tuple containing:
            - Multi-hot encoded tensor, where each position of dim 2
              specifies whether the entity corresponding to that index occurs in
              the particular document along dim 1.
            - Idem for class labels
            - List of relations indexed to document identifiers
        """
        entity_targets, class_targets = super().ground_truth(batch)

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

        return entity_targets, class_targets, relation_targets

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
            try:
                subj_ix = int(self.entity_to_index[tr.subject])
                obj_ix = int(self.entity_to_index[tr.object])
            except KeyError:
                continue  # gold refers to entity not mapped in this doc/batch
            gold_by_key[(int(tr.docix), subj_ix, obj_ix)].append(int(tr.label))

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
    ) -> tuple[
        Float[Tensor, "sequence entities"],
        Float[Tensor, "sequence classes"],
        tuple[dict[str, Tensor], Float[Tensor, "pairs relations"]] | None,
    ]:
        token_embeddings, token_att_mask = self.get_token_embeddings(batch)

        entity_logits, class_logits, relation_index_logits = self(
            token_embeddings,
            token_att_mask,
            gold_relations=gold_relations,
        )

        return (
            entity_logits,
            class_logits,
            relation_index_logits,
        )

    def compute_batch_losses(
        self, batch: Sequence[BatchItem]
    ) -> tuple[
        Float[Tensor, ""],
        Float[Tensor, ""],
        Float[Tensor, ""],
        Float[Tensor, ""] | None,
    ]:
        """Compute loss for a batch."""
        ent_true, class_true, rel_true = self.ground_truth(batch)
        token_embeddings, token_att_mask = self.get_token_embeddings(batch)
        entity_logits, class_logits, relation_index_logits = self(
            token_embeddings,
            token_att_mask,
            gold_relations=rel_true,
        )

        ent_loss, class_loss = self.compute_entity_loss(
            predictions=(entity_logits, class_logits),
            targets=(ent_true, class_true),
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

        return (
            ent_loss,
            class_loss,
            relation_loss,
            self.compute_token_loss(batch, token_embeddings, token_att_mask),
        )

    def compute_batch_true_x_pred(
        self, batch: Sequence[BatchItem]
    ) -> dict[str, dict[str, np.ndarray]]:
        """Returns y_true, y_pred arrays for each task tackled by the model."""
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
        rel_truth: list[IndexedRelation]
        entity_truth, class_truth, rel_truth = self.ground_truth(batch)
        relations_true = np.array([], dtype=int)
        relations_pred = np.array([], dtype=int)

        def _none_predictions():
            """Return none predictions for every gold label in this batch."""
            relations_true = np.array(
                [rel.label for rel in rel_truth], dtype=int
            )
            relations_pred = np.full(
                len(rel_truth), int(self.relations_none_index), dtype=int
            )
            return relations_true, relations_pred

        if rel_truth:
            if relation_index_logits:
                rel_meta: dict[str, Tensor]
                rel_logits: Float[Tensor, "pairs relations"]
                rel_meta, rel_logits = relation_index_logits
                aligned_rel_preds = self.align_relation_predictions(
                    true_relations=rel_truth,
                    rel_meta=rel_meta,
                    rel_logits=rel_logits,
                )
                if aligned_rel_preds is not None:
                    _, preds, targets = aligned_rel_preds
                    relations_true = (
                        targets.numpy(force=True).reshape(-1).astype(int)
                    )
                    relations_pred = preds.numpy(force=True)
                    relations_pred = (
                        relations_pred.argmax(axis=-1).reshape(-1).astype(int)
                    )
                else:
                    relations_true, relations_pred = _none_predictions()
            else:
                relations_true, relations_pred = _none_predictions()

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
        """
        Compute relation logits for all valid entity pairs.
        Returns:
            - dict of raw tensors: {
                "doc": LongTensor[n_pairs],
                "arg_pred_i": LongTensor[n_pairs],
                "arg_pred_j": LongTensor[n_pairs],
            }
            - logits: FloatTensor[n_pairs, n_relations]
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
    ) -> tuple[
        BatchedLogits,
        BatchedLogits,
        tuple[
            dict[str, Tensor],
            Float[Tensor, "pairs relations"],
        ]
        | None,
    ]:
        """Forward pass

        :return: tuple containing:
            - Entity logits pooled by document.
            - Class logits pooled by document.
            - Tuple containing:
                - Index of entity A, where dim=-1 corresponds to the entity
                  selected in entity_index
                - Index of entity B
                - Relation type logits
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
                    docix = int(tr.docix)
                    subj = int(self.entity_to_index.get(tr.subject, -1))
                    obj = int(self.entity_to_index.get(tr.object, -1))
                    if subj < 0 or obj < 0:
                        continue
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
                    doc_ix = int(tr.docix)
                    doc_reps = soft_repr_by_doc.get(doc_ix)
                    if not doc_reps:
                        continue
                    subj = int(self.entity_to_index.get(tr.subject, -1))
                    obj = int(self.entity_to_index.get(tr.object, -1))
                    if subj in doc_reps and obj in doc_reps:
                        key = (doc_ix, subj, obj)
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

            return (
                self._pool_logits(entity_logits),
                self._pool_logits(class_logits),
                merged,
            )

    def evaluate_model(
        self,
        test_data: DataLoader,
        tau_ids: float = 0.5,
        tau_cls: float = 0.5,
        topk_ids: int | None = None,
    ) -> dict[str, float]:
        """
        Evaluate the end-to-end model from *document-level pooled logits*.
        - tau_ids / tau_cls: global thresholds for multilabel binarization
        - topk_ids: also keep top-K entity IDs per document

        Returns what it prints and logs the same dict to the active tracking
        run; a dict carrying nothing but the coverage counts means the split
        produced no samples at all.
        """
        self.eval()
        metrics: dict[str, float] = {}
        all_id_logits, all_id_true = [], []
        all_cls_logits, all_cls_true = [], []
        all_rel_logits, all_rel_true = [], []  # we'll argmax rel later
        detection = self._detection_accumulator()

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
                id_true_doc, cls_true_doc, rel_true_list = self.ground_truth(
                    batch
                )  # id_true_doc: [B,num_ids], cls_true_doc: [B,num_classes], rel_true_list: list[...]

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
                if rel_meta_logits is not None:
                    rel_meta, rel_logits = rel_meta_logits  # [N_pairs,R]
                    aligned = self.align_relation_predictions(
                        true_relations=rel_true_list,
                        rel_meta=rel_meta,
                        rel_logits=rel_logits,
                    )
                    if aligned is not None:
                        _, rel_logits_aligned, rel_targets = aligned
                        all_rel_logits.append(rel_logits_aligned.detach().cpu())
                        all_rel_true.append(rel_targets.detach().cpu())

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

        # Relations (multiclass over candidate pairs)
        if all_rel_logits:
            rel_logits_np = torch.cat(all_rel_logits, dim=0).numpy()
            rel_true = torch.cat(all_rel_true, dim=0).numpy().astype(int)
            rel_pred = rel_logits_np.argmax(axis=1)

            logger.info(
                "\n=== Relation metrics (multiclass over candidate pairs) ==="
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
