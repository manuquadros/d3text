"""End-to-end model: entity IDs, entity classes and the relations between them.

Extends `entity_linking.BrendaClassificationModel` with a biaffine relation
head. The candidate pairs it scores are the ones the *entity* head proposed —
token positions whose entity prediction is confident and not UNK — merged, in
training, with the gold pairs. That is what makes the relation metrics delicate:
gold the entity head never proposed leaves no row to be scored on, so
`unscored_gold_relations` exists to add it back, and every relation metric here
goes through it.
"""

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import torch
from jaxtyping import Bool, Float, Int16, Int64
from sklearn.metrics import (
    classification_report,
    f1_score,
    label_ranking_average_precision_score,
)
from torch import Tensor
from torch.autograd.profiler import record_function
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base import Step
from .entity_linking import BrendaClassificationModel
from .heads import BiaffineRelationClassifier
from .model_types import BatchedLogits, BatchItem, IndexedRelation

__all__ = [
    "ETEBrendaModel",
    "balanced_class_weights",
    "focal_cross_entropy",
    "get_batch_entities",
]


def get_batch_entities(
    batch: Sequence[BatchItem], device: str = "cuda"
) -> tuple[Int16[Tensor, " entities"], ...]:
    """Get tuple indicating the entities tagged for each document.

    :return: Tuple whose positions correspond to sequences found in
        the batch. Each sequence mapped to the entities found in its
        respective document.
    """
    seqs = []
    for doc in batch:
        entities = (
            doc["entities"].nonzero()[:, 1].to(device=device, dtype=torch.int16)
        )
        seqs.append(entities)

    return tuple(seqs)


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


class ETEBrendaModel(BrendaClassificationModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # The relation head's columns are the schema's relation types, in the
        # order it declares them — the order the corpus's one-hot label vectors
        # are in, whose argmax the losses and the metrics take as the target
        # index. Naming them here as well would be a second source of truth for
        # one ordering, with nothing to check the two agree.
        self.relations = self.schema.relation_names
        self.relations_none_index = self.schema.none_relation_index
        self.num_relations = len(self.relations)
        self.relation_classifier = BiaffineRelationClassifier(
            hidden_size=self.hidden_block_output_size,
            num_relations=self.num_relations,
            separate_predicate_layer=self.config.separate_predicate_layer,
            biaffine_hidden_size=self.config.biaffine_hidden_size,
        )

        self.relation_label_smoothing = self.config.relation_label_smoothing
        self.relation_loss_weighting = self.config.relation_loss_weighting
        self.relation_focal_gamma = self.config.relation_focal_gamma

    def relation_loss_weight(self, epoch: int, w0: float = 0.1) -> float:
        """The relation loss' weight at `epoch`, ramping linearly from `w0` to
        1.0 over `ramp_epochs` (which, at 0, means no ramp at all).

        The schedule holds the relation head back until the entity head proposes
        usable pairs to classify — it is this model's alone. No other objective
        rides it, here or in any other model: the entity and class losses train
        at full weight from the first epoch.
        """
        if not self.ramp_epochs:
            return 1.0
        t = min(1.0, epoch / float(self.ramp_epochs))
        return w0 + (1.0 - w0) * t

    def on_epoch_start(self, step: Step, epoch: int) -> None:
        w_rel = self.relation_loss_weight(epoch)
        tqdm.write(f"Epoch {epoch}: w_rel={w_rel:.3f}")

    def compute_losses(
        self, batch: Sequence[BatchItem], epoch: int
    ) -> dict[str, Float[Tensor, ""]]:
        """Both entity-head losses train at full weight; only the relation loss
        ramps."""
        ent_loss, class_loss, rel_loss = self.compute_batch_losses(batch)

        return {
            "entity": ent_loss,
            "class": class_loss,
            "relation": rel_loss * self.relation_loss_weight(epoch),
        }

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
            try:
                doc_relations = doc.get("relations", [{}])[0]
            except IndexError:
                continue

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

        def _as_list(x: Tensor):
            return x.detach().cpu().tolist()

        seq_list = _as_list(rel_meta["sequence"])
        subj_list = _as_list(rel_meta["arg_pred_i"])
        obj_list = _as_list(rel_meta["arg_pred_j"])

        n_rows = rel_logits.size(0)
        assert (
            len(seq_list) == n_rows
            and len(subj_list) == n_rows
            and len(obj_list) == n_rows
        ), "rel_meta fields must align with rel_logits rows"

        device = rel_logits.device

        # Build grouping of row indices per (doc, subj_ix, obj_ix)
        groups = defaultdict(list)
        for row_idx, (d, i, j) in enumerate(zip(seq_list, subj_list, obj_list)):
            groups[(int(d), int(i), int(j))].append(row_idx)

        if not groups:
            return None

        # Build a quick lookup of gold labels per triple
        gold_by_key = defaultdict(list)
        for tr in true_relations:
            try:
                subj_ix = int(self.entity_to_index[tr.subject])
                obj_ix = int(self.entity_to_index[tr.object])
            except KeyError:
                continue  # gold refers to entity not mapped in this doc/batch
            gold_by_key[(int(tr.docix), subj_ix, obj_ix)].append(int(tr.label))

        # Prepare pooled outputs
        pooled_logits = []
        pooled_targets = []
        pooled_seq = []
        pooled_subj = []
        pooled_obj = []

        none_idx = self.relations_none_index

        # Pool each group's logits and assign target
        for (d, i, j), row_idxs in groups.items():
            group_logits = rel_logits[row_idxs]  # lives on device

            pooled = self._pool_logits(group_logits, dim=0)

            # Target: default none, overwrite if gold(s) exist
            labels = gold_by_key.get((d, i, j))
            if labels:
                # If multiple labels exist, prefer any non-none; else first.
                if any(lbl != none_idx for lbl in labels):
                    target = next(lbl for lbl in labels if lbl != none_idx)
                else:
                    target = labels[0]
            else:
                target = int(none_idx)

            pooled_logits.append(pooled)
            pooled_targets.append(target)
            pooled_seq.append(d)
            pooled_subj.append(i)
            pooled_obj.append(j)

        stacked_logits = torch.stack(pooled_logits, dim=0).to(device)
        target_tensor = torch.tensor(
            pooled_targets, dtype=torch.long, device=device
        )

        pooled_meta = {
            "sequence": torch.tensor(
                pooled_seq, dtype=torch.long, device=device
            ),
            "arg_pred_i": torch.tensor(
                pooled_subj, dtype=torch.long, device=device
            ),
            "arg_pred_j": torch.tensor(
                pooled_obj, dtype=torch.long, device=device
            ),
        }

        return pooled_meta, stacked_logits, target_tensor

    def unscored_gold_relations(
        self,
        true_relations: Sequence[IndexedRelation],
        scored_meta: dict[str, Tensor] | None,
    ) -> tuple[list[int], list[int]]:
        """Gold relations that no scored row can account for.

        `align_relation_predictions` builds its rows out of the *candidate*
        pairs the entity head proposed, and gold only ever relabels a row that
        already exists. Gold whose triple was never proposed therefore leaves no
        row at all, and so cannot show up in any metric computed over those
        rows: it is not a false negative, it is absent, and the denominator
        becomes whatever the entity head chose to propose. A caller computing
        metrics must add these back as misses.

        This is deliberately not folded into `align_relation_predictions`: the
        loss path consumes that function, and these relations carry no logits to
        backpropagate.

        :param scored_meta: the meta of the rows actually scored -- the pooled
            meta the aligner returned, or None when it returned nothing.
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
            try:
                key = (
                    int(relation.docix),
                    int(self.entity_to_index[relation.subject]),
                    int(self.entity_to_index[relation.object]),
                )
            except KeyError:
                out_of_vocabulary.append(int(relation.label))
                continue

            if key not in scored:
                not_proposed.append(int(relation.label))

        return not_proposed, out_of_vocabulary

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
    ) -> tuple[
        Float[Tensor, "sequence entities"],
        Float[Tensor, "sequence classes"],
        tuple[dict[str, Tensor], Float[Tensor, "pairs relations"]] | None,
    ]:
        token_embeddings, token_att_mask = self.get_token_embeddings(batch)
        entities_in_batch = get_batch_entities(batch)

        entity_logits, class_logits, relation_index_logits = self(
            token_embeddings,
            token_att_mask,
            entities_in_batch,
            gold_relations=gold_relations,
        )

        return (
            entity_logits,
            class_logits,
            relation_index_logits,
        )

    def compute_batch_losses(
        self, batch: Sequence[BatchItem]
    ) -> tuple[Float[Tensor, ""], Float[Tensor, ""], Float[Tensor, ""]]:
        """Compute loss for a batch."""
        ent_true, class_true, rel_true = self.ground_truth(batch)
        entity_logits, class_logits, relation_index_logits = (
            self.get_batch_logits(batch, gold_relations=rel_true)
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

        return ent_loss, class_loss, relation_loss

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
            print(
                f"relations_true {relations_true.shape} "
                f"!= relations_pred {relations_pred.shape}"
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
        entities_in_batch: tuple[Int16[Tensor, " entities"], ...],
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

            # Find entity positions
            entity_probs: Float[Tensor, "document token ent_probs"] = (
                torch.softmax(entity_logits, dim=-1)
            )
            entropy = -(
                entity_probs * (entity_probs.clamp_min(1e-9)).log()
            ).sum(-1)

            max_indices = entity_probs.argmax(dim=-1)
            hard_entity_mask: Bool[Tensor, "document token"]
            hard_entity_mask = (max_indices != self.unk_index) & (
                entropy <= self.entity_entropy_threshold
            )

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
                for tr in gold_relations:
                    doc_ix = int(tr.docix)
                    doc_reps = soft_repr_by_doc.get(doc_ix)
                    if not doc_reps:
                        continue
                    subj = int(self.entity_to_index.get(tr.subject, -1))
                    obj = int(self.entity_to_index.get(tr.object, -1))
                    if subj in doc_reps and obj in doc_reps:
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
    ) -> None:
        """
        Evaluate the end-to-end model from *document-level pooled logits*.
        - tau_ids / tau_cls: global thresholds for multilabel binarization
        - topk_ids: also keep top-K entity IDs per document
        """
        self.eval()
        all_id_logits, all_id_true = [], []
        all_cls_logits, all_cls_true = [], []
        all_rel_logits, all_rel_true = [], []
        gold_relations = 0
        missed_not_proposed: list[int] = []
        missed_out_of_vocabulary: list[int] = []

        with torch.no_grad():
            # do NOT autocast around metric collection; keep numerics simple
            for batch in tqdm(test_data, desc="Evaluating"):
                id_logits_doc, cls_logits_doc, rel_meta_logits = (
                    self.get_batch_logits(batch)
                )

                id_true_doc, cls_true_doc, rel_true_list = self.ground_truth(
                    batch
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

                # Reuse the training-time aligner so eval and training pool
                # duplicates and assign targets identically (one row per
                # (doc, subj, obj) triple).
                aligned = None
                if rel_meta_logits is not None:
                    rel_meta, rel_logits = rel_meta_logits
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

        if not all_id_logits:
            print("No samples found.")
            return

        id_logits = torch.cat(all_id_logits, dim=0).numpy()
        id_true = torch.cat(all_id_true, dim=0).numpy().astype(int)
        cls_logits = torch.cat(all_cls_logits, dim=0).numpy()
        cls_true = torch.cat(all_cls_true, dim=0).numpy().astype(int)

        id_probs = 1.0 / (1.0 + np.exp(-id_logits))
        id_pred = (id_probs >= tau_ids).astype(int)
        if topk_ids is not None and topk_ids > 0:
            # ensure at least top-K positives per doc (in addition to threshold)
            topk_idx = np.argpartition(
                -id_probs, kth=min(topk_ids, id_probs.shape[1] - 1), axis=1
            )[:, :topk_ids]
            rows = np.arange(id_probs.shape[0])[:, None]
            id_pred[rows, topk_idx] = 1

        cls_probs = 1.0 / (1.0 + np.exp(-cls_logits))
        cls_pred = (cls_probs >= tau_cls).astype(int)

        print(
            f"\n[Entities] gold positives: {int(id_true.sum())} | predicted positives: {int(id_pred.sum())} | classes with any preds: {int((id_pred.sum(axis=0) > 0).sum())}"
        )
        print(
            f"[Classes ] gold positives: {int(cls_true.sum())} | predicted positives: {int(cls_pred.sum())}"
        )
        scored_pairs = sum(int(true.numel()) for true in all_rel_true)
        print(
            f"[Relations] gold: {gold_relations} | candidate pairs scored: {scored_pairs} "
            f"| missed, never proposed: {len(missed_not_proposed)} "
            f"| missed, entity out of vocabulary: {len(missed_out_of_vocabulary)}"
        )

        # Entities (6k+ labels): prefer micro-F1 + LRAP; macro over frequent
        # labels only
        print("\n=== Entity ID metrics (multilabel, document-level) ===")
        try:
            print(
                "micro-F1:",
                f1_score(id_true, id_pred, average="micro", zero_division=0),
            )
        except ValueError:
            print("micro-F1: (no positive labels or predictions) 0.0")

        try:
            print(
                "LRAP:",
                label_ranking_average_precision_score(id_true, id_probs),
            )
        except ValueError:
            print("LRAP: undefined (no positives)")

        support = id_true.sum(axis=0)
        keep = np.where(support >= 10)[0]
        if keep.size > 0:
            print(
                "macro-F1 (support>=10):",
                f1_score(
                    id_true[:, keep],
                    id_pred[:, keep],
                    average="macro",
                    zero_division=0,
                ),
            )
        else:
            print(
                "macro-F1 (support>=10): n/a (no labels meet support threshold)"
            )

        print("\n=== Entity CLASS metrics (multilabel, document-level) ===")
        print(
            "micro-F1:",
            f1_score(cls_true, cls_pred, average="micro", zero_division=0),
        )
        print(
            classification_report(
                y_true=cls_true,
                y_pred=cls_pred,
                target_names=self.known_classes,
                zero_division=0,
            )
        )

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
            print(
                "\n=== Relation metrics (multiclass over candidate pairs "
                "and missed gold) ==="
            )
            labels = np.arange(len(self.relations))
            print(
                classification_report(
                    y_true=rel_true,
                    y_pred=rel_pred,
                    labels=labels,
                    target_names=list(self.relations),
                    zero_division=0,
                )
            )
        else:
            print("\n(No relation pairs produced on this split.)")
