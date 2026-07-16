"""The BRENDA-family model: entity classes, entity IDs, and — optionally — the
relations between them.

One class, not two. It carries an entity head and a class head always, and a
`RelationExtractor` only when its config asks for one; every method keeps a
single signature either way, with the relation part absent from the *value*
(`Targets.relations` empty, `Logits.relations` `None`) rather than from the
arity. The end-to-end model used to be a subclass that widened `ground_truth`,
`get_batch_logits`, `compute_batch_losses` and `forward` — so a caller holding
the declared type could not know whether it was about to be handed two things or
three, and the type checker said so four times over.

`d3text.factory` is what turns the two names a config may carry
(`BrendaClassificationModel`, `ETEBrendaModel`) into this one class with and
without a relation extractor.
"""

import operator
from collections.abc import Mapping, Sequence
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Bool, Float
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    label_ranking_average_precision_score,
)
from torch import Tensor
from torch.autograd.profiler import record_function
from torch.utils.data import DataLoader
from tqdm import tqdm

from d3text.schema import Schema

from .base import Model, Step, label_columns, load_base_model
from .config import ModelConfig, embedding_dims
from .heads import ClassificationHead
from .model_types import BatchedLogits, BatchItem, IndexedRelation
from .relations import RelationExtractor, RelationPairs

__all__ = ["BrendaModel", "Logits", "Targets", "ordered_entities"]


class Targets(NamedTuple):
    """One batch's ground truth, for every head this model might carry."""

    entities: Float[Tensor, "batch entities"]
    classes: Float[Tensor, "batch classes"]
    # Empty for a model with no relation extractor: there is nothing to supervise.
    relations: list[IndexedRelation]


class Logits(NamedTuple):
    """One batch's predictions, for every head this model might carry."""

    entities: BatchedLogits
    classes: BatchedLogits
    # `None` with no relation extractor, and also when the batch offered the
    # extractor no candidate pair at all.
    relations: RelationPairs | None


def ordered_entities(entity_index: Mapping[str, int]) -> list[str]:
    """Entity names ordered by the logit column they are scored in.

    The model treats an entity's index as a *position* in the entity logit
    vector, so the indices must be exactly ``0..N-1``; anything else would make
    ``entities[i]`` name a different entity than column ``i`` scores.
    """
    ordered = sorted(entity_index.items(), key=operator.itemgetter(1))
    if [index for _, index in ordered] != list(range(len(ordered))):
        raise ValueError(
            "entity_index must map its "
            f"{len(entity_index)} names onto contiguous indices 0..N-1, "
            f"got {sorted(entity_index.values())}"
        )
    return [name for name, _ in ordered]


class BrendaModel(Model):
    """Document-level multi-label entity linking, with optional relations.

    :param extract_relations: build a `RelationExtractor`. Without one the model
        detects and links entities and nothing else, and `relations` is `None`.
    """

    # Registered buffers; annotated so access resolves to Tensor, not Module.
    class_matrix: Tensor
    entity_pos_weight: Tensor
    class_pos_weight: Tensor
    entity_columns: Tensor

    classifier: ClassificationHead
    relations: RelationExtractor | None

    def __init__(
        self,
        schema: Schema,
        class_matrix: Float[Tensor, "entity class"],
        entity_index: dict[str, int],
        config: None | ModelConfig = None,
        entity_freqs: Float[Tensor, " entities"] | None = None,
        class_freqs: Float[Tensor, " classes"] | None = None,
        device: str | None = None,
        extract_relations: bool = False,
    ) -> None:
        super().__init__(config, device=device)
        self.schema = schema

        # The corpus has no class for a token belonging to none of the schema's
        # types, so we add one.
        self.classes = list(schema.class_names) + ["OOS"]

        # Derived from `entity_index`, not from the schema, so that
        # `entities[i]` is always the entity scored by entity logit column `i`.
        # Flattening the per-class entity sets only yields that order while
        # they stay disjoint.
        self.entities = ordered_entities(entity_index) + ["UNK"]

        self.num_of_entities = len(self.entities)
        self.num_of_classes = len(self.classes)

        self.register_entity_columns()
        self.register_class_columns()

        self.build_layers(embedding_size=embedding_dims[self.config.base_model])

        self.base_model = load_base_model(self.config.base_model)

        for param in self.base_model.parameters():
            param.requires_grad = False

        self.enable_gradient_checkpointing()

        self.entity_to_index = entity_index
        self.register_buffer("class_matrix", class_matrix)

        if entity_freqs is not None:
            entity_pos_w = (
                (1 - entity_freqs).clamp(1e-5, 1 - 1e-5)
                / entity_freqs.clamp(1e-5, 1 - 1e-5)
            ).clamp(max=50.0)
        else:
            entity_pos_w = torch.ones(len(entity_index))
        if class_freqs is not None:
            class_pos_w = (
                (1 - class_freqs).clamp(1e-5, 1 - 1e-5)
                / class_freqs.clamp(1e-5, 1 - 1e-5)
            ).clamp(max=20.0)
        else:
            class_pos_w = torch.ones(len(schema.class_names))

        self.register_buffer("entity_pos_weight", entity_pos_w)
        self.register_buffer("class_pos_weight", class_pos_w)

        self.classifier = ClassificationHead(
            input_size=self.hidden_block_output_size,
            n_entities=self.num_of_entities,
            n_classes=self.num_of_classes,
            entity_freqs=entity_freqs,
            class_freqs=class_freqs,
            unk_index=self.unk_index,
            oos_index=self.oos_index,
        )

        self.relations = (
            RelationExtractor(
                hidden_size=self.hidden_block_output_size,
                schema=schema,
                entity_to_index=entity_index,
                unk_index=self.unk_index,
                config=self.config,
            )
            if extract_relations
            else None
        )

        self.consistency_weight = getattr(
            self.config, "consistency_weight", 0.1
        )

    def register_entity_columns(self) -> None:
        """Find the UNK column and remember the others. Call once
        `self.entities` is set.

        Non-persistent: derived from `self.entities`, so it must not enter the
        checkpoint (an older checkpoint would then be missing the key).
        """
        self.unk_index, entity_columns = label_columns(self.entities, "UNK")
        self.register_buffer("entity_columns", entity_columns, persistent=False)

    def drop_unk(
        self, entity_logits: Float[Tensor, "... entity"]
    ) -> Float[Tensor, "... entity"]:
        """Entity logits without the UNK column, to the width of the targets."""
        return entity_logits.index_select(-1, self.entity_columns)

    @property
    def known_entities(self) -> list[str]:
        """Entity names in column order, minus UNK: the columns `drop_unk`
        keeps, aligned with `entity_index` and with `class_matrix`'s rows."""
        return [
            self.entities[column] for column in self.entity_columns.tolist()
        ]

    def _consistency_loss(
        self, entity_logits: torch.Tensor, class_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Penalize cases where an entity is predicted but the class head
        does not agree with that entity's class.

        Uses only the 'proper' columns: drops UNK (entity) and OOS (class),
        leveraging self.class_matrix [E-1, C-1].
        """
        if self.consistency_weight <= 0:
            return torch.tensor(
                0.0, device=entity_logits.device, dtype=entity_logits.dtype
            )

        with torch.autocast(device_type=self.device, enabled=False):
            # probabilities in fp32 for stable reductions
            pe = torch.sigmoid(self.drop_unk(entity_logits)).float()
            pc = torch.sigmoid(self.drop_oos(class_logits)).float()

            # Each row of class_matrix is one-hot, so the matmul gathers, for
            # every entity column, the class head's probability for that
            # entity's class.
            pc_for_entity = pc @ self.class_matrix.T  # [B, E-1]

            penalty = pe * (1.0 - pc_for_entity)  # [B, E-1]

            cons = penalty.mean()

        return cons.to(entity_logits.dtype)

    @property
    def entity_loss_fn(self) -> nn.Module:
        return nn.BCEWithLogitsLoss(
            reduction="mean", pos_weight=self.entity_pos_weight
        )

    @property
    def class_loss_fn(self) -> nn.Module:
        return nn.BCEWithLogitsLoss(
            reduction="mean", pos_weight=self.class_pos_weight
        )

    def compute_entity_loss(
        self,
        predictions: tuple[Tensor, Tensor],
        targets: tuple[Tensor, Tensor],
    ) -> tuple[Float[Tensor, ""], Float[Tensor, ""]]:
        entity_loss = self.entity_loss_fn(
            self.drop_unk(predictions[0]).float(),
            targets[0].float(),
        )
        class_loss = self.class_loss_fn(
            self.drop_oos(predictions[1]).float(),
            targets[1].float(),
        )

        cons = self._consistency_loss(predictions[0], predictions[1])
        class_loss = class_loss + self.consistency_weight * cons

        return entity_loss, class_loss

    def compute_batch_losses(
        self, batch: Sequence[BatchItem]
    ) -> dict[str, Float[Tensor, ""]]:
        """The entity and class losses, plus the relation loss if this model
        extracts relations. Unweighted — `compute_losses` applies the ramp."""
        truth = self.ground_truth(batch)
        predicted = self.get_batch_logits(batch, gold_relations=truth.relations)

        entity_loss, class_loss = self.compute_entity_loss(
            predictions=(predicted.entities, predicted.classes),
            targets=(truth.entities, truth.classes),
        )
        losses = {"entity": entity_loss, "class": class_loss}

        if self.relations is not None:
            losses["relation"] = self.relations.loss(
                truth.relations, predicted.relations
            )

        return losses

    def compute_losses(
        self, batch: Sequence[BatchItem], epoch: int
    ) -> dict[str, Float[Tensor, ""]]:
        """Both entity-head losses train at full weight from the first epoch;
        only the relation loss ramps, and only when there is one."""
        losses = self.compute_batch_losses(batch)

        if self.relations is not None:
            losses["relation"] = losses[
                "relation"
            ] * self.relations.loss_weight(epoch)

        return losses

    def on_epoch_start(self, step: Step, epoch: int) -> None:
        if self.relations is not None:
            weight = self.relations.loss_weight(epoch)
            tqdm.write(f"Epoch {epoch}: w_rel={weight:.3f}")

    def ground_truth(self, batch: Sequence[BatchItem]) -> Targets:
        """Each document's gold entities, classes and relations.

        The relations are read only when there is a relation extractor to
        supervise; without one they would be built every batch and dropped.
        """
        entity_targets = torch.stack(
            tuple(doc["entities"] for doc in batch),
        ).to(self.device)

        class_targets = torch.stack(
            tuple(doc["classes"] for doc in batch),
        ).to(self.device)

        relations: list[IndexedRelation] = []
        if self.relations is not None:
            for docix, doc in enumerate(batch):
                for args, label in doc.get("relations", [{}])[0].items():
                    relations.append(
                        IndexedRelation(
                            docix=docix,
                            subject=args[0],
                            object=args[1],
                            label=label.argmax(),
                        )
                    )

        return Targets(
            entities=entity_targets.float(),
            classes=class_targets.float(),
            relations=relations,
        )

    def get_batch_logits(
        self,
        batch: Sequence[BatchItem],
        gold_relations: Sequence[IndexedRelation] | None = None,
    ) -> Logits:
        token_embeddings, token_att_mask = self.get_token_embeddings(batch)

        return self(
            token_embeddings.to(self.device, non_blocking=True),
            token_att_mask.to(self.device, non_blocking=True),
            gold_relations=gold_relations,
        )

    @record_function("forward")
    def forward(
        self,
        embeddings: Float[Tensor, "document token embedding"],
        attention_mask: Bool[Tensor, "document token"],
        gold_relations: Sequence[IndexedRelation] | None = None,
    ) -> Logits:
        """Entity and class logits pooled by document, and the relation head's
        logits for every candidate pair the batch offers.

        :param gold_relations: passed straight to the relation extractor, which
            scores the gold pairs alongside the ones the entity head proposed.
            Training supplies them; evaluation does not.
        """
        with self.autocast_context():
            hidden_output: Float[Tensor, "document token features"] = (
                self.hidden(embeddings)
            )
            unmasked_entity_logits, unmasked_class_logits = self.classifier(
                hidden_output
            )

            token_mask = attention_mask.unsqueeze(-1)
            entity_logits = torch.where(
                token_mask, unmasked_entity_logits, self._neg_inf
            )
            class_logits = torch.where(
                token_mask, unmasked_class_logits, self._neg_inf
            )

            pairs = (
                None
                if self.relations is None
                else self.relations(
                    hidden=hidden_output,
                    entity_logits=entity_logits,
                    unmasked_entity_logits=unmasked_entity_logits,
                    attention_mask=attention_mask,
                    gold_relations=gold_relations,
                )
            )

            return Logits(
                entities=self._pool_logits(entity_logits),
                classes=self._pool_logits(class_logits),
                relations=pairs,
            )

    def compute_batch_true_x_pred(
        self, batch: Sequence[BatchItem]
    ) -> dict[str, dict[str, np.ndarray]]:
        """Returns y_true, y_pred arrays for each task tackled by the model."""
        predicted = self.get_batch_logits(batch)
        truth = self.ground_truth(batch)

        relations_true = np.array([], dtype=int)
        relations_pred = np.array([], dtype=int)

        if self.relations is not None and truth.relations:
            aligned = self.relations.align(truth.relations, predicted.relations)

            scored_meta = None
            if aligned is not None:
                scored_meta = aligned.meta
                relations_true = (
                    aligned.targets.numpy(force=True).reshape(-1).astype(int)
                )
                relations_pred = (
                    aligned.logits.numpy(force=True)
                    .argmax(axis=-1)
                    .reshape(-1)
                    .astype(int)
                )

            # Gold the entity head never proposed has no row to be scored on,
            # so without this it would vanish from the metrics rather than
            # count against them.
            not_proposed, out_of_vocabulary = self.relations.unscored_gold(
                truth.relations, scored_meta
            )
            missed_true, missed_pred = self.relations.none_predictions(
                not_proposed + out_of_vocabulary
            )
            relations_true = np.concatenate([relations_true, missed_true])
            relations_pred = np.concatenate([relations_pred, missed_pred])

        return {
            "entities": {
                "true": truth.entities.numpy(force=True),
                "pred": torch.sigmoid(predicted.entities.float())
                .round()
                .numpy(force=True),
            },
            "classes": {
                "true": truth.classes.numpy(force=True),
                "pred": torch.sigmoid(predicted.classes.float())
                .round()
                .numpy(force=True),
            },
            "relations": {
                "true": relations_true.reshape(-1),
                "pred": relations_pred.reshape(-1),
            },
        }

    def evaluate_model(
        self,
        test_data: DataLoader,
        tau_ids: float = 0.5,
        tau_cls: float = 0.5,
        topk_ids: int | None = None,
    ) -> None:
        """Document-level multilabel evaluation of the entity IDs and classes,
        plus the relations when this model extracts them.

        :param tau_ids, tau_cls: thresholds for binarizing the pooled logits.
        :param topk_ids: also keep the top-K entity IDs per document, in addition
            to whatever clears `tau_ids`.
        """
        self.eval()
        all_id_logits, all_id_true = [], []
        all_cls_logits, all_cls_true = [], []
        all_rel_logits, all_rel_true = [], []
        gold_relations = 0
        missed_not_proposed: list[int] = []
        missed_out_of_vocabulary: list[int] = []

        with torch.no_grad():
            # Do NOT autocast around metric collection; keep the numerics simple.
            for batch in tqdm(test_data, desc="Evaluating"):
                predicted = self.get_batch_logits(batch)
                truth = self.ground_truth(batch)

                all_id_logits.append(
                    self.drop_unk(predicted.entities).detach().float().cpu()
                )
                all_id_true.append(
                    truth.entities.detach().to(torch.int64).cpu()
                )

                all_cls_logits.append(
                    self.drop_oos(predicted.classes).detach().float().cpu()
                )
                all_cls_true.append(
                    truth.classes.detach().to(torch.int64).cpu()
                )

                if self.relations is None:
                    continue

                # The training-time aligner, so that eval and training pool
                # duplicates and assign targets identically (one row per
                # (doc, subj, obj) triple).
                aligned = self.relations.align(
                    truth.relations, predicted.relations
                )

                scored_meta = None
                if aligned is not None:
                    scored_meta = aligned.meta
                    all_rel_logits.append(aligned.logits.detach().cpu())
                    all_rel_true.append(aligned.targets.detach().cpu())

                # The scored rows are the pairs the entity head proposed, so gold
                # it missed leaves no row and would otherwise never be counted
                # against the model -- the metric would be conditioned on the
                # entity head having already found both arguments.
                gold_relations += len(truth.relations)
                not_proposed, out_of_vocabulary = self.relations.unscored_gold(
                    truth.relations, scored_meta
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
            topk_idx = np.argpartition(
                -id_probs, kth=min(topk_ids, id_probs.shape[1] - 1), axis=1
            )[:, :topk_ids]
            rows = np.arange(id_probs.shape[0])[:, None]
            id_pred[rows, topk_idx] = 1

        cls_probs = 1.0 / (1.0 + np.exp(-cls_logits))
        cls_pred = (cls_probs >= tau_cls).astype(int)

        print(
            f"\n[Entities] gold positives: {int(id_true.sum())} "
            f"| predicted positives: {int(id_pred.sum())} "
            f"| classes with any preds: {int((id_pred.sum(axis=0) > 0).sum())}"
        )
        print(
            f"[Classes ] gold positives: {int(cls_true.sum())} "
            f"| predicted positives: {int(cls_pred.sum())}"
        )
        if self.relations is not None:
            scored_pairs = sum(int(true.numel()) for true in all_rel_true)
            print(
                f"[Relations] gold: {gold_relations} "
                f"| candidate pairs scored: {scored_pairs} "
                f"| missed, never proposed: {len(missed_not_proposed)} "
                f"| missed, entity out of vocabulary: "
                f"{len(missed_out_of_vocabulary)}"
            )

        self._report_entities(id_true, id_pred, id_probs)
        self._report_classes(cls_true, cls_pred, cls_probs)

        if self.relations is not None:
            self._report_relations(
                all_rel_logits,
                all_rel_true,
                missed_not_proposed + missed_out_of_vocabulary,
            )

    def _report_entities(
        self, id_true: np.ndarray, id_pred: np.ndarray, id_probs: np.ndarray
    ) -> None:
        """Thousands of labels, so: micro-F1 and LRAP over all of them, macro-F1
        over the ones with enough support to mean anything."""
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
            print(
                "micro-AP:",
                average_precision_score(id_true, id_probs, average="micro"),
            )
        except ValueError:
            print("LRAP / micro-AP: undefined (no positives)")

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

    def _report_classes(
        self, cls_true: np.ndarray, cls_pred: np.ndarray, cls_probs: np.ndarray
    ) -> None:
        print("\n=== Entity CLASS metrics (multilabel, document-level) ===")
        print(
            "micro-F1:",
            f1_score(cls_true, cls_pred, average="micro", zero_division=0),
        )
        try:
            print(
                "micro-AP:",
                average_precision_score(cls_true, cls_probs, average="micro"),
            )
        except ValueError:
            print("micro-AP: undefined (no positives)")
        print(
            classification_report(
                y_true=cls_true,
                y_pred=cls_pred,
                target_names=self.known_classes,
                zero_division=0,
            )
        )

    def _report_relations(
        self,
        all_rel_logits: list[Tensor],
        all_rel_true: list[Tensor],
        missed: list[int],
    ) -> None:
        """The candidate pairs, plus every gold relation that never became one,
        scored as the `none` prediction the model effectively made by not
        proposing it."""
        assert self.relations is not None

        missed_true, missed_pred = self.relations.none_predictions(missed)
        if all_rel_logits:
            rel_true = torch.cat(all_rel_true, dim=0).numpy().astype(int)
            rel_pred = torch.cat(all_rel_logits, dim=0).numpy().argmax(axis=1)
        else:
            rel_true = np.array([], dtype=int)
            rel_pred = np.array([], dtype=int)

        rel_true = np.concatenate([rel_true, missed_true])
        rel_pred = np.concatenate([rel_pred, missed_pred])

        if not rel_true.size:
            print("\n(No relation pairs produced on this split.)")
            return

        print(
            "\n=== Relation metrics (multiclass over candidate pairs "
            "and missed gold) ==="
        )
        print(
            classification_report(
                y_true=rel_true,
                y_pred=rel_pred,
                labels=np.arange(len(self.relations.labels)),
                target_names=list(self.relations.labels),
                zero_division=0,
            )
        )
