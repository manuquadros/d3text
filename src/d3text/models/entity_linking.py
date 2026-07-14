"""Entity-linking model: entity IDs and entity classes, both at document level.

Adds the two-headed classifier on top of `base.Model` — one head over the
corpus' entity IDs, one over the schema's classes — plus the consistency term
that ties them together. `ete.ETEBrendaModel` extends this with a relation head.
"""

import operator
from collections.abc import Mapping, Sequence

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

__all__ = ["BrendaClassificationModel", "ordered_entities"]


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


class BrendaClassificationModel(Model):
    # Registered buffers; annotated so access resolves to Tensor, not Module.
    class_matrix: Tensor
    entity_pos_weight: Tensor
    class_pos_weight: Tensor
    entity_columns: Tensor

    def __init__(
        self,
        schema: Schema,
        class_matrix: Float[Tensor, "entity class"],
        entity_index: dict[str, int],
        config: None | ModelConfig = None,
        entity_freqs: Float[Tensor, " entities"] | None = None,
        class_freqs: Float[Tensor, " classes"] | None = None,
        device: str | None = None,
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

        # Initialize class matrix mapping each entity index to its entity
        # class index.
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

        self.entity_entropy_threshold = self.config.entity_entropy_threshold
        self.consistency_weight = getattr(
            self.config, "consistency_weight", 0.1
        )
        self.evaluation = False

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

    def run_epoch(
        self, data: DataLoader, step: Step, epoch: int
    ) -> tuple[dict[str, float], int]:
        """Process all batches, computing loss and printing diagnostics.

        :param epoch: epoch number
        :param train_data: DataLoader for the training data
        :returns: combined losses for epoch and the denominator for loss
            averaging.
        """
        epoch_ent_loss = 0.0
        epoch_class_loss = 0.0
        n_batches = 0

        w_ent, w_class = self.get_loss_weights(epoch)

        for batch in tqdm(
            data,
            dynamic_ncols=True,
            position=1,
            desc="Batches",
            leave=False,
        ):
            if step == Step.TRAINING:
                self.optimizer.zero_grad(set_to_none=True)

            ent_loss, class_loss = self.compute_batch_losses(batch)
            n_batches += 1

            ent_loss_scaled = ent_loss * w_ent
            class_loss_scaled = class_loss * w_class

            if step == Step.TRAINING:
                self._update(ent_loss_scaled, class_loss_scaled)

            epoch_ent_loss += ent_loss_scaled.detach().cpu().item()
            epoch_class_loss += class_loss_scaled.detach().cpu().item()
            del ent_loss, class_loss, ent_loss_scaled, class_loss_scaled

        losses = {
            "entity": epoch_ent_loss,
            "class": epoch_class_loss,
        }

        return losses, n_batches

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
        class_scale: float = 1,
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
    ) -> tuple[Float[Tensor, ""], Float[Tensor, ""]]:
        ent_true, class_true = self.ground_truth(batch)
        entity_logits, class_logits = self.get_batch_logits(batch)

        return self.compute_entity_loss(
            predictions=(entity_logits, class_logits),
            targets=(ent_true, class_true),
        )

    def get_batch_logits(
        self,
        batch: Sequence[BatchItem],
        gold_relations: list[IndexedRelation] | None = None,
    ) -> tuple[
        Float[Tensor, "sequence entities"],
        Float[Tensor, "sequence classes"],
    ]:
        token_embeddings, token_att_mask = self.get_token_embeddings(batch)
        token_embeddings = token_embeddings.to(self.device, non_blocking=True)
        token_att_mask = token_att_mask.to(self.device, non_blocking=True)

        entity_logits, class_logits = self(
            token_embeddings,
            token_att_mask,
        )

        return (
            entity_logits,
            class_logits,
        )

    def ground_truth(
        self,
        batch: Sequence[BatchItem],
    ) -> tuple[
        Float[Tensor, "batch entities"],
        Float[Tensor, "batch classes"],
    ]:
        """Get ground truth for each document in the batch

        :param: Batch of documents.
        :return: Tuple containing:
            - Multi-hot encoded tensor, where each position of dim 2
              specifies whether the entity corresponding to that index occurs in
              the particular document along dim 1.
            - Idem for class labels
        """
        entity_targets = torch.concat(
            tuple(doc["entities"] for doc in batch)
        ).to(self.device)

        class_targets = torch.concat(tuple(doc["classes"] for doc in batch)).to(
            self.device
        )

        return entity_targets.float(), class_targets.float()

    def evaluate_model(
        self, test_data: DataLoader, tau_ids: float = 0.5, tau_cls: float = 0.5
    ) -> None:
        """Document-level multilabel evaluation for entity IDs and classes."""
        self.eval()
        all_id_logits, all_id_true = [], []
        all_cls_logits, all_cls_true = [], []

        with torch.no_grad():
            for batch in tqdm(test_data, desc="Evaluating"):
                id_logits_doc, cls_logits_doc = self.get_batch_logits(batch)
                id_true_doc, cls_true_doc = self.ground_truth(batch)

                # logits, narrowed to the columns the targets carry
                all_id_logits.append(
                    self.drop_unk(id_logits_doc).detach().float().cpu()
                )
                all_cls_logits.append(
                    self.drop_oos(cls_logits_doc).detach().float().cpu()
                )

                all_id_true.append(id_true_doc.detach().to(torch.int64).cpu())
                all_cls_true.append(cls_true_doc.detach().to(torch.int64).cpu())

        if not all_id_logits:
            print("No samples found.")
            return

        id_logits = torch.cat(all_id_logits, dim=0).numpy()
        id_true = torch.cat(all_id_true, dim=0).numpy().astype(int)

        cls_logits = torch.cat(all_cls_logits, dim=0).numpy()
        cls_true = torch.cat(all_cls_true, dim=0).numpy().astype(int)

        id_probs = 1.0 / (1.0 + np.exp(-id_logits))
        cls_probs = 1.0 / (1.0 + np.exp(-cls_logits))

        id_pred = (id_probs >= tau_ids).astype(int)
        cls_pred = (cls_probs >= tau_cls).astype(int)

        print("\n=== Entity ID metrics (multilabel, document-level) ===")
        try:
            print(
                "micro-F1:",
                f1_score(id_true, id_pred, average="micro", zero_division=0),
            )
        except ValueError:
            print("micro-F1: (no positives or predictions) 0.0")

        # Probability-aware multilabel metrics (no threshold)
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

        # macro-F1 over frequent IDs only
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
            "micro-AP:",
            average_precision_score(cls_true, cls_probs, average="micro"),
        )
        print(
            classification_report(
                y_true=cls_true,
                y_pred=cls_pred,
                target_names=self.known_classes,
                zero_division=0,
            )
        )

    @record_function("forward")
    def forward(
        self,
        embeddings: Float[Tensor, "document token embedding"],
        attention_mask: Bool[Tensor, "document token"],
    ) -> tuple[
        BatchedLogits,
        BatchedLogits,
    ]:
        """Forward pass

        :return: tuple containing:
            - Entity logits pooled by document.
            - Class logits pooled by document.
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

            return (
                self._pool_logits(entity_logits),
                self._pool_logits(class_logits),
            )
