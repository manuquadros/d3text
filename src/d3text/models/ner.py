"""Entity-class detection without entity linking.

The same document-level multi-label setup as `entity_linking`, minus the entity
head: this model says which of the schema's classes a document mentions, not
which database entities. It therefore needs no `entity_index` and no class
matrix, and accepts both only so that every model in the registry can be built
from one config.
"""

from collections.abc import Sequence
from typing import cast

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Bool, Float
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
)
from torch import Tensor
from torch.autograd.profiler import record_function
from torch.utils.data import DataLoader
from tqdm import tqdm

from d3text.schema import Schema

from .base import Model, load_base_model
from .config import ModelConfig, embedding_dims
from .heads import initialize_classifier_bias
from .model_types import BatchedLogits, BatchItem

__all__ = ["NERClassificationModel"]


class NERClassificationModel(Model):
    """Simplified model for Named Entity Recognition (NER) without entity linking.

    This model predicts entity classes/types for each token in a document,
    aggregating predictions at the document level. Unlike BrendaClassificationModel,
    it does not perform entity linking (mapping to specific entity IDs).
    """

    # Registered buffer; annotated so access resolves to Tensor, not Module.
    class_pos_weight: Tensor

    def __init__(
        self,
        schema: Schema,
        config: None | ModelConfig = None,
        class_freqs: Float[Tensor, " classes"] | None = None,
        # Accept but ignore entity-linking arguments for compatibility
        class_matrix: Float[Tensor, "entity class"] | None = None,
        entity_index: dict[str, int] | None = None,
        entity_freqs: Float[Tensor, " entities"] | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__(config, device=device)
        self.schema = schema

        self.classes = list(schema.class_names) + ["OOS"]
        self.num_of_classes = len(self.classes)

        self.register_class_columns()

        self.build_layers(embedding_size=embedding_dims[self.config.base_model])

        self.base_model = load_base_model(self.config.base_model)

        for param in self.base_model.parameters():
            param.requires_grad = False

        self.enable_gradient_checkpointing()

        if class_freqs is not None:
            class_pos_w = (
                (1 - class_freqs).clamp(1e-5, 1 - 1e-5)
                / class_freqs.clamp(1e-5, 1 - 1e-5)
            ).clamp(max=20.0)
        else:
            class_pos_w = torch.ones(len(schema.class_names))

        self.register_buffer("class_pos_weight", class_pos_w)

        self.classifier = nn.Sequential(
            nn.Linear(
                in_features=self.hidden_block_output_size,
                out_features=self.hidden_block_output_size,
                bias=True,
            ),
            nn.GELU(),
            nn.Dropout(self.config.dropout)
            if self.config.dropout
            else nn.Identity(),
            nn.Linear(self.hidden_block_output_size, self.num_of_classes),
        )

        if class_freqs is not None:
            initialize_classifier_bias(
                linear=cast(nn.Linear, self.classifier[-1]),
                freqs=class_freqs,
                sentinel_index=self.oos_index,
                sentinel_prior=0.9,
            )

    @property
    def class_loss_fn(self) -> nn.Module:
        """Binary cross-entropy loss for multilabel classification."""
        return nn.BCEWithLogitsLoss(
            reduction="mean", pos_weight=self.class_pos_weight
        )

    def compute_losses(
        self, batch: Sequence[BatchItem], epoch: int
    ) -> dict[str, Float[Tensor, ""]]:
        """The class loss is the whole objective here: with no entity head
        there is nothing for the ramp schedule to trade it off against."""
        return {"class": self.compute_batch_losses(batch)}

    def compute_batch_losses(
        self, batch: Sequence[BatchItem]
    ) -> Float[Tensor, ""]:
        """Compute loss for a batch."""
        class_true = self.ground_truth(batch)
        class_logits = self.get_batch_logits(batch)

        class_loss = self.class_loss_fn(
            self.drop_oos(class_logits).float(),
            class_true.float(),
        )

        return class_loss

    def get_batch_logits(
        self,
        batch: Sequence[BatchItem],
    ) -> Float[Tensor, "sequence classes"]:
        """Get class logits for a batch."""
        token_embeddings, token_att_mask = self.get_token_embeddings(batch)
        token_embeddings = token_embeddings.to(self.device, non_blocking=True)
        token_att_mask = token_att_mask.to(self.device, non_blocking=True)

        class_logits = self(token_embeddings, token_att_mask)

        return class_logits

    def ground_truth(
        self,
        batch: Sequence[BatchItem],
    ) -> Float[Tensor, "batch classes"]:
        """Get ground truth class labels for each document in the batch.

        :param batch: Batch of documents.
        :return: Multi-hot encoded tensor, where each position specifies
                 whether the class corresponding to that index occurs in
                 the particular document.
        """
        class_targets = torch.concat(tuple(doc["classes"] for doc in batch)).to(
            self.device
        )

        return class_targets.float()

    @record_function("forward")
    def forward(
        self,
        embeddings: Float[Tensor, "document token embedding"],
        attention_mask: Bool[Tensor, "document token"],
    ) -> BatchedLogits:
        """Forward pass for NER classification.

        :param embeddings: Token embeddings from base model
        :param attention_mask: Attention mask for valid tokens
        :return: Class logits pooled by document
        """
        with self.autocast_context():
            hidden_output: Float[Tensor, "document token features"] = (
                self.hidden(embeddings)
            )

            unmasked_class_logits = self.classifier(hidden_output)

            token_mask = attention_mask.unsqueeze(-1)
            class_logits = torch.where(
                token_mask, unmasked_class_logits, self._neg_inf
            )

            return self._pool_logits(class_logits)

    def evaluate_model(
        self, test_data: DataLoader, tau_cls: float = 0.5
    ) -> None:
        """Document-level multilabel evaluation for entity classes."""
        self.eval()
        all_cls_logits, all_cls_true = [], []

        with torch.no_grad():
            for batch in tqdm(test_data, desc="Evaluating"):
                cls_logits_doc = self.get_batch_logits(batch)
                cls_true_doc = self.ground_truth(batch)

                all_cls_logits.append(cls_logits_doc.detach().float().cpu())
                all_cls_true.append(cls_true_doc.detach().to(torch.int64).cpu())

        if not all_cls_logits:
            print("No samples found.")
            return

        cls_logits = torch.cat(all_cls_logits, dim=0).numpy()
        cls_true = torch.cat(all_cls_true, dim=0).numpy().astype(int)

        if cls_logits.shape[1] != cls_true.shape[1]:
            cls_logits = cls_logits[:, : cls_true.shape[1]]

        cls_probs = 1.0 / (1.0 + np.exp(-cls_logits))

        cls_pred = (cls_probs >= tau_cls).astype(int)

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
