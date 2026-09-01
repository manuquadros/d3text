"""`NERClassificationModel` — entity class detection without linking."""

import logging
from collections.abc import Sequence
from typing import cast

import numpy as np
import torch
import torch.nn as nn
from d3text import tracking
from d3text.progress import batch_progress
from d3text.schema import Schema
from jaxtyping import Bool, Float
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
)
from torch import Tensor
from torch.autograd.profiler import record_function
from torch.utils.data import DataLoader

from . import base
from .base import Model, Step, coverage_metrics, support_metrics
from .config import ModelConfig, embedding_dims
from .heads import initialize_classifier_bias
from .model_types import BatchedLogits, BatchItem

logger = logging.getLogger(__name__)


class NERClassificationModel(Model):
    """Entity class detection without entity linking.

    Predicts entity types per token and pools them to the document, but never
    maps a mention to a specific entity ID.
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

        # Add "OOS" (out-of-scope) class for tokens that don't belong to any entity class
        self.classes = list(schema.class_names) + ["OOS"]
        self.num_of_classes = len(self.classes)

        self.register_class_columns()

        # Build hidden layers
        self.build_layers(embedding_size=embedding_dims[self.config.base_model])

        # Initialize transformer base model
        self.base_model = base.load_base_model(self.config.base_model)

        # Freeze base model parameters initially
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.enable_gradient_checkpointing()

        # Setup class weights for handling imbalanced data
        if class_freqs is not None:
            class_pos_w = (
                (1 - class_freqs).clamp(1e-5, 1 - 1e-5)
                / class_freqs.clamp(1e-5, 1 - 1e-5)
            ).clamp(max=20.0)
        else:
            class_pos_w = torch.ones(len(schema.class_names))

        self.register_buffer("class_pos_weight", class_pos_w)

        # Simple classification head
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

        # Initialize classifier bias if frequencies provided
        if class_freqs is not None:
            initialize_classifier_bias(
                linear=cast(nn.Linear, self.classifier[-1]),
                freqs=class_freqs,
                sentinel_index=self.oos_index,
                sentinel_prior=0.9,
            )

    @property
    def class_loss_fn(self) -> nn.Module:
        """Binary cross-entropy loss for multilabel classification.

        :return: the loss module.
        """
        return nn.BCEWithLogitsLoss(
            reduction="mean", pos_weight=self.class_pos_weight
        )

    def compute_losses(
        self,
        batch: Sequence[BatchItem],
        step: Step,
        epoch: int,
    ) -> dict[str, Tensor]:
        """This batch's class loss.

        One objective, and no schedule rides it: `step` and `epoch` are taken
        only to match the shared signature.

        :param batch: the batch to run.
        :param step: whether this is a training or a validation pass.
        :param epoch: the epoch number, unused here.
        :return: the loss, under the key `class`.
        """
        return {"class": self.compute_batch_losses(batch)}

    def compute_batch_losses(
        self, batch: Sequence[BatchItem]
    ) -> Float[Tensor, ""]:
        """This batch's class loss.

        :param batch: the batch to run.
        :return: the scalar loss.
        """
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
        """Class logits for a batch.

        :param batch: the batch to run.
        :return: the pooled class logits.
        """
        token_embeddings, token_att_mask = self.get_token_embeddings(batch)
        token_embeddings = token_embeddings.to(self.device, non_blocking=True)
        token_att_mask = token_att_mask.to(self.device, non_blocking=True)

        class_logits = self(token_embeddings, token_att_mask)

        return class_logits

    def ground_truth(
        self,
        batch: Sequence[BatchItem],
    ) -> Float[Tensor, "batch classes"]:
        """The gold classes of each document in the batch.

        :param batch: the batch to read.
        :return: a multi-hot tensor, one position per class.
        """
        class_targets = torch.stack(tuple(doc["classes"] for doc in batch)).to(
            self.device
        )

        return class_targets.float()

    @record_function("forward")
    def forward(
        self,
        embeddings: Float[Tensor, "document token embedding"],
        attention_mask: Bool[Tensor, "document token"],
    ) -> BatchedLogits:
        """Class logits for one batch, pooled by document.

        :param embeddings: the batch's token embeddings.
        :param attention_mask: which positions carry a real token.
        :return: the pooled class logits.
        """
        with self.autocast_context():
            # Pass through hidden layers
            hidden_output: Float[Tensor, "document token features"] = (
                self.hidden(embeddings)
            )

            # Get class logits
            unmasked_class_logits = self.classifier(hidden_output)

            # Mask invalid positions
            token_mask = attention_mask.unsqueeze(-1)
            class_logits = torch.where(
                token_mask, unmasked_class_logits, self._neg_inf
            )

            return self._pool_logits(class_logits, mask=attention_mask)

    def evaluate_model(
        self, test_data: DataLoader, tau_cls: float = 0.5
    ) -> dict[str, float]:
        """Document-level multilabel evaluation for entity classes.

        Returns what it prints and logs the same dict to the active tracking
        run.

        :param test_data: the split to score.
        :param tau_cls: threshold binarizing the class logits.
        :return: the scores; a dict carrying nothing but the coverage counts
            means the split produced no samples at all.
        """
        self.eval()
        metrics: dict[str, float] = {}
        all_cls_logits, all_cls_true = [], []

        with torch.no_grad():
            for batch in batch_progress(
                test_data, desc="Evaluating", position=0, leave=True
            ):
                cls_logits_doc = self.get_batch_logits(batch)
                cls_true_doc = self.ground_truth(batch)

                # logits
                all_cls_logits.append(cls_logits_doc.detach().float().cpu())

                # TRUE LABELS
                all_cls_true.append(cls_true_doc.detach().to(torch.int64).cpu())

        if not all_cls_logits:
            logger.warning("No samples found.")
            metrics.update(coverage_metrics(test_data, 0))
            tracking.log_metrics(metrics)
            return metrics

        # concat
        cls_logits = torch.cat(all_cls_logits, dim=0).numpy()
        cls_true = torch.cat(all_cls_true, dim=0).numpy().astype(int)

        if cls_logits.shape[1] != cls_true.shape[1]:
            cls_logits = cls_logits[:, : cls_true.shape[1]]

        # probabilities
        cls_probs = 1.0 / (1.0 + np.exp(-cls_logits))

        # binarize for F1 / report
        cls_pred = (cls_probs >= tau_cls).astype(int)

        # ======= METRICS =======

        metrics.update(coverage_metrics(test_data, cls_true.shape[0]))
        metrics.update(support_metrics({"class": (cls_true, cls_pred)}))

        logger.info(
            "\n=== Entity CLASS metrics (multilabel, document-level) ==="
        )
        metrics["test/class_micro_f1"] = f1_score(
            cls_true, cls_pred, average="micro", zero_division=0
        )
        logger.info("micro-F1: %s", metrics["test/class_micro_f1"])
        metrics["test/class_micro_ap"] = average_precision_score(
            cls_true, cls_probs, average="micro"
        )
        logger.info("micro-AP: %s", metrics["test/class_micro_ap"])
        report = classification_report(
            y_true=cls_true,
            y_pred=cls_pred,
            target_names=self.known_classes,
            zero_division=0,
        )
        logger.info(report)
        tracking.log_text(str(report), "test/class_report.txt")

        tracking.log_metrics(metrics)

        return metrics
