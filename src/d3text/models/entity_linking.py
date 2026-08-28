"""`BrendaClassificationModel` — entity ID and class detection.

Split out of what used to be `models.py`.
"""

import logging
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
from d3text import tracking
from d3text.mention_metrics import DetectionAccumulator
from d3text.progress import batch_progress
from d3text.schema import Schema
from d3text.token_labels import IGNORE_INDEX
from jaxtyping import Bool, Float, Int64
from sklearn.metrics import (
    average_precision_score,
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
    coverage_metrics,
    masked_bce_with_logits,
    masked_token_cross_entropy,
    ordered_entities,
    support_metrics,
)
from .config import ModelConfig, embedding_dims
from .heads import ClassificationHead
from .model_types import (
    BatchItem,
    BatchLogits,
    BatchLosses,
    GroundTruth,
    IndexedRelation,
)
from .token_supervision import (
    TokenLabelReader,
    document_lengths,
    padded_targets,
)

logger = logging.getLogger(__name__)


class BrendaClassificationModel(Model):
    # Registered buffers; annotated so access resolves to Tensor, not Module.
    class_matrix: Tensor
    entity_pos_weight: Tensor
    class_pos_weight: Tensor
    # Submodule (or its absence); annotated so access resolves past
    # nn.Module.__getattr__.
    token_tagger: nn.Linear | None

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
        self.classes = list(schema.class_names) + ["OOS"]

        # Derived from `entity_index`, not from `classes`, so that
        # `entities[i]` is always the entity scored by entity logit column `i`.
        # Flattening `classes.values()` only yields that order while the
        # per-class entity sets stay disjoint.
        self.entities = ordered_entities(entity_index) + ["UNK"]

        # The dataset does not include a `none` class, so we add one.
        self.num_of_entities = len(self.entities)
        self.num_of_classes = len(self.classes)

        self.register_entity_columns()
        self.register_class_columns()

        self.build_layers(embedding_size=embedding_dims[self.config.base_model])

        self.base_model = base.load_base_model(self.config.base_model)

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

        self.entity_threshold = self.config.entity_entropy_threshold
        self.consistency_weight = getattr(
            self.config, "consistency_weight", 0.1
        )
        self.evaluation = False

        # The token-level span tagger, present only when a label store is
        # configured — so a config without one builds (and checkpoints)
        # exactly the model it always did. One column per entity type plus
        # OUTSIDE, in the store's own code order: column c scores code c, so
        # the targets need no translation and the store's recorded space
        # (verified by the reader at open) is the head's geometry.
        self.token_tagger = None
        self._token_labels: TokenLabelReader | None = None
        self._unlabelled_documents: set[int] = set()
        if self.config.token_labels_store:
            self._token_labels = TokenLabelReader(
                self.config.token_labels_store
            )
            self.token_tagger = nn.Linear(
                self.hidden_block_output_size,
                1 + len(self._token_labels.space.types),
            )

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

            # pick, for each entity row, its class probability from class head:
            # pc_for_entity: [B, E-1] where each column i = pc[:, class_of_entity_i]
            # class_matrix: [E-1, C-1]; do a gather via matmul because rows are one-hot
            pc_for_entity = pc @ self.class_matrix.T  # [B, E-1]

            penalty = pe * (1.0 - pc_for_entity)  # [B, E-1]

            # average over batch and entities (avoid NaNs)
            cons = penalty.mean()

        return cons.to(entity_logits.dtype)

    def compute_losses(
        self,
        batch: Sequence[BatchItem],
        step: Step,
        epoch: int,
    ) -> dict[str, Tensor]:
        """Neither loss is ramped, so `step` and `epoch` are unused here —
        taken only to match the shared signature. `token` is absent from the
        returned dict without a configured label store; without one the
        returned keys — and every number derived from them — are exactly
        what they always were.
        """
        batch_losses = self.compute_batch_losses(batch)

        losses = {"entity": batch_losses.entity, "class": batch_losses.class_}
        if batch_losses.token is not None:
            # Unramped: the token targets are supervision available from
            # epoch 0, like the entity BCE, not a late-phase objective.
            losses["token"] = batch_losses.token

        return losses

    def epoch_loss_weights(self, epoch: int) -> dict[str, float]:
        """Every objective at full weight: this model has no relation head
        whose ramp either of its losses could ride."""
        weights = {"entity": 1.0, "class": 1.0}
        if getattr(self, "token_tagger", None) is not None:
            weights["token"] = 1.0
        return weights

    @property
    def entity_loss_fn(self) -> nn.Module:
        # weights = torch.ones(self.num_of_entities - 1, device=self.device)
        return nn.BCEWithLogitsLoss(
            reduction="mean", pos_weight=self.entity_pos_weight
        )

    def compute_entity_loss(
        self,
        predictions: tuple[Tensor, Tensor],
        targets: tuple[Tensor, Tensor],
        class_scale: float = 1,
        class_abstain: Bool[Tensor, "document class"] | None = None,
    ) -> tuple[Float[Tensor, ""], Float[Tensor, ""]]:
        entity_loss = self.entity_loss_fn(
            self.drop_unk(predictions[0]).float(),
            targets[0].float(),
        )
        class_loss = masked_bce_with_logits(
            self.drop_oos(predictions[1]).float(),
            targets[1].float(),
            abstain=class_abstain,
            pos_weight=self.class_pos_weight,
        )

        cons = self._consistency_loss(predictions[0], predictions[1])
        class_loss = class_loss + self.consistency_weight * cons

        return entity_loss, class_loss

    def class_negative_abstain_mask(
        self,
        batch: Sequence[BatchItem],
        class_true: Float[Tensor, "document class"],
    ) -> Bool[Tensor, "document class"] | None:
        """Which document-level class negatives to stop asserting (DEC-04).

        `None` when `class_negative_abstention` is off — the ordinary case,
        and `compute_entity_loss` reduces to a plain masked-nowhere BCE.
        Otherwise, `True` at `(document, class)` where the document is a
        gold negative for that class (`class_true == 0`) yet
        `token_labels_store`'s dictionary matched a surface form of that
        class's type somewhere in the document — at least
        `class_negative_abstention_min_chars` characters long — gold-linked
        or not. The length gate is what keeps this from abstaining on a
        one- or two-character incidental match; DEC-04 measured the
        ungated version as collapsing strains and bacteria toward
        predicting positive on nearly every document. Reuses the tagger's
        own matches rather than a second dictionary pass, so it is exactly
        the mask `token_targets` already abstains at the token level, one
        level up.

        The class-head column order is `schema.class_names`, the same
        declaration order `token_labels.LabelSpace` assigns its codes 1..n
        from, so column `j` is type code `j + 1` with no lookup needed.
        """
        if not self.config.class_negative_abstention:
            return None
        reader = self._token_labels
        assert reader is not None  # config validation requires the store

        mask = torch.zeros_like(class_true, dtype=torch.bool)
        for row, item in enumerate(batch):
            mentioned = reader.mentioned_types(
                int(item["id"].item()),
                min_chars=self.config.class_negative_abstention_min_chars,
            )
            if not mentioned:
                continue
            for code in mentioned:
                column = code - 1
                if 0 <= column < mask.shape[1]:
                    mask[row, column] = True
        return mask & (class_true == 0)

    def compute_batch_losses(self, batch: Sequence[BatchItem]) -> BatchLosses:
        ground_truth = self.ground_truth(batch)
        token_embeddings, token_att_mask = self.get_token_embeddings(batch)
        logits = self(token_embeddings, token_att_mask)

        ent_loss, class_loss = self.compute_entity_loss(
            predictions=(logits.entities, logits.classes),
            targets=(ground_truth.entities, ground_truth.classes),
            class_abstain=self.class_negative_abstain_mask(
                batch, ground_truth.classes
            ),
        )
        return BatchLosses(
            entity=ent_loss,
            class_=class_loss,
            token=self.compute_token_loss(
                batch, token_embeddings, token_att_mask
            ),
        )

    def compute_token_loss(
        self,
        batch: Sequence[BatchItem],
        embeddings: Float[Tensor, "document token embedding"],
        attention_mask: Bool[Tensor, "document token"],
    ) -> Float[Tensor, ""] | None:
        """The span tagger's masked cross-entropy, or None without a tagger.

        Additive to the document-level losses, never a replacement: the
        pooled terms carry the gold links that are never named in the text,
        which no distant supervision reaches, and this term supplies the
        localization the pooled loss cannot. The mask (`IGNORE_INDEX`) covers
        the tokens matching entities BRENDA did not link to the document, the
        padding, and any document the store has no targets for — all skipped
        by `masked_token_cross_entropy`, whose divisor is the unmasked count.
        """
        if self.token_tagger is None:
            return None

        targets = self.token_targets(batch, attention_mask)
        with self.autocast_context():
            token_logits = self.token_tagger(self.hidden(embeddings))
        return masked_token_cross_entropy(
            token_logits.reshape(-1, token_logits.shape[-1]).float(),
            targets.reshape(-1),
            weighting=self.config.token_loss_weighting,
            focal_gamma=self.config.token_focal_gamma,
        )

    def token_targets(
        self,
        batch: Sequence[BatchItem],
        attention_mask: Bool[Tensor, "document token"],
    ) -> Int64[Tensor, "document token"]:
        """The batch's token targets, padded to the embeddings' geometry.

        A document the store does not hold gets an all-`IGNORE_INDEX` row —
        skipped by the loss, warned about once per document — because a split
        wider than the labelling run is a data gap, not a modelling error. A
        document whose stored row *disagrees in length* with its embeddings
        raises instead: that store was built against other encodings, and
        every one of its codes would land on the wrong token.
        """
        reader = self._token_labels
        assert reader is not None

        rows: list[Int64[Tensor, " token"]] = []
        for item, length in zip(batch, document_lengths(attention_mask)):
            pubmed_id = int(item["id"].item())
            codes = reader.document_codes(
                pubmed_id, item["sequence"]["attention_mask"]
            )
            if codes is None:
                if pubmed_id not in self._unlabelled_documents:
                    self._unlabelled_documents.add(pubmed_id)
                    logger.warning(
                        "%s has no token labels in %s; its tokens are "
                        "masked out of the tagger loss.",
                        pubmed_id,
                        self.config.token_labels_store,
                    )
                codes = torch.full((length,), IGNORE_INDEX, dtype=torch.int64)
            elif codes.shape[0] != length:
                msg = (
                    f"document {pubmed_id} aggregates to {length} tokens but "
                    f"its stored labels aggregate to {codes.shape[0]}; the "
                    "label store and the encodings disagree — regenerate the "
                    "store"
                )
                raise ValueError(msg)
            rows.append(codes)

        return padded_targets(rows, attention_mask.shape[1]).to(self.device)

    def score_token_detection(
        self,
        batch: Sequence[BatchItem],
        embeddings: Float[Tensor, "document token embedding"],
        attention_mask: Bool[Tensor, "document token"],
        accumulator: DetectionAccumulator,
    ) -> None:
        """Add one batch's span detections to `accumulator`.

        Token-axis spans: the tagger's argmax runs against the stored codes'
        runs, with the `ignore` set masked and counted — the numbers
        `DetectionAccumulator.metrics` reports as what they are.
        """
        reader = self._token_labels
        assert reader is not None and self.token_tagger is not None

        with self.autocast_context():
            token_logits = self.token_tagger(self.hidden(embeddings))
        predictions = token_logits.float().argmax(dim=-1).cpu()

        for item, predicted, length in zip(
            batch, predictions, document_lengths(attention_mask)
        ):
            pubmed_id = int(item["id"].item())
            gold = reader.document_codes(
                pubmed_id, item["sequence"]["attention_mask"]
            )
            if gold is None:
                accumulator.missing_documents += 1
                continue
            if gold.shape[0] != length:
                msg = (
                    f"document {pubmed_id} aggregates to {length} tokens but "
                    f"its stored labels aggregate to {gold.shape[0]}; the "
                    "label store and the encodings disagree — regenerate the "
                    "store"
                )
                raise ValueError(msg)
            accumulator.add_document(predicted[:length].numpy(), gold.numpy())

    def get_batch_logits(
        self,
        batch: Sequence[BatchItem],
        gold_relations: list[IndexedRelation] | None = None,
    ) -> BatchLogits:
        token_embeddings, token_att_mask = self.get_token_embeddings(batch)
        token_embeddings = token_embeddings.to(self.device, non_blocking=True)
        token_att_mask = token_att_mask.to(self.device, non_blocking=True)

        return self(token_embeddings, token_att_mask)

    def ground_truth(
        self,
        batch: Sequence[BatchItem],
    ) -> GroundTruth:
        """Get ground truth for each document in the batch

        :param: Batch of documents.
        :return: `GroundTruth` with `relations=None` — this model has no
            relation head to supervise.
        """
        entity_targets = torch.stack(
            tuple(doc["entities"] for doc in batch)
        ).to(self.device)

        class_targets = torch.stack(tuple(doc["classes"] for doc in batch)).to(
            self.device
        )

        return GroundTruth(entity_targets.float(), class_targets.float())

    def evaluate_model(
        self, test_data: DataLoader, tau_ids: float = 0.5, tau_cls: float = 0.5
    ) -> dict[str, float]:
        """Document-level multilabel evaluation for entity IDs and classes.

        Returns what it prints, and logs the same dict to the active tracking
        run — the `print_epoch_stats` contract, for the same reason: a number
        computed twice is a number that can disagree with itself. A dict
        carrying nothing but the coverage counts means the split produced no
        samples at all.
        """
        self.eval()
        metrics: dict[str, float] = {}
        all_id_logits, all_id_true = [], []
        all_cls_logits, all_cls_true = [], []
        detection = self._detection_accumulator()

        with torch.no_grad():
            for batch in batch_progress(
                test_data, desc="Evaluating", position=0, leave=True
            ):
                if detection is None:
                    doc_logits = self.get_batch_logits(batch)
                else:
                    embeddings, token_mask = self.get_token_embeddings(batch)
                    doc_logits = self(embeddings, token_mask)
                    self.score_token_detection(
                        batch, embeddings, token_mask, detection
                    )
                ground_truth = self.ground_truth(batch)

                # logits, narrowed to the columns the targets carry
                all_id_logits.append(
                    self.drop_unk(doc_logits.entities).detach().float().cpu()
                )
                all_cls_logits.append(
                    self.drop_oos(doc_logits.classes).detach().float().cpu()
                )

                # TRUE LABELS (fix the bug: append *_true, not logits)
                all_id_true.append(
                    ground_truth.entities.detach().to(torch.int64).cpu()
                )
                all_cls_true.append(
                    ground_truth.classes.detach().to(torch.int64).cpu()
                )

        if not all_id_logits:
            logger.warning("No samples found.")
            metrics.update(coverage_metrics(test_data, 0))
            tracking.log_metrics(metrics)
            return metrics

        # concat
        id_logits = torch.cat(all_id_logits, dim=0).numpy()
        id_true = torch.cat(all_id_true, dim=0).numpy().astype(int)

        cls_logits = torch.cat(all_cls_logits, dim=0).numpy()
        cls_true = torch.cat(all_cls_true, dim=0).numpy().astype(int)

        # probabilities
        id_probs = 1.0 / (1.0 + np.exp(-id_logits))
        cls_probs = 1.0 / (1.0 + np.exp(-cls_logits))

        # binarize for F1 / report
        id_pred = (id_probs >= tau_ids).astype(int)
        cls_pred = (cls_probs >= tau_cls).astype(int)

        # ======= METRICS =======

        metrics.update(coverage_metrics(test_data, id_true.shape[0]))
        metrics.update(
            support_metrics(
                {"entity": (id_true, id_pred), "class": (cls_true, cls_pred)}
            )
        )

        logger.info("\n=== Entity ID metrics (multilabel, document-level) ===")
        try:
            metrics["test/entity_micro_f1"] = f1_score(
                id_true, id_pred, average="micro", zero_division=0
            )
            logger.info("micro-F1: %s", metrics["test/entity_micro_f1"])
        except ValueError:
            logger.info("micro-F1: (no positives or predictions) 0.0")

        # Probability-aware multilabel metrics (no threshold)
        try:
            metrics["test/entity_lrap"] = label_ranking_average_precision_score(
                id_true, id_probs
            )
            logger.info("LRAP: %s", metrics["test/entity_lrap"])
            metrics["test/entity_micro_ap"] = average_precision_score(
                id_true, id_probs, average="micro"
            )
            logger.info("micro-AP: %s", metrics["test/entity_micro_ap"])
        except ValueError:
            logger.info("LRAP / micro-AP: undefined (no positives)")

        # macro-F1 over frequent IDs only
        support = id_true.sum(axis=0)
        keep = np.where(support >= 10)[0]
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
        metrics["test/class_micro_ap"] = average_precision_score(
            cls_true, cls_probs, average="micro"
        )
        logger.info("micro-AP: %s", metrics["test/class_micro_ap"])
        report = classification_report(
            y_true=cls_true,
            y_pred=cls_pred,  # <- must be binary indicators
            target_names=self.known_classes,
            zero_division=0,
        )
        logger.info(report)
        tracking.log_text(str(report), "test/class_report.txt")

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

    def _detection_accumulator(self) -> DetectionAccumulator | None:
        """A fresh accumulator when this model has a span tagger to score."""
        if getattr(self, "token_tagger", None) is None:
            return None
        reader = self._token_labels
        assert reader is not None
        return DetectionAccumulator(reader.space)

    @record_function("forward")
    def forward(
        self,
        embeddings: Float[Tensor, "document token embedding"],
        attention_mask: Bool[Tensor, "document token"],
    ) -> BatchLogits:
        """Forward pass. `relations` is always `None`: this model has no
        relation head."""
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

            return BatchLogits(
                self._pool_logits(entity_logits),
                self._pool_logits(class_logits),
            )
