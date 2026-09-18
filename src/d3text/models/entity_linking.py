"""`BrendaClassificationModel` — entity-class detection over pooled logits."""

import logging
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
from d3text import tracking
from d3text.constraints import FREQUENCY_CLAMP_EPS, UnitInterval
from d3text.mention_metrics import (
    DetectionAccumulator,
    token_gold_mentions_with_entities,
    token_predicted_mentions,
)
from d3text.progress import batch_progress
from d3text.schema import Schema
from d3text.token_labels import IGNORE_INDEX
from jaxtyping import Bool, Float, Int64
from sklearn.metrics import classification_report, f1_score
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
    micro_ap_metrics,
    support_metrics,
)
from .config import ModelConfig
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
    # Registered buffer; annotated so access resolves to Tensor, not Module.
    class_pos_weight: Tensor
    # Submodule (or its absence); annotated so access resolves past
    # nn.Module.__getattr__.
    token_tagger: nn.Linear | None

    def __init__(
        self,
        schema: Schema,
        config: None | ModelConfig = None,
        class_freqs: Float[Tensor, " classes"] | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__(config, device=device)
        self.schema = schema
        self.classes = list(schema.class_names) + ["OOS"]

        # The dataset does not include a `none` class, so we add one.
        self.num_of_classes = len(self.classes)

        self.register_class_columns()

        self.base_model = base.load_base_model(self.config.base_model)
        self.build_layers(embedding_size=self.base_model.config.hidden_size)
        self.freeze_base_model()

        if self.config.gradient_checkpointing:
            self.enable_gradient_checkpointing()

        if class_freqs is not None:
            class_pos_w = (
                (1 - class_freqs).clamp(
                    FREQUENCY_CLAMP_EPS, 1 - FREQUENCY_CLAMP_EPS
                )
                / class_freqs.clamp(
                    FREQUENCY_CLAMP_EPS, 1 - FREQUENCY_CLAMP_EPS
                )
            ).clamp(max=20.0)
        else:
            class_pos_w = torch.ones(len(schema.class_names))

        self.register_buffer("class_pos_weight", class_pos_w)

        self.classifier = ClassificationHead(
            input_size=self.hidden_block_output_size,
            n_classes=self.num_of_classes,
            class_freqs=class_freqs,
            oos_index=self.oos_index,
        )

        # The token-level span tagger, present only when a label store is
        # configured — so a config without one builds (and checkpoints)
        # exactly the model it always did. One column per entity type plus
        # OUTSIDE, in the store's own code order: column c scores code c, so
        # the targets need no translation and the store's recorded space
        # (verified by the reader at open) is the head's geometry.
        self.token_tagger = None
        self._token_labels: TokenLabelReader | None = None
        self._unlabelled_documents: set[int] = set()
        # The entity IDs the training split named, set from outside
        # (typically by `evaluate.py`) before scoring; None leaves
        # `_detection_accumulator` building one that reports no novelty split.
        self.training_entity_ids: frozenset[str] | None = None
        if self.config.token_labels_store:
            self._token_labels = TokenLabelReader(
                self.config.token_labels_store
            )
            self.token_tagger = nn.Linear(
                self.hidden_block_output_size,
                1 + len(self._token_labels.space.types),
            )

    def compute_losses(
        self,
        batch: Sequence[BatchItem],
        step: Step,
        epoch: int,
    ) -> dict[str, Tensor]:
        """This batch's class and (optional) token losses.

        Neither loss is ramped, so `step` and `epoch` are taken only to match
        the shared signature.

        :param batch: the batch to run.
        :param step: whether this is a training or a validation pass.
        :param epoch: the epoch number, unused here.
        :return: one loss per objective, `token` present only with a label
            store.
        """
        batch_losses = self.compute_batch_losses(batch)

        losses = {"class": batch_losses.class_}
        if batch_losses.token is not None:
            # Unramped: the token targets are supervision available from
            # epoch 0, not a late-phase objective.
            losses["token"] = batch_losses.token

        return losses

    def epoch_loss_weights(self, epoch: int) -> dict[str, float]:
        """Every objective at full weight: there is no relation head to ramp.

        :param epoch: the epoch about to run.
        :return: each objective's multiplier.
        """
        weights = {"class": 1.0}
        if getattr(self, "token_tagger", None) is not None:
            weights["token"] = 1.0
        return weights

    def compute_class_loss(
        self,
        logits: Float[Tensor, "document class"],
        targets: Float[Tensor, "document class"],
        class_abstain: Bool[Tensor, "document class"] | None = None,
    ) -> Float[Tensor, ""]:
        """The document-level class BCE, `OOS` dropped to the targets' width.

        :param logits: the class head's full-width pooled logits.
        :param targets: the gold class indicators.
        :param class_abstain: which gold negatives to stop asserting.
        :return: the scalar loss.
        """
        return masked_bce_with_logits(
            self.drop_oos(logits).float(),
            targets.float(),
            abstain=class_abstain,
            pos_weight=self.class_pos_weight,
            downweight=self.config.class_negative_downweight,
        )

    def class_negative_abstain_mask(
        self,
        batch: Sequence[BatchItem],
        class_true: Float[Tensor, "document class"],
    ) -> Bool[Tensor, "document class"] | None:
        """Which document-level class negatives to stop asserting.

        `True` where the document is a gold negative for a class yet the label
        store's dictionary matched a surface form of that class's type in it,
        at least that class's own length cutoff, gold-linked or not. The length
        gate keeps an incidental one- or two-character match from abstaining;
        it is overridable per class because a uniform cutoff collapses
        `bacteria` toward predicting positive almost everywhere while rescuing
        `strains` and `other_organisms`. Reuses the tagger's own matches, so it
        is the token-level abstention one level up.

        :param batch: the batch to read.
        :param class_true: the batch's gold class targets.
        :return: the mask, or None when class-negative abstention is off.
        """
        if not self.config.class_negative_abstention:
            return None
        reader = self._token_labels
        assert reader is not None  # config validation requires the store

        overrides = self.config.class_negative_abstention_min_chars_by_class
        default_cutoff = self.config.class_negative_abstention_min_chars
        min_chars_by_code = {
            column + 1: overrides.get(name, default_cutoff)
            for column, name in enumerate(self.classes[:-1])  # drop OOS
        }

        # Built on the CPU and transferred once: writing individual elements
        # of a device tensor launches one kernel per write.
        mask = torch.zeros_like(class_true, dtype=torch.bool, device="cpu")
        for row, item in enumerate(batch):
            mentioned = reader.mentioned_types(
                int(item["id"].item()), min_chars=min_chars_by_code
            )
            if not mentioned:
                continue
            for code in mentioned:
                column = code - 1
                if 0 <= column < mask.shape[1]:
                    mask[row, column] = True
        return mask.to(class_true.device) & (class_true == 0)

    def compute_batch_losses(self, batch: Sequence[BatchItem]) -> BatchLosses:
        ground_truth = self.ground_truth(batch)
        token_embeddings, token_att_mask = self.get_token_embeddings(batch)

        # Computed once here, up front, only when the tagger loss will need
        # it — `forward` and `compute_token_loss` both take it instead of
        # each running the projection over the same embeddings themselves.
        hidden_output = None
        if self.token_tagger is not None:
            with self.autocast_context():
                hidden_output = self.hidden(token_embeddings)

        logits = self(
            token_embeddings, token_att_mask, hidden_output=hidden_output
        )

        class_loss = self.compute_class_loss(
            logits.classes,
            ground_truth.classes,
            class_abstain=self.class_negative_abstain_mask(
                batch, ground_truth.classes
            ),
        )
        return BatchLosses(
            class_=class_loss,
            token=self.compute_token_loss(
                batch,
                token_embeddings,
                token_att_mask,
                hidden_output=hidden_output,
            ),
        )

    def compute_token_loss(
        self,
        batch: Sequence[BatchItem],
        embeddings: Float[Tensor, "document token embedding"],
        attention_mask: Bool[Tensor, "document token"],
        hidden_output: Float[Tensor, "document token features"] | None = None,
        token_logits: Float[Tensor, "document token codes"] | None = None,
        lengths: list[int] | None = None,
    ) -> Float[Tensor, ""] | None:
        """The span tagger's masked cross-entropy, or None without a tagger.

        Additive to the document-level losses, never a replacement: the pooled
        terms carry the gold links never named in the text, and this term
        supplies the localization the pooled loss cannot.

        :param batch: the batch to run.
        :param embeddings: the batch's token embeddings.
        :param attention_mask: which positions carry a real token.
        :param hidden_output: `self.hidden(embeddings)`, already computed by
            the caller; recomputed here only when not supplied.
        :param token_logits: `self.token_tagger(hidden_output)`, already
            computed by the caller; recomputed here only when not supplied.
        :param lengths: `document_lengths(attention_mask)`, already computed
            by the caller; recomputed here only when not supplied, and shared
            with `token_targets` and `token_ambiguous_mask` rather than each
            re-reading the mask off the device.
        :return: the scalar loss, or None.
        """
        if self.token_tagger is None:
            return None
        if lengths is None:
            lengths = document_lengths(attention_mask)

        targets = self.token_targets(batch, attention_mask, lengths=lengths)
        with self.autocast_context():
            if token_logits is None:
                if hidden_output is None:
                    hidden_output = self.hidden(embeddings)
                token_logits = self.token_tagger(hidden_output)
        return masked_token_cross_entropy(
            token_logits.reshape(-1, token_logits.shape[-1]).float(),
            targets.reshape(-1),
            weighting=self.config.token_loss_weighting,
            focal_gamma=self.config.token_focal_gamma,
            ambiguous=self.token_ambiguous_mask(
                batch, attention_mask, lengths=lengths
            ).reshape(-1),
            downweight=self.config.token_ambiguous_downweight,
        )

    def token_targets(
        self,
        batch: Sequence[BatchItem],
        attention_mask: Bool[Tensor, "document token"],
        lengths: list[int] | None = None,
    ) -> Int64[Tensor, "document token"]:
        """The batch's token targets, padded to the embeddings' geometry.

        A document the store does not hold gets an all-`IGNORE_INDEX` row,
        warned about once, because a split wider than the labelling run is a
        data gap rather than a modelling error.

        :param batch: the batch to read.
        :param attention_mask: which positions carry a real token.
        :param lengths: `document_lengths(attention_mask)`, already computed
            by the caller; recomputed here only when not supplied.
        :return: one target per token.
        :raises ValueError: if a stored row disagrees in length with its
            embeddings, which means the store was built against other encodings
            and every code would land on the wrong token.
        """
        reader = self._token_labels
        assert reader is not None
        if lengths is None:
            lengths = document_lengths(attention_mask)

        rows: list[Int64[Tensor, " token"]] = []
        for item, length in zip(batch, lengths):
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

    def token_ambiguous_mask(
        self,
        batch: Sequence[BatchItem],
        attention_mask: Bool[Tensor, "document token"],
        lengths: list[int] | None = None,
    ) -> Bool[Tensor, "document token"]:
        """Which tokens sit in an ambiguous, comma-joined mention.

        `False` -- never down-weighted -- is the harmless default for a
        document the store lacks, or whose stored row disagrees in length
        with the encodings: this mask only ever softens the tagger loss,
        never invents a target the way an all-`IGNORE_INDEX` row does in
        `token_targets`.

        :param batch: the batch to read.
        :param attention_mask: which positions carry a real token.
        :param lengths: `document_lengths(attention_mask)`, already computed
            by the caller; recomputed here only when not supplied.
        :return: one flag per token.
        """
        reader = self._token_labels
        assert reader is not None
        if lengths is None:
            lengths = document_lengths(attention_mask)

        mask = torch.zeros(attention_mask.shape, dtype=torch.bool)
        for row, (item, length) in enumerate(zip(batch, lengths)):
            pubmed_id = int(item["id"].item())
            ambiguous = reader.document_ambiguous(
                pubmed_id, item["sequence"]["attention_mask"]
            )
            if ambiguous is not None and ambiguous.shape[0] == length:
                mask[row, :length] = ambiguous
        return mask.to(self.device)

    def score_token_detection(
        self,
        batch: Sequence[BatchItem],
        embeddings: Float[Tensor, "document token embedding"],
        attention_mask: Bool[Tensor, "document token"],
        accumulator: DetectionAccumulator,
        hidden_output: Float[Tensor, "document token features"] | None = None,
        token_logits: Float[Tensor, "document token codes"] | None = None,
        lengths: list[int] | None = None,
    ) -> None:
        """Add one batch's span detections to `accumulator`.

        Token-axis spans: the tagger's argmax runs against the stored codes'
        runs, with the ignored set masked and counted. Each assertable gold
        span carries the entity IDs the label store anchors there, so
        `accumulator` can split detection by novelty when it was built with a
        training vocabulary; fuzzy matches carry no entity here, the same
        exclusion the store's precompute makes.

        :param batch: the batch to score.
        :param embeddings: the batch's token embeddings.
        :param attention_mask: which positions carry a real token.
        :param accumulator: collects the counts across batches.
        :param hidden_output: `self.hidden(embeddings)`, already computed by
            the caller; recomputed here only when not supplied.
        :param token_logits: `self.token_tagger(hidden_output)`, already
            computed by the caller; recomputed here only when not supplied.
        :param lengths: `document_lengths(attention_mask)`, already computed
            by the caller; recomputed here only when not supplied.
        """
        reader = self._token_labels
        assert reader is not None and self.token_tagger is not None
        if lengths is None:
            lengths = document_lengths(attention_mask)

        with self.autocast_context():
            if token_logits is None:
                if hidden_output is None:
                    hidden_output = self.hidden(embeddings)
                token_logits = self.token_tagger(hidden_output)
        predictions = token_logits.float().argmax(dim=-1).cpu()

        for item, predicted, length in zip(batch, predictions, lengths):
            pubmed_id = int(item["id"].item())
            mask = item["sequence"]["attention_mask"]
            gold = reader.document_codes(pubmed_id, mask)
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

            entity_positions = reader._gold_entity_positions(pubmed_id, mask)
            gold_mentions = token_gold_mentions_with_entities(
                gold.numpy(),
                {
                    entity_id: positions.tolist()
                    for entity_id, positions in entity_positions.items()
                },
            )
            accumulator.add_mentions(
                token_predicted_mentions(predicted[:length].numpy()),
                gold_mentions,
            )

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
        """The gold classes of each document in the batch.

        :param batch: the batch to read.
        :return: the targets, `relations=None` since this model has no relation
            head to supervise.
        """
        class_targets = torch.stack(tuple(doc["classes"] for doc in batch)).to(
            self.device
        )

        return GroundTruth(class_targets.float())

    def evaluate_model(
        self,
        test_data: DataLoader,
        tau_cls: UnitInterval = 0.5,
    ) -> dict[str, float]:
        """Document-level multilabel evaluation for entity classes.

        Returns what it prints and logs the same dict to the active tracking
        run: a number computed twice is a number that can disagree with itself.

        :param test_data: the split to score.
        :param tau_cls: threshold binarizing the class logits.
        :return: the scores; a dict carrying nothing but the coverage counts
            means the split produced no samples at all.
        """
        self.eval()
        metrics: dict[str, float] = {}
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
                    with self.autocast_context():
                        hidden_output = self.hidden(embeddings)
                    doc_logits = self(
                        embeddings, token_mask, hidden_output=hidden_output
                    )
                    self.score_token_detection(
                        batch,
                        embeddings,
                        token_mask,
                        detection,
                        hidden_output=hidden_output,
                    )
                ground_truth = self.ground_truth(batch)

                # logits, narrowed to the columns the targets carry
                all_cls_logits.append(
                    self.drop_oos(doc_logits.classes).detach().float().cpu()
                )
                all_cls_true.append(
                    ground_truth.classes.detach().to(torch.int64).cpu()
                )

        if not all_cls_logits:
            logger.warning("No samples found.")
            metrics.update(coverage_metrics(test_data, 0))
            tracking.log_metrics(metrics)
            return metrics

        cls_logits = torch.cat(all_cls_logits, dim=0).numpy()
        cls_true = torch.cat(all_cls_true, dim=0).numpy().astype(int)

        cls_probs = 1.0 / (1.0 + np.exp(-cls_logits))
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
        metrics.update(micro_ap_metrics("class", cls_true, cls_probs))
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
        """A fresh accumulator when this model has a span tagger to score.

        Carries `self.training_entity_ids` through, so a caller that set it
        gets the novelty split; one that never set it gets the accumulator's
        own default of no split, unchanged from before.
        """
        if getattr(self, "token_tagger", None) is None:
            return None
        reader = self._token_labels
        assert reader is not None
        return DetectionAccumulator(
            reader.space, training_entity_ids=self.training_entity_ids
        )

    @record_function("forward")
    def forward(
        self,
        embeddings: Float[Tensor, "document token embedding"],
        attention_mask: Bool[Tensor, "document token"],
        hidden_output: Float[Tensor, "document token features"] | None = None,
    ) -> BatchLogits:
        """Class logits for one batch.

        :param embeddings: the batch's token embeddings.
        :param attention_mask: which positions carry a real token.
        :param hidden_output: `self.hidden(embeddings)`, already computed by
            the caller; recomputed here only when not supplied.
        :return: the pooled logits, `relations` always None.
        """
        with self.autocast_context():
            if hidden_output is None:
                hidden_output = self.hidden(embeddings)
            class_logits = self.classifier(hidden_output)
            token_mask = attention_mask.unsqueeze(-1)
            class_logits.masked_fill_(~token_mask, self._neg_inf)

            return BatchLogits(
                self._pool_logits(class_logits, mask=attention_mask)
            )
