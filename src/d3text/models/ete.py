"""`ETEBrendaModel` — entity-class detection + relation extraction."""

import itertools
import logging
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch
from d3text import tracking
from d3text.constraints import NonNegative, UnitInterval
from d3text.mention_metrics import token_predicted_mentions
from d3text.progress import batch_progress
from d3text.schema import Schema
from jaxtyping import Bool, Float, Int64
from sklearn.metrics import classification_report, f1_score
from torch import Tensor, nn
from torch.autograd.profiler import record_function
from torch.utils.data import DataLoader

from .base import (
    Model,
    Step,
    balanced_class_weights,
    coverage_metrics,
    focal_cross_entropy,
    relation_metrics,
    support_metrics,
    typed_relation_f1,
)
from .config import ModelConfig
from .entity_linking import BrendaClassificationModel
from .heads import BiaffineRelationClassifier, ClassificationHead
from .model_types import (
    BatchItem,
    BatchLogits,
    BatchLosses,
    GroundTruth,
    IndexedRelation,
    RelationCandidates,
)
from .token_supervision import (
    StoredMention,
    TokenLabelReader,
    document_lengths,
    resolve_mentions,
)

logger = logging.getLogger(__name__)


class ArgumentGroups:
    """Batch-local integer ids for the candidate sets relation rows key on.

    An argument is a candidate set — every entity the store's mentions leave a
    tagged span able to name — so two mentions carrying the same set are one
    argument. The pair keys have to stay integer tensors for the aligner to
    group and join them on the device, hence the interning; it happens where
    the sets already are, on the host. An id means nothing outside the batch
    that interned it, exactly as a `sequence` index does not.
    """

    def __init__(self) -> None:
        self._ids: dict[frozenset[str], int] = {}

    def intern(self, candidates: frozenset[str]) -> int:
        """This set's id, allocated in first-seen order.

        :param candidates: the argument's candidate entity IDs.
        :return: its id.
        """
        return self._ids.setdefault(candidates, len(self._ids))

    @property
    def sets(self) -> tuple[frozenset[str], ...]:
        """Every interned set, indexed by its own id.

        :return: the sets, `sets[id]` being the set that `id` stands for.
        """
        return tuple(self._ids)

    def by_entity(self) -> dict[str, frozenset[int]]:
        """Which arguments an entity could be.

        :return: entity ID -> the ids of every interned set holding it. One
            entity sits in several where a set narrowed to it and a wider set
            containing it were both proposed.
        """
        holders: dict[str, set[int]] = {}
        for candidates, group in self._ids.items():
            for entity_id in candidates:
                holders.setdefault(entity_id, set()).add(group)
        return {
            entity_id: frozenset(groups)
            for entity_id, groups in holders.items()
        }


class RelationRow(NamedTuple):
    """One candidate pair, before the relation classifier runs over it.

    `argument_i` and `argument_j` are `ArgumentGroups` ids in ascending order,
    and the two representations are ordered with them.
    """

    docix: int
    argument_i: int
    argument_j: int
    repr_i: Float[Tensor, " features"]
    repr_j: Float[Tensor, " features"]


class PredictedRelation(NamedTuple):
    """One pair the relation head labelled, as inference keeps it.

    The arguments are candidate sets, not single entities: an argument is
    whatever entity ids the tagger's span grounded to, and collapsing a set
    onto one id is the choice the grounding rule refuses to make. They are
    ordered by the batch's interning table, which carries no subject/object
    role — the schema's relation type is what says which argument type fills
    which role.
    """

    predicate: str
    arguments: tuple[frozenset[str], frozenset[str]]


class ETEBrendaModel(Model):
    """Entity-class detection + relation extraction.

    Composes a `BrendaClassificationModel` for the class-head machinery rather
    than subclassing it, so both return the same typed containers instead of
    widening them. `__getattr__` reaches through to that model for what this
    class does not declare, and is read-only by construction: a value that
    must reach it on a write needs its own property.
    """

    # What this class reads through the reach-through, declared so mypy
    # resolves it here and not via `__getattr__`, which types every name it
    # cannot find. The methods it reaches for are declared in `__init__`.
    two_head: BrendaClassificationModel
    classifier: ClassificationHead
    token_tagger: nn.Linear | None

    # `object`, not the supertype's `Tensor | Module`: beartype enforces the
    # annotation at runtime, and the reach-through hands back whatever the
    # composed model holds, plain functions and floats among it. `object` is
    # still what makes a misspelt name an error rather than `Any`.
    def __getattr__(self, name: str) -> object:  # type: ignore[override]
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

    @property
    def training_entity_ids(self) -> frozenset[str] | None:
        return self.two_head.training_entity_ids

    @training_entity_ids.setter
    def training_entity_ids(self, entity_ids: frozenset[str] | None) -> None:
        self.two_head.training_entity_ids = entity_ids

    def __init__(
        self,
        schema: Schema,
        config: ModelConfig | None = None,
        class_freqs: Float[Tensor, " classes"] | None = None,
        device: str | None = None,
    ) -> None:
        """Compose the class head and build the relation classifier atop it.

        :param schema: the entity, class and relation vocabulary to build
            heads for.
        :param config: hyperparameters; defaults to a fresh `ModelConfig()`.
        :param class_freqs: per-class positive frequency, forwarded to the
            composed class head's loss weighting; None trains it unweighted.
        :param device: torch device to build on; None picks CUDA if
            available, else CPU.
        """
        config = config if config is not None else ModelConfig()
        super().__init__(config, device=device)

        self.schema = schema
        self.two_head = BrendaClassificationModel(
            schema=schema,
            config=config,
            class_freqs=class_freqs,
            device=device,
        )
        if TYPE_CHECKING:
            # Typed as `two_head`'s bound methods, which is what the
            # reach-through returns, so a signature change there is seen
            # here. Nothing is bound at runtime: `object.__setattr__` on an
            # instance still shadows them, as the evaluation stubs rely on.
            self.compute_class_loss = self.two_head.compute_class_loss
            self.class_negative_abstain_mask = (
                self.two_head.class_negative_abstain_mask
            )
            self.compute_token_loss = self.two_head.compute_token_loss
            self.score_token_detection = self.two_head.score_token_detection
            self._detection_accumulator = self.two_head._detection_accumulator

        # What the `arg_pred_*` integers of the last `forward` stood for. The
        # pair keys are interned candidate sets, and the loss and the metrics
        # have to read that mapping back to join gold against them; `forward`
        # is the only writer and it rewrites both on every call, so an id is
        # never read against another batch's table.
        self._argument_sets: tuple[frozenset[str], ...] = ()
        self._argument_groups: dict[str, frozenset[int]] = {}

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

    def relation_loss_weight(
        self, epoch: NonNegative, w0: UnitInterval = 0.1
    ) -> float:
        """The relation loss' weight at `epoch`, ramping `w0` to 1.0.

        The ramp runs over `ramp_epochs`, which at 0 means no ramp at all. It
        holds the relation head back until the span tagger proposes usable
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
        """This batch's class, relation and (optional) token losses.

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
            "class": batch_losses.class_,
            "relation": batch_losses.relation * w_rel,
        }
        token_loss = batch_losses.token
        if token_loss is not None:
            # Unramped: the token targets are supervision available from
            # epoch 0, not a late-phase objective.
            losses["token"] = token_loss

        return losses

    def epoch_loss_weights(self, epoch: int) -> dict[str, float]:
        """Only the relation loss is scheduled.

        :param epoch: the epoch about to run.
        :return: every objective's multiplier, the rest at the full weight they
            train under so each has a curve.
        """
        weights = {
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
        """The gold classes and relations of the batch's documents.

        :param batch: the batch to read.
        :return: the targets, `relations` always a (possibly empty) list here.
        """
        class_targets = self.two_head.ground_truth(batch).classes

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

        return GroundTruth(class_targets, relation_targets)

    @torch.compiler.disable
    def _gold_pair_key(self, relation: IndexedRelation) -> tuple[int, str, str]:
        """`(doc, argument, argument)` for a gold relation, arguments sorted.

        A gold pair's own identity, in the entity-ID strings the corpus states
        it in. Sorted rather than taken on trust from the corpus, so a triple
        repeated with its arguments reversed is recognised as the one pair it
        is; the label is directional by argument *type*, not by argument order.

        `@torch.compiler.disable`d because dynamo guards on string *values*:
        it would specialise this frame on each entity ID it saw and recompile
        until the limit, for a helper that runs no tensor op at all.
        """
        first, second = sorted((relation.subject, relation.object))
        return int(relation.docix), first, second

    def _covering_row_keys(
        self, relation: IndexedRelation
    ) -> list[tuple[int, int, int]]:
        """Every `(doc, argument, argument)` row key that covers this gold pair.

        A candidate argument is a *set* of entity IDs, so a row covers a gold
        pair when one of its arguments could be the subject and the other the
        object — the intersection rule the linking scores already use, not the
        equality a column pair admitted. One gold pair can therefore key
        several rows, and each of them is trained toward its label.

        :param relation: the gold triple.
        :return: the covering keys, arguments ascending, in ascending order.
        """
        groups = self._argument_groups
        docix = int(relation.docix)
        return sorted(
            {
                (docix, min(first, second), max(first, second))
                for first in groups.get(relation.subject, frozenset())
                for second in groups.get(relation.object, frozenset())
                if first != second
            }
        )

    def _gold_entity_positions(
        self,
        batch: Sequence[BatchItem],
        gold_relations: Sequence[IndexedRelation],
    ) -> dict[int, dict[str, Int64[Tensor, " positions"]]]:
        """Aggregated-token positions of each gold argument's own mention(s).

        Looked up from the configured label store, not learned: a gold
        relation argument's representation is pooled from where its own
        surface form was matched in the document, never from the span
        tagger's detections. Reuses `compute_token_loss`'s optional dependency
        -- a model built with no `config.token_labels_store` represents no
        gold argument at all, which `forward`'s existing "representation
        unavailable" drop already turns into a `none`-labeled miss via
        `unscored_gold_relations`. It doubles as the anchor test that
        bookkeeping reads: an argument absent here is one the store places
        nowhere in the document.

        :param batch: the batch's items, for each document's pubmed id and
            window-level attention mask.
        :param gold_relations: the batch's gold relation triples.
        :return: docix -> entity ID -> its own mention's token positions on
            the aggregated per-document axis. An entity absent from an entry
            has no matched span; a docix absent has none at all.
        """
        if not gold_relations:
            return {}
        reader = self._token_labels
        if reader is None:
            return {}

        needed: dict[int, set[str]] = {}
        for relation in gold_relations:
            docix = int(relation.docix)
            needed.setdefault(docix, set()).update(
                (relation.subject, relation.object)
            )

        positions: dict[int, dict[str, Tensor]] = {}
        for docix, entity_ids in needed.items():
            item = batch[docix]
            pubmed_id = int(item["id"].item())
            window_mask = item["sequence"]["attention_mask"]
            doc_positions: dict[str, Tensor] = {}
            for entity_id in entity_ids:
                found = reader.entity_positions(
                    pubmed_id, entity_id, window_mask
                )
                if found is not None:
                    doc_positions[entity_id] = found
            if doc_positions:
                positions[docix] = doc_positions
        return positions

    def _stored_mentions(
        self, batch: Sequence[BatchItem]
    ) -> dict[int, tuple[StoredMention, ...]]:
        """Every exact mention the store holds for each document of the batch.

        `forward` takes these as an argument because it has no document
        identity of its own — no pubmed ids and no batch — the same reason
        `_gold_entity_positions` is looked up out here. Gold plays no part:
        these are every dictionary match, which is what lets a detected span
        ground in an entity the document is not linked to.

        :param batch: the batch's items, for each document's pubmed id and
            window-level attention mask.
        :return: docix -> its mentions. A document the store holds nothing for
            is absent, and so proposes no candidate at all; its gold relations
            have no anchor either and are counted under that.
        """
        reader = self._token_labels
        if reader is None:
            return {}

        mentions: dict[int, tuple[StoredMention, ...]] = {}
        for docix, item in enumerate(batch):
            found = reader.exact_mentions(
                int(item["id"].item()), item["sequence"]["attention_mask"]
            )
            if found:
                mentions[docix] = found
        return mentions

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
        """One row per candidate pair, with the target gold gives it.

        Rows repeating a pair are pooled into one and the gold label of every
        pair a row covers becomes that row's target; a row no gold covers is
        trained toward `none`, which is the prediction the corpus makes about a
        pair it does not hold.

        :param true_relations: the batch's gold triples.
        :param rel_meta: the candidate rows' `sequence` and the two
            `ArgumentGroups` ids of each row's arguments.
        :param rel_logits: those rows' relation logits.
        :return: the pooled rows' meta, logits and targets, in the order the
            rows first appeared; None when there are no rows.
        """
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
            for key in self._covering_row_keys(tr):
                gold_by_key[key].append(int(tr.label))

        gold_triples: list[tuple[int, int, int]] = []
        gold_labels: list[int] = []
        for key, labels in gold_by_key.items():
            gold_triples.append(key)
            gold_labels.append(self._missed_gold_label(labels))
        gold_index = torch.tensor(
            gold_triples, dtype=torch.long, device=device
        ).reshape(-1, 3)

        # Pack (sequence, subject, object) into one int64 so that grouping is a
        # single `torch.unique` and the gold join a single `searchsorted`.
        # The radices are read off the data instead of being fixed bit widths:
        # an argument id counts the distinct candidate sets this batch
        # proposed, which is a property of the batch rather than of this
        # function. Their product is bounded by batch x |arguments|^2 and stays
        # far inside int64.
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

    @staticmethod
    def _meta_rows(
        meta: dict[str, Tensor] | None,
    ) -> list[tuple[int, int, int]] | None:
        """`(sequence, arg_pred_i, arg_pred_j)` triples, host data, one read.

        Stacks the three columns before the single `.tolist()` so a caller
        needing the pooled meta on the host pays one device sync for the
        whole table, rather than one per column per consumer.
        """
        if meta is None:
            return None
        return [
            (seq, i, j)
            for seq, i, j in torch.stack(
                [meta["sequence"], meta["arg_pred_i"], meta["arg_pred_j"]],
                dim=1,
            ).tolist()
        ]

    def unscored_gold_relations(
        self,
        true_relations: Sequence[IndexedRelation],
        scored_rows: Sequence[tuple[int, int, int]] | None,
        anchored: Mapping[int, Mapping[str, Tensor]],
    ) -> tuple[list[int], list[int]]:
        """Gold relations that no scored row can account for.

        The aligner builds its rows out of the candidate pairs the span tagger's
        detections were grounded into, so gold that was never proposed leaves no
        row and cannot appear in any metric over those rows. A caller computing
        metrics must add these back as misses. Kept out of the aligner because
        the loss path consumes that function and these relations carry no logits
        to backpropagate.

        :param true_relations: the document's gold relations.
        :param scored_rows: the `(sequence, arg_pred_i, arg_pred_j)` triples of
            the rows actually scored, already read to the host once by the
            caller, or None when the aligner returned nothing.
        :param anchored: each document's gold arguments that the store places
            in its text, as `_gold_entity_positions` returns them.
        :return: the labels of the missed gold, as `(not_proposed, no_anchor)`.
            No anchor means the store holds no mention of one argument in that
            document — nothing could have proposed the pair, and no detector
            reaches it — while the rest were simply not proposed. A gold triple
            repeated across a document's pair-dicts yields one entry in either
            list.
        """
        scored: set[tuple[int, int, int]] = (
            set(scored_rows) if scored_rows else set()
        )

        missed_by_key: dict[tuple[int, str, str], list[int]] = defaultdict(list)
        no_anchor_by_key: dict[tuple[int, str, str], list[int]] = defaultdict(
            list
        )

        for relation in true_relations:
            key = self._gold_pair_key(relation)
            positions = anchored.get(int(relation.docix), {})
            if not {relation.subject, relation.object} <= positions.keys():
                no_anchor_by_key[key].append(int(relation.label))
            elif not any(
                row in scored for row in self._covering_row_keys(relation)
            ):
                missed_by_key[key].append(int(relation.label))

        return (
            [
                self._missed_gold_label(labels)
                for labels in missed_by_key.values()
            ],
            [
                self._missed_gold_label(labels)
                for labels in no_anchor_by_key.values()
            ],
        )

    def _strict_relation_targets(
        self,
        true_relations: Sequence[IndexedRelation],
        scored_rows: Sequence[tuple[int, int, int]] | None,
    ) -> tuple[Int64[Tensor, " rows"], list[int]]:
        """The scored rows' targets under the strict rule, and what it misses.

        The rule the rows are trained and scored under is intersection: a row
        takes a gold label as soon as one of its arguments *could* be the
        subject and the other the object. Strict asks for each argument to be
        that entity alone, so the gap between the two scores is what the
        linking left undisambiguated rather than anything the relation head
        did. Gold no row covers strictly is returned for the caller to score as
        `none`, the way the other misses are.

        :param scored_rows: the `(sequence, arg_pred_i, arg_pred_j)` triples of
            the rows actually scored, already read to the host once by the
            caller, or None when the aligner returned nothing.
        """
        none_index = int(self.relations_none_index)
        rows = len(scored_rows) if scored_rows is not None else 0
        targets = torch.full((rows,), none_index, dtype=torch.int64)
        if not true_relations:
            return targets, []

        singletons = {
            next(iter(candidates)): group
            for group, candidates in enumerate(self._argument_sets)
            if len(candidates) == 1
        }
        row_of_key: dict[tuple[int, int, int], int] = {}
        if scored_rows is not None:
            row_of_key = {key: row for row, key in enumerate(scored_rows)}

        labels_by_row: dict[int, list[int]] = defaultdict(list)
        missed_by_key: dict[tuple[int, str, str], list[int]] = defaultdict(list)
        for relation in true_relations:
            first = singletons.get(relation.subject)
            second = singletons.get(relation.object)
            row = None
            if first is not None and second is not None and first != second:
                row = row_of_key.get(
                    (
                        int(relation.docix),
                        min(first, second),
                        max(first, second),
                    )
                )
            if row is None:
                missed_by_key[self._gold_pair_key(relation)].append(
                    int(relation.label)
                )
            else:
                labels_by_row[row].append(int(relation.label))

        for row, labels in labels_by_row.items():
            targets[row] = self._missed_gold_label(labels)

        return targets, [
            self._missed_gold_label(labels) for labels in missed_by_key.values()
        ]

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
        """This batch's relation loss, rows aligned against gold first.

        :param true_relations: the batch's gold triples.
        :param rel_meta: the candidate rows' `sequence` and argument-group
            ids, as `align_relation_predictions` expects.
        :param rel_logits: those rows' relation logits, or None when the
            batch proposed no pair.
        :return: the scalar loss; `0.0` when there is nothing to align.

        The loss form follows `self.relation_loss_weighting`: focal,
        class-balanced, or plain mean cross-entropy, all under the
        configured label smoothing.
        """
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
        """Class and relation logits for a batch, embeddings fetched here.

        :param batch: the batch to score.
        :param gold_relations: gold pairs to fall back to a row for,
            forwarded to `forward`; None scores detected pairs only.
        :return: the pooled logits, as `forward` returns them.
        """
        token_embeddings, token_att_mask = self.get_token_embeddings(batch)

        return self(
            token_embeddings,
            token_att_mask,
            gold_relations=gold_relations,
            gold_entity_positions=self._gold_entity_positions(
                batch, gold_relations or []
            ),
            stored_mentions=self._stored_mentions(batch),
        )

    def compute_batch_losses(self, batch: Sequence[BatchItem]) -> BatchLosses:
        """This batch's losses, one field per objective.

        :param batch: the batch to run.
        :return: the class, relation and token losses.
        """
        class_true, rel_true = self.ground_truth(batch)
        rel_true = rel_true or []
        token_embeddings, token_att_mask = self.get_token_embeddings(batch)

        # Computed once here, up front, only when the tagger loss will need
        # it — `forward` (via `_tagged_arguments`) and `compute_token_loss`
        # both take the shared hidden state, the tagger's own logits and the
        # mask's unpadded lengths instead of each recomputing them (the
        # lengths off the same device tensor, a host sync each time) over the
        # same batch themselves.
        hidden_output = None
        token_logits = None
        lengths = None
        if self.token_tagger is not None:
            with self.autocast_context():
                hidden_output = self.hidden(token_embeddings)
                token_logits = self.token_tagger(hidden_output)
            lengths = document_lengths(token_att_mask)

        class_logits, relation_index_logits = self(
            token_embeddings,
            token_att_mask,
            gold_relations=rel_true,
            gold_entity_positions=self._gold_entity_positions(batch, rel_true),
            stored_mentions=self._stored_mentions(batch),
            hidden_output=hidden_output,
            token_logits=token_logits,
            lengths=lengths,
        )

        class_loss = self.compute_class_loss(
            class_logits,
            class_true,
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
            class_=class_loss,
            relation=relation_loss,
            token=self.compute_token_loss(
                batch,
                token_embeddings,
                token_att_mask,
                hidden_output=hidden_output,
                token_logits=token_logits,
                lengths=lengths,
            ),
        )

    def predicted_relations(
        self, batch: Sequence[BatchItem]
    ) -> list[PredictedRelation] | None:
        """Every candidate pair the relation head gave a non-null label.

        The inference counterpart of the aligner `compute_batch_true_x_pred`
        runs: with no gold to align against, a row's label is just its
        argmax, and the two argument ids are read back through the batch's
        own interning table, which is the only thing that knows which
        candidate sets they stand for.

        :param batch: the batch to run; no gold relation is passed to
            `forward`, so every scored row is one the tagger's own
            groundings proposed.
        :return: one entry per row labelled anything but the null relation,
            empty where the head was put pairs and called every one of them
            null, and None where it was put none at all — a document with
            fewer than two grounded arguments, or one the token-label store
            holds no mention to ground against, which is not the head
            ruling a relation out.
        """
        candidates = self.get_batch_logits(batch).relations
        if candidates is None:
            return None
        meta, logits = candidates
        rows = self._meta_rows(meta)
        assert rows is not None  # `meta` is not None
        sets = self._argument_sets
        none_index = int(self.relations_none_index)
        names = self.schema.relation_names
        return [
            PredictedRelation(
                predicate=names[label],
                arguments=(sets[argument_i], sets[argument_j]),
            )
            for (_, argument_i, argument_j), label in zip(
                rows, logits.argmax(dim=-1).tolist(), strict=True
            )
            if label != none_index
        ]

    def compute_batch_true_x_pred(
        self, batch: Sequence[BatchItem]
    ) -> dict[str, dict[str, np.ndarray]]:
        """Gold and predicted arrays for each task the model tackles.

        :param batch: the batch to score.
        :return: task name -> its `y_true` and `y_pred` arrays.
        """
        class_logits: Float[Tensor, "sequence classes"]
        relation_index_logits: (
            tuple[dict[str, Tensor], Float[Tensor, "pairs relations"]] | None
        )
        class_logits, relation_index_logits = self.get_batch_logits(batch)

        class_truth: Float[Tensor, "batch classes"]
        class_truth, rel_truth_optional = self.ground_truth(batch)
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

            scored_rows = None
            if aligned_rel_preds is not None:
                scored_meta, preds, targets = aligned_rel_preds
                scored_rows = self._meta_rows(scored_meta)
                relations_true = (
                    targets.numpy(force=True).reshape(-1).astype(int)
                )
                relations_pred = preds.numpy(force=True)
                relations_pred = (
                    relations_pred.argmax(axis=-1).reshape(-1).astype(int)
                )

            # Gold no candidate pair covers has no row to be scored on, so
            # without this it would vanish from the metrics rather than count
            # against them.
            not_proposed, no_anchor = self.unscored_gold_relations(
                rel_truth,
                scored_rows,
                self._gold_entity_positions(batch, rel_truth),
            )
            missed_true, missed_pred = self._missed_gold_predictions(
                not_proposed + no_anchor
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

    def _tagged_arguments(
        self,
        hidden_output: Float[Tensor, "document token features"],
        attention_mask: Bool[Tensor, "document token"],
        stored_mentions: Mapping[int, Sequence[StoredMention]],
        token_logits: Float[Tensor, "document token codes"] | None = None,
        lengths: list[int] | None = None,
    ) -> dict[int, dict[frozenset[str], Int64[Tensor, " positions"]]]:
        """Each document's detected relation arguments, by candidate set.

        The tagger runs over the hidden states `forward` already computed, its
        argmax is cut into typed spans, and each span is grounded in the
        mentions the store holds for that document. Spans carrying the same
        candidate set are one argument, and a NIL span carries none at all, so
        it proposes nothing: relations train on grounded arguments only.

        :param hidden_output: the shared hidden block's output.
        :param attention_mask: which positions carry a real token.
        :param stored_mentions: each document's exact mentions, from
            `_stored_mentions`.
        :param token_logits: `self.token_tagger(hidden_output)`, already
            computed by the caller; recomputed here only when not supplied.
        :param lengths: `document_lengths(attention_mask)`, already computed
            by the caller; recomputed here only when not supplied.
        :return: docix -> candidate set -> the tokens its mentions cover. A
            document with fewer than two arguments is absent, since no pair can
            come out of it.
        """
        tagger = self.token_tagger
        reader = self._token_labels
        if tagger is None or reader is None or not stored_mentions:
            return {}

        # Nothing differentiable leaves the tagger here: its argmax only picks
        # which hidden states to pool, and the relation head's gradient reaches
        # them by indexing `hidden_output`, never through these logits.
        with torch.no_grad():
            if token_logits is None:
                token_logits = tagger(hidden_output)
            codes = token_logits.float().argmax(dim=-1).cpu()

        if lengths is None:
            lengths = document_lengths(attention_mask)

        arguments: dict[int, dict[frozenset[str], Tensor]] = {}
        for docix, length in enumerate(lengths):
            stored = stored_mentions.get(docix)
            if not stored:
                continue
            covered: dict[frozenset[str], list[int]] = {}
            for span in resolve_mentions(
                token_predicted_mentions(codes[docix, :length].numpy()),
                stored,
                space=reader.space,
            ):
                if span.entity_ids:
                    covered.setdefault(span.entity_ids, []).extend(
                        range(span.start, span.end)
                    )
            if len(covered) > 1:
                arguments[docix] = {
                    candidates: torch.tensor(positions, dtype=torch.int64)
                    for candidates, positions in covered.items()
                }
        return arguments

    def _pooled_arguments(
        self,
        doc_hidden: Float[Tensor, "token features"],
        by_set: Mapping[frozenset[str], Int64[Tensor, " positions"]],
        groups: ArgumentGroups,
    ) -> list[tuple[int, frozenset[str], Float[Tensor, " features"]]]:
        """One document's arguments as `(id, candidates, representation)`.

        Ascending by id, which is what makes every pair built out of them carry
        its arguments in the order the aligner and the merge both key on.
        """
        pooled = [
            (
                groups.intern(candidates),
                candidates,
                doc_hidden.index_select(
                    0, positions.to(device=doc_hidden.device, dtype=torch.long)
                ).mean(dim=0),
            )
            for candidates, positions in by_set.items()
            if positions.numel()
        ]
        return sorted(pooled, key=lambda argument: argument[0])

    def _detected_rows(
        self,
        arguments: Mapping[
            int, Mapping[frozenset[str], Int64[Tensor, " positions"]]
        ],
        hidden_output: Float[Tensor, "document token features"],
        groups: ArgumentGroups,
    ) -> list[RelationRow]:
        """Every admitted pair of one batch's detected arguments.

        An argument is a set of candidate IDs, so nothing here can identify one
        by a single predicted entity: collapsing a narrowed set onto one ID
        would be the choice the linking rule refuses to make, which is why the
        pairing runs over explicit `(positions, id)` groups.

        :param arguments: each document's arguments, from `_tagged_arguments`.
        :param hidden_output: the shared hidden block's output.
        :param groups: the batch's interning table, extended in place.
        :return: the rows, documents in batch order.
        """
        rows: list[RelationRow] = []
        for docix, by_set in arguments.items():
            pooled = self._pooled_arguments(
                hidden_output[docix], by_set, groups
            )
            for (first, left, repr_i), (
                second,
                right,
                repr_j,
            ) in itertools.combinations(pooled, 2):
                # An argument's type is its candidates' shared ID prefix — the
                # span's own tagged type, which is what the grounding filtered
                # them to. No relation type admits most type pairings, and
                # their label is fixed `none` by the schema alone.
                if not self.schema.admits_relation(
                    next(iter(left)), next(iter(right))
                ):
                    continue
                rows.append(RelationRow(docix, first, second, repr_i, repr_j))
        return rows

    def _gold_rows(
        self,
        gold_relations: Sequence[IndexedRelation],
        gold_entity_positions: Mapping[
            int, Mapping[str, Int64[Tensor, " positions"]]
        ],
        detected: Mapping[
            int, Mapping[frozenset[str], Int64[Tensor, " positions"]]
        ],
        hidden_output: Float[Tensor, "document token features"],
        groups: ArgumentGroups,
    ) -> list[RelationRow]:
        """A row for each gold pair no detected pair already covers.

        Detected first, gold as the fallback: a detected row whose sets cover
        the pair is the representation the evaluation scores, so training the
        gold row instead left that representation with no positive to learn
        from. A gold argument is one entity, hence a singleton set, and is
        pooled from its own stored mention positions.

        :param gold_relations: the batch's gold triples.
        :param gold_entity_positions: each gold argument's own positions, from
            `_gold_entity_positions`; an argument absent here has no anchor and
            gets no row.
        :param detected: each document's detected arguments, whose sets decide
            what is already covered.
        :param hidden_output: the shared hidden block's output.
        :param groups: the batch's interning table, extended in place.
        :return: the rows, one per uncovered gold pair.
        """
        rows: list[RelationRow] = []
        for pair in dict.fromkeys(
            self._gold_pair_key(relation) for relation in gold_relations
        ):
            docix, subject, object_ = pair
            anchors = gold_entity_positions.get(docix, {})
            if not {subject, object_} <= anchors.keys():
                continue
            if self._covered_by_detection(
                detected.get(docix, {}), subject, object_
            ):
                continue
            pooled = self._pooled_arguments(
                hidden_output[docix],
                {
                    frozenset({subject}): anchors[subject],
                    frozenset({object_}): anchors[object_],
                },
                groups,
            )
            if len(pooled) != 2:
                continue
            (first, _, repr_i), (second, _, repr_j) = pooled
            rows.append(RelationRow(docix, first, second, repr_i, repr_j))
        return rows

    @staticmethod
    def _covered_by_detection(
        by_set: Mapping[frozenset[str], Int64[Tensor, " positions"]],
        subject: str,
        object_: str,
    ) -> bool:
        """Whether two detected arguments could be this gold pair.

        Coverage, not key equality: a gold pair names two entities while a
        detected argument may carry a set of several, so the two sides never
        share a key even when the detected pair is exactly the gold one.
        """
        holders = [
            [candidates for candidates in by_set if entity_id in candidates]
            for entity_id in (subject, object_)
        ]
        return any(left != right for left in holders[0] for right in holders[1])

    def _score_rows(
        self, rows: Sequence[RelationRow]
    ) -> RelationCandidates | None:
        """The relation logits of one batch's candidate pairs.

        :param rows: the pairs, detected ones before gold fallbacks.
        :return: the rows' `sequence` and argument ids beside their logits, or
            None when the batch proposed no pair at all.
        """
        if not rows:
            return None

        logits = self.relation_classifier(
            torch.stack([row.repr_i for row in rows], dim=0),
            torch.stack([row.repr_j for row in rows], dim=0),
        )
        # One host-to-device upload for the whole [rows, 3] table, rather than
        # one column at a time.
        ids = torch.tensor(
            [(row.docix, row.argument_i, row.argument_j) for row in rows],
            device=self.device,
            dtype=torch.long,
        )
        meta = {
            "sequence": ids[:, 0],
            "arg_pred_i": ids[:, 1],
            "arg_pred_j": ids[:, 2],
        }
        return meta, logits

    @record_function("forward")
    def forward(
        self,
        embeddings: Float[Tensor, "document token embedding"],
        attention_mask: Bool[Tensor, "document token"],
        gold_relations: list[IndexedRelation] | None = None,
        gold_entity_positions: dict[int, dict[str, Tensor]] | None = None,
        stored_mentions: dict[int, tuple[StoredMention, ...]] | None = None,
        hidden_output: Float[Tensor, "document token features"] | None = None,
        token_logits: Float[Tensor, "document token codes"] | None = None,
        lengths: list[int] | None = None,
    ) -> BatchLogits:
        """Class and relation logits for one batch.

        :param embeddings: the batch's token embeddings.
        :param attention_mask: which positions carry a real token.
        :param gold_relations: gold pairs to fall back to a row for, on a
            training pass.
        :param gold_entity_positions: each gold argument's own mention token
            positions, from `_gold_entity_positions`; an argument absent here
            gets no gold-side row, which is the miss the bookkeeping counts as
            having no anchor.
        :param stored_mentions: each document's exact mentions, from
            `_stored_mentions`; without them the tagger's spans cannot be
            grounded and the batch proposes no detected pair at all.
        :param hidden_output: `self.hidden(embeddings)`, already computed by
            the caller; recomputed here only when not supplied.
        :param token_logits: `self.token_tagger(hidden_output)`, already
            computed by the caller, forwarded to `_tagged_arguments`;
            recomputed there only when not supplied.
        :param lengths: `document_lengths(attention_mask)`, already computed
            by the caller, forwarded to `_tagged_arguments`; recomputed there
            only when not supplied.
        :return: the pooled logits, `relations` carrying which sequence and
            which pair of candidate-set ids each scored row belongs to, beside
            its logits.
        """
        with self.autocast_context():
            if hidden_output is None:
                hidden_output = self.hidden(embeddings)
            class_logits = self.classifier(hidden_output)
            self._mask_padding(class_logits, attention_mask)

            groups = ArgumentGroups()
            detected = self._tagged_arguments(
                hidden_output,
                attention_mask,
                stored_mentions or {},
                token_logits=token_logits,
                lengths=lengths,
            )
            rows = self._detected_rows(detected, hidden_output, groups)
            rows += self._gold_rows(
                gold_relations or (),
                gold_entity_positions or {},
                detected,
                hidden_output,
                groups,
            )
            # Published for the aligner and the metrics: they join gold against
            # these rows by what each argument id could be, which only the
            # table knows.
            self._argument_sets = groups.sets
            self._argument_groups = groups.by_entity()

            return BatchLogits(
                self._pool_logits(class_logits, mask=attention_mask),
                self._score_rows(rows),
            )

    def evaluate_model(
        self,
        test_data: DataLoader,
        tau_cls: UnitInterval = 0.5,
    ) -> dict[str, float]:
        """Evaluate the end-to-end model from document-level pooled logits.

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
        all_rel_logits, all_rel_true = [], []
        all_rel_strict: list[Int64[Tensor, " rows"]] = []
        detection = self._detection_accumulator()
        gold_relations = 0
        missed_not_proposed: list[int] = []
        missed_no_anchor: list[int] = []
        missed_strictly: list[int] = []
        argument_ids = argument_count = 0

        with torch.no_grad():
            # do NOT autocast around metric collection; keep numerics simple
            for batch in batch_progress(
                test_data, desc="Evaluating", position=0, leave=True
            ):
                # shapes: [B, num_classes], (meta, [N_pairs, R]) or None
                if detection is None:
                    cls_logits_doc, rel_meta_logits = self.get_batch_logits(
                        batch
                    )
                else:
                    # One embedding fetch serves the pooled head and the
                    # tagger; `get_batch_logits` would hide it. The tagger's
                    # own projection is likewise shared between the detection
                    # branch below and `_tagged_arguments` inside `forward`,
                    # rather than each running it over the same hidden state.
                    embeddings, token_mask = self.get_token_embeddings(batch)
                    assert (
                        self.token_tagger is not None
                    )  # detection is not None
                    with self.autocast_context():
                        hidden_output = self.hidden(embeddings)
                        token_logits = self.token_tagger(hidden_output)
                    lengths = document_lengths(token_mask)
                    cls_logits_doc, rel_meta_logits = self(
                        embeddings,
                        token_mask,
                        stored_mentions=self._stored_mentions(batch),
                        hidden_output=hidden_output,
                        token_logits=token_logits,
                        lengths=lengths,
                    )
                    self.score_token_detection(
                        batch,
                        embeddings,
                        token_mask,
                        detection,
                        hidden_output=hidden_output,
                        token_logits=token_logits,
                        lengths=lengths,
                    )

                cls_true_doc, rel_true_list_optional = self.ground_truth(batch)
                rel_true_list: list[IndexedRelation] = (
                    rel_true_list_optional or []
                )

                # logits narrowed to the columns the targets carry
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

                # The pooled meta is read to the host once here and shared by
                # every consumer below, instead of each re-fetching its three
                # columns off the device.
                scored_rows = None
                if aligned is not None:
                    scored_meta, rel_logits_aligned, rel_targets = aligned
                    all_rel_logits.append(rel_logits_aligned.detach().cpu())
                    all_rel_true.append(rel_targets.detach().cpu())
                    scored_rows = self._meta_rows(scored_meta)
                    assert scored_rows is not None  # scored_meta is not None
                    sets = self._argument_sets
                    for _, arg_i, arg_j in scored_rows:
                        for group in (arg_i, arg_j):
                            argument_ids += len(sets[group])
                            argument_count += 1

                strict_targets, strict_missed = self._strict_relation_targets(
                    rel_true_list, scored_rows
                )
                all_rel_strict.append(strict_targets)
                missed_strictly.extend(strict_missed)

                # The scored rows are the pairs the tagger's groundings were
                # paired into, so gold they miss leaves no row and would
                # otherwise never be counted against the model -- the metric
                # would be conditioned on detection having already found both
                # arguments.
                gold_relations += len(rel_true_list)
                not_proposed, no_anchor = self.unscored_gold_relations(
                    rel_true_list,
                    scored_rows,
                    self._gold_entity_positions(batch, rel_true_list),
                )
                missed_not_proposed.extend(not_proposed)
                missed_no_anchor.extend(no_anchor)

        if not all_cls_logits:
            logger.warning("No samples found.")
            metrics.update(coverage_metrics(test_data, 0))
            tracking.log_metrics(metrics)
            return metrics

        cls_logits = torch.cat(all_cls_logits, dim=0).numpy()
        cls_true = torch.cat(all_cls_true, dim=0).numpy().astype(int)

        # ---- CLASSES: probs -> binarize
        cls_probs = 1.0 / (1.0 + np.exp(-cls_logits))
        cls_pred = (cls_probs >= tau_cls).astype(int)

        metrics.update(coverage_metrics(test_data, cls_true.shape[0]))
        metrics.update(support_metrics({"class": (cls_true, cls_pred)}))
        logger.info(
            "\n[Classes ] gold positives: %d | predicted positives: %d",
            int(cls_true.sum()),
            int(cls_pred.sum()),
        )
        scored_pairs = sum(int(true.numel()) for true in all_rel_true)
        logger.info(
            "[Relations] gold: %d | candidate pairs scored: %d "
            "| missed, never proposed: %d "
            "| missed, no stored mention of an argument: %d",
            gold_relations,
            scored_pairs,
            len(missed_not_proposed),
            len(missed_no_anchor),
        )
        metrics["test/relation_gold"] = float(gold_relations)
        metrics["test/relation_missed_not_proposed"] = float(
            len(missed_not_proposed)
        )
        metrics["test/relation_missed_no_anchor"] = float(len(missed_no_anchor))
        if argument_count:
            metrics["test/relation_argument_set_size"] = (
                argument_ids / argument_count
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
            missed_not_proposed + missed_no_anchor
        )
        if all_rel_logits:
            rel_logits_np = torch.cat(all_rel_logits, dim=0).numpy()
            row_true = torch.cat(all_rel_true, dim=0).numpy().astype(int)
            row_true_strict = (
                torch.cat(all_rel_strict, dim=0).numpy().astype(int)
            )
            row_pred = rel_logits_np.argmax(axis=1)
        else:
            row_true = np.array([], dtype=int)
            row_true_strict = np.array([], dtype=int)
            row_pred = np.array([], dtype=int)

        rel_true = np.concatenate([row_true, missed_true])
        rel_pred = np.concatenate([row_pred, missed_pred])

        if rel_true.size:
            logger.info(
                "\n=== Relation metrics (multiclass over candidate pairs "
                "and missed gold) ==="
            )
            labels = np.arange(len(self.relations))
            none_index = int(self.relations_none_index)
            metrics.update(
                relation_metrics(
                    true=rel_true,
                    pred=rel_pred,
                    labels=labels,
                    none_index=none_index,
                )
            )
            strict_missed_true, strict_missed_pred = (
                self._missed_gold_predictions(missed_strictly)
            )
            metrics.update(
                typed_relation_f1(
                    true=np.concatenate([row_true_strict, strict_missed_true]),
                    pred=np.concatenate([row_pred, strict_missed_pred]),
                    labels=labels,
                    none_index=none_index,
                    suffix="_strict",
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
