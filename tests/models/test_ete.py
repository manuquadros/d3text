"""Pure unit tests for `d3text.models.ete.ETEBrendaModel`.

The relation-loss ramp, config wiring, relation alignment and its bookkeeping
of gold no candidate pair covers, the reported metrics, relation-loss class
weighting, and the relation half of `ground_truth`. Candidate proposal itself is
`test_detected_candidates.py`. CPU only, through the `stub` fixture, bar the
handful using `patch_base_model`.
"""

import logging

import pytest
import torch
from torch.utils.data import DataLoader
from pydantic import ValidationError

from d3text.models.config import ModelConfig
from d3text.models.entity_linking import BrendaClassificationModel
from d3text.models.ete import ETEBrendaModel
from d3text.models.model_types import IndexedRelation
from d3text.schema import EntityType, RelationType, Schema

# Three relations, matching what `ETEBrendaModel` used to hardcode as
# `("HasEnzyme", "HasSpecies", "none")` — `test_config_knobs_reach_the_ete_model`
# pins the relation head's width at exactly this count.
SCHEMA = Schema(
    entity_types=(
        EntityType(name="enzymes", prefix="enz"),
        EntityType(name="bacteria", prefix="bac"),
    ),
    relation_types=(
        RelationType(
            name="HasEnzyme", subject_types=("bacteria",), object_type="enzymes"
        ),
        RelationType(
            name="HasSpecies",
            subject_types=("bacteria",),
            object_type="enzymes",
        ),
        RelationType(name="none", is_none=True),
    ),
)
# `test_forward_dedups_repeated_gold_relation_pairs` links both its entities
# under the one "enzymes" class.
SINGLE_CLASS_SCHEMA = Schema(
    entity_types=(EntityType(name="enzymes", prefix="enz"),),
    relation_types=(
        RelationType(
            name="HasEnzyme",
            subject_types=("enzymes",),
            object_type="enzymes",
        ),
        RelationType(name="none", is_none=True),
    ),
)


# --------------------------------------------------------------------------- #
# ETEBrendaModel.relation_loss_weight                                          #
# --------------------------------------------------------------------------- #
def test_relation_loss_weight_without_ramp(stub):
    m = stub(ETEBrendaModel, ramp_epochs=0)
    assert m.relation_loss_weight(0) == 1.0
    assert m.relation_loss_weight(50) == 1.0


def test_relation_loss_weight_ramps_monotonically(stub):
    m = stub(ETEBrendaModel, ramp_epochs=4)
    weights = [m.relation_loss_weight(e) for e in range(6)]
    assert weights == sorted(weights)  # non-decreasing
    assert weights[0] == pytest.approx(0.1)  # starts at w0
    assert weights[-1] == pytest.approx(1.0)  # saturates at 1.0


def test_ramp_epochs_ramps_from_a_real_config(
    patch_base_model, empty_token_label_store
):
    """`ModelConfig`'s lower bound on `ramp_epochs` must not disturb a valid
    schedule: 0.1 -> 0.55 -> 1.0 over two epochs, read off a model built from
    an actual config rather than the `stub` fixture that bypasses it."""
    model = ETEBrendaModel(
        schema=SCHEMA,
        config=ModelConfig(
            base_model="prajjwal1/bert-mini",
            hidden_layers=[8],
            ramp_epochs=2,
            token_labels_store=str(empty_token_label_store),
        ),
        device="cpu",
    )
    weights = [model.relation_loss_weight(e) for e in range(3)]
    assert weights == pytest.approx([0.1, 0.55, 1.0])


def test_only_the_relation_head_owns_a_schedule(stub):
    """The model without a relation head has no ramp to expose at all."""
    assert not hasattr(
        stub(BrendaClassificationModel, ramp_epochs=4), "relation_loss_weight"
    )


def test_epoch_loss_weights_name_the_objective_each_weight_scales(stub):
    """The keys are what make a logged weight readable beside the loss it
    scaled. Only the relation loss is ever scheduled, so it is the only one
    whose weight moves with the epoch."""
    epoch = 2  # half way through a four-epoch ramp: 0.1 + 0.9 * 0.5

    parent = stub(BrendaClassificationModel, ramp_epochs=4)
    assert parent.epoch_loss_weights(epoch) == {"class": 1.0}

    ete = stub(ETEBrendaModel, ramp_epochs=4)
    assert ete.epoch_loss_weights(epoch) == {
        "class": 1.0,
        "relation": pytest.approx(0.55),
    }


# --------------------------------------------------------------------------- #
# ModelConfig knobs reaching the ETE model's relation classifier               #
# --------------------------------------------------------------------------- #
def test_config_knobs_reach_the_ete_model(
    patch_base_model, empty_token_label_store
):
    """biaffine_hidden_size is a ModelConfig field that must reach the relation
    classifier's projection width, rather than the former hardcoded 32."""
    model = ETEBrendaModel(
        schema=SCHEMA,
        config=ModelConfig(
            base_model="prajjwal1/bert-mini",
            hidden_layers=[8],
            biaffine_hidden_size=16,
            token_labels_store=str(empty_token_label_store),
        ),
        device="cpu",
    )
    assert tuple(model.relation_classifier.bilinear.shape) == (3, 16, 16)


def test_separate_predicate_layer_reaches_the_relation_classifier(
    patch_base_model, empty_token_label_store
):
    """ModelConfig.separate_predicate_layer must reach the biaffine
    classifier's constructor: with it set, the x/y projections are two
    distinct modules rather than the same one aliased under both names."""
    model = ETEBrendaModel(
        schema=SCHEMA,
        config=ModelConfig(
            base_model="prajjwal1/bert-mini",
            hidden_layers=[8],
            separate_predicate_layer=True,
            token_labels_store=str(empty_token_label_store),
        ),
        device="cpu",
    )
    assert (
        model.relation_classifier.hidden_linear_y
        is not model.relation_classifier.hidden_linear
    )


def test_forward_dedups_repeated_gold_relation_pairs(
    patch_base_model, empty_token_label_store
):
    """A `(subject, object)` pair named in two of a document's relation dicts
    must reach the biaffine classifier as one gold row, not two: the
    classifier still runs once per gold row, so a duplicate is a wasted
    launch, and the aligner counts one row per triple, so a duplicated row
    is a shape the loss path never sees. Under `logsumexp` pooling a
    duplicate would also add a spurious +log(2) to that pair's logits, but
    that pooling is no longer the model's default."""
    torch.manual_seed(0)
    config = ModelConfig(
        base_model="prajjwal1/bert-mini",
        hidden_layers=[8],
        token_labels_store=str(empty_token_label_store),
    )
    model = ETEBrendaModel(
        schema=SINGLE_CLASS_SCHEMA,
        config=config,
        device="cpu",
    )
    model.eval()

    embeddings = torch.randn(1, 6, 256)
    attention_mask = torch.ones(1, 6, dtype=torch.bool)

    single = [
        IndexedRelation(docix=0, subject="A", object="B", label=torch.tensor(0))
    ]
    duplicated = single + [
        IndexedRelation(docix=0, subject="A", object="B", label=torch.tensor(0))
    ]
    # Stands in for what `_gold_entity_positions` would look up from the label
    # store: each entity's own mention token positions in this one document.
    gold_entity_positions = {
        0: {"A": torch.tensor([0, 1]), "B": torch.tensor([2, 3])}
    }

    with torch.no_grad():
        _, single_out = model.forward(
            embeddings,
            attention_mask,
            gold_relations=single,
            gold_entity_positions=gold_entity_positions,
        )
        _, dup_out = model.forward(
            embeddings,
            attention_mask,
            gold_relations=duplicated,
            gold_entity_positions=gold_entity_positions,
        )

    assert single_out is not None and dup_out is not None
    single_meta, single_logits = single_out
    dup_meta, dup_logits = dup_out

    assert single_meta["sequence"].shape[0] == 1
    assert dup_meta["sequence"].shape[0] == 1  # not 2, despite the repeat
    torch.testing.assert_close(dup_logits, single_logits)


def test_gold_representation_is_pooled_from_the_entitys_own_mentions(
    patch_base_model, empty_token_label_store
):
    """A gold argument's representation must come from where *that* entity's
    own mention sits in the document, not from a per-type or learned signal
    that cannot tell two same-type entities apart.

    `enz1` and `enz2` are both entities of the model's one class, mentioned at
    disjoint token positions of the same document. `hidden_layers=[]` keeps
    `self.hidden` a pass-through and `self.relation_classifier` is replaced
    with the identity, so the value `forward` hands back *is* the pooled
    representation, letting this compare it against a hand-computed mean over
    each entity's own positions exactly -- the failure mode the design
    rejected (reading the class head's per-token representation) would give
    both entities the same vector regardless of which positions this test
    names.
    """
    torch.manual_seed(0)
    config = ModelConfig(
        base_model="prajjwal1/bert-mini",
        hidden_layers=[],  # keep `self.hidden` a pass-through
        token_labels_store=str(empty_token_label_store),
    )
    model = ETEBrendaModel(
        schema=SINGLE_CLASS_SCHEMA,
        config=config,
        device="cpu",
    )
    model.eval()
    # Expose the pooled representation itself as `forward`'s output, rather
    # than a biaffine projection of it.
    object.__setattr__(model, "relation_classifier", lambda rep_i, rep_j: rep_i)

    embeddings = torch.randn(1, 6, 256)
    attention_mask = torch.ones(1, 6, dtype=torch.bool)
    enz1_positions = torch.tensor([0, 1])
    enz2_positions = torch.tensor([4, 5])

    gold_relations = [
        IndexedRelation(
            docix=0, subject="enz1", object="enz2", label=torch.tensor(0)
        )
    ]

    with torch.no_grad():
        _, out = model.forward(
            embeddings,
            attention_mask,
            gold_relations=gold_relations,
            gold_entity_positions={
                0: {"enz1": enz1_positions, "enz2": enz2_positions}
            },
        )

    assert out is not None
    _, pooled = out
    torch.testing.assert_close(
        pooled[0], embeddings[0, enz1_positions].mean(dim=0)
    )

    # The two entities share the model's one class, so nothing but their own
    # mention positions could tell their representations apart.
    assert not torch.allclose(
        embeddings[0, enz1_positions].mean(dim=0),
        embeddings[0, enz2_positions].mean(dim=0),
    )

    with torch.no_grad():
        _, swapped_out = model.forward(
            embeddings,
            attention_mask,
            gold_relations=gold_relations,
            gold_entity_positions={
                0: {"enz1": enz2_positions, "enz2": enz1_positions}
            },
        )

    assert swapped_out is not None
    _, swapped_pooled = swapped_out
    assert not torch.allclose(pooled[0], swapped_pooled[0])


# --------------------------------------------------------------------------- #
# Gold relations no candidate pair covers                                     #
#                                                                             #
# The aligner scores only the pairs the detections were paired into, so gold  #
# no pair covers leaves no row. Unless the metrics add it back, it is not a   #
# false negative -- it is absent, and relation F1 is computed over a          #
# denominator the model chose for itself.                                     #
# --------------------------------------------------------------------------- #
def _missed_stub(stub):
    return stub(
        ETEBrendaModel,
        # `evaluate_model` now wraps its loop in
        # `prefetch_layer_boundary_reads`, which reads
        # `config.unfrozen_top_layers` -- 0 either way, so the class name
        # here doesn't matter, only that `config` exists.
        config=ModelConfig(model_class="NERClassificationModel"),
        entity_logits_pooling="logsumexp",
        _argument_groups={
            "A": frozenset({0}),
            "B": frozenset({1}),
            "C": frozenset({2}),
        },
        _argument_sets=(
            frozenset({"A"}),
            frozenset({"B"}),
            frozenset({"C"}),
        ),
        relations=("HasEnzyme", "HasSpecies", "none"),
        relations_none_index=2,
    )


# Every argument of documents 0 and 1 placed in the text, as the label store
# would place it: what separates a pair the detections missed from one no
# detector could have reached.
ANCHORED = {
    docix: {
        entity_id: torch.tensor([position])
        for position, entity_id in enumerate(("A", "B", "C"))
    }
    for docix in (0, 1)
}


HAS_ENZYME, HAS_SPECIES, NONE = 0, 1, 2


def _gold(subject, object, label, docix=0):
    return IndexedRelation(
        docix=docix, subject=subject, object=object, label=torch.tensor(label)
    )


def test_gold_with_a_scored_row_is_not_missed(stub):
    m = _missed_stub(stub)
    # (sequence, arg_pred_i, arg_pred_j) triples, as `unscored_gold_relations`
    # now takes them -- the host-side form the caller reads once.
    scored = [(0, 0, 1)]
    assert m.unscored_gold_relations(
        [_gold("A", "B", HAS_ENZYME)], scored, ANCHORED
    ) == ([], [])


def test_gold_never_proposed_is_missed_even_when_other_pairs_were(stub):
    m = _missed_stub(stub)
    # (A, B) was proposed; (A, C) was not -- one scored row is not licence to
    # forget the other gold relation.
    scored = [(0, 0, 1)]
    not_proposed, no_anchor = m.unscored_gold_relations(
        [_gold("A", "B", HAS_ENZYME), _gold("A", "C", HAS_SPECIES)],
        scored,
        ANCHORED,
    )
    assert not_proposed == [HAS_SPECIES]
    assert no_anchor == []


def test_gold_in_another_document_is_missed(stub):
    m = _missed_stub(stub)
    # Same (subject, object), different document: the row scored for doc 0 says
    # nothing about doc 1.
    scored = [(0, 0, 1)]
    not_proposed, _ = m.unscored_gold_relations(
        [_gold("A", "B", HAS_ENZYME, docix=1)], scored, ANCHORED
    )
    assert not_proposed == [HAS_ENZYME]


def test_gold_the_store_places_nowhere_is_reported_as_having_no_anchor(stub):
    """An argument the store holds no mention of is a miss no detector can
    fix: a proposer reading that store could never have placed the pair, so
    counting it beside the pairs detection merely failed to propose would
    charge span recall for a gap in the dictionary."""
    m = _missed_stub(stub)
    not_proposed, no_anchor = m.unscored_gold_relations(
        [_gold("Z", "B", HAS_ENZYME)], None, ANCHORED
    )
    assert not_proposed == []
    assert no_anchor == [HAS_ENZYME]


def test_gold_in_a_document_the_store_lacks_has_no_anchor_either(stub):
    """The store holds nothing at all for document 2, so neither argument is
    anchored and the pair is not charged to detection."""
    m = _missed_stub(stub)
    not_proposed, no_anchor = m.unscored_gold_relations(
        [_gold("A", "B", HAS_ENZYME, docix=2)], None, ANCHORED
    )
    assert not_proposed == []
    assert no_anchor == [HAS_ENZYME]


def test_every_gold_is_missed_when_nothing_was_scored(stub):
    m = _missed_stub(stub)
    not_proposed, no_anchor = m.unscored_gold_relations(
        [_gold("A", "B", HAS_ENZYME), _gold("A", "C", HAS_SPECIES)],
        None,
        ANCHORED,
    )
    assert not_proposed == [HAS_ENZYME, HAS_SPECIES]
    assert no_anchor == []


def test_gold_repeated_across_pair_dicts_is_missed_once(stub):
    m = _missed_stub(stub)
    # A document carries a list of pair-dicts and the same triple may appear in
    # more than one of them. It could only ever have matched a single candidate
    # row, so it is one miss, not two -- which is how the aligner and the gold
    # rows in `forward` count it.
    not_proposed, no_anchor = m.unscored_gold_relations(
        [_gold("A", "B", HAS_ENZYME), _gold("A", "B", HAS_ENZYME)],
        None,
        ANCHORED,
    )
    assert not_proposed == [HAS_ENZYME]
    assert no_anchor == []


def test_repeated_gold_is_missed_under_its_non_none_label(stub):
    m = _missed_stub(stub)
    # Reversed arguments are the same pair, and the aligner would have labelled
    # the row it built for them non-none: the miss carries the same label.
    not_proposed, _ = m.unscored_gold_relations(
        [_gold("B", "A", NONE), _gold("A", "B", HAS_SPECIES)], None, ANCHORED
    )
    assert not_proposed == [HAS_SPECIES]


def test_unanchored_gold_repeated_is_reported_once(stub):
    m = _missed_stub(stub)
    # Counted per occurrence, repetitions inflate the reported coverage gap and
    # the false-negative total both halves feed.
    not_proposed, no_anchor = m.unscored_gold_relations(
        [_gold("Z", "B", HAS_ENZYME), _gold("Z", "B", HAS_ENZYME)],
        None,
        ANCHORED,
    )
    assert not_proposed == []
    assert no_anchor == [HAS_ENZYME]


def test_repeated_unanchored_gold_keeps_its_non_none_label(stub):
    m = _missed_stub(stub)
    # The string key sorts its arguments, so reversed arguments are one pair
    # here too -- and it keeps the non-none label, since a miss counted as
    # `none` leaves the typed metrics instead of counting against the model.
    _, no_anchor = m.unscored_gold_relations(
        [_gold("B", "Z", NONE), _gold("Z", "B", HAS_SPECIES)], None, ANCHORED
    )
    assert no_anchor == [HAS_SPECIES]


# --------------------------------------------------------------------------- #
# ETEBrendaModel._score_gold_relations (the row/gold mapping is many-to-many) #
#                                                                             #
# Under intersection a wide argument set lets one row cover several gold     #
# pairs, and a narrowed singleton beside the wider set it came from lets     #
# one gold pair be covered by several rows. Neither side may be scored       #
# independently of the other, or a gold relation either drops out or is      #
# counted twice.                                                             #
# --------------------------------------------------------------------------- #
def test_one_row_covering_two_golds_scores_both(stub):
    """One row spans {E}x{B1,B2}: it covers (E, B1) and (E, B2) alike, and
    both have to be scored against it, not just whichever the row's own
    single resolved target used to keep."""
    m = stub(
        ETEBrendaModel,
        relations_none_index=NONE,
        _argument_groups={
            "E": frozenset({0}),
            "B1": frozenset({1}),
            "B2": frozenset({1}),
        },
    )
    gold = [_gold("E", "B1", HAS_ENZYME), _gold("E", "B2", HAS_SPECIES)]
    row_pred_by_key = {(0, 0, 1): HAS_ENZYME}

    matched_true, matched_pred, covered = m._score_gold_relations(
        gold, row_pred_by_key
    )

    assert matched_true == [HAS_ENZYME, HAS_SPECIES]
    # Both are scored against the one row there is, so neither vanishes even
    # though only one of the two labels can possibly be right.
    assert matched_pred == [HAS_ENZYME, HAS_ENZYME]
    assert covered == {(0, 0, 1)}


def test_a_gold_covered_by_two_rows_is_scored_once(stub):
    """{E}, {B1}, {B1,B2}: the narrowed singleton {B1} and the wider set it
    came from both cover the one gold pair (E, B1). It must be scored
    against exactly one of them, preferring whichever got it right, and the
    other row must not add a second entry for the same gold relation."""
    m = stub(
        ETEBrendaModel,
        relations_none_index=NONE,
        _argument_groups={
            "E": frozenset({0}),
            "B1": frozenset({1, 2}),
            "B2": frozenset({2}),
        },
    )
    gold = [_gold("E", "B1", HAS_ENZYME)]
    row_pred_by_key = {(0, 0, 1): HAS_ENZYME, (0, 0, 2): HAS_SPECIES}

    matched_true, matched_pred, covered = m._score_gold_relations(
        gold, row_pred_by_key
    )

    assert matched_true == [HAS_ENZYME]
    assert matched_pred == [HAS_ENZYME]
    # Both covering rows are exempt from the "covers no gold" bucket, even
    # though only the first was picked to carry the gold's own score.
    assert covered == {(0, 0, 1), (0, 0, 2)}


# --------------------------------------------------------------------------- #
# ETEBrendaModel.evaluate_model (the reported test metrics)                    #
# --------------------------------------------------------------------------- #
def _true_x_pred_stub(stub, relation_index_logits, gold, anchored=ANCHORED):
    m = _missed_stub(stub)
    class_logits = torch.zeros(1, 3)
    object.__setattr__(
        m,
        "get_batch_logits",
        lambda batch: (class_logits, relation_index_logits),
    )
    # Stands in for the label-store lookup: which gold arguments the store
    # places in the document, which is what the bookkeeping reads.
    object.__setattr__(
        m, "_gold_entity_positions", lambda batch, relations: anchored
    )
    object.__setattr__(
        m, "ground_truth", lambda batch: (torch.zeros(1, 2), gold)
    )
    return m


def _candidate_pair_favouring_has_enzyme():
    """One candidate row for (doc 0, A, B), predicted HasEnzyme."""
    meta = {
        "sequence": torch.tensor([0]),
        "arg_pred_i": torch.tensor([0]),
        "arg_pred_j": torch.tensor([1]),
    }
    return meta, torch.tensor([[10.0, 0.0, 0.0]])


def _evaluate_stub(stub, relation_index_logits, gold):
    """A model whose only real behaviour is the relation bookkeeping.

    Classes are ``enzyme species OOS``, so `drop_oos` narrows the logits to
    the width the targets carry.
    """
    m = _true_x_pred_stub(stub, relation_index_logits, gold)
    object.__setattr__(m, "eval", lambda: None)
    object.__setattr__(m, "_detection_accumulator", lambda: None)
    object.__setattr__(m, "classes", ["enzyme", "species", "OOS"])
    object.__setattr__(m, "class_columns", torch.tensor([0, 1]))
    object.__setattr__(
        m,
        "ground_truth",
        lambda batch: (torch.tensor([[1.0, 1.0]]), gold),
    )
    return m


def _single_batch_loader():
    """One batch of one (empty) document.

    `evaluate_model` is typed to a real `DataLoader` and beartype enforces it;
    the stubbed `get_batch_logits` and `ground_truth` ignore its contents.
    """
    return DataLoader([{}], batch_size=1, collate_fn=list)


def test_evaluate_scores_unproposed_gold_against_the_model(stub):
    # The head proposes (A, B) and labels it HasEnzyme correctly; the gold
    # HasSpecies pair (A, C) it never proposed at all.
    gold = [_gold("A", "B", HAS_ENZYME), _gold("A", "C", HAS_SPECIES)]
    m = _evaluate_stub(stub, _candidate_pair_favouring_has_enzyme(), gold)

    metrics = m.evaluate_model(_single_batch_loader())

    assert metrics["test/relation_gold"] == 2
    assert metrics["test/relation_missed_not_proposed"] == 1
    assert metrics["test/relation_missed_no_anchor"] == 0

    # The missed relation must reach the score as a false negative: without
    # it, HasEnzyme's correct call alone would put micro-F1 at 1.0.
    assert metrics["test/relation_micro_f1_typed"] == pytest.approx(2 / 3)

    # Both arguments of the scored row are single entities, so the strict rule
    # keeps the same target and the two scores coincide.
    assert metrics["test/relation_argument_set_size"] == 1.0
    assert metrics["test/relation_micro_f1_typed_strict"] == pytest.approx(
        2 / 3
    )


def test_evaluate_reports_gold_when_no_pairs_were_proposed(stub):
    gold = [_gold("A", "B", HAS_ENZYME)]
    m = _evaluate_stub(stub, None, gold)

    metrics = m.evaluate_model(_single_batch_loader())

    assert metrics["test/relation_missed_not_proposed"] == 1
    # A split on which the head proposes nothing scores zero, rather than
    # silently reporting no relations at all.
    assert metrics["test/relation_micro_f1_typed"] == 0.0


def test_evaluate_separates_unanchored_gold_from_unproposed_gold(
    stub,
):
    gold = [_gold("Z", "B", HAS_ENZYME), _gold("A", "C", HAS_SPECIES)]
    m = _evaluate_stub(stub, _candidate_pair_favouring_has_enzyme(), gold)

    metrics = m.evaluate_model(_single_batch_loader())

    assert metrics["test/relation_missed_not_proposed"] == 1
    assert metrics["test/relation_missed_no_anchor"] == 1


def test_evaluate_scores_both_golds_a_shared_row_covers(stub):
    """End to end through `evaluate_model`: one candidate row spans {E}x
    {B1,B2} and gets (E, B1)'s label right but (E, B2)'s wrong. Before the
    fix the second pair was neither the row's target nor a miss -- it
    dropped out, `relation_gold` stopped matching the population the typed
    score was computed over, and the wrong call went unpunished."""
    gold = [_gold("E", "B1", HAS_ENZYME), _gold("E", "B2", HAS_SPECIES)]
    m = _evaluate_stub(stub, _candidate_pair_favouring_has_enzyme(), gold)
    anchored = {
        0: {eid: torch.tensor([i]) for i, eid in enumerate(("E", "B1", "B2"))}
    }
    object.__setattr__(
        m, "_gold_entity_positions", lambda batch, relations: anchored
    )
    object.__setattr__(
        m,
        "_argument_groups",
        {
            "E": frozenset({0}),
            "B1": frozenset({1}),
            "B2": frozenset({1}),
        },
    )
    object.__setattr__(
        m, "_argument_sets", (frozenset({"E"}), frozenset({"B1", "B2"}))
    )

    metrics = m.evaluate_model(_single_batch_loader())

    assert metrics["test/relation_gold"] == 2
    assert metrics["test/relation_candidate_pairs"] == 2
    assert metrics["test/relation_missed_not_proposed"] == 0
    assert metrics["test/relation_missed_no_anchor"] == 0
    # One pair right, one wrong: a correct call alone would put this at 1.0.
    assert metrics["test/relation_micro_f1_typed"] == pytest.approx(0.5)


def test_evaluate_does_not_double_count_a_gold_two_rows_cover(stub):
    """End to end through `evaluate_model`: {E}, {B1}, {B1,B2}. A narrowed
    singleton row and the wider set it came from both cover the same one
    gold relation, one predicting it right and the other wrong. Before the
    fix both rows counted as separate typed instances, turning one gold
    relation into two and inflating `relation_candidate_pairs` past
    `relation_gold`."""
    gold = [_gold("E", "B1", HAS_ENZYME)]
    meta = {
        "sequence": torch.tensor([0, 0]),
        "arg_pred_i": torch.tensor([0, 0]),
        "arg_pred_j": torch.tensor([1, 2]),
    }
    logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    m = _evaluate_stub(stub, (meta, logits), gold)
    anchored = {0: {"E": torch.tensor([0]), "B1": torch.tensor([1])}}
    object.__setattr__(
        m, "_gold_entity_positions", lambda batch, relations: anchored
    )
    object.__setattr__(
        m,
        "_argument_groups",
        {
            "E": frozenset({0}),
            "B1": frozenset({1, 2}),
            "B2": frozenset({2}),
        },
    )
    object.__setattr__(
        m,
        "_argument_sets",
        (frozenset({"E"}), frozenset({"B1"}), frozenset({"B1", "B2"})),
    )

    metrics = m.evaluate_model(_single_batch_loader())

    assert metrics["test/relation_gold"] == 1
    assert metrics["test/relation_candidate_pairs"] == 1
    assert metrics["test/relation_micro_f1_typed"] == pytest.approx(1.0)


def test_evaluate_model_logs_the_relation_report_only_when_asked(stub, caplog):
    """`log_reports` gates the relation `classification_report` in the
    console log the same way it gates `class_report_metrics`'s; the
    one-line relation-count summary logs at INFO regardless."""
    gold = [_gold("A", "B", HAS_ENZYME), _gold("A", "C", HAS_SPECIES)]
    m = _evaluate_stub(stub, _candidate_pair_favouring_has_enzyme(), gold)

    caplog.set_level(logging.INFO)
    m.evaluate_model(_single_batch_loader(), log_reports=False)
    assert "[Relations]" in caplog.text
    assert "precision" not in caplog.text

    caplog.clear()
    m.evaluate_model(_single_batch_loader(), log_reports=True)
    assert "precision" in caplog.text


def _count_tolist_calls(model, loader) -> int:
    """How many `Tensor.tolist` reads one `evaluate_model` run makes."""
    original = torch.Tensor.tolist
    calls = 0

    def counting(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original(self, *args, **kwargs)

    torch.Tensor.tolist = counting
    try:
        model.evaluate_model(loader)
    finally:
        torch.Tensor.tolist = original
    return calls


def test_evaluate_reads_the_scored_meta_to_the_host_once_per_batch(stub):
    """The scored meta must be read to the host once, not eight times.

    Comparing a run with a scored candidate row against one with none isolates
    the reads the meta itself costs from every other `.tolist()` call
    `evaluate_model` makes regardless (e.g. `known_classes`), which fire
    identically either way and cancel out of the difference. Before the fix,
    `unscored_gold_relations`, `_strict_relation_targets` and the
    argument-set-size loop each re-fetched the meta's three columns off the
    device, an eight-call difference; the meta is shared host-side now.
    """
    gold = [_gold("A", "B", HAS_ENZYME), _gold("A", "C", HAS_SPECIES)]

    without_pairs = _evaluate_stub(stub, None, gold)
    with_pairs = _evaluate_stub(
        stub, _candidate_pair_favouring_has_enzyme(), gold
    )

    baseline_calls = _count_tolist_calls(without_pairs, _single_batch_loader())
    scored_calls = _count_tolist_calls(with_pairs, _single_batch_loader())

    assert scored_calls - baseline_calls == 1


# --------------------------------------------------------------------------- #
# Relation-loss class weighting                                                #
# --------------------------------------------------------------------------- #
def _relation_loss_stub(stub, weighting):
    return stub(
        ETEBrendaModel,
        device="cpu",
        entity_logits_pooling="logsumexp",
        _argument_groups={"A": frozenset({0}), "B": frozenset({1})},
        relations_none_index=2,
        num_relations=3,
        relation_label_smoothing=0.0,
        relation_loss_weighting=weighting,
        relation_focal_gamma=2.0,
    )


def _imbalanced_pairs(n_none):
    """One mispredicted positive plus `n_none` confidently-correct pairs.

    Mimics what a document full of detected spans proposes: a flood of easy
    negatives around sparse gold. Every triple is distinct, so alignment pools
    them 1:1.
    """
    gold = [
        IndexedRelation(docix=0, subject="A", object="B", label=torch.tensor(0))
    ]
    meta = {
        "sequence": torch.zeros(n_none + 1, dtype=torch.long),
        "arg_pred_i": torch.tensor([0] + [k + 2 for k in range(n_none)]),
        "arg_pred_j": torch.tensor([1] + [k + 3 for k in range(n_none)]),
    }
    logits = torch.tensor(
        [[-6.0, 0.0, 6.0]]  # gold "HasEnzyme", confidently called `none`
        + [[-6.0, -6.0, 6.0]] * n_none  # `none`, confidently correct
    )
    return gold, meta, logits


def test_unweighted_relation_loss_is_diluted_by_none_pairs(stub):
    """The smell itself: the same mistake on the same gold relation costs the
    model ~8x less once the mask floods the batch with easy negatives."""
    m = _relation_loss_stub(stub, "unweighted")
    few = m.compute_relation_loss(*_imbalanced_pairs(3))
    many = m.compute_relation_loss(*_imbalanced_pairs(30))
    assert many < few / 5


@pytest.mark.parametrize("weighting", ("balanced", "focal"))
def test_weighting_keeps_the_positive_from_being_diluted(stub, weighting):
    m = _relation_loss_stub(stub, weighting)
    few = m.compute_relation_loss(*_imbalanced_pairs(3))
    many = m.compute_relation_loss(*_imbalanced_pairs(30))
    assert torch.isclose(few, many, rtol=0.02)


def test_relation_loss_weighting_defaults_to_unweighted():
    assert (
        ModelConfig(
            token_labels_store="/fake/store.hdf5"
        ).relation_loss_weighting
        == "unweighted"
    )


def test_relation_loss_weighting_rejects_an_unknown_scheme():
    with pytest.raises(ValidationError):
        ModelConfig(relation_loss_weighting="bogus")


# --------------------------------------------------------------------------- #
# ETEBrendaModel.ground_truth (relation loop)                                  #
# --------------------------------------------------------------------------- #
def test_ground_truth_builds_indexed_relation_from_argmax(stub):
    m = stub(
        ETEBrendaModel,
        device="cpu",
        two_head=stub(BrendaClassificationModel, device="cpu"),
    )
    batch = [
        {
            "classes": torch.tensor([1, 0]),
            "relations": [{("A", "B"): torch.tensor([0, 1, 0])}],  # argmax == 1
        }
    ]
    _, relations = m.ground_truth(batch)
    assert len(relations) == 1
    rel = relations[0]
    assert (rel.docix, rel.subject, rel.object) == (0, "A", "B")
    assert int(rel.label) == 1


def test_ground_truth_reads_every_relations_dict_of_a_document(stub):
    m = stub(
        ETEBrendaModel,
        device="cpu",
        two_head=stub(BrendaClassificationModel, device="cpu"),
    )
    batch = [
        {
            "classes": torch.tensor([1, 0]),
            "relations": [
                {("A", "B"): torch.tensor([0, 1, 0])},
                {("C", "D"): torch.tensor([1, 0, 0])},
            ],
        }
    ]
    _, relations = m.ground_truth(batch)
    assert {(r.subject, r.object, int(r.label)) for r in relations} == {
        ("A", "B", 1),
        ("C", "D", 0),
    }


def test_ground_truth_yields_no_relations_for_an_empty_relations_list(stub):
    m = stub(
        ETEBrendaModel,
        device="cpu",
        two_head=stub(BrendaClassificationModel, device="cpu"),
    )
    batch = [
        {
            "classes": torch.tensor([1, 0]),
            "relations": [],
        }
    ]
    _, relations = m.ground_truth(batch)
    assert relations == []


def test_ground_truth_yields_no_relations_for_empty_dict(stub):
    m = stub(
        ETEBrendaModel,
        device="cpu",
        two_head=stub(BrendaClassificationModel, device="cpu"),
    )
    batch = [
        {
            "classes": torch.tensor([1, 0]),
            "relations": [{}],
        }
    ]
    _, relations = m.ground_truth(batch)
    assert relations == []
