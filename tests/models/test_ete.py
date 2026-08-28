"""Pure unit tests for `d3text.models.ete.ETEBrendaModel` — the relation-loss
ramp, config wiring into the relation classifier, relation alignment and its
bookkeeping of gold relations the entity head never proposed, the reported
evaluation metrics, relation-loss class weighting, the vectorised relation
candidate builder, and the relation-loop half of `ground_truth`.

Every test here runs on CPU with tiny synthetic tensors and no data, network,
or GPU (bar the handful using `patch_base_model`, which swaps in a tiny random
BERT rather than downloading one). Methods are exercised through the `stub`
fixture (see `tests/conftest.py`), which supplies only the attributes each
method reads.
"""

import logging

import pytest
import torch
from torch.utils.data import DataLoader
from pydantic import ValidationError

from d3text import logs
from d3text.models.config import ModelConfig
from d3text.models.entity_linking import BrendaClassificationModel
from d3text.models.ete import ETEBrendaModel
from d3text.models.heads import BiaffineRelationClassifier
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
    assert parent.epoch_loss_weights(epoch) == {"entity": 1.0, "class": 1.0}

    ete = stub(ETEBrendaModel, ramp_epochs=4)
    assert ete.epoch_loss_weights(epoch) == {
        "entity": 1.0,
        "class": 1.0,
        "relation": pytest.approx(0.55),
    }


# --------------------------------------------------------------------------- #
# ModelConfig knobs reaching the ETE model's relation classifier               #
# --------------------------------------------------------------------------- #
def test_config_knobs_reach_the_ete_model(patch_base_model):
    """entity_entropy_threshold and biaffine_hidden_size are ModelConfig fields
    that must reach the entropy-mask cutoff and the relation classifier's
    projection width, rather than the former hardcoded 0.8 / 32."""
    model = ETEBrendaModel(
        schema=SCHEMA,
        class_matrix=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        entity_index={"enz1": 0, "bac1": 1},
        config=ModelConfig(
            base_model="prajjwal1/bert-mini",
            hidden_layers=[8],
            entity_entropy_threshold=0.5,
            biaffine_hidden_size=16,
        ),
        device="cpu",
    )
    assert model.entity_threshold == 0.5
    assert tuple(model.relation_classifier.bilinear.shape) == (3, 16, 16)


def test_separate_predicate_layer_reaches_the_relation_classifier(
    patch_base_model,
):
    """ModelConfig.separate_predicate_layer must reach the biaffine
    classifier's constructor: with it set, the x/y projections are two
    distinct modules rather than the same one aliased under both names."""
    model = ETEBrendaModel(
        schema=SCHEMA,
        class_matrix=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        entity_index={"enz1": 0, "bac1": 1},
        config=ModelConfig(
            base_model="prajjwal1/bert-mini",
            hidden_layers=[8],
            separate_predicate_layer=True,
        ),
        device="cpu",
    )
    assert (
        model.relation_classifier.hidden_linear_y
        is not model.relation_classifier.hidden_linear
    )


# --------------------------------------------------------------------------- #
# ETEBrendaModel.align_relation_predictions                                    #
# --------------------------------------------------------------------------- #
def _align_stub(stub):
    return stub(
        ETEBrendaModel,
        entity_logits_pooling="logsumexp",
        entity_to_index={"A": 0, "B": 1},
        relations_none_index=2,
    )


def _rel_meta():
    # two candidate rows for the same (doc=0, subj=0, obj=1) triple
    return {
        "sequence": torch.tensor([0, 0]),
        "arg_pred_i": torch.tensor([0, 0]),
        "arg_pred_j": torch.tensor([1, 1]),
    }


def test_align_pools_duplicate_rows_and_uses_gold_label(stub):
    m = _align_stub(stub)
    rel_logits = torch.randn(2, 3)
    gold = [
        IndexedRelation(docix=0, subject="A", object="B", label=torch.tensor(0))
    ]
    meta, pooled_logits, targets = m.align_relation_predictions(
        gold, _rel_meta(), rel_logits
    )
    assert pooled_logits.shape[0] == 1  # two rows pooled into one
    assert pooled_logits.shape[1] == 3  # relation width preserved
    assert targets.tolist() == [0]  # gold "HasEnzyme"
    assert meta["arg_pred_i"].tolist() == [0]
    assert meta["arg_pred_j"].tolist() == [1]


def test_align_defaults_to_none_when_gold_entity_not_indexed(stub):
    m = _align_stub(stub)
    # subject "Z" is absent from entity_to_index -> gold is dropped
    gold = [
        IndexedRelation(docix=0, subject="Z", object="B", label=torch.tensor(0))
    ]
    _, _, targets = m.align_relation_predictions(
        gold, _rel_meta(), torch.randn(2, 3)
    )
    assert targets.tolist() == [2]  # relations_none_index


def test_align_returns_none_for_empty_logits(stub):
    m = _align_stub(stub)
    assert m.align_relation_predictions([], _rel_meta(), None) is None


def test_forward_dedups_repeated_gold_relation_pairs(patch_base_model):
    """A `(subject, object)` pair named in two of a document's relation dicts
    must reach the biaffine classifier as one gold row, not two -- otherwise
    the default logsumexp pooling adds a spurious +log(2) to that pair's
    logits, exactly what the hard/gold merge above this branch exists to
    avoid."""
    torch.manual_seed(0)
    entity_index = {"A": 0, "B": 1}
    config = ModelConfig(
        base_model="prajjwal1/bert-mini",
        hidden_layers=[8],
        entity_entropy_threshold=0.0,  # keep the hard-mask path silent
    )
    model = ETEBrendaModel(
        schema=SINGLE_CLASS_SCHEMA,
        class_matrix=torch.tensor([[1.0], [1.0]]),
        entity_index=entity_index,
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

    with torch.no_grad():
        _, _, single_out = model.forward(
            embeddings, attention_mask, gold_relations=single
        )
        _, _, dup_out = model.forward(
            embeddings, attention_mask, gold_relations=duplicated
        )

    assert single_out is not None and dup_out is not None
    single_meta, single_logits = single_out
    dup_meta, dup_logits = dup_out

    assert single_meta["sequence"].shape[0] == 1
    assert dup_meta["sequence"].shape[0] == 1  # not 2, despite the repeat
    torch.testing.assert_close(dup_logits, single_logits)


# --------------------------------------------------------------------------- #
# Gold relations the entity head never proposed                                #
#                                                                              #
# The aligner scores only the pairs the entity head proposed, so gold it never #
# proposed leaves no row. Unless the metrics add it back, it is not a false    #
# negative -- it is absent, and relation F1 is computed over a denominator the #
# model chose for itself.                                                      #
# --------------------------------------------------------------------------- #
def _missed_stub(stub):
    return stub(
        ETEBrendaModel,
        entity_logits_pooling="logsumexp",
        entity_to_index={"A": 0, "B": 1, "C": 2},
        relations=("HasEnzyme", "HasSpecies", "none"),
        relations_none_index=2,
    )


HAS_ENZYME, HAS_SPECIES, NONE = 0, 1, 2


def _gold(subject, object, label, docix=0):
    return IndexedRelation(
        docix=docix, subject=subject, object=object, label=torch.tensor(label)
    )


def test_gold_with_a_scored_row_is_not_missed(stub):
    m = _missed_stub(stub)
    scored = {
        "sequence": torch.tensor([0]),
        "arg_pred_i": torch.tensor([0]),
        "arg_pred_j": torch.tensor([1]),
    }
    assert m.unscored_gold_relations([_gold("A", "B", HAS_ENZYME)], scored) == (
        [],
        [],
    )


def test_gold_never_proposed_is_missed_even_when_other_pairs_were(stub):
    m = _missed_stub(stub)
    # (A, B) was proposed; (A, C) was not -- one scored row is not licence to
    # forget the other gold relation.
    scored = {
        "sequence": torch.tensor([0]),
        "arg_pred_i": torch.tensor([0]),
        "arg_pred_j": torch.tensor([1]),
    }
    not_proposed, out_of_vocabulary = m.unscored_gold_relations(
        [_gold("A", "B", HAS_ENZYME), _gold("A", "C", HAS_SPECIES)], scored
    )
    assert not_proposed == [HAS_SPECIES]
    assert out_of_vocabulary == []


def test_gold_in_another_document_is_missed(stub):
    m = _missed_stub(stub)
    # Same (subject, object), different document: the row scored for doc 0 says
    # nothing about doc 1.
    scored = {
        "sequence": torch.tensor([0]),
        "arg_pred_i": torch.tensor([0]),
        "arg_pred_j": torch.tensor([1]),
    }
    not_proposed, _ = m.unscored_gold_relations(
        [_gold("A", "B", HAS_ENZYME, docix=1)], scored
    )
    assert not_proposed == [HAS_ENZYME]


def test_gold_with_unindexed_entity_is_reported_out_of_vocabulary(stub):
    m = _missed_stub(stub)
    # "Z" is absent from entity_to_index, so no relation head could ever
    # predict this pair: a real miss, but not one the relation head can fix.
    not_proposed, out_of_vocabulary = m.unscored_gold_relations(
        [_gold("Z", "B", HAS_ENZYME)], None
    )
    assert not_proposed == []
    assert out_of_vocabulary == [HAS_ENZYME]


def test_every_gold_is_missed_when_nothing_was_scored(stub):
    m = _missed_stub(stub)
    not_proposed, out_of_vocabulary = m.unscored_gold_relations(
        [_gold("A", "B", HAS_ENZYME), _gold("A", "C", HAS_SPECIES)], None
    )
    assert not_proposed == [HAS_ENZYME, HAS_SPECIES]
    assert out_of_vocabulary == []


# --------------------------------------------------------------------------- #
# ETEBrendaModel.compute_batch_true_x_pred (the validation path)               #
# --------------------------------------------------------------------------- #
def _true_x_pred_stub(stub, relation_index_logits, gold):
    m = _missed_stub(stub)
    entity_logits = torch.zeros(1, 4)
    class_logits = torch.zeros(1, 3)
    object.__setattr__(
        m,
        "get_batch_logits",
        lambda batch: (entity_logits, class_logits, relation_index_logits),
    )
    object.__setattr__(
        m,
        "ground_truth",
        lambda batch: (torch.zeros(1, 3), torch.zeros(1, 2), gold),
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


def test_true_x_pred_counts_unproposed_gold_as_a_false_negative(stub):
    gold = [_gold("A", "B", HAS_ENZYME), _gold("A", "C", HAS_SPECIES)]
    m = _true_x_pred_stub(stub, _candidate_pair_favouring_has_enzyme(), gold)

    relations = m.compute_batch_true_x_pred([{}])["relations"]

    # The proposed pair is scored on its logits; the unproposed one counts as
    # `none`, rather than disappearing because it has no row.
    assert relations["true"].tolist() == [HAS_ENZYME, HAS_SPECIES]
    assert relations["pred"].tolist() == [HAS_ENZYME, NONE]


def test_true_x_pred_counts_out_of_vocabulary_gold_as_a_false_negative(stub):
    gold = [_gold("A", "B", HAS_ENZYME), _gold("Z", "B", HAS_SPECIES)]
    m = _true_x_pred_stub(stub, _candidate_pair_favouring_has_enzyme(), gold)

    relations = m.compute_batch_true_x_pred([{}])["relations"]

    assert relations["true"].tolist() == [HAS_ENZYME, HAS_SPECIES]
    assert relations["pred"].tolist() == [HAS_ENZYME, NONE]


def test_true_x_pred_counts_all_gold_when_no_pairs_were_proposed(stub):
    gold = [_gold("A", "B", HAS_ENZYME), _gold("A", "C", HAS_SPECIES)]
    m = _true_x_pred_stub(stub, None, gold)

    relations = m.compute_batch_true_x_pred([{}])["relations"]

    assert relations["true"].tolist() == [HAS_ENZYME, HAS_SPECIES]
    assert relations["pred"].tolist() == [NONE, NONE]


# --------------------------------------------------------------------------- #
# ETEBrendaModel.evaluate_model (the reported test metrics)                    #
# --------------------------------------------------------------------------- #
def _evaluate_stub(stub, relation_index_logits, gold):
    """A model whose only real behaviour is the relation bookkeeping.

    Entities are ``A B C UNK`` and classes ``enzyme species OOS``, so `drop_unk`
    and `drop_oos` narrow the logits to the width the targets carry.
    """
    m = _true_x_pred_stub(stub, relation_index_logits, gold)
    object.__setattr__(m, "eval", lambda: None)
    object.__setattr__(m, "_detection_accumulator", lambda: None)
    object.__setattr__(m, "classes", ["enzyme", "species", "OOS"])
    object.__setattr__(m, "entity_columns", torch.tensor([0, 1, 2]))
    object.__setattr__(m, "class_columns", torch.tensor([0, 1]))
    object.__setattr__(
        m,
        "ground_truth",
        lambda batch: (
            torch.tensor([[1.0, 1.0, 0.0]]),
            torch.tensor([[1.0, 1.0]]),
            gold,
        ),
    )
    return m


def _single_batch_loader():
    """One batch of one (empty) document.

    `evaluate_model` is typed to a real `DataLoader` and beartype enforces it,
    so the batch has to arrive through one; the stubbed `get_batch_logits` and
    `ground_truth` ignore its contents.
    """
    return DataLoader([{}], batch_size=1, collate_fn=list)


def _relation_report_row(output, label):
    row = next(
        line
        for line in output.splitlines()
        if line.strip().startswith(f"{label} ")
    )
    _, precision, recall, f1, support = row.split()
    # sklearn formats support as an int or a float depending on the report.
    return float(precision), float(recall), float(f1), int(float(support))


@pytest.fixture
def console(restore_package_logger, capsys):
    """The evaluation's report reaches the console through the package logger,
    not `print`, so these assertions need a configured handler; without one
    they pass only when an earlier test happens to have installed it."""
    logs.configure(logging.INFO)
    return capsys


def test_evaluate_scores_unproposed_gold_against_the_model(stub, console):
    # The head proposes (A, B) and labels it HasEnzyme correctly; the gold
    # HasSpecies pair (A, C) it never proposed at all.
    gold = [_gold("A", "B", HAS_ENZYME), _gold("A", "C", HAS_SPECIES)]
    m = _evaluate_stub(stub, _candidate_pair_favouring_has_enzyme(), gold)

    m.evaluate_model(_single_batch_loader())

    out = console.readouterr().out
    assert "gold: 2" in out
    assert "missed, never proposed: 1" in out

    # The missed relation must reach the report as a false negative: without it
    # the model scores a perfect HasEnzyme and HasSpecies is simply not there.
    _, recall, _, support = _relation_report_row(out, "HasSpecies")
    assert support == 1
    assert recall == 0.0


def test_evaluate_reports_gold_when_no_pairs_were_proposed(stub, console):
    gold = [_gold("A", "B", HAS_ENZYME)]
    m = _evaluate_stub(stub, None, gold)

    m.evaluate_model(_single_batch_loader())

    out = console.readouterr().out
    assert "missed, never proposed: 1" in out
    # A split on which the head proposes nothing scores zero, rather than
    # silently reporting no relations at all.
    _, recall, _, support = _relation_report_row(out, "HasEnzyme")
    assert support == 1
    assert recall == 0.0


def test_evaluate_separates_out_of_vocabulary_gold_from_unproposed_gold(
    stub, console
):
    gold = [_gold("Z", "B", HAS_ENZYME), _gold("A", "C", HAS_SPECIES)]
    m = _evaluate_stub(stub, _candidate_pair_favouring_has_enzyme(), gold)

    m.evaluate_model(_single_batch_loader())

    out = console.readouterr().out
    assert "missed, never proposed: 1" in out
    assert "missed, entity out of vocabulary: 1" in out


# --------------------------------------------------------------------------- #
# Relation-loss class weighting                                                #
# --------------------------------------------------------------------------- #
def _relation_loss_stub(stub, weighting):
    return stub(
        ETEBrendaModel,
        device="cpu",
        entity_logits_pooling="logsumexp",
        entity_to_index={"A": 0, "B": 1},
        relations_none_index=2,
        num_relations=3,
        relation_label_smoothing=0.0,
        relation_loss_weighting=weighting,
        relation_focal_gamma=2.0,
    )


def _imbalanced_pairs(n_none):
    """One mispredicted positive plus `n_none` confidently-correct `none` pairs.

    Mimics what the entropy hard mask actually proposes: a flood of easy
    negatives around the sparse gold relations. Every triple is distinct, so
    alignment pools them 1:1 and the loss sees exactly these rows.
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
    assert ModelConfig().relation_loss_weighting == "unweighted"


def test_relation_loss_weighting_rejects_an_unknown_scheme():
    with pytest.raises(ValidationError):
        ModelConfig(relation_loss_weighting="bogus")


# --------------------------------------------------------------------------- #
# ETEBrendaModel._compute_relations_vectorized                                 #
# --------------------------------------------------------------------------- #
def _relations_stub(stub):
    return stub(
        ETEBrendaModel,
        device="cpu",
        relation_classifier=BiaffineRelationClassifier(
            hidden_size=8, num_relations=3
        ),
    )


def test_compute_relations_one_pair_for_two_distinct_entities(stub):
    m = _relations_stub(stub)
    positions = torch.tensor(
        [[0, 0], [0, 1]], dtype=torch.int64
    )  # doc 0, tokens 0/1
    reprs = torch.randn(2, 8)
    max_indices = torch.tensor(
        [[5, 7]], dtype=torch.int64
    )  # token 0->5, token 1->7
    meta, logits = m._compute_relations_vectorized(
        positions, reprs, max_indices
    )
    assert tuple(logits.shape) == (1, 3)
    assert meta["arg_pred_i"].tolist() == [5]
    assert meta["arg_pred_j"].tolist() == [7]


def test_compute_relations_none_for_single_entity(stub):
    m = _relations_stub(stub)
    positions = torch.tensor([[0, 0], [0, 1]], dtype=torch.int64)
    reprs = torch.randn(2, 8)
    max_indices = torch.tensor(
        [[5, 5]], dtype=torch.int64
    )  # both tokens -> entity 5
    assert (
        m._compute_relations_vectorized(positions, reprs, max_indices) is None
    )


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
            "entities": torch.tensor([1, 0]),
            "classes": torch.tensor([1, 0]),
            "relations": [{("A", "B"): torch.tensor([0, 1, 0])}],  # argmax == 1
        }
    ]
    _, _, relations = m.ground_truth(batch)
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
            "entities": torch.tensor([1, 0]),
            "classes": torch.tensor([1, 0]),
            "relations": [
                {("A", "B"): torch.tensor([0, 1, 0])},
                {("C", "D"): torch.tensor([1, 0, 0])},
            ],
        }
    ]
    _, _, relations = m.ground_truth(batch)
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
            "entities": torch.tensor([1, 0]),
            "classes": torch.tensor([1, 0]),
            "relations": [],
        }
    ]
    _, _, relations = m.ground_truth(batch)
    assert relations == []


def test_ground_truth_yields_no_relations_for_empty_dict(stub):
    m = stub(
        ETEBrendaModel,
        device="cpu",
        two_head=stub(BrendaClassificationModel, device="cpu"),
    )
    batch = [
        {
            "entities": torch.tensor([1, 0]),
            "classes": torch.tensor([1, 0]),
            "relations": [{}],
        }
    ]
    _, _, relations = m.ground_truth(batch)
    assert relations == []
