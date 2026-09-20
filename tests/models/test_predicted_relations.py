"""Reading the relation head's own rows back, with no gold to align against.

`compute_batch_true_x_pred` and `evaluate_model` both put the head's rows
through the aligner, which needs gold. Inference has none, so the only thing
left is the argmax and the batch's interning table — and the table is what
says which candidate sets the two argument ids stand for.
"""

import torch
from d3text.models.ete import ETEBrendaModel, PredictedRelation
from d3text.models.model_types import BatchLogits
from d3text.schema import EntityType, RelationType, Schema

SCHEMA = Schema(
    entity_types=(
        EntityType(name="bacteria", prefix="bac"),
        EntityType(name="enzymes", prefix="enz"),
    ),
    relation_types=(
        RelationType(
            name="produces", subject_types=("bacteria",), object_type="enzymes"
        ),
        RelationType(name="none", is_none=True),
    ),
)

PRODUCES, NONE = 0, 1

ARGUMENTS = (
    frozenset({"bac1"}),
    frozenset({"enz7", "enz9"}),
    frozenset({"enz3"}),
)


def _model(stub, rows: list[tuple[int, int, int]], labels: list[int]):
    """An `ETEBrendaModel` whose forward is already done: `rows` scored, each
    one's argmax landing on the matching entry of `labels`."""
    meta = {
        "sequence": torch.tensor([row[0] for row in rows]),
        "arg_pred_i": torch.tensor([row[1] for row in rows]),
        "arg_pred_j": torch.tensor([row[2] for row in rows]),
    }
    logits = torch.zeros(len(rows), len(SCHEMA.relation_names))
    for position, label in enumerate(labels):
        logits[position, label] = 9.0

    return stub(
        ETEBrendaModel,
        schema=SCHEMA,
        relations_none_index=NONE,
        _argument_sets=ARGUMENTS,
        get_batch_logits=lambda batch: BatchLogits(
            torch.zeros(1, 2), (meta, logits)
        ),
    )


def test_a_labelled_pair_carries_its_predicate_and_both_candidate_sets(stub):
    """An argument is the whole set its span grounded to: collapsing it onto
    one id here would make the choice the grounding rule refuses to make."""
    model = _model(stub, rows=[(0, 0, 1)], labels=[PRODUCES])

    assert model.predicted_relations([]) == [
        PredictedRelation(
            predicate="produces",
            arguments=(ARGUMENTS[0], ARGUMENTS[1]),
        )
    ]


def test_a_pair_the_head_called_null_is_not_a_prediction(stub):
    """Every admitted pair of a document's arguments is scored, so most rows
    come back null; writing them out would report a relation per pairing."""
    model = _model(stub, rows=[(0, 0, 1), (0, 0, 2)], labels=[NONE, PRODUCES])

    assert [row.arguments[1] for row in model.predicted_relations([])] == [
        ARGUMENTS[2]
    ]


def test_pairs_the_head_called_all_null_are_still_an_answer(stub):
    """`[]` says the head scored this document's pairs and rejected every
    one, which is a finding; `None` says it was put none at all. Collapsing
    the two loses the only evidence the head ran."""
    model = _model(stub, rows=[(0, 0, 1), (0, 0, 2)], labels=[NONE, NONE])

    assert model.predicted_relations([]) == []


def test_a_batch_that_proposed_no_pair_makes_no_claim(stub):
    """`forward` returns no relation candidates at all for a document with
    fewer than two grounded arguments -- not an error, and not the head
    having looked at pairs and rejected them, which is what an empty list
    would say."""
    model = stub(
        ETEBrendaModel,
        schema=SCHEMA,
        relations_none_index=NONE,
        _argument_sets=(),
        get_batch_logits=lambda batch: BatchLogits(torch.zeros(1, 2), None),
    )

    assert model.predicted_relations([]) is None
