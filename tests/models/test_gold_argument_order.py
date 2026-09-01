"""Gold relation argument order versus candidate pair order.

Candidate pairs arrive in ascending entity-head column order from
`torch.combinations`; gold arrives sorted lexicographically by entity-ID
string. The two disagree for every HasSpecies pair — `bac…` sorts before `str…`
while the strain columns precede the bacteria ones — so a join on the raw
triple could never match such gold. Everything here uses a vocabulary where the
gold subject's column exceeds its object's.
"""

import pytest
import torch
from d3text.models.config import ModelConfig
from d3text.models.ete import ETEBrendaModel
from d3text.models.model_types import IndexedRelation
from d3text.schema import EntityType, RelationType, Schema

_HAS_SPECIES = 0
_NONE_INDEX = 1

# Column order: the strain before the bacterium, as in the BRENDA schema.
_ENTITY_TO_INDEX = {"str1": 0, "bac1": 1}


def _gold_has_species():
    # Argument order as preprocessing stores it: lexicographic, so the
    # bacterium (the higher column) comes first.
    return [
        IndexedRelation(
            docix=0,
            subject="bac1",
            object="str1",
            label=torch.tensor(_HAS_SPECIES),
        )
    ]


def _candidate_meta():
    # The one candidate pair, in the ascending column order
    # `torch.combinations` emits.
    return {
        "sequence": torch.tensor([0]),
        "arg_pred_i": torch.tensor([0]),
        "arg_pred_j": torch.tensor([1]),
    }


def _aligner_model(stub):
    return stub(
        ETEBrendaModel,
        entity_logits_pooling="logsumexp",
        entity_to_index=_ENTITY_TO_INDEX,
        relations_none_index=_NONE_INDEX,
    )


def test_align_scores_gold_whose_string_order_reverses_column_order(stub):
    model = _aligner_model(stub)

    _, _, targets = model.align_relation_predictions(
        _gold_has_species(), _candidate_meta(), torch.randn(1, 2)
    )

    assert targets.tolist() == [_HAS_SPECIES]


def test_unscored_gold_ignores_a_scored_column_reversed_pair(stub):
    model = _aligner_model(stub)

    not_proposed, out_of_vocabulary = model.unscored_gold_relations(
        _gold_has_species(), _candidate_meta()
    )

    assert not_proposed == []
    assert out_of_vocabulary == []


@pytest.fixture
def strain_species_ete(patch_base_model):
    schema = Schema(
        entity_types=(
            EntityType(name="strains", prefix="str"),
            EntityType(name="bacteria", prefix="bac"),
        ),
        relation_types=(
            RelationType(
                name="HasSpecies",
                subject_types=("strains",),
                object_type="bacteria",
            ),
            RelationType(name="none", is_none=True),
        ),
    )
    model = ETEBrendaModel(
        schema=schema,
        class_matrix=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        entity_index=dict(_ENTITY_TO_INDEX),
        config=ModelConfig(
            base_model="prajjwal1/bert-mini", hidden_layers=[8], ramp_epochs=0
        ),
        device="cpu",
    )
    model.eval()
    return model


def test_forward_emits_gold_rows_in_column_order(strain_species_ete):
    """The gold-path rows must carry the same argument order as the
    hard-mask candidates, or the merge dedup and the aligner treat one pair
    as two, supervising it with contradictory targets."""
    torch.manual_seed(0)
    embeddings = torch.randn(1, 10, 256)
    mask = torch.ones(1, 10, dtype=torch.bool)

    with torch.no_grad():
        *_, rel = strain_species_ete(
            embeddings, mask, gold_relations=_gold_has_species()
        )

    assert rel is not None
    meta, _ = rel
    rows = set(
        zip(
            meta["sequence"].tolist(),
            meta["arg_pred_i"].tolist(),
            meta["arg_pred_j"].tolist(),
        )
    )
    assert (0, 0, 1) in rows
    assert (0, 1, 0) not in rows
