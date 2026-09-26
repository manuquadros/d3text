"""Gold relation argument order versus candidate pair order.

A candidate pair carries its two candidate-set ids ascending, since that is the
order the pairing builds them in; gold arrives sorted lexicographically by
entity-ID string. The two disagree whenever the lexicographic order reverses the
order the two sets were interned in — `bac…` sorts before `str…` while a strain
detected first is interned first — so a join on the raw gold order could never
match such a pair. Everything here uses a document where it reverses.
"""

import pytest
import torch
from d3text.models.config import ModelConfig
from d3text.models.ete import ETEBrendaModel
from d3text.models.model_types import IndexedRelation
from d3text.schema import EntityType, RelationType, Schema

_HAS_SPECIES = 0
_NONE_INDEX = 1

# Interning order: the strain's set before the bacterium's.
_ARGUMENT_GROUPS = {"str1": frozenset({0}), "bac1": frozenset({1})}


def _gold_has_species():
    # Argument order as preprocessing stores it: lexicographic, so the
    # bacterium — the later-interned set — comes first.
    return [
        IndexedRelation(
            docix=0,
            subject="bac1",
            object="str1",
            label=torch.tensor(_HAS_SPECIES),
        )
    ]


def _candidate_meta():
    # The one candidate pair, in the ascending argument-id order the pairing
    # emits.
    return {
        "sequence": torch.tensor([0]),
        "arg_pred_i": torch.tensor([0]),
        "arg_pred_j": torch.tensor([1]),
    }


def _anchored():
    """Both arguments placed in the document, as the label store would."""
    return {0: {"str1": torch.tensor([0, 1]), "bac1": torch.tensor([5, 6])}}


def _aligner_model(stub):
    return stub(
        ETEBrendaModel,
        entity_logits_pooling="logsumexp",
        _argument_groups=_ARGUMENT_GROUPS,
        relations_none_index=_NONE_INDEX,
    )


def test_align_scores_gold_whose_string_order_reverses_argument_order(stub):
    model = _aligner_model(stub)

    _, _, targets = model.align_relation_predictions(
        _gold_has_species(), _candidate_meta(), torch.randn(1, 2)
    )

    assert targets.tolist() == [_HAS_SPECIES]


def test_unscored_gold_ignores_a_scored_order_reversed_pair(stub):
    model = _aligner_model(stub)

    # `unscored_gold_relations` takes the pooled meta as host-side
    # (sequence, arg_pred_i, arg_pred_j) triples, matching `_candidate_meta`.
    not_proposed, no_anchor = model.unscored_gold_relations(
        _gold_has_species(), [(0, 0, 1)], _anchored()
    )

    assert not_proposed == []
    assert no_anchor == []


@pytest.fixture
def strain_species_ete(patch_base_model, empty_token_label_store):
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
        config=ModelConfig(
            base_model="prajjwal1/bert-mini",
            hidden_layers=[8],
            ramp_epochs=0,
            token_supervision=True,
        ),
        device="cpu",
    )
    model.eval()
    return model


def test_the_aligner_finds_the_gold_row_forward_emitted(strain_species_ete):
    """End to end: the row `forward` builds for a gold pair has to be the row
    the aligner then labels.

    A disagreement over argument order between the two sides costs the pair its
    label — it is supervised and scored as `none` instead — silently, and in the
    one direction that reads as a model which learnt nothing.
    """
    torch.manual_seed(0)
    embeddings = torch.randn(1, 10, 256)
    mask = torch.ones(1, 10, dtype=torch.bool)
    gold = _gold_has_species()

    with torch.no_grad():
        *_, rel = strain_species_ete(
            embeddings,
            mask,
            gold_relations=gold,
            gold_entity_positions=_anchored(),
        )

    assert rel is not None
    meta, logits = rel
    assert meta["arg_pred_i"].tolist() < meta["arg_pred_j"].tolist()

    _, _, targets = strain_species_ete.align_relation_predictions(
        gold, meta, logits
    )
    assert targets.tolist() == [_HAS_SPECIES]
