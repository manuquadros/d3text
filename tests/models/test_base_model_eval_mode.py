"""The frozen base model stays in eval mode whatever the heads are doing.

`no_grad` does not disable dropout, so a base model carried along by
`model.train()` would hand a fresh draw of a document's activations to every
epoch while the CPU cache and the precomputed store, each written once, served
one draw forever. Real models with a tiny random BERT injected (see the
``patch_base_model`` fixture), so there is no download.
"""

import pytest
import torch
from d3text.models.config import ModelConfig
from d3text.models.entity_linking import BrendaClassificationModel
from d3text.models.ete import ETEBrendaModel
from d3text.models.ner import NERClassificationModel
from d3text.schema import EntityType, RelationType, Schema

SCHEMA = Schema(
    entity_types=(
        EntityType(name="enzymes", prefix="enz"),
        EntityType(name="bacteria", prefix="bac"),
    ),
    relation_types=(
        RelationType(
            name="HasEnzyme", subject_types=("bacteria",), object_type="enzymes"
        ),
        RelationType(name="none", is_none=True),
    ),
)


def _config() -> ModelConfig:
    return ModelConfig(
        base_model="prajjwal1/bert-mini", hidden_layers=[8], ramp_epochs=0
    )


def _ner() -> NERClassificationModel:
    return NERClassificationModel(schema=SCHEMA, config=_config(), device="cpu")


def _entity_linking() -> BrendaClassificationModel:
    return BrendaClassificationModel(
        schema=SCHEMA,
        class_matrix=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        entity_index={"enz1": 0, "bac1": 1},
        config=_config(),
        device="cpu",
    )


def _ete() -> ETEBrendaModel:
    return ETEBrendaModel(
        schema=SCHEMA,
        class_matrix=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        entity_index={"enz1": 0, "bac1": 1},
        config=_config(),
        device="cpu",
    )


@pytest.fixture(params=[_ner, _entity_linking, _ete], ids=lambda f: f.__name__)
def model(request, patch_base_model):
    """One of the three model classes, on CPU.

    `ETEBrendaModel` owns no `base_model` of its own — it composes a
    `BrendaClassificationModel` that does — so the invariant has to hold
    through that composition too.
    """
    return request.param()


def test_base_model_is_frozen_and_in_eval_mode_after_train(model):
    model.train()

    assert model.training is True
    assert model.classifier.training is True
    assert model.base_model.training is False
    assert not any(p.requires_grad for p in model.base_model.parameters())


def test_two_forwards_agree_while_the_heads_are_training(
    patch_base_model, monkeypatch
):
    """The observable consequence: one document, one draw.

    Both other sources are disabled here, so the comparison is between two
    live base-model forwards rather than between a forward and a cache hit.
    """
    monkeypatch.setattr("d3text.models.base.cpu_embeddings_cache", None)
    monkeypatch.setattr("d3text.models.base.embeddings_store", lambda _: None)

    model = _ner()
    model.train()
    batch = [
        {
            "id": torch.tensor(777),
            "doc_id": torch.zeros(1, dtype=torch.uint8),
            "sequence": {
                "input_ids": torch.randint(0, 999, (1, 24)),
                "attention_mask": torch.ones(1, 24, dtype=torch.long),
            },
        }
    ]

    first, _ = model.get_token_embeddings(batch)
    second, _ = model.get_token_embeddings(batch)

    assert torch.equal(first, second)
