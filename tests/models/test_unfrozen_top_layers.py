"""`config.unfrozen_top_layers` leaves the base model's top layers trainable.

Real models with a tiny 2-layer random BERT injected (`patch_base_model`), so
there is no download. `unfrozen_top_layers=0` (the default) must stay
byte-identical to the fully-frozen behaviour `test_base_model_eval_mode.py`
already pins; these tests are about what changes once it is not 0.
"""

import contextlib

import pytest
import torch
from d3text.models.config import ModelConfig
from d3text.models.ner import NERClassificationModel
from d3text.schema import EntityType, Schema

SCHEMA = Schema(entity_types=(EntityType(name="enzymes", prefix="enz"),))


def _ner(unfrozen_top_layers: int) -> NERClassificationModel:
    return NERClassificationModel(
        schema=SCHEMA,
        config=ModelConfig(
            model_class="NERClassificationModel",
            base_model="prajjwal1/bert-mini",
            hidden_layers=[8],
            unfrozen_top_layers=unfrozen_top_layers,
        ),
        device="cpu",
    )


def _batch() -> list:
    return [
        {
            "id": torch.tensor(777),
            "doc_id": torch.zeros(1, dtype=torch.uint8),
            "sequence": {
                "input_ids": torch.randint(0, 999, (1, 24)),
                "attention_mask": torch.ones(1, 24, dtype=torch.long),
            },
        }
    ]


def test_zero_unfrozen_layers_keeps_the_whole_trunk_frozen(patch_base_model):
    model = _ner(0)
    assert not any(p.requires_grad for p in model.base_model.parameters())


def test_unfreezing_the_top_layer_leaves_the_bottom_one_frozen(
    patch_base_model,
):
    """The injected BERT has 2 encoder layers; `unfrozen_top_layers=1` must
    train the last one and nothing else."""
    model = _ner(1)
    layers = model.base_model.encoder.layer

    assert not any(
        p.requires_grad for p in model.base_model.embeddings.parameters()
    )
    assert not any(p.requires_grad for p in layers[0].parameters())
    assert all(p.requires_grad for p in layers[1].parameters())


def test_unfreezing_more_layers_than_exist_is_rejected(patch_base_model):
    with pytest.raises(ValueError, match="exceeds"):
        _ner(3)


def test_train_puts_a_partially_unfrozen_trunk_in_train_mode(patch_base_model):
    """`test_base_model_is_frozen_and_in_eval_mode_after_train` pins the
    opposite for `unfrozen_top_layers=0`; a trainable trunk needs its dropout
    active, so it must not be pinned to eval the same way."""
    model = _ner(1)
    model.train()

    assert model.base_model.training is True


def test_backward_reaches_the_unfrozen_layer_and_not_the_frozen_one(
    patch_base_model, monkeypatch
):
    """The mechanism actually works: a gradient reaches the unfrozen top
    layer's parameters and not the frozen bottom layer's, through the same
    `get_token_embeddings` path training calls every batch."""
    monkeypatch.setattr("d3text.models.base.cpu_embeddings_cache", None)
    monkeypatch.setattr("d3text.models.base.embeddings_store", lambda _: None)

    model = _ner(1)
    model.train()
    layers = model.base_model.encoder.layer

    with contextlib.nullcontext():
        embeddings, _ = model.get_token_embeddings(_batch())
    embeddings.sum().backward()

    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in layers[1].parameters()
    )
    assert all(p.grad is None for p in layers[0].parameters())


def test_a_trainable_trunk_bypasses_both_embedding_caches(
    patch_base_model, monkeypatch
):
    """Both caches assume a fixed trunk; reading or writing either through a
    training step would serve one step's weights to the next."""
    cache_calls: list[str] = []

    class _SpyCache:
        def get(self, *_args, **_kwargs):
            cache_calls.append("get")
            return None

        def set(self, *_args, **_kwargs):
            cache_calls.append("set")

        def full(self):
            return False

    def _spy_store(_base_model):
        cache_calls.append("store")
        return None

    monkeypatch.setattr("d3text.models.base.cpu_embeddings_cache", _SpyCache())
    monkeypatch.setattr("d3text.models.base.embeddings_store", _spy_store)

    model = _ner(1)
    model.train()
    model.get_token_embeddings(_batch())

    assert cache_calls == []
