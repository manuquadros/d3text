"""A layer-boundary cache hit must agree with a live forward under `train()`.

The frozen bottom is cacheable only if deterministic, so its dropout must
stay off under `train()`, the only mode the trainer embeds in.
"""

import torch
from d3text.embeddings_store import LayerBoundaryStore
from d3text.models.config import ModelConfig
from d3text.models.ner import NERClassificationModel
from d3text.schema import EntityType, Schema
from transformers.masking_utils import create_bidirectional_mask

SCHEMA = Schema(entity_types=(EntityType(name="enzymes", prefix="enz"),))

# The injected BERT has 2 encoder layers (see `patch_base_model`); one
# trainable top layer leaves exactly one frozen bottom layer to cache.
UNFROZEN_TOP_LAYERS = 1


def _ner() -> NERClassificationModel:
    return NERClassificationModel(
        schema=SCHEMA,
        config=ModelConfig(
            model_class="NERClassificationModel",
            base_model="prajjwal1/bert-mini",
            hidden_layers=[8],
            unfrozen_top_layers=UNFROZEN_TOP_LAYERS,
        ),
        device="cpu",
    )


def _batch() -> list:
    """One document, two windows, the second heavily padded."""
    input_ids = torch.randint(0, 999, (2, 24))
    attention_mask = torch.ones(2, 24, dtype=torch.long)
    attention_mask[1, 10:] = 0
    return [
        {
            "id": torch.tensor(777),
            "doc_id": torch.zeros(2, dtype=torch.uint8),
            "sequence": {
                "input_ids": input_ids.unsqueeze(0),
                "attention_mask": attention_mask.unsqueeze(0),
            },
        }
    ]


def _cached_prefix(model: NERClassificationModel, item: dict) -> torch.Tensor:
    """The layer-boundary prefix a real store would hold for `item`.

    Computed as `embed_document_and_prefix`'s frozen-layers path does,
    under the model's autocast, and cast to `amp_dtype`, the precision a
    store round-trips.
    """
    encoder_layers = model.base_model.get_submodule("encoder.layer")
    frozen_layers = len(encoder_layers) - model.config.unfrozen_top_layers
    input_ids = item["sequence"]["input_ids"].reshape(-1, 24)
    attention_mask = item["sequence"]["attention_mask"].reshape(-1, 24)

    with torch.inference_mode(), model.autocast_context():
        hidden = model.base_model.get_submodule("embeddings")(
            input_ids=input_ids
        )
        extended_mask = create_bidirectional_mask(
            config=model.base_model.config,
            inputs_embeds=hidden,
            attention_mask=attention_mask,
        )
        for layer in encoder_layers[:frozen_layers]:
            hidden = layer(hidden, extended_mask)

    return hidden.clone().to(model.amp_dtype)


def test_train_mode_cached_forward_agrees_with_a_live_forward(
    patch_base_model, monkeypatch
):
    """Under `train()`, a cached replay matches a live forward up to bf16.

    Frozen-layer dropout in a live forward would consume RNG a replay skips,
    putting the top layer's dropout out of step even with a perfect prefix.
    """
    monkeypatch.setattr("d3text.models.base.cpu_embeddings_cache", None)

    model = _ner()
    batch = _batch()
    prefix = _cached_prefix(model, batch[0])

    model.train()

    class _FakeStore(LayerBoundaryStore):
        def __init__(self) -> None:
            pass  # skip LayerBoundaryStore.__init__'s LMDB open

        def get(self, document_id: int, expected_windows: int) -> torch.Tensor:
            return prefix.clone()

    monkeypatch.setattr(
        "d3text.models.base.layer_boundary_store", lambda *_: _FakeStore()
    )
    torch.manual_seed(0)
    cached_out, cached_mask = model.get_token_embeddings(batch)

    monkeypatch.setattr(
        "d3text.models.base.layer_boundary_store", lambda *_: None
    )
    torch.manual_seed(0)
    live_out, live_mask = model.get_token_embeddings(batch)

    assert torch.equal(cached_mask, live_mask)
    assert torch.allclose(cached_out, live_out, rtol=0.05, atol=0.05)
