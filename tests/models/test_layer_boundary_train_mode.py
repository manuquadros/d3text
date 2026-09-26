"""A layer-boundary cache hit must agree with a live forward under `train()`.

`unfrozen_top_layers` makes the frozen bottom of the trunk a candidate for
caching only if its output is a pure function of the input -- which needs
its dropout off even while `Model.train()` is engaged, since that is the
only mode the trainer ever calls `get_token_embeddings` in. A model built
with a tiny random BERT injected (`patch_base_model`), so there is no
download.
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

    Computed the way `precompute_embeddings.embed_document_layer_prefix`
    computes it -- through the embeddings and the frozen bottom layer only,
    under the model's own autocast policy -- and cast down to `amp_dtype`,
    the precision a real store round-trips through (`tensor_to_bytes` is
    always bf16). `base_model` is already in eval mode here:
    `freeze_base_model` leaves it that way at construction, before
    `model.train()` is ever called, exactly as the offline precompute tool
    runs it.
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
    """Before the fix, `Model.train()` left the whole trunk -- including the
    frozen bottom layer -- in train mode, so a live forward drew fresh
    dropout noise there that a cached-prefix replay, skipping straight to
    the top layer, never consumed. That RNG-stream shift alone would put
    the top layer's own dropout out of step between the two calls even with
    an otherwise perfect cached prefix, so this comparison would not have
    held under the old `train()`. After the fix the frozen layer is
    deterministic under `train()`, so what is left between a cached and a
    live forward is only the bf16 rounding a real store's compression adds
    -- the two agree up to that.
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
