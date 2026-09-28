"""A training run fills a configured layer-boundary store it finds missing.

The store is created on first use, every document whose prefix the trunk
computes goes in, and a later pass reads it back instead of running the
frozen layers again.
"""

import pytest
import torch
from d3text.models import base
from d3text.models.config import ModelConfig
from d3text.models.ner import NERClassificationModel
from d3text.schema import EntityType, Schema

SCHEMA = Schema(entity_types=(EntityType(name="enzymes", prefix="enz"),))
BASE_MODEL = "prajjwal1/bert-mini"
WINDOW_TOKENS = 8


def _item(doc_id: int, n_windows: int) -> dict:
    generator = torch.Generator().manual_seed(doc_id)
    return {
        "id": torch.tensor(doc_id),
        "doc_id": torch.zeros(n_windows, dtype=torch.uint8),
        "sequence": {
            "input_ids": torch.randint(
                1, 1000, (1, n_windows, WINDOW_TOKENS), generator=generator
            ),
            "attention_mask": torch.ones(
                1, n_windows, WINDOW_TOKENS, dtype=torch.long
            ),
        },
    }


@pytest.fixture
def store_path(monkeypatch, tmp_path):
    path = tmp_path / "layer-boundary"
    monkeypatch.setitem(base.mconfig.embeddings_store, BASE_MODEL, str(path))
    base.layer_boundary_store.cache_clear()
    yield path
    base.layer_boundary_store.cache_clear()


def test_the_building_pass_matches_the_pass_that_reads_it_back(
    patch_base_model, store_path, monkeypatch
):
    """The pass that builds the store sees each prefix rounded through the
    store's bf16, so its embeddings are the ones every later pass replays
    from the store. Items of different window counts pin that each gets its
    own windows, not a neighbour's."""
    # CPU autocast is bf16 already, where the rounding is a no-op; fp32 is
    # the precision it has to reconcile with the store.
    monkeypatch.setattr(base, "select_amp_dtype", lambda _device: torch.float32)
    model = NERClassificationModel(
        schema=SCHEMA,
        config=ModelConfig(
            model_class="NERClassificationModel",
            base_model=BASE_MODEL,
            hidden_layers=[8],
            unfrozen_top_layers=1,
        ),
        device="cpu",
    )
    model.eval()
    batch = [_item(111, 2), _item(222, 3)]

    with torch.no_grad():
        built, _ = model.get_token_embeddings(batch)
        store = model._layer_boundary_store()
        assert store is not None and store.writable
        assert store.written == 2
        replayed, _ = model.get_token_embeddings(batch)

    assert store.hits == 2
    assert store_path.exists()
    assert torch.equal(built, replayed)
    store.close()
