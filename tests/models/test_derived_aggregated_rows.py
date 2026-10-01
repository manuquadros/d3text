"""A frozen run reads its rows from the one env its base model's entry names.

With no aggregated sub-database there, the rows are derived from the stored
boundary with the fewest unfrozen layers: those layers are replayed over the
stored prefix and the windows aggregated as `utils.embed_document` does. A
run that trains its top layers resolves its own boundary by unfrozen count.
"""

import types

import pytest
import torch
from d3text.embeddings_store import LayerBoundaryStore, StoreProvenance
from d3text.models import base
from d3text.models.config import ModelConfig
from d3text.models.ner import NERClassificationModel
from d3text.schema import EntityType, Schema
from d3text.utils import WINDOW_LENGTH, WINDOW_STRIDE

SCHEMA = Schema(entity_types=(EntityType(name="enzymes", prefix="enz"),))
BASE_MODEL = "prajjwal1/bert-mini"
# Longer than WINDOW_STRIDE, so a multi-window document really is stitched.
WINDOW_TOKENS = 32
PROVENANCE = StoreProvenance(
    base_model=BASE_MODEL, max_length=WINDOW_LENGTH, stride=WINDOW_STRIDE
)


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


def _model(unfrozen_top_layers: int) -> NERClassificationModel:
    model = NERClassificationModel(
        schema=SCHEMA,
        config=ModelConfig(
            model_class="NERClassificationModel",
            base_model=BASE_MODEL,
            hidden_layers=[8],
            unfrozen_top_layers=unfrozen_top_layers,
        ),
        device="cpu",
    )
    return model.eval()


@pytest.fixture
def env(monkeypatch, tmp_path):
    """The env's path, configured only once a test calls `configure`: a
    frozen model looks its store up while it is built, and would otherwise
    create an aggregated sub-database before any boundary is stored."""
    path = tmp_path / "embeddings"
    # fp32, so the only rounding left between a derived row and a live one
    # is the store's own bf16.
    monkeypatch.setattr(base, "select_amp_dtype", lambda _device: torch.float32)

    def configure() -> None:
        monkeypatch.setitem(
            base.mconfig.embeddings_store, BASE_MODEL, str(path)
        )
        base.embeddings_store.cache_clear()
        base.layer_boundary_store.cache_clear()

    base.embeddings_store.cache_clear()
    base.layer_boundary_store.cache_clear()
    yield types.SimpleNamespace(path=path, configure=configure)
    base.embeddings_store.cache_clear()
    base.layer_boundary_store.cache_clear()


def _store_prefixes(path, model, batch, unfrozen_counts) -> None:
    """Each item's per-window hidden states at each boundary, from the
    model's own `hidden_states`, stored as `precompute-embeddings` would."""
    layers = model.base_model.config.num_hidden_layers
    for unfrozen in unfrozen_counts:
        store = LayerBoundaryStore.create(path, PROVENANCE, unfrozen)
        for item in batch:
            windows = item["sequence"]
            with torch.no_grad():
                hidden = model.base_model(
                    input_ids=windows["input_ids"][0],
                    attention_mask=windows["attention_mask"][0],
                    output_hidden_states=True,
                ).hidden_states
            store.put(int(item["id"]), hidden[layers - unfrozen])
        store.close()


def test_a_frozen_run_derives_its_rows_from_the_lowest_boundary(
    patch_base_model, env
):
    """Replaying the fewest layers is the cheapest derivation, and the rows
    it yields must be the live forward's: a skipped replay, or windows
    stitched at another stride, gives rows of another shape or value."""
    model = _model(0)
    batch = [_item(111, 1), _item(222, 3)]
    _store_prefixes(env.path, model, batch, unfrozen_counts=(2, 1))
    env.configure()

    store = base.embeddings_store(BASE_MODEL)
    assert isinstance(store, LayerBoundaryStore)
    assert store.unfrozen_top_layers == 1

    with torch.no_grad():
        derived, derived_mask = model.get_token_embeddings(batch)
    assert store.hits == len(batch)

    base.embeddings_store.cache_clear()
    live_model = _model(0)
    live_model.load_state_dict(model.state_dict())
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(base, "embeddings_store", lambda _base_model: None)
        with torch.no_grad():
            live, live_mask = live_model.get_token_embeddings(batch)

    assert torch.equal(derived_mask, live_mask)
    torch.testing.assert_close(
        derived.float(), live.float(), rtol=2e-2, atol=2e-2
    )


def test_a_trainable_run_reads_the_boundary_its_unfrozen_count_names(
    patch_base_model, env
):
    model = _model(1)
    _store_prefixes(env.path, model, [_item(111, 2)], unfrozen_counts=(1, 2))
    env.configure()

    store = model._layer_boundary_store()

    assert store is not None
    assert store.unfrozen_top_layers == 1
    assert store.writable
