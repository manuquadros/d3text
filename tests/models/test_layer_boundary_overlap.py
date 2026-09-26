"""A later batch item's store read overlaps an earlier item's replay.

`_resolve_layer_boundary_cached` submits every hit's `store.get` to a single
background thread before replaying any of them, so the thread can be reading
item `n + 1` while the trainable top layers replay item `n`'s prefix. A
version that reads and replays one item at a time -- what a bare loop over
`store.get` would do -- can never let a later item's read complete while an
earlier item's replay is still running: this pins that difference by making
the first item's (faked) replay block on an `Event` only the second item's
(faked) store read can set.
"""

import threading

import torch
from d3text.embeddings_store import LayerBoundaryStore
from d3text.models.config import ModelConfig
from d3text.models.ner import NERClassificationModel
from d3text.schema import EntityType, Schema

SCHEMA = Schema(entity_types=(EntityType(name="enzymes", prefix="enz"),))

# The injected BERT has 2 encoder layers (see `patch_base_model`); one
# trainable top layer leaves exactly one frozen bottom layer to cache.
UNFROZEN_TOP_LAYERS = 1
HIDDEN_SIZE = 256
WINDOW_TOKENS = 8
N_WINDOWS = 2


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


def _item(doc_id: int) -> dict:
    attention_mask = torch.ones(N_WINDOWS, WINDOW_TOKENS, dtype=torch.long)
    return {
        "id": torch.tensor(doc_id),
        "doc_id": torch.zeros(N_WINDOWS, dtype=torch.uint8),
        "sequence": {
            "input_ids": torch.zeros(
                N_WINDOWS, WINDOW_TOKENS, dtype=torch.long
            ).unsqueeze(0),
            "attention_mask": attention_mask.unsqueeze(0),
        },
    }


def test_the_second_items_read_runs_while_the_first_items_replay_is_in_flight(
    patch_base_model, monkeypatch
):
    model = _ner()
    prefixes = {
        111: torch.zeros(
            N_WINDOWS, WINDOW_TOKENS, HIDDEN_SIZE, dtype=model.amp_dtype
        ),
        222: torch.zeros(
            N_WINDOWS, WINDOW_TOKENS, HIDDEN_SIZE, dtype=model.amp_dtype
        ),
    }
    second_item_read = threading.Event()

    class _FakeStore(LayerBoundaryStore):
        def __init__(self) -> None:
            pass  # skip LayerBoundaryStore.__init__'s LMDB open

        def get(self, document_id: int, expected_windows: int):
            if document_id == 222:
                second_item_read.set()
            return prefixes[document_id]

    replayed: list[torch.Tensor] = []

    def fake_replay_top_layers(
        self, prefix, attention_mask, attention_mask_cpu
    ):
        replayed.append(prefix)
        if len(replayed) == 1:
            assert second_item_read.wait(timeout=2), (
                "the second item's store read never ran while the first "
                "item's replay was in flight"
            )
        return prefix

    monkeypatch.setattr(
        "d3text.models.base.layer_boundary_store",
        lambda *_a, **_k: _FakeStore(),
    )
    monkeypatch.setattr(
        NERClassificationModel, "_replay_top_layers", fake_replay_top_layers
    )

    embeddings, mask = model.get_token_embeddings([_item(111), _item(222)])

    assert len(replayed) == 2
    assert embeddings.shape[0] == 2
    assert mask.shape[0] == 2
