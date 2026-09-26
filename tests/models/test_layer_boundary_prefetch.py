"""`prefetch_layer_boundary_reads`: cross-batch overlap, and no cross-talk.

Batch `k + 1`'s reads are submitted before batch `k` is yielded. A park is
claimed only by identity and cleared in `finally`, so no unrelated call can
mistake it for its own.
"""

import threading
import time

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


def test_the_next_batchs_read_runs_while_this_batchs_replay_is_in_flight(
    patch_base_model, monkeypatch
):
    """The next batch's read runs during this batch's replay.

    Without cross-batch prefetch it would start only after the replay
    returned, and this would time out on the event.
    """
    model = _ner()
    prefixes = {
        111: torch.zeros(
            N_WINDOWS, WINDOW_TOKENS, HIDDEN_SIZE, dtype=model.amp_dtype
        ),
        222: torch.zeros(
            N_WINDOWS, WINDOW_TOKENS, HIDDEN_SIZE, dtype=model.amp_dtype
        ),
    }
    second_batch_read = threading.Event()

    class _FakeStore(LayerBoundaryStore):
        def __init__(self) -> None:
            pass  # skip LayerBoundaryStore.__init__'s LMDB open

        def get(self, document_id: int, expected_windows: int):
            if document_id == 222:
                second_batch_read.set()
            return prefixes[document_id]

    replayed: list[torch.Tensor] = []

    def fake_replay_top_layers(
        self, prefix, attention_mask, attention_mask_cpu
    ):
        replayed.append(prefix)
        if len(replayed) == 1:
            assert second_batch_read.wait(timeout=2), (
                "the second batch's store read never ran while the first "
                "batch's replay was in flight"
            )
        return prefix

    monkeypatch.setattr(
        "d3text.models.base.layer_boundary_store",
        lambda *_a, **_k: _FakeStore(),
    )
    monkeypatch.setattr(
        NERClassificationModel, "_replay_top_layers", fake_replay_top_layers
    )

    batch_111 = [_item(111)]
    batch_222 = [_item(222)]
    outputs = [
        model.get_token_embeddings(batch)
        for batch in model.prefetch_layer_boundary_reads(
            iter([batch_111, batch_222])
        )
    ]

    assert len(replayed) == 2
    assert len(outputs) == 2


def test_closing_early_never_leaves_a_park_a_later_call_can_reuse(
    patch_base_model, monkeypatch
):
    """Closing the generator after one batch must clear the park, so a
    later direct `get_token_embeddings` on an unrelated, same-length batch
    serves its own document (batch_200's 2.0, not batch_100's 1.0) rather
    than the parked batch's prefix under its mask."""
    model = _ner()
    prefixes = {
        100: torch.full(
            (N_WINDOWS, WINDOW_TOKENS, HIDDEN_SIZE),
            1.0,
            dtype=model.amp_dtype,
        ),
        200: torch.full(
            (N_WINDOWS, WINDOW_TOKENS, HIDDEN_SIZE),
            2.0,
            dtype=model.amp_dtype,
        ),
    }

    class _FakeStore(LayerBoundaryStore):
        def __init__(self) -> None:
            pass  # skip LayerBoundaryStore.__init__'s LMDB open

        def get(self, document_id: int, expected_windows: int):
            return prefixes[document_id]

    monkeypatch.setattr(
        "d3text.models.base.layer_boundary_store",
        lambda *_a, **_k: _FakeStore(),
    )
    monkeypatch.setattr(
        NERClassificationModel,
        "_replay_top_layers",
        lambda self, prefix, attention_mask, attention_mask_cpu: prefix,
    )

    batch_100 = [_item(100)]
    gen = model.prefetch_layer_boundary_reads(iter([batch_100]))
    yielded = next(gen)
    assert yielded is batch_100
    gen.close()

    assert model._parked_layer_boundary_reads is None

    batch_200 = [_item(200)]  # same length as batch_100, a different document
    embeddings, _ = model.get_token_embeddings(batch_200)

    assert torch.equal(embeddings, torch.full_like(embeddings, 2.0))


def test_a_live_parks_futures_are_not_consumed_by_a_mismatched_batch(
    patch_base_model, monkeypatch
):
    """A live park is matched by identity, not by length.

    The park stays live (no `close()`, whose `finally` would clear it): a
    same-length batch of another document must read its own value, not the
    parked one.
    """
    model = _ner()
    prefixes = {
        100: torch.full(
            (N_WINDOWS, WINDOW_TOKENS, HIDDEN_SIZE),
            1.0,
            dtype=model.amp_dtype,
        ),
        200: torch.full(
            (N_WINDOWS, WINDOW_TOKENS, HIDDEN_SIZE),
            2.0,
            dtype=model.amp_dtype,
        ),
    }

    class _FakeStore(LayerBoundaryStore):
        def __init__(self) -> None:
            pass  # skip LayerBoundaryStore.__init__'s LMDB open

        def get(self, document_id: int, expected_windows: int):
            return prefixes[document_id]

    monkeypatch.setattr(
        "d3text.models.base.layer_boundary_store",
        lambda *_a, **_k: _FakeStore(),
    )
    monkeypatch.setattr(
        NERClassificationModel,
        "_replay_top_layers",
        lambda self, prefix, attention_mask, attention_mask_cpu: prefix,
    )

    batch_100 = [_item(100)]
    gen = model.prefetch_layer_boundary_reads(iter([batch_100]))
    yielded = next(gen)
    assert yielded is batch_100
    parked = model._parked_layer_boundary_reads
    assert parked is not None and parked[0] is batch_100

    batch_200 = [_item(200)]  # same length as batch_100, a different document
    embeddings, _ = model.get_token_embeddings(batch_200)

    assert torch.equal(embeddings, torch.full_like(embeddings, 2.0))
    gen.close()


def test_a_discarded_stale_park_never_reads_the_store_from_two_threads(
    patch_base_model, monkeypatch
):
    """A discarded park's still-running read never overlaps a fresh one.

    `Future.cancel()` cannot stop a started read, so a second pool would race
    `LayerBoundaryStore.get`'s unlocked counters. The stale read is held open
    on an `Event`; the single shared worker must queue the fresh one behind.
    """
    model = _ner()
    lock = threading.Lock()
    active = 0
    max_active = 0
    batch_100_started = threading.Event()
    release_batch_100 = threading.Event()

    class _FakeStore(LayerBoundaryStore):
        def __init__(self) -> None:
            pass  # skip LayerBoundaryStore.__init__'s LMDB open

        def get(self, document_id: int, expected_windows: int):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            try:
                if document_id == 100:
                    batch_100_started.set()
                    assert release_batch_100.wait(timeout=2)
                    # Held past batch_200's submit, so a second thread
                    # would overlap this read rather than miss it by luck.
                    time.sleep(0.05)
                return torch.zeros(
                    N_WINDOWS,
                    WINDOW_TOKENS,
                    HIDDEN_SIZE,
                    dtype=model.amp_dtype,
                )
            finally:
                with lock:
                    active -= 1

    monkeypatch.setattr(
        "d3text.models.base.layer_boundary_store",
        lambda *_a, **_k: _FakeStore(),
    )
    monkeypatch.setattr(
        NERClassificationModel,
        "_replay_top_layers",
        lambda self, prefix, attention_mask, attention_mask_cpu: prefix,
    )

    batch_100 = [_item(100)]
    gen = model.prefetch_layer_boundary_reads(iter([batch_100]))
    next(gen)  # yields batch_100, parking its (now in-flight) read

    assert batch_100_started.wait(timeout=2), "batch_100's read never started"
    release_batch_100.set()

    batch_200 = [_item(200)]  # mismatched: discards batch_100's stale park
    model.get_token_embeddings(batch_200)
    gen.close()

    assert max_active <= 1
