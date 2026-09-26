"""`prefetch_layer_boundary_reads`: cross-batch overlap, and no cross-talk.

`_resolve_layer_boundary_cached` (see `test_layer_boundary_overlap.py`)
already overlaps one item's store read with the previous item's replay
*within* a batch. `prefetch_layer_boundary_reads` extends that across
batches: it submits batch `k + 1`'s reads before yielding batch `k`, so they
decompress on the background thread while batch `k`'s items replay on the
main thread. `_resolve_layer_boundary_cached` picks up a batch's park only
by identity (`is`), and the wrapper clears it in `finally`, so an early exit
never leaves a park a later, unrelated call could mistake for its own.
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
    """Drives two batches through `prefetch_layer_boundary_reads` the way
    `run_epoch` and each `evaluate_model` override do -- one
    `get_token_embeddings` call per yielded batch. Without a cross-batch
    prefetch (only the within-batch overlap
    `test_layer_boundary_overlap.py` pins), the second batch's read would
    only start once its own turn began, after the first batch's replay had
    already returned, and this would time out waiting on the event."""
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
    """`test_closing_early_never_leaves_a_park_a_later_call_can_reuse` closes
    the generator before the mismatched call, which already clears the park
    through `finally` alone -- so a regression that degraded the identity
    check (`is`) to a length comparison, while keeping that clear, would
    still pass it. This keeps the park live (no `close()`): batch_100's
    futures are still parked when batch_200 (same length, a different
    document) is resolved directly. A length match would hand
    `_resolve_layer_boundary_cached` batch_100's own still-parked futures
    for batch_200 to consume, serving batch_100's value (1.0) under
    batch_200's mask; matching by identity discards them instead and reads
    batch_200's own value (2.0)."""
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
    """`Future.cancel()` cannot stop a read that has already started, so
    discarding a mismatched park's futures (see the test above) cannot
    guarantee batch_100's read has actually stopped. If the fallback branch
    of `_resolve_layer_boundary_cached` opened a second, fresh pool for
    batch_200 instead of reusing `_layer_boundary_worker`'s single thread,
    that still-running stale read and the fresh one could call
    `LayerBoundaryStore.get` concurrently and race its unlocked
    hit/miss/mismatch counters. Document 100's read is held open on an
    `Event` past the point batch_200's read is submitted, so any second
    thread would overlap it; the single shared worker instead queues
    batch_200's read behind it."""
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
                    # Stay "active" past the point batch_200's read is
                    # submitted, so a second thread reading concurrently
                    # would actually overlap this one instead of missing it
                    # by luck.
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
