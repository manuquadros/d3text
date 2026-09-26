"""Whether a pass actually waited on prefetched layer-boundary reads.

`_resolve_layer_boundary_cached` (`src/d3text/models/base.py`) blocks each
batch item's `store.get` future on `.result()`; nothing recorded how long
that took, so a run could not say whether the reads keep ahead of the GPU.
This pins `log_pass_stats`'s new wait line: a store whose `get` sleeps must
show up as a wait at least as long as the sleep, and an instant store's
wait must stay near zero.
"""

import logging
import re

import torch
from d3text.embeddings_store import LayerBoundaryStore
from d3text.models.base import Step
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
SLEEP_SECONDS = 0.2


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


def _resolve_one_batch(model, monkeypatch, sleep_seconds: float) -> None:
    """Run one batch through `get_token_embeddings` against a fake store
    whose `get` blocks for `sleep_seconds` before returning a stored prefix.
    """
    import time

    class _FakeStore(LayerBoundaryStore):
        def __init__(self) -> None:
            # Skip LayerBoundaryStore.__init__'s LMDB open, but still set
            # what `summary()` (called from `log_pass_stats`) reads.
            self.path = "<fake>"
            self.hits = 0
            self.misses = 0
            self.mismatches = 0

        def get(self, document_id: int, expected_windows: int):
            if sleep_seconds:
                time.sleep(sleep_seconds)
            self.hits += 1
            return torch.zeros(
                N_WINDOWS, WINDOW_TOKENS, HIDDEN_SIZE, dtype=model.amp_dtype
            )

    monkeypatch.setattr(
        "d3text.models.base.layer_boundary_store",
        lambda *_a, **_k: _FakeStore(),
    )
    monkeypatch.setattr(
        NERClassificationModel,
        "_replay_top_layers",
        lambda self, prefix, attention_mask, attention_mask_cpu: prefix,
    )
    model.get_token_embeddings([_item(1)])


def _logged_wait_seconds(caplog) -> float:
    """The wait `log_pass_stats` reported, parsed out of its log record."""
    matches = [
        m
        for record in caplog.records
        if (m := re.search(r"[Ww]aited ([\d.]+) s", record.getMessage()))
    ]
    assert len(matches) == 1, (
        "expected exactly one 'waited ... s' log line, "
        f"got {[r.getMessage() for r in caplog.records]}"
    )
    return float(matches[0].group(1))


def test_a_slow_store_makes_the_logged_wait_positive(
    patch_base_model, monkeypatch, caplog
):
    """A `store.get` that sleeps must make the reported wait at least half
    the sleep -- a generous margin against scheduling jitter, since the
    invariant under test is "the wait tracks real blocking time", not an
    exact measurement."""
    model = _ner()
    _resolve_one_batch(model, monkeypatch, SLEEP_SECONDS)

    with caplog.at_level(logging.INFO, logger="d3text.models.base"):
        model.log_pass_stats(Step.TRAINING)

    assert _logged_wait_seconds(caplog) >= SLEEP_SECONDS / 2


def test_an_instant_store_keeps_the_logged_wait_near_zero(
    patch_base_model, monkeypatch, caplog
):
    """A `store.get` that returns immediately must not accumulate anything
    resembling `SLEEP_SECONDS` of wait -- otherwise the metric would always
    read "waiting" regardless of the store's actual latency."""
    model = _ner()
    _resolve_one_batch(model, monkeypatch, 0.0)

    with caplog.at_level(logging.INFO, logger="d3text.models.base"):
        model.log_pass_stats(Step.TRAINING)

    assert _logged_wait_seconds(caplog) < SLEEP_SECONDS / 2


def test_the_wait_resets_after_being_logged(
    patch_base_model, monkeypatch, caplog
):
    """`log_pass_stats` must not let one pass's wait bleed into the next --
    otherwise a fast pass following a slow one would still report the slow
    pass's total."""
    model = _ner()
    _resolve_one_batch(model, monkeypatch, SLEEP_SECONDS)

    with caplog.at_level(logging.INFO, logger="d3text.models.base"):
        model.log_pass_stats(Step.TRAINING)
    assert _logged_wait_seconds(caplog) >= SLEEP_SECONDS / 2

    caplog.clear()
    _resolve_one_batch(model, monkeypatch, 0.0)
    with caplog.at_level(logging.INFO, logger="d3text.models.base"):
        model.log_pass_stats(Step.TRAINING)
    assert _logged_wait_seconds(caplog) < SLEEP_SECONDS / 2
