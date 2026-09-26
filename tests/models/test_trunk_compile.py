"""`compile_trunk` compiles the trainable top encoder layers, not the model.

A real, tiny 4-layer `BertModel` (`patch_base_model` only injects 2, so this
builds its own), top 2 layers trainable: a frozen prefix and a trainable
top both exist, the shape that makes `BertLayer.forward`'s one code object
guard on `requires_grad`, mask, dtype and kwargs-vs-positional if each
`encoder.layer` is compiled in place instead of through one wrapper.
`test_compile_trunk_compiles_only_the_top_layers` pins that compiling the
one wrapper both trunk paths call, through `Model.compile_trunk`, holds one
guard set across window counts, padding, train and eval instead;
`test_ete_resolves_the_trunk_wrapper_through_two_head` pins that
`ETEBrendaModel`, which never calls `freeze_base_model` itself, resolves
its `_trunk_top` through its composed `two_head` rather than through the
`None` `Model.__init__` sets.

`ETEBrendaModel.forward`'s orchestration is never compiled, so this suite has
nothing there to test.
"""

from pathlib import Path

import h5py
import torch
import torch._dynamo as dynamo
import pytest
from d3text import runtime, token_labels, utils
from d3text.embeddings_store import LayerBoundaryStore
from d3text.models.config import ModelConfig
from d3text.models.ete import ETEBrendaModel
from d3text.models.ner import NERClassificationModel
from d3text.schema import EntityType, RelationType, Schema
from torch._dynamo.testing import CompileCounter
from transformers import BertConfig, BertModel
from transformers.masking_utils import create_bidirectional_mask

SCHEMA = Schema(entity_types=(EntityType(name="enzymes", prefix="enz"),))
ETE_SCHEMA = Schema(
    entity_types=(EntityType(name="enzymes", prefix="enz"),),
    relation_types=(
        RelationType(
            name="HasEnzyme", subject_types=("enzymes",), object_type="enzymes"
        ),
        RelationType(name="none", is_none=True),
    ),
)

# 4 layers, top 2 trainable: a frozen prefix and a trainable top both exist,
# the shape an unfrozen-trunk training run compiles.
NUM_LAYERS = 4
UNFROZEN_TOP_LAYERS = 2
HIDDEN_SIZE = 32
WINDOW_TOKENS = 16


def _tiny_bert(*_args: object, **_kwargs: object) -> BertModel:
    return BertModel(
        BertConfig(
            vocab_size=1000,
            hidden_size=HIDDEN_SIZE,
            num_hidden_layers=NUM_LAYERS,
            num_attention_heads=4,
            intermediate_size=64,
        )
    )


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> NERClassificationModel:
    """A real tiny 4-layer trunk, top 2 layers trainable, both embedding
    caches off so `get_token_embeddings` always drives a real forward."""
    monkeypatch.setattr("d3text.models.base.load_base_model", _tiny_bert)
    monkeypatch.setattr("d3text.models.base.cpu_embeddings_cache", None)
    monkeypatch.setattr("d3text.models.base.embeddings_store", lambda _: None)
    return NERClassificationModel(
        schema=SCHEMA,
        config=ModelConfig(
            model_class="NERClassificationModel",
            base_model="tiny",
            hidden_layers=[8],
            unfrozen_top_layers=UNFROZEN_TOP_LAYERS,
        ),
        device="cpu",
    )


@pytest.fixture
def compile_counter(monkeypatch: pytest.MonkeyPatch) -> CompileCounter:
    """Compile through dynamo with a counting backend instead of Triton, so
    a `.compile()` call runs on CPU: `torch.compile` is looked up at call
    time, the same reason `test_trainer.py`'s
    `test_compiling_puts_the_trainers_forward_on_the_compiled_path` can
    stand a recording stand-in in for it. Also pins the production
    recompile budget and turns a hit into a hard failure instead of a
    silent eager fallback, so a regression fails the test instead of
    passing quietly slower.
    """
    counter = CompileCounter()
    real_compile = torch.compile

    def counting_compile(*args: object, **kwargs: object) -> object:
        return real_compile(*args, **{**kwargs, "backend": counter})

    monkeypatch.setattr(torch, "compile", counting_compile)
    monkeypatch.setattr(runtime, "is_triton_compatible", lambda: True)
    monkeypatch.setenv(runtime.COMPILE_VARIABLE, "1")
    monkeypatch.setattr(dynamo.config, "recompile_limit", 8)
    monkeypatch.setattr(dynamo.config, "fail_on_recompile_limit_hit", True)
    dynamo.reset()
    return counter


class _FakeLayerBoundaryStore(LayerBoundaryStore):
    """Stands in for `embeddings_store.layer_boundary_store`'s reader."""

    def __init__(self, prefix: torch.Tensor) -> None:
        self._prefix = prefix  # skip LayerBoundaryStore.__init__'s LMDB open

    def get(self, document_id: int, expected_windows: int) -> torch.Tensor:
        return self._prefix.clone()


def _batch(n_windows: int, *, pad_last: bool, doc_id: int) -> list[dict]:
    input_ids = torch.randint(0, 999, (n_windows, WINDOW_TOKENS))
    attention_mask = torch.ones(n_windows, WINDOW_TOKENS, dtype=torch.long)
    if pad_last:
        attention_mask[-1, WINDOW_TOKENS // 2 :] = 0
    return [
        {
            "id": torch.tensor(doc_id),
            "doc_id": torch.zeros(n_windows, dtype=torch.uint8),
            "sequence": {
                "input_ids": input_ids.unsqueeze(0),
                "attention_mask": attention_mask.unsqueeze(0),
            },
        }
    ]


def _layer_boundary_prefix(
    model: NERClassificationModel, n_windows: int
) -> torch.Tensor:
    """A prefix the way a real layer-boundary store would hold it, for the
    cached-replay half of the trunk (`_resolve_layer_boundary_cached`)."""
    encoder_layers = model.base_model.get_submodule("encoder.layer")
    frozen_layers = len(encoder_layers) - model.config.unfrozen_top_layers
    input_ids = torch.randint(0, 999, (n_windows, WINDOW_TOKENS))
    attention_mask = torch.ones(n_windows, WINDOW_TOKENS, dtype=torch.long)

    with torch.no_grad(), model.autocast_context():
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

    return hidden.to(model.amp_dtype)


def _drive_both_trunk_paths(
    model: NERClassificationModel,
    monkeypatch: pytest.MonkeyPatch,
    doc_id_start: int,
) -> None:
    """Both trunk paths, several window counts including 1, padded and
    unpadded, in train mode with backward, then an eval/no_grad pass --
    the combination of route, requires_grad, dtype and mask that would
    guard a shared per-layer code object into exceeding the recompile
    budget.
    """
    doc_id = doc_id_start
    for n_windows in (1, 2, 3, 5):
        for pad_last in (False, True):
            doc_id += 1
            embeddings, _ = model.get_token_embeddings(
                _batch(n_windows, pad_last=pad_last, doc_id=doc_id)
            )
            embeddings.sum().backward()
            model.zero_grad()

    for n_windows in (1, 2, 3):
        doc_id += 1
        prefix = _layer_boundary_prefix(model, n_windows)
        monkeypatch.setattr(
            "d3text.models.base.layer_boundary_store",
            lambda *_a, prefix=prefix, **_k: _FakeLayerBoundaryStore(prefix),
        )
        embeddings, _ = model.get_token_embeddings(
            _batch(n_windows, pad_last=False, doc_id=doc_id)
        )
        embeddings.sum().backward()
        model.zero_grad()
    monkeypatch.setattr(
        "d3text.models.base.layer_boundary_store", lambda *_a, **_k: None
    )

    model.eval()
    with torch.no_grad():
        for n_windows in (1, 2, 4):
            doc_id += 1
            model.get_token_embeddings(
                _batch(n_windows, pad_last=False, doc_id=doc_id)
            )


def test_compile_trunk_compiles_only_the_top_layers(
    model: NERClassificationModel,
    compile_counter: CompileCounter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One wrapper both trunk paths call, compiled once, holds its guard
    set across window counts, padding, train-with-backward and
    eval/no_grad -- where compiling each `encoder.layer` module in place
    instead would share one guarded code object across every layer and
    both routes and exceed the recompile budget on the same scenario."""
    assert model.compile_trunk() is True
    assert model.trunk_is_compiled() is True

    model.train()
    _drive_both_trunk_paths(model, monkeypatch, doc_id_start=0)

    # Not zero (the wrapper really compiled) and small (well under the
    # eight-recompile budget `fail_on_recompile_limit_hit` would have
    # raised on otherwise): a handful of frames for dtype/requires_grad/
    # window-dim transitions, not one per window count and path.
    assert 0 < compile_counter.frame_count <= 4


def test_a_frozen_trunk_compiles_nothing(
    compile_counter: CompileCounter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`unfrozen_top_layers=0` (the default) builds no wrapper: the trunk
    runs under `no_grad` or answers from a store, neither of which has a
    recompile to save, so `compile_trunk` is a documented no-op rather than
    tracing anything."""
    monkeypatch.setattr("d3text.models.base.load_base_model", _tiny_bert)
    frozen_model = NERClassificationModel(
        schema=SCHEMA,
        config=ModelConfig(
            model_class="NERClassificationModel",
            base_model="tiny",
            hidden_layers=[8],
        ),
        device="cpu",
    )

    assert frozen_model.compile_trunk() is False
    assert frozen_model.trunk_is_compiled() is False
    assert compile_counter.frame_count == 0


def test_the_compiled_wrapper_agrees_with_the_eager_computation(
    compile_counter: CompileCounter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compiling must not change what the trunk computes.

    Compares the compiled wrapper (`_replay_top_layers`, dispatching
    through `_trunk_top`) against the uncompiled computation it wraps
    (`_replay_top_layers_eager`) on the same input, and separately compares
    `_embed_missing`'s split -- frozen layers eagerly, then the same
    wrapper -- against one whole `self.base_model(...)` call. Built with
    `amp_dtype` pinned to fp32, so no bf16 cast or autocast is involved:
    the bf16 cast at the frozen/trainable boundary moves the output more
    than a deleted attention mask does, so only a tight fp32 tolerance
    catches a masking bug. Both comparisons include a padded window.
    """
    monkeypatch.setattr("d3text.models.base.load_base_model", _tiny_bert)
    monkeypatch.setattr(
        "d3text.models.base.select_amp_dtype", lambda _device: torch.float32
    )
    monkeypatch.setattr("d3text.models.base.cpu_embeddings_cache", None)
    monkeypatch.setattr("d3text.models.base.embeddings_store", lambda _: None)
    model = NERClassificationModel(
        schema=SCHEMA,
        config=ModelConfig(
            model_class="NERClassificationModel",
            base_model="tiny",
            hidden_layers=[8],
            unfrozen_top_layers=UNFROZEN_TOP_LAYERS,
        ),
        device="cpu",
    )
    assert model.compile_trunk() is True
    model.eval()

    prefix = torch.randn(2, WINDOW_TOKENS, HIDDEN_SIZE)
    attention_mask = torch.ones(2, WINDOW_TOKENS, dtype=torch.long)
    attention_mask[-1, WINDOW_TOKENS // 2 :] = 0  # a padded window
    with torch.no_grad():
        compiled_out = model._replay_top_layers(prefix.clone(), attention_mask)
        eager_out = model._replay_top_layers_eager(
            prefix.clone(), attention_mask
        )
    assert torch.allclose(compiled_out, eager_out, rtol=1e-5, atol=1e-5)

    input_ids = torch.randint(0, 999, (3, WINDOW_TOKENS))
    attention_mask = torch.ones(3, WINDOW_TOKENS, dtype=torch.long)
    attention_mask[-1, WINDOW_TOKENS // 2 :] = 0
    with torch.no_grad():
        whole = model.base_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        split = model._embed_missing_trainable_trunk(
            input_ids, attention_mask, attention_mask
        )
    assert torch.allclose(whole, split, rtol=1e-5, atol=1e-5)


def test_embed_missing_trainable_trunk_skips_the_device_mask_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`create_bidirectional_mask` would otherwise read the mask on the
    device to decide whether it can skip building one, once per call.
    `_embed_missing_trainable_trunk` decides that from the host mask
    instead: `None` plus the default skip check for an unpadded batch --
    the same mask the check would itself have resolved to, only without
    touching the device for it -- and the real mask with the check
    disabled otherwise, since a real check would have found padding and
    materialized the mask anyway, so disabling it only removes the read. A
    revert to always forwarding the device mask with the default kwargs
    would call `create_bidirectional_mask` identically in both cases.
    """
    monkeypatch.setattr("d3text.models.base.load_base_model", _tiny_bert)
    monkeypatch.setattr(
        "d3text.models.base.select_amp_dtype", lambda _device: torch.float32
    )
    monkeypatch.setattr("d3text.models.base.cpu_embeddings_cache", None)
    monkeypatch.setattr("d3text.models.base.embeddings_store", lambda _: None)
    # Isolates the frozen-prefix mask this method builds itself from the
    # one `_replay_top_layers_eager` builds for the top layers, unmodified
    # and out of scope here, so it does not also land in `calls`.
    monkeypatch.setattr(
        NERClassificationModel,
        "_replay_top_layers",
        lambda self, prefix, attention_mask: prefix,
    )
    model = NERClassificationModel(
        schema=SCHEMA,
        config=ModelConfig(
            model_class="NERClassificationModel",
            base_model="tiny",
            hidden_layers=[8],
            unfrozen_top_layers=UNFROZEN_TOP_LAYERS,
        ),
        device="cpu",
    )
    model.eval()

    calls: list[tuple[bool, bool]] = []

    def spy(*, attention_mask=None, allow_is_bidirectional_skip=True, **kwargs):
        calls.append((attention_mask is None, allow_is_bidirectional_skip))
        return create_bidirectional_mask(
            attention_mask=attention_mask,
            allow_is_bidirectional_skip=allow_is_bidirectional_skip,
            **kwargs,
        )

    monkeypatch.setattr("d3text.models.base.create_bidirectional_mask", spy)

    input_ids = torch.randint(0, 999, (2, WINDOW_TOKENS))
    unpadded = torch.ones(2, WINDOW_TOKENS, dtype=torch.long)
    padded = unpadded.clone()
    padded[-1, WINDOW_TOKENS // 2 :] = 0

    with torch.no_grad():
        model._embed_missing_trainable_trunk(input_ids, unpadded, unpadded)
        model._embed_missing_trainable_trunk(input_ids, padded, padded)

    assert calls == [(True, True), (False, False)]


def test_ete_resolves_the_trunk_wrapper_through_two_head(
    compile_counter: CompileCounter,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """`ETEBrendaModel` composes `two_head` and never calls
    `freeze_base_model` itself, so the `_trunk_top = None`
    `Model.__init__` sets in its own `__dict__` must not shadow
    `two_head`'s wrapper. Unresolved, `get_token_embeddings` on an
    unfrozen trunk fails the `_replay_top_layers` assert on its first
    batch.
    """
    monkeypatch.setattr("d3text.models.base.load_base_model", _tiny_bert)
    monkeypatch.setattr("d3text.models.base.cpu_embeddings_cache", None)
    monkeypatch.setattr("d3text.models.base.embeddings_store", lambda _: None)

    label_store = tmp_path / "empty_labels.hdf5"
    with h5py.File(label_store, "w") as store:
        token_labels.write_label_space(
            store,
            token_labels.BRENDA_LABELS,
            stamp=token_labels.IndexStamp(digest="empty-store"),
            tokenizer=token_labels.TokenizerStamp(
                base_model="tiny",
                digest="test-tokenizer",
                window_length=utils.WINDOW_LENGTH,
                window_stride=utils.WINDOW_STRIDE,
            ),
        )

    model = ETEBrendaModel(
        schema=ETE_SCHEMA,
        config=ModelConfig(
            model_class="ETEBrendaModel",
            base_model="tiny",
            hidden_layers=[8],
            unfrozen_top_layers=UNFROZEN_TOP_LAYERS,
            token_labels_store=str(label_store),
        ),
        device="cpu",
    )

    embeddings, mask = model.get_token_embeddings(
        _batch(2, pad_last=True, doc_id=0)
    )
    assert embeddings.shape[0] == 1
    assert embeddings.shape[-1] == HIDDEN_SIZE
    assert mask.shape == embeddings.shape[:2]

    assert model.compile_trunk() is True
    assert model.trunk_is_compiled() is True
