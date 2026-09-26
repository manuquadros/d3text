"""The frozen trunk holds its linear weights in the autocast dtype.

Autocast caches a cast only for a leaf with `requires_grad=True`, so a frozen
`nn.Linear` re-cast its weight every forward. `LayerNorm` and `Embedding` must
stay fp32. Both `select_amp_dtype` outcomes (bf16, fp16) are forced, so a CPU
covers both kinds of card.
"""

import copy

import pytest
import torch
import torch.nn as nn
from d3text.models.config import ModelConfig
from d3text.models.ner import NERClassificationModel
from d3text.schema import EntityType, Schema
from transformers import BertConfig, BertModel

SCHEMA = Schema(entity_types=(EntityType(name="enzymes", prefix="enz"),))


@pytest.fixture(params=[torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def amp_dtype(request, monkeypatch):
    """Build every model under one of the two dtypes a real card selects.

    `select_amp_dtype` is the sole authority on which one a machine gets, so
    forcing its answer is what stands in for running on an Ada card and on a
    P100 or T4 without needing either.
    """
    monkeypatch.setattr(
        "d3text.models.base.select_amp_dtype", lambda _device: request.param
    )
    return request.param


def _ner(unfrozen_top_layers: int = 0) -> NERClassificationModel:
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


def _linears(module: nn.Module) -> list[nn.Linear]:
    return [child for child in module.modules() if isinstance(child, nn.Linear)]


def test_frozen_trunk_linears_are_stored_in_the_autocast_dtype(
    patch_base_model, amp_dtype
):
    model = _ner()
    linears = _linears(model.base_model)

    assert linears
    assert all(
        linear.weight.dtype is amp_dtype and linear.bias.dtype is amp_dtype
        for linear in linears
    )


def test_layernorm_and_the_embedding_table_stay_fp32(
    patch_base_model, amp_dtype
):
    """Narrowing either would move values rather than only the cast:
    autocast's fp32 policy would copy a narrowed `LayerNorm` weight back up
    every forward, and it never casts `Embedding` at all, so a narrowed table
    would change what the first `LayerNorm` sees."""
    model = _ner()
    parameters = [
        parameter
        for module in model.base_model.modules()
        if isinstance(module, nn.LayerNorm | nn.Embedding)
        for parameter in module.parameters(recurse=False)
    ]

    assert parameters
    assert all(parameter.dtype is torch.float32 for parameter in parameters)


def test_a_trainable_top_layer_keeps_its_fp32_master_weights(
    patch_base_model, amp_dtype
):
    """The optimizer steps these, so they must not be narrowed; the frozen
    layer below them still is."""
    model = _ner(unfrozen_top_layers=1)
    bottom, top = model.base_model.encoder.layer

    assert _linears(top) and _linears(bottom)
    assert all(linear.weight.dtype is torch.float32 for linear in _linears(top))
    assert all(linear.weight.dtype is amp_dtype for linear in _linears(bottom))


def test_the_cast_trunk_returns_what_autocast_returned_before(
    monkeypatch, amp_dtype
):
    """The stored weight is bit-identical to the one autocast produced from
    fp32 — the same cast, once instead of once per forward — so the trunk's
    hidden states must match an fp32-weighted reference run under the same
    autocast exactly, not merely closely. This is what carries the fp16
    machines too: anything that overflows or flushes to zero on the way down
    did so in autocast's own copy already."""
    torch.manual_seed(0)
    reference = BertModel(
        BertConfig(
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=512,
        )
    ).eval()
    monkeypatch.setattr(
        "d3text.models.base.load_base_model",
        lambda *_args, **_kwargs: copy.deepcopy(reference),
    )

    model = _ner()
    input_ids = torch.randint(0, 999, (2, 24))
    attention_mask = torch.ones(2, 24, dtype=torch.long)

    with torch.no_grad(), model.autocast_context():
        expected = reference(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        actual = model.base_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state

    assert torch.equal(actual, expected)


def test_a_checkpoint_crosses_machines_and_predates_the_cast(
    patch_base_model, monkeypatch
):
    """A frozen weight loads as exactly its cast, whatever dtype wrote it.

    Compared against the cast, not the source: fp16's subnormal step is
    coarser than bf16's, so some near-zero weights move by one step.
    """

    def built_for(dtype: torch.dtype) -> NERClassificationModel:
        monkeypatch.setattr(
            "d3text.models.base.select_amp_dtype", lambda _device: dtype
        )
        return _ner()

    key = "base_model.encoder.layer.0.attention.self.query.weight"
    on_ada = built_for(torch.bfloat16)
    on_p100 = built_for(torch.float16)
    # Cloned, not referenced: `state_dict` hands back the live parameters,
    # so the second load below would rewrite the first one's source.
    written_on_ada = {
        name: value.clone() for name, value in on_ada.state_dict().items()
    }
    written_as_fp32 = {
        name: value.float() if value.is_floating_point() else value.clone()
        for name, value in on_p100.state_dict().items()
    }

    on_p100.load_state_dict(written_on_ada, strict=True)
    on_ada.load_state_dict(written_as_fp32, strict=True)

    assert torch.equal(
        on_p100.state_dict()[key], written_on_ada[key].to(torch.float16)
    )
    assert torch.equal(
        on_ada.state_dict()[key], written_as_fp32[key].to(torch.bfloat16)
    )
