"""The padding fill must fit the logits' dtype.

`forward` used to fill padded positions with a fixed -1e9, which bfloat16
absorbs and float16 cannot hold: `masked_fill_` refuses the value outright, so
the first training batch on an fp16-only GPU died before pooling. CPU autocast
in float16 reproduces the refusal without a GPU.
"""

import pytest
import torch
from d3text.models.base import Model
from d3text.models.config import ModelConfig
from d3text.models.ner import NERClassificationModel
from d3text.schema import EntityType, Schema

pytestmark = pytest.mark.slow

SCHEMA = Schema(
    entity_types=(EntityType(name="enzymes", prefix="enz"),),
    relation_types=(),
)


def test_forward_masks_padding_under_float16_autocast(patch_base_model):
    """Under float16 the masked forward must run at all, and padding must
    stay invisible: with `max` pooling, whatever sits in the padded positions
    cannot move a document's pooled logits."""
    model = NERClassificationModel(
        schema=SCHEMA,
        config=ModelConfig(
            model_class="NERClassificationModel",
            base_model="prajjwal1/bert-mini",
            hidden_layers=[8],
            entity_logits_pooling="max",
        ),
        device="cpu",
    )
    model.eval()
    model.amp_dtype = torch.float16

    torch.manual_seed(0)
    embeddings = torch.randn(2, 6, 256)
    mask = torch.ones(2, 6, dtype=torch.bool)
    mask[1, 3:] = False
    with torch.no_grad():
        pooled = model(embeddings, mask)
        embeddings[1, 3:] = torch.randn(3, 256) * 50
        repooled = model(embeddings, mask)

    assert pooled.dtype == torch.float16
    assert torch.isfinite(pooled).all()
    assert torch.equal(pooled, repooled)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32]
)
def test_mask_padding_fills_with_the_dtype_minimum(dtype):
    """The fill is the lowest finite value of the tensor's own dtype, so it
    always fits and never becomes -inf, which the masked mean would turn into
    a NaN. Real positions are untouched."""
    logits = torch.zeros(2, 4, 3, dtype=dtype)
    mask = torch.ones(2, 4, dtype=torch.bool)
    mask[0, 2:] = False

    Model._mask_padding(logits, mask)

    assert torch.isfinite(logits).all()
    assert bool((logits[0, 2:] == torch.finfo(dtype).min).all())
    assert bool((logits[mask] == 0).all())
