"""Masking the class head's logits in place must not change the answer.

`forward` in `ete.py`, `entity_linking.py` and `ner.py` used to build a
second full-size `[document, token, classes]` tensor with
`torch.where(token_mask, unmasked_class_logits, self._neg_inf)`, keeping the
unmasked tensor alive with no reader left for it. The fix masks the
classifier's own output in place with `masked_fill_` instead. `nn.Linear`'s
backward saves its input and weight, not its output, so the in-place write
is expected to be autograd-safe — proven here, not asserted, by calling each
model's real (fixed) `forward` and comparing it against an independently
built, out-of-place `torch.where` reference sharing the same weights and
inputs: identical pooled logits and identical gradients, on a batch with a
padded document and a fully-masked one.
"""

from collections.abc import Callable

import pytest
import torch
from torch import Tensor
from d3text.models.base import Model
from d3text.models.config import ModelConfig
from d3text.models.entity_linking import BrendaClassificationModel
from d3text.models.ete import ETEBrendaModel
from d3text.models.ner import NERClassificationModel
from d3text.schema import EntityType, RelationType, Schema

pytestmark = pytest.mark.slow

SCHEMA = Schema(
    entity_types=(
        EntityType(name="enzymes", prefix="enz"),
        EntityType(name="bacteria", prefix="bac"),
    ),
    relation_types=(
        RelationType(
            name="HasEnzyme", subject_types=("bacteria",), object_type="enzymes"
        ),
        RelationType(name="none", is_none=True),
    ),
)

BATCH, TOKENS, HIDDEN = 3, 6, 256
# bf16 autocast (the CPU default, see `select_amp_dtype`) loses precision
# next to the fp32 reference computed under the same autocast region.
ATOL = 2e-2


def _inputs() -> tuple[Tensor, Tensor]:
    """A batch with one full document, one padded one, one fully-masked one."""
    embeddings = torch.randn(BATCH, TOKENS, HIDDEN, requires_grad=True)
    mask = torch.ones(BATCH, TOKENS, dtype=torch.bool)
    mask[1, TOKENS // 2 :] = False  # padded
    mask[2, :] = False  # fully masked
    return embeddings, mask


def _old_reference(model: Model, embeddings: Tensor, mask: Tensor) -> Tensor:
    """The pooled class logits as the removed two-tensor version computed
    them: a fresh classifier call, masked out-of-place with `torch.where`,
    under the same autocast region `forward` itself runs under."""
    with model.autocast_context():
        hidden_output = model.hidden(embeddings)
        old_unmasked = model.classifier(hidden_output)
        token_mask = mask.unsqueeze(-1)
        old_logits = torch.where(
            token_mask, old_unmasked, torch.finfo(old_unmasked.dtype).min
        )
        return model._pool_logits(old_logits, mask=mask)


def _assert_masking_is_behavior_preserving(
    model: Model, real_forward: Callable[[Tensor, Tensor], Tensor]
) -> None:
    embeddings, mask = _inputs()

    new_pooled = real_forward(embeddings, mask)
    old_pooled = _old_reference(model, embeddings, mask)
    assert torch.allclose(new_pooled, old_pooled, atol=ATOL)

    params = [p for p in model.classifier.parameters() if p.requires_grad]
    new_grads = torch.autograd.grad(new_pooled.sum(), [embeddings, *params])
    old_grads = torch.autograd.grad(old_pooled.sum(), [embeddings, *params])
    for new_grad, old_grad in zip(new_grads, old_grads):
        assert torch.allclose(new_grad, old_grad, atol=ATOL)


def test_ner_masking_preserves_logits_and_gradients(patch_base_model):
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
    _assert_masking_is_behavior_preserving(model, model)


def test_entity_linking_masking_preserves_logits_and_gradients(
    patch_base_model,
):
    model = BrendaClassificationModel(
        schema=SCHEMA,
        config=ModelConfig(
            model_class="BrendaClassificationModel",
            base_model="prajjwal1/bert-mini",
            hidden_layers=[8],
            entity_logits_pooling="max",
        ),
        device="cpu",
    )
    model.eval()
    _assert_masking_is_behavior_preserving(
        model, lambda e, m: model(e, m).classes
    )


def test_ete_masking_preserves_logits_and_gradients(
    patch_base_model, empty_token_label_store
):
    model = ETEBrendaModel(
        schema=SCHEMA,
        config=ModelConfig(
            base_model="prajjwal1/bert-mini",
            hidden_layers=[8],
            ramp_epochs=0,
            token_labels_store=str(empty_token_label_store),
            entity_logits_pooling="max",
        ),
        device="cpu",
    )
    model.eval()
    _assert_masking_is_behavior_preserving(
        model, lambda e, m: model(e, m).classes
    )
