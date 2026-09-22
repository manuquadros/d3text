"""A trunk forward keeps its unpadded windows off the padded ones' mask.

`BertModel` hands SDPA a mask whenever any window in the batch is padded,
and SDPA cannot pick the flash kernel once a mask exists, so one padded
window used to cost every window in the batch the faster kernel. These pin
the split: full windows go through with no mask, padded ones with theirs,
and the stitched output is what a single masked forward would have given.
Random-init 2-layer BERT on CPU (`patch_base_model`), so no download.
"""

from unittest.mock import patch

import torch
from d3text.models import base
from d3text.models.config import ModelConfig
from d3text.models.ner import NERClassificationModel
from d3text.schema import EntityType, Schema

SCHEMA = Schema(entity_types=(EntityType(name="enzymes", prefix="enz"),))
WINDOW = 16


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


def _item(doc_id: int, n_windows: int, last_window_tokens: int) -> dict:
    """One document of `n_windows` windows; only the last may be padded."""
    mask = torch.ones(n_windows, WINDOW, dtype=torch.long)
    mask[-1, last_window_tokens:] = 0
    return {
        "id": torch.tensor(doc_id),
        "doc_id": torch.zeros(n_windows, dtype=torch.uint8),
        "sequence": {
            "input_ids": torch.randint(1, 999, (n_windows, WINDOW)),
            "attention_mask": mask,
        },
    }


def _masks_of(spy) -> list:
    return [call.kwargs["attention_mask"] for call in spy.call_args_list]


def _one_masked_forward(model, batch):
    inputs = model.batch_input_tensors(batch)
    with torch.no_grad(), model.autocast_context():
        return model.base_model(
            input_ids=inputs["input_ids"].to(torch.int),
            attention_mask=inputs["attention_mask"],
        ).last_hidden_state


def test_full_and_padded_windows_are_forwarded_apart(patch_base_model):
    """Two calls: the full windows with no mask, the padded ones with
    exactly their rows of the mask. Output equality alone would hold on the
    old single masked call too; the call shape is what pins the fix."""
    torch.manual_seed(0)
    model = _ner()
    batch = [_item(1, 3, WINDOW // 2), _item(2, 2, WINDOW)]
    padded_mask = batch[0]["sequence"]["attention_mask"][-1:]

    with patch.object(
        model.base_model, "forward", wraps=model.base_model.forward
    ) as spy:
        embeddings, _ = model.get_token_embeddings(batch)

    masks = _masks_of(spy)
    assert len(masks) == 2
    assert masks[0] is None
    assert torch.equal(masks[1], padded_mask)

    expected = _one_masked_forward(model, batch)
    inputs = model.batch_input_tensors(batch)
    with torch.no_grad(), model.autocast_context():
        per_window = model._forward_windows(
            lambda rows, mask: model.base_model(
                input_ids=inputs["input_ids"][rows].to(torch.int),
                attention_mask=mask,
            ).last_hidden_state,
            inputs["attention_mask"],
        )
    torch.testing.assert_close(per_window, expected)
    assert embeddings.shape[0] == len(batch)


def test_a_batch_with_no_padded_window_makes_one_maskless_call(
    patch_base_model,
):
    torch.manual_seed(0)
    model = _ner()
    batch = [_item(1, 2, WINDOW), _item(2, 1, WINDOW)]

    with patch.object(
        model.base_model, "forward", wraps=model.base_model.forward
    ) as spy:
        model.get_token_embeddings(batch)

    assert _masks_of(spy) == [None]


def test_a_batch_of_only_padded_windows_makes_one_masked_call(
    patch_base_model,
):
    torch.manual_seed(0)
    model = _ner()
    batch = [_item(1, 1, WINDOW - 3), _item(2, 1, WINDOW - 5)]
    inputs = model.batch_input_tensors(batch)

    with patch.object(
        model.base_model, "forward", wraps=model.base_model.forward
    ) as spy:
        model.get_token_embeddings(batch)

    masks = _masks_of(spy)
    assert len(masks) == 1
    assert torch.equal(masks[0], inputs["attention_mask"])


def test_top_layer_replay_splits_the_same_way(patch_base_model):
    """The layer-boundary replay builds its own mask; it must split too."""
    torch.manual_seed(0)
    model = _ner(unfrozen_top_layers=1)
    hidden = model.base_model.config.hidden_size
    prefix = torch.randn(3, WINDOW, hidden)
    mask = torch.ones(3, WINDOW, dtype=torch.long)
    mask[1, WINDOW // 2 :] = 0

    with patch.object(
        base,
        "create_bidirectional_mask",
        wraps=base.create_bidirectional_mask,
    ) as spy:
        replayed = model._replay_top_layers(prefix, mask)

    masks = _masks_of(spy)
    assert len(masks) == 2
    assert masks[0] is None
    assert torch.equal(masks[1], mask[1:2])

    top = model.base_model.encoder.layer[-1]
    expected = top(
        prefix,
        base.create_bidirectional_mask(
            config=model.base_model.config,
            inputs_embeds=prefix,
            attention_mask=mask,
        ),
    )
    torch.testing.assert_close(replayed, expected, rtol=1e-4, atol=1e-4)
    assert replayed.requires_grad
