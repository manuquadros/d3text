"""The ETE hard-entity mask is computed a token-slice at a time.

The softmax, its clamp, its log and the product are four more tensors the size
of the entity logits, and they set the peak of the step even under `no_grad`.
Every row of a softmax over the last dimension is independent, so the slice
width cannot change a value — which is what these assert: the production width
and a width larger than the whole tensor must agree bitwise, candidate pairs
included.
"""

import pytest
import torch

import d3text.models.base as models
from d3text.models.config import ModelConfig
from d3text.models.ete import ETEBrendaModel
from d3text.models.model_types import IndexedRelation
from d3text.schema import EntityType, RelationType, Schema

pytestmark = pytest.mark.slow

TOKENS = 5000  # spans three slices at the production width of 2048

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


@pytest.fixture
def masking_ete(patch_base_model, device):
    """An `ETEBrendaModel` whose hard-entity mask actually fires.

    On a randomly initialised head the softmax entropy is near its maximum, so
    at the configured threshold nothing is selected and the path under test
    would not run.
    """
    model = ETEBrendaModel(
        schema=SCHEMA,
        class_matrix=torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        entity_index={"enz1": 0, "enz2": 1, "bac1": 2},
        config=ModelConfig(
            base_model="prajjwal1/bert-mini", hidden_layers=[8], ramp_epochs=0
        ),
        device=device,
    )
    model.to(device)
    model.eval()
    model.entity_threshold = 1e9
    return model


def _run(model, chunk, monkeypatch):
    monkeypatch.setattr(
        models, "pool_chunk_tokens", lambda documents, width: chunk
    )
    torch.manual_seed(0)
    embeddings = torch.randn(2, TOKENS, 256, device=model.device)
    mask = torch.ones(2, TOKENS, dtype=torch.bool, device=model.device)
    gold = [
        IndexedRelation(
            docix=0, subject="enz1", object="bac1", label=torch.tensor(0)
        )
    ]
    with torch.no_grad():
        return model(embeddings, mask, gold_relations=gold)


@pytest.mark.parametrize("chunk", [512, 2048, 4096, TOKENS, 10**9])
def test_hard_mask_is_invariant_to_the_slice_width(
    masking_ete, chunk, monkeypatch
):
    """The behavioural pin: a slice wider than the tensor is the unsliced
    expression, and every narrower width must agree with it bitwise."""
    whole = _run(masking_ete, 10**9, monkeypatch)
    sliced = _run(masking_ete, chunk, monkeypatch)

    assert torch.equal(whole[0], sliced[0])  # pooled entity logits
    assert torch.equal(whole[1], sliced[1])  # pooled class logits

    assert (whole[2] is None) == (sliced[2] is None)
    if whole[2] is not None:
        whole_meta, whole_logits = whole[2]
        sliced_meta, sliced_logits = sliced[2]
        assert set(whole_meta) == set(sliced_meta)
        for key in whole_meta:
            assert torch.equal(whole_meta[key], sliced_meta[key])
        assert torch.equal(whole_logits, sliced_logits)


def test_the_mask_actually_fires_in_this_fixture(masking_ete, monkeypatch):
    """Guard against the invariance tests passing vacuously: if the mask never
    selected a token, the hard-pair path would not run and every slice width
    would trivially agree on nothing."""
    _, _, relations = _run(masking_ete, 2048, monkeypatch)
    assert relations is not None
    meta, _ = relations
    # more pairs than the single gold pair -> the hard mask contributed
    assert meta["sequence"].numel() > 1
