"""``ETEBrendaModel.get_batch_logits`` must stay on the model's device.

A prior version of ``get_batch_logits`` built a per-document entity tensor
(via a helper that defaulted its device to ``"cuda"``) and fed it into
``forward``, so on a CPU-only machine the whole ETE path raised before the
forward ran, and on a machine that *has* a GPU a CPU-built model silently got
those tensors on the wrong device. That entity tensor was never read inside
``forward`` — nothing in the relation-candidate logic consulted it — so it was
later deleted outright rather than merely fixed; this test now just pins that
``get_batch_logits`` keeps running, and keeps its outputs on the model's
device, with that dead plumbing gone. Every existing test that reaches
``get_batch_logits`` stubs it, which is why nothing caught the original bug.
"""

import pytest
import torch
from d3text.models.config import ModelConfig
from d3text.models.ete import ETEBrendaModel
from d3text.schema import EntityType, RelationType, Schema

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

pytestmark = pytest.mark.slow


@pytest.fixture
def cpu_ete(patch_base_model):
    """A real ETEBrendaModel built and placed on CPU."""
    model = ETEBrendaModel(
        schema=SCHEMA,
        class_matrix=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        entity_index={"enz1": 0, "bac1": 1},
        config=ModelConfig(
            base_model="prajjwal1/bert-mini", hidden_layers=[8], ramp_epochs=0
        ),
        device="cpu",
    )
    model.to("cpu")
    model.eval()
    return model


def _batch(tokens: int = 8):
    """Two single-chunk documents, shaped as the DataLoader yields them."""
    return [
        {
            "id": torch.tensor(pmid),
            "doc_id": torch.tensor([[pmid]]),
            "sequence": {
                "input_ids": torch.randint(0, 1000, (1, tokens)),
                "attention_mask": torch.ones((1, tokens), dtype=torch.int64),
            },
            "entities": torch.tensor([entities], dtype=torch.uint8),
            "classes": torch.tensor([[1.0, 0.0]]),
        }
        for pmid, entities in ((10, [1, 0]), (20, [0, 1]))
    ]


def test_get_batch_logits_runs_unstubbed_on_a_cpu_model(cpu_ete):
    with torch.no_grad():
        entity_logits, class_logits, _ = cpu_ete.get_batch_logits(_batch())

    assert entity_logits.device.type == "cpu"
    assert class_logits.device.type == "cpu"
    assert tuple(entity_logits.shape) == (2, cpu_ete.num_of_entities)
