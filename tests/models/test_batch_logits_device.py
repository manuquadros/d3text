"""``ETEBrendaModel.get_batch_logits`` must stay on the model's device.

``get_batch_entities`` used to default ``device`` to ``"cuda"`` and
``get_batch_logits`` called it bare, so on a CPU-only machine the whole ETE
path raised before the forward ran, and on a machine that *has* a GPU a
CPU-built model silently got its entity indices on the wrong device. Every
existing test that reaches ``get_batch_logits`` stubs it, which is why nothing
caught it.
"""

import pytest
import torch
from d3text.models.config import ModelConfig
from d3text.models.models import ETEBrendaModel

pytestmark = pytest.mark.slow


@pytest.fixture
def cpu_ete(patch_base_model):
    """A real ETEBrendaModel built and placed on CPU."""
    model = ETEBrendaModel(
        classes={"enzymes": {"enz1"}, "bacteria": {"bac1"}},
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


def _capture_forward(model):
    """Record the ``entities_in_batch`` argument the forward is called with."""
    captured: list[tuple[torch.Tensor, ...]] = []
    real_forward = model.forward

    def spy(embeddings, attention_mask, entities_in_batch, **kwargs):
        captured.append(entities_in_batch)
        return real_forward(
            embeddings, attention_mask, entities_in_batch, **kwargs
        )

    model.forward = spy
    return captured


def test_get_batch_logits_runs_unstubbed_on_a_cpu_model(cpu_ete):
    with torch.no_grad():
        entity_logits, class_logits, _ = cpu_ete.get_batch_logits(_batch())

    assert entity_logits.device.type == "cpu"
    assert class_logits.device.type == "cpu"
    assert tuple(entity_logits.shape) == (2, cpu_ete.num_of_entities)


def test_get_batch_logits_builds_entities_on_the_model_device(cpu_ete):
    captured = _capture_forward(cpu_ete)

    with torch.no_grad():
        cpu_ete.get_batch_logits(_batch())

    (entities_in_batch,) = captured
    assert len(entities_in_batch) == 2
    assert all(
        entities.device.type == cpu_ete.device for entities in entities_in_batch
    )
