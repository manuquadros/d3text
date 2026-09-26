"""A frozen run with no embeddings store says what it is paying for.

`unfrozen_top_layers = 0` makes the trunk's output constant, so every epoch
after the first recomputes what it already computed. Nothing failed, so only
a warning distinguishes that run from one reading a store; these pin that it
is emitted where the cost is paid and nowhere else.
"""

import logging
import re

import pytest
import torch
from d3text.models import base
from d3text.models.config import ModelConfig
from d3text.models.ner import NERClassificationModel
from d3text.schema import EntityType, Schema
from d3text.training.update import BatchUpdate
from torch.utils.data import DataLoader

SCHEMA = Schema(entity_types=(EntityType(name="enzymes", prefix="enz"),))


def _ner(unfrozen_top_layers: int) -> NERClassificationModel:
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


@pytest.fixture
def no_store(monkeypatch):
    monkeypatch.setattr("d3text.models.base.embeddings_store", lambda _: None)


def _warnings(caplog) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING
    ]


def test_a_frozen_trunk_without_a_store_warns(
    no_store, patch_base_model, caplog
):
    """The one message has to be true of a single-pass run as well.

    `evaluate` builds through the same factory and never runs an epoch, so
    naming the training loop's repetition promises it a cost it does not
    pay; it does pay the trunk once, which is why the warning fires there.
    """
    with caplog.at_level(logging.WARNING, logger="d3text.models.base"):
        _ner(0)

    (frozen,) = [m for m in _warnings(caplog) if "unfrozen_top_layers=0" in m]
    assert "precompute-embeddings" in frozen
    assert not re.search(r"epoch|validation", frozen)


def test_the_warning_names_the_trunk_and_fires_once(
    no_store, patch_base_model, caplog
):
    """One line per model built, not one per batch: the lookup it reports on
    runs every batch, and a warning there would drown the log it belongs in."""
    with caplog.at_level(logging.WARNING, logger="d3text.models.base"):
        model = _ner(0)
        for _ in range(3):
            model.get_token_embeddings(
                [
                    {
                        "id": torch.tensor(777),
                        "doc_id": torch.zeros(1, dtype=torch.uint8),
                        "sequence": {
                            "input_ids": torch.randint(0, 999, (1, 24)),
                            "attention_mask": torch.ones(
                                1, 24, dtype=torch.long
                            ),
                        },
                    }
                ]
            )

    frozen = [m for m in _warnings(caplog) if "unfrozen_top_layers=0" in m]
    assert len(frozen) == 1
    assert "prajjwal1/bert-mini" in frozen[0]


def test_a_trainable_trunk_does_not_warn(no_store, patch_base_model, caplog):
    """A trunk that trains recomputes because it must, not because a store is
    missing, and a store would be wrong to read."""
    with caplog.at_level(logging.WARNING, logger="d3text.models.base"):
        _ner(1)

    assert not [m for m in _warnings(caplog) if "unfrozen_top_layers=0" in m]


def test_a_frozen_trunk_with_a_store_does_not_warn(
    patch_base_model, caplog, monkeypatch
):
    monkeypatch.setattr(
        "d3text.models.base.embeddings_store", lambda _: object()
    )

    with caplog.at_level(logging.WARNING, logger="d3text.models.base"):
        _ner(0)

    assert not [m for m in _warnings(caplog) if "unfrozen_top_layers=0" in m]


def test_each_pass_reports_what_the_store_has_served(stub, caplog, monkeypatch):
    """The store's counters otherwise surface only in `close`, at process
    exit, so a multi-hour run cannot tell whether it is being answered."""

    class _Store:
        def summary(self) -> str:
            return "/data/store served 7 of 9 documents"

    monkeypatch.setattr(
        "d3text.models.base.embeddings_store", lambda _: _Store()
    )
    model = stub(
        base.Model,
        config=ModelConfig(
            model_class="NERClassificationModel", unfrozen_top_layers=0
        ),
    )
    trainable = torch.nn.Linear(1, 1)
    update = BatchUpdate(
        trainable, torch.optim.SGD(trainable.parameters(), lr=0.1), "cpu"
    )

    with caplog.at_level(logging.INFO, logger="d3text.models.base"):
        model.run_epoch(DataLoader([]), epoch=0, update=update)

    assert any(
        "/data/store served 7 of 9 documents" in record.getMessage()
        and "cumulative" in record.getMessage()
        for record in caplog.records
    )
