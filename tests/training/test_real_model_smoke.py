"""`Trainer.fit` against a real model's `run_epoch`, not a stub.

`tests/training/test_trainer.py` drives `_ScriptedModel`, which defines its
own `run_epoch` and so never binds `fit`'s call to the signature a production
model actually exposes. A `run_epoch` that reverted to its old three-argument
form would still pass that whole file, and the suite would only flag it
through an unrelated structural assertion. This file puts one real
`BrendaClassificationModel` batch through `Trainer.fit`, on a tiny injected
BERT and a tiny on-disk encodings file, so the call itself — `update=`
included — is what is under test.
"""

import h5py
import numpy
import pandas as pd
import pytest
import torch
from d3text.data.data import BrendaDataset, get_batch_loader
from d3text.models.config import ModelConfig
from d3text.models.models import BrendaClassificationModel
from d3text.training.trainer import Trainer
from transformers import BertConfig, BertModel

pytestmark = pytest.mark.slow

WINDOW = 16


@pytest.fixture(autouse=True)
def _offline_no_dropout(monkeypatch):
    """A tiny, dropout-free random BERT: `Model.__init__` loads no real base
    model, and dropout would make `Trainer.fit`'s gradient-scale calibration
    (below) run-to-run flaky rather than a fixed number of skipped steps."""

    def tiny_bert(*_args, **_kwargs):
        return BertModel(
            BertConfig(
                vocab_size=1000,
                hidden_size=256,
                num_hidden_layers=2,
                num_attention_heads=4,
                intermediate_size=512,
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
            )
        )

    monkeypatch.setattr("d3text.models.models.load_base_model", tiny_bert)
    torch.manual_seed(0)


@pytest.fixture
def corpus(tmp_path):
    """Encodings for two tiny synthetic documents."""
    path = tmp_path / "encodings.hdf5"
    with h5py.File(path, "w") as handle:
        for pmid in ("1", "2"):
            group = handle.create_group(pmid)
            group.create_dataset(
                "input_ids",
                data=numpy.arange(WINDOW, dtype=numpy.int64).reshape(1, -1),
            )
            group.create_dataset(
                "attention_mask",
                data=numpy.ones((1, WINDOW), dtype=numpy.int64),
            )
    frame = pd.DataFrame(
        {
            "pubmed_id": [1, 2],
            "relations": pd.Series([[], []]),
            "entities": [numpy.array([1, 0], dtype=numpy.uint8)] * 2,
            "classes": [numpy.array([1, 0], dtype=numpy.float32)] * 2,
        }
    )
    return BrendaDataset(frame, encodings=path)


def loader_over(dataset):
    return get_batch_loader(
        dataset,
        batch_size=2,
        sampler=torch.utils.data.SequentialSampler(range(len(dataset))),
    )


def test_fit_trains_a_real_models_run_epoch(corpus):
    """`Trainer.fit` must call the real, four-argument `run_epoch` — a
    reverted three-argument signature raises `TypeError` on the first batch,
    which this catches and a stubbed `run_epoch` cannot.

    `num_epochs=3`: on CPU this model's `amp_dtype` is `float16`, and the
    `GradScaler`'s fixed initial scale overflows this tiny network's first
    gradient and skips that step by design — deterministically, since dropout
    is off above — before halving the scale and succeeding on the second. Two
    epochs would already clear that; the third is margin, not superstition.
    """
    model = BrendaClassificationModel(
        classes={"enzymes": {"enz1"}, "bacteria": {"bac1"}},
        class_matrix=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        entity_index={"enz1": 0, "bac1": 1},
        config=ModelConfig(
            base_model="prajjwal1/bert-mini",
            hidden_layers=[8],
            num_epochs=3,
            ramp_epochs=0,
            lr=0.1,
        ),
        device="cpu",
    )
    before = model.hidden_layers[0][0].weight.detach().clone()

    result = Trainer(model).fit(
        train_data=loader_over(corpus), save_checkpoint=False
    )

    assert result is None  # no validation data: nothing to snapshot
    assert not torch.equal(model.hidden_layers[0][0].weight.detach(), before)
