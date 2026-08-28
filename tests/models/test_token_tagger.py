"""The span tagger head: additive loss, real gradients, scored detection.

A real ETEBrendaModel over the tiny injected BERT (``patch_base_model``), a
tiny on-disk encodings file, and a matching token-label store — so the whole
path from `config.token_labels_store` to the `test/detection_*` metrics runs
without the BRENDA data or a checkpoint anywhere near it.
"""

import logging

import h5py
import numpy
import pandas as pd
import pytest
import torch
from d3text import token_labels
from d3text.data.data import BrendaDataset, get_batch_loader
from d3text.models.config import ModelConfig
from d3text.models.base import Step
from d3text.models.entity_linking import BrendaClassificationModel
from d3text.models.ete import ETEBrendaModel
from d3text.models.token_supervision import TokenLabelReader
from d3text.schema import EntityType, RelationType, Schema
from d3text.token_labels import (
    BRENDA_LABELS,
    IGNORE_INDEX,
    DocumentLabels,
)
from d3text.training.update import BatchUpdate

pytestmark = pytest.mark.slow

BACTERIA = BRENDA_LABELS.by_prefix["bac"]
WINDOW = 32
TOKENS = WINDOW - 2  # [CLS] and [SEP] are stripped by the aggregation
NO_SPANS = numpy.zeros((0, token_labels.SPAN_COLUMNS), dtype=numpy.int32)

SCHEMA = Schema(
    entity_types=(
        EntityType(name="enzymes", prefix="enz"),
        EntityType(name="bacteria", prefix="bac"),
    )
)
# `ETEBrendaModel` reads its relation set off the schema, so its fixture needs
# one even though this file's assertions never touch a relation.
ETE_SCHEMA = Schema(
    entity_types=SCHEMA.entity_types,
    relation_types=(
        RelationType(
            name="HasEnzyme", subject_types=("bacteria",), object_type="enzymes"
        ),
        RelationType(
            name="HasSpecies",
            subject_types=("bacteria",),
            object_type="enzymes",
        ),
        RelationType(name="none", is_none=True),
    ),
)


@pytest.fixture
def corpus(tmp_path):
    """Encodings for pmids 11/12/13 and a frame over them."""
    path = tmp_path / "encodings.hdf5"
    with h5py.File(path, "w") as handle:
        for pmid in ("11", "12", "13"):
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
            "pubmed_id": [11, 12, 13],
            "relations": pd.Series([[], [], []]),
            "entities": [numpy.array([1, 0], dtype=numpy.uint8)] * 3,
            "classes": [numpy.array([1, 0], dtype=numpy.float32)] * 3,
        }
    )
    return BrendaDataset(frame, encodings=path)


def write_store(path, documents):
    with h5py.File(path, "w") as store:
        token_labels.write_label_space(store, BRENDA_LABELS)
        for pmid, codes in documents.items():
            token_labels.store_token_labels(
                store,
                pmid,
                DocumentLabels(
                    codes=numpy.asarray(codes, dtype=numpy.int8),
                    spans=NO_SPANS,
                    text_length=0,
                ),
            )
    return path


@pytest.fixture
def label_store(tmp_path):
    """Doc 11: a bacteria mention and an ignore run. Doc 12: all bacteria.
    Doc 13 deliberately absent."""
    doc_11 = numpy.zeros((1, WINDOW), dtype=numpy.int8)
    doc_11[0, 6:11] = BACTERIA  # aggregated positions 5..9
    doc_11[0, 16:19] = IGNORE_INDEX  # aggregated positions 15..17
    doc_12 = numpy.full((1, WINDOW), BACTERIA, dtype=numpy.int8)
    return write_store(tmp_path / "labels.hdf5", {"11": doc_11, "12": doc_12})


def build_model(patch_base_model, store=None):
    return ETEBrendaModel(
        schema=ETE_SCHEMA,
        class_matrix=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        entity_index={"enz1": 0, "bac1": 1},
        config=ModelConfig(
            base_model="prajjwal1/bert-mini",
            hidden_layers=[8],
            ramp_epochs=0,
            token_labels_store=str(store) if store else "",
        ),
        device="cpu",
    )


def loader_over(dataset, indices=None):
    return get_batch_loader(
        dataset,
        batch_size=1,
        sampler=torch.utils.data.SequentialSampler(
            range(len(dataset)) if indices is None else indices
        ),
    )


def one_batch(corpus):
    return next(iter(loader_over(corpus)))


# --------------------------------------------------------------------------- #
# Construction and checkpoint shape                                            #
# --------------------------------------------------------------------------- #
def test_without_a_store_the_model_is_unchanged(patch_base_model) -> None:
    """No head, no new state-dict keys: old checkpoints keep loading."""
    model = build_model(patch_base_model)

    assert model.token_tagger is None
    assert not any("token_tagger" in key for key in model.state_dict())
    assert model.epoch_loss_weights(0) == {
        "entity": 1.0,
        "class": 1.0,
        "relation": 1.0,
    }


def test_with_a_store_the_head_matches_the_label_space(
    patch_base_model, label_store
) -> None:
    model = build_model(patch_base_model, label_store)

    assert model.token_tagger is not None
    # One column per entity type plus OUTSIDE, so column c scores code c.
    assert model.token_tagger.out_features == 1 + len(BRENDA_LABELS.types)
    assert "token_tagger.weight" in model.state_dict()
    assert model.epoch_loss_weights(0)["token"] == 1.0


# --------------------------------------------------------------------------- #
# The loss reads the labels                                                    #
# --------------------------------------------------------------------------- #
def test_token_loss_is_none_without_a_store(patch_base_model, corpus) -> None:
    model = build_model(patch_base_model)

    *_, token_loss = model.compute_batch_losses(one_batch(corpus))

    assert token_loss is None


def test_token_loss_changes_when_the_labels_change(
    patch_base_model, corpus, label_store, tmp_path
) -> None:
    """The head is trained on the store's targets, not on a constant: the
    same weights over the same document must lose differently under
    different labels."""
    model = build_model(patch_base_model, label_store)
    batch = one_batch(corpus)  # doc 11

    *_, original = model.compute_batch_losses(batch)

    flipped = numpy.zeros((1, WINDOW), dtype=numpy.int8)
    flipped[0, 6:11] = BRENDA_LABELS.by_prefix["enz"]
    model._token_labels = TokenLabelReader(
        write_store(tmp_path / "flipped.hdf5", {"11": flipped})
    )
    *_, relabelled = model.compute_batch_losses(batch)

    assert original is not None and relabelled is not None
    assert original.item() != pytest.approx(relabelled.item())


def test_all_masked_labels_cost_exactly_nothing(
    patch_base_model, corpus, tmp_path
) -> None:
    """A document that is one ignore region contributes a differentiable
    zero — the divisor is the unmasked count, and there is none."""
    masked = numpy.full((1, WINDOW), IGNORE_INDEX, dtype=numpy.int8)
    store = write_store(tmp_path / "masked.hdf5", {"11": masked})
    model = build_model(patch_base_model, store)

    *_, token_loss = model.compute_batch_losses(one_batch(corpus))

    assert token_loss is not None
    assert token_loss.item() == 0.0


def test_token_gradient_reaches_the_head_and_the_trunk(
    patch_base_model, corpus, label_store
) -> None:
    """The term must train the tagger and shape the shared hidden block —
    the localization signal the pooled loss cannot supply."""
    model = build_model(patch_base_model, label_store)

    *_, token_loss = model.compute_batch_losses(one_batch(corpus))
    assert token_loss is not None
    token_loss.backward()

    assert model.token_tagger is not None
    assert model.token_tagger.weight.grad is not None
    assert model.token_tagger.weight.grad.abs().sum() > 0
    assert model.hidden_layers[0][0].weight.grad is not None
    assert model.hidden_layers[0][0].weight.grad.abs().sum() > 0


def test_a_stale_store_fails_loudly(patch_base_model, corpus, tmp_path) -> None:
    """Labels of the wrong window geometry would land on the wrong tokens."""
    store = write_store(
        tmp_path / "stale.hdf5",
        {"11": numpy.zeros((2, WINDOW), dtype=numpy.int8)},
    )
    model = build_model(patch_base_model, store)

    with pytest.raises(ValueError, match="regenerate"):
        model.compute_batch_losses(one_batch(corpus))


# --------------------------------------------------------------------------- #
# The epoch carries the term                                                   #
# --------------------------------------------------------------------------- #
def test_run_epoch_reports_and_trains_on_the_token_loss(
    patch_base_model, corpus, label_store
) -> None:
    model = build_model(patch_base_model, label_store)
    update = BatchUpdate(
        model, torch.optim.SGD(model.parameters(), lr=0.5), "cpu"
    )
    assert model.token_tagger is not None
    before = model.token_tagger.weight.detach().clone()

    losses, _ = model.run_epoch(
        data=loader_over(corpus, indices=[0, 1]),
        step=Step.TRAINING,
        epoch=0,
        update=update,
    )

    assert "token" in losses
    assert losses["token"] > 0
    # The update stepped on the token term: only it reaches the tagger head.
    assert not torch.equal(before, model.token_tagger.weight.detach())


def test_run_epoch_keys_are_unchanged_without_a_store(
    patch_base_model, corpus
) -> None:
    model = build_model(patch_base_model)
    update = BatchUpdate(
        model, torch.optim.SGD(model.parameters(), lr=0.0), "cpu"
    )

    losses, _ = model.run_epoch(
        data=loader_over(corpus, indices=[0]),
        step=Step.VALIDATION,
        epoch=0,
        update=update,
    )

    assert set(losses) == {"entity", "class", "relation"}


# --------------------------------------------------------------------------- #
# Evaluation scores detection, ignore set applied and reported                 #
# --------------------------------------------------------------------------- #
def test_evaluate_model_scores_detection_against_the_store(
    patch_base_model, corpus, label_store
) -> None:
    """The tagger is rigged to say `bacteria` on every token, so every score
    below is arithmetic, not luck: doc 12 (all-bacteria gold) is the one TP;
    doc 11's full-document span misses the short gold mention but overlaps
    the ignore run, so it is masked and counted, not charged; doc 13 has no
    labels and is reported missing rather than silently skipped."""
    model = build_model(patch_base_model, label_store)
    assert model.token_tagger is not None
    with torch.no_grad():
        model.token_tagger.weight.zero_()
        model.token_tagger.bias.zero_()
        model.token_tagger.bias[BACTERIA] = 10.0

    metrics = model.evaluate_model(loader_over(corpus))

    assert metrics["test/detection_true_positives"] == 1.0
    assert metrics["test/detection_false_positives"] == 0.0
    assert metrics["test/detection_false_negatives"] == 1.0
    assert metrics["test/detection_ignored_predictions"] == 1.0
    assert metrics["test/detection_precision"] == pytest.approx(1.0)
    assert metrics["test/detection_recall"] == pytest.approx(0.5)
    assert metrics["test/detection_f1"] == pytest.approx(2 / 3)
    assert metrics["test/detection_ignore_regions"] == 1.0
    assert metrics["test/detection_ignore_firing_rate"] == pytest.approx(1.0)
    assert metrics["test/detection_documents"] == 2.0
    assert metrics["test/detection_documents_missing_labels"] == 1.0
    assert metrics["test/detection_bacteria_recall"] == pytest.approx(0.5)


def build_brenda_model(patch_base_model, store=None):
    return BrendaClassificationModel(
        schema=SCHEMA,
        class_matrix=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        entity_index={"enz1": 0, "bac1": 1},
        config=ModelConfig(
            base_model="prajjwal1/bert-mini",
            hidden_layers=[8],
            ramp_epochs=0,
            token_labels_store=str(store) if store else "",
        ),
        device="cpu",
    )


@pytest.mark.parametrize(
    "build, logger_name",
    [
        (build_model, "d3text.models.ete"),
        (build_brenda_model, "d3text.models.entity_linking"),
    ],
)
def test_evaluate_model_prints_the_detection_report_it_returns(
    patch_base_model,
    corpus,
    label_store,
    caplog,
    monkeypatch,
    build,
    logger_name,
) -> None:
    """The detection block must reach the console the way the entity, class
    and relation blocks already do, not just MLflow — the one sink built to
    fail open and silently drop it when no tracking server is reachable."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    model = build(patch_base_model, label_store)

    with caplog.at_level(logging.INFO, logger=logger_name):
        metrics = model.evaluate_model(loader_over(corpus))

    assert "test/detection_precision" in metrics
    assert "test/detection_recall" in metrics
    assert "test/detection_f1" in metrics

    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "Detection metrics" in logged
    assert str(metrics["test/detection_precision"]) in logged
    assert str(metrics["test/detection_recall"]) in logged
    assert str(metrics["test/detection_f1"]) in logged


def test_evaluate_model_emits_no_detection_keys_without_a_store(
    patch_base_model, corpus
) -> None:
    """Scored with the parent classification model: without a store, no
    detection key exists to be misread as a measurement of nothing."""
    model = BrendaClassificationModel(
        schema=SCHEMA,
        class_matrix=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        entity_index={"enz1": 0, "bac1": 1},
        config=ModelConfig(
            base_model="prajjwal1/bert-mini", hidden_layers=[8], ramp_epochs=0
        ),
        device="cpu",
    )

    metrics = model.evaluate_model(loader_over(corpus))

    assert not any(key.startswith("test/detection") for key in metrics)
