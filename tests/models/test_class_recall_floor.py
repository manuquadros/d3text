"""No class channel is allowed to go dead.

Correct pooling wiring can still leave a low-prevalence channel predicting
nothing, and the pooled loss hides it: never firing is near-optimal on the
documents lacking the class. Only a test that trains can see it. The floors
are for a `--limit` run, not the whole training split.
"""

import pytest
import torch

from d3text import data, factory
from d3text.datasets.brenda import BRENDA_SCHEMA, brenda_dataset
from d3text.training.trainer import Trainer
from d3text.models.config import ModelConfig, encodings_path
from d3text.models.ete import ETEBrendaModel

# `--limit` picks the entity vocabulary as well as the documents, so the
# floors below are only valid at this value; change it and re-measure.
LIMIT = 500
THRESHOLD = 0.5

SEED = 0

# Low-prevalence floor: sqrt(lowest live recall * highest dead one), rounded
# down, the cutoff furthest from both in ratio. High-prevalence classes were
# never seen dead, so theirs sits just below 1.0, which one document fails.
RECALL_FLOORS = {
    "enzymes": 0.90,
    "other_organisms": 0.90,
    "bacteria": 0.06,
    "strains": 0.03,
}


def training_config() -> ModelConfig:
    """`tests/best_config_so_far.toml` at a batch budget this fits in 6 GB.

    Its `class_negative_abstention`, `unfrozen_top_layers` and
    `base_model_lr` are left out: they change what is measured, and the
    floors were not calibrated with them.

    `entity_logits_pooling` is deliberately not set: the shipped default is
    what is under test.
    """
    return ModelConfig(
        model_class="ETEBrendaModel",
        base_model="michiyasunaga/BioLinkBERT-base",
        optimizer="nadam",
        lr=0.001,
        lr_scheduler="exponential",
        dropout=0.2,
        hidden_layers=[128],
        normalization="layer",
        batch_size=8,
        batch_max_chunks=64,
        num_epochs=6,
        patience=10,
        relation_label_smoothing=0,
        common_hidden_block=True,
        ramp_epochs=4,
        separate_predicate_layer=True,
        token_supervision=True,
    )


@pytest.fixture(scope="module")
def trained_run():
    """A short training run and the validation loader to score it on.

    Module-scoped, since the run is the expensive part. Seeded here rather than
    left to conftest's function-scoped autouse fixture, which pytest would set
    up *after* the training it is meant to make reproducible.
    """
    torch.manual_seed(SEED)
    config = training_config()
    dataset = brenda_dataset(
        schema=BRENDA_SCHEMA,
        encodings=encodings_path(config.base_model),
        limit=LIMIT,
    )
    train_split = dataset.data["train"]

    model = factory.build_model(
        config,
        BRENDA_SCHEMA,
        class_freqs=data.compute_frequencies(train_split, column="classes"),
    )
    model.to(model.device)

    train_data = data.get_batch_loader(
        dataset=train_split,
        batch_size=config.batch_size,
        max_chunks=config.batch_max_chunks,
    )
    val_data = data.get_batch_loader(
        dataset=dataset.data["val"],
        batch_size=config.batch_size,
        max_chunks=config.batch_max_chunks,
    )

    # `patience` exceeds `num_epochs`, so nothing stops early and the head
    # scored is the last epoch's, as `train` would write, with no snapshot.
    Trainer(model).fit(
        train_data=train_data, val_data=val_data, save_checkpoint=False
    )

    return model, val_data


def document_recall(model, val_data, threshold=THRESHOLD) -> dict[str, float]:
    """Per class, the share of validation documents carrying it that fire.

    Counted per batch rather than accumulated as logits, so the split's size
    does not get in the way of a test that already trains.
    """
    model.eval()
    names = model.known_classes
    positives = dict.fromkeys(names, 0)
    hits = dict.fromkeys(names, 0)

    with torch.no_grad():
        for batch in val_data:
            class_logits = model.get_batch_logits(batch).classes
            probs = torch.sigmoid(model.drop_oos(class_logits).float()).cpu()
            gold = (
                model.ground_truth(batch)
                .classes[:, : probs.shape[1]]
                .bool()
                .cpu()
            )
            fired = probs >= threshold

            for column, name in enumerate(names):
                positives[name] += int(gold[:, column].sum())
                hits[name] += int((gold[:, column] & fired[:, column]).sum())

    return {
        name: hits[name] / positives[name] for name in names if positives[name]
    }


@pytest.mark.integration
@pytest.mark.slow
def test_no_class_channel_is_dead(trained_run):
    """Every class detects the documents it belongs to, above its floor.

    One assertion for all four rather than four tests: they share a training
    run, and a collapse takes the low-prevalence pair together, so a report
    naming every channel that fell is what makes a failure readable.
    """
    model, val_data = trained_run
    recall = document_recall(model, val_data)

    assert set(recall) == set(RECALL_FLOORS), (
        "the class head's columns are not the four classes the floors were "
        f"measured on: {sorted(recall)}"
    )

    # Printed on a pass too: the floors are calibrated off this table, and a
    # value drifting toward its floor is the warning before the failure.
    print(
        "\nper-class document recall at p >= "
        f"{THRESHOLD}\n"
        + "\n".join(
            f"  {name:<18}{measured:.3f}  (floor {RECALL_FLOORS[name]:.2f})"
            for name, measured in sorted(recall.items())
        )
    )

    dead = {
        name: (measured, RECALL_FLOORS[name])
        for name, measured in recall.items()
        if measured < RECALL_FLOORS[name]
    }
    assert not dead, "document recall below floor: " + ", ".join(
        f"{name} {measured:.3f} < {floor:.2f}"
        for name, (measured, floor) in sorted(dead.items())
    )


def test_training_config_is_accepted():
    """The integration run's config must pass `ModelConfig` validation.

    The run is deselected by default, so a refusal added to `ModelConfig`
    would otherwise only surface in a twenty-minute run's fixture.
    """
    assert training_config().token_supervision


def test_document_recall_reads_the_class_logits(
    patch_base_model, empty_token_label_store, monkeypatch
):
    """`document_recall` scores class logits, not the relation half.

    `get_batch_logits` and `ground_truth` return `(classes, relations)`
    tuples; indexing the wrong one only fails once a trained model reaches
    it, so a tiny untrained model over a hand-built batch pins it here.
    """
    empty_token_label_store(None)
    model = ETEBrendaModel(
        schema=BRENDA_SCHEMA,
        config=ModelConfig(
            base_model="prajjwal1/bert-mini",
            hidden_layers=[8],
            ramp_epochs=0,
            token_supervision=True,
        ),
        device="cpu",
    )
    tokens = 10
    monkeypatch.setattr(
        model,
        "get_token_embeddings",
        lambda batch: (
            torch.randn(len(batch), tokens, 256),
            torch.ones(len(batch), tokens, dtype=torch.bool),
        ),
    )
    monkeypatch.setattr(model, "_stored_mentions", lambda batch: {})
    gold = torch.zeros(model.num_of_classes)
    gold[model.class_columns[0]] = 1
    batch = [{"classes": gold}, {"classes": gold}]

    recall = document_recall(model, [batch])

    assert set(recall) == {model.known_classes[0]}
    assert 0.0 <= recall[model.known_classes[0]] <= 1.0
