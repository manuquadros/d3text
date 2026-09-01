"""No class channel is allowed to go dead.

`test_pooling_default.py` pins the wiring, which is a different guarantee: a
correct, distinct, correctly-ordered pooling can still leave the bacteria and
strains channels predicting nothing. The pooled loss hides it — a channel that
never fires is near-optimal on the 75-83% of documents where its class is
absent — which is what makes this worth a test that trains. The floors separate
a live channel from a dead one and are indifferent to which pooling produced
it. Read the numbers as a `--limit 500` measurement: the same channels reach
0.83-0.93 on the whole training split.
"""

import pytest
import torch

from d3text import data, factory
from d3text.datasets.brenda import BRENDA_SCHEMA, brenda_dataset
from d3text.training.trainer import Trainer
from d3text.models.config import ModelConfig, encodings

pytestmark = [pytest.mark.integration, pytest.mark.slow]

# The arm the floors below were measured on. `--limit` picks the entity
# vocabulary as well as the documents, so it is part of the run's identity and
# not free to change without re-measuring; 500 curated documents plus the
# split's 450 noise articles is what keeps this to about three quarters of an
# hour rather than the seven a full-corpus arm costs.
LIMIT = 500
THRESHOLD = 0.5

SEED = 0

# Each low-prevalence floor is the geometric midpoint of the two states it has
# to tell apart -- sqrt(lowest live measurement * highest dead one), rounded
# down -- which is the cutoff furthest from both in ratio, and a rule a later
# reader can recompute rather than a number someone once observed.
#
#   strains    live 0.264  dead 0.005  -> sqrt(0.264*0.005) = 0.036
#   bacteria   live 0.607  dead 0.007  -> sqrt(0.607*0.007) = 0.065
#
# The live pair is the lowest of two pre-`06e36cf` `logmeanexp` runs, the dead
# pair the `logsumexp` run beside them; no arm at HEAD reaches either state,
# which is why the floors are calibrated against those and not against what
# this test currently measures (0.07 and 0.10-0.13, comfortably clear).
#
# The high-prevalence pair has no dead observation to bracket: both sit at
# 1.000 under every pooling and every scale measured. Their floor is set just
# clear of 1.000, because a floor *of* 1.000 fails on a single document.
RECALL_FLOORS = {
    "enzymes": 0.90,
    "other_organisms": 0.90,
    "bacteria": 0.06,
    "strains": 0.03,
}


def training_config() -> ModelConfig:
    """`tests/best_config_so_far.toml` at a batch budget this fits in 6 GB.

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
        encodings=encodings[config.base_model],
        limit=LIMIT,
    )
    train_split = dataset.data["train"]

    model = factory.build_model(
        config,
        dataset,
        BRENDA_SCHEMA,
        entity_freqs=data.compute_frequencies(train_split, column="entities"),
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

    # `save_checkpoint=False` keeps the best-epoch CPU snapshot out of
    # the run. `patience` exceeds `num_epochs`, so nothing stops early
    # and there is no best-epoch state to restore: the head scored
    # below is the last epoch's, which is what `train` would write out.
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
            class_logits = model.get_batch_logits(batch)[1]
            probs = torch.sigmoid(model.drop_oos(class_logits).float()).cpu()
            gold = (
                model.ground_truth(batch)[1][:, : probs.shape[1]].bool().cpu()
            )
            fired = probs >= threshold

            for column, name in enumerate(names):
                positives[name] += int(gold[:, column].sum())
                hits[name] += int((gold[:, column] & fired[:, column]).sum())

    return {
        name: hits[name] / positives[name] for name in names if positives[name]
    }


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

    # A test that costs three quarters of an hour hands back what it measured
    # rather than a bare pass: the floors below are calibrated by reading this
    # table off a green run, and a number drifting toward its floor is the
    # warning that comes before the failure.
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
