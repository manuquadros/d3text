"""No class channel is allowed to go dead.

`tests/models/test_pooling_default.py` pins the *wiring*: the shipped pooling is
length-invariant, and duplicating a document token for token moves neither head.
That is not the same guarantee. A change could leave every class a correct,
distinct, correctly-ordered pooling and still leave the bacteria and strains
channels predicting nothing, and every wiring assertion would stay green.

Nothing else would catch it either. The pooled validation loss is what hid such
a collapse the first time: a channel that never fires is near-optimal on the
75-83% of validation documents where its class is absent, so when the two
low-prevalence channels fell to a document recall of 0.005 and 0.007 the printed
training loss barely moved. That is the failure this file exists to reject, and
it is silent by construction -- which is what makes it worth a test that trains.

**What this does not do is compare poolings.** The dead channels above were
measured on checkpoints built before `06e36cf`; at HEAD, inverting
`entity_logits_pooling` to `logsumexp` moves these numbers by about 1.2x and
kills nothing, and at full `--limit` the two poolings tie within noise on every
class (`169d373`). So the floors here separate a live channel from a dead one
and are indifferent to which pooling produced it. A future change that revives
the collapse will trip them whatever its cause.

Read the numbers with the caveat that half the bacteria class-negative documents
and about a third of the other-organism ones name an entity of that type anyway,
so the ceiling on these classes is not 1.0 and a recall of 0.5 is not half a
failure. Read them, too, as a small-corpus measurement: `--limit 500` is what
keeps this test to three quarters of an hour, and the same channels reach
0.83-0.93 on the whole training split.
"""

import pytest
import torch

from d3text import data, factory
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

    `entity_logits_pooling` is deliberately **not** set: the shipped default is
    what is under test, so naming it here would make the test pass whatever the
    default became.
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
        base_layers_to_unfreeze=0,
        entity_loss_scaling_factor=1,
        relation_label_smoothing=0,
        common_hidden_block=True,
        ramp_epochs=4,
        separate_predicate_layer=True,
    )


@pytest.fixture(scope="module")
def trained_run():
    """A short training run and the validation loader to score it on.

    Module-scoped: the run is the expensive part, and every assertion below
    reads the same trained head.

    Seeded here rather than left to conftest's autouse `deterministic_rng`.
    That fixture is function-scoped, and pytest sets a module-scoped fixture up
    *first*, so the seeding would land after the training it is meant to make
    reproducible -- which is how the first calibration run trained against an
    unseeded generator.
    """
    torch.manual_seed(SEED)
    config = training_config()
    dataset = data.brenda_dataset(
        encodings=encodings[config.base_model], limit=LIMIT
    )
    train_split = dataset.data["train"]

    model = factory.build_model(
        config,
        dataset,
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

    Counted per batch rather than accumulated as logits: what is asserted is
    four ratios, and holding the whole split's probabilities to compute them
    would put the split's size in the way of a test that already trains.
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

    One assertion per class rather than four separate tests: they share a
    training run, and a report naming every channel that fell is what makes a
    failure here readable -- a collapse takes the low-prevalence pair together.
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
