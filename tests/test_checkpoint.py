"""What a `.pt` file carries, and what it does when it carries less.

The failure this format exists to stop is not exotic: `train --limit 250`
followed by `evaluate --limit 1000` builds a 776-column entity head against a
303-column checkpoint, because `--limit` truncates the training split and the
vocabulary is derived from it. That one fails loudly. The same corpus drift at
an unchanged width does not, which is why the vocabulary travels with the
weights instead of being rebuilt beside them.
"""

import pandas as pd
import pytest
import torch
from torch import nn

from d3text import checkpoint, factory
from d3text.datasets import brenda
from d3text.models.config import ModelConfig
from d3text.schema import EntityType, Schema
from d3text.vocabulary import Vocabulary

SCHEMA = Schema(
    entity_types=(
        EntityType(name="enzymes", prefix="ec"),
        EntityType(name="bacteria", prefix="taxon"),
    )
)

VOCABULARY = Vocabulary.from_class_map(
    {"enzymes": {"ec7", "ec2"}, "bacteria": {"taxon42"}}
)


class _Head(nn.Module):
    """A stand-in for an entity head: as wide as its vocabulary, and nothing
    about it says which entity owns which column."""

    def __init__(self, entities: int) -> None:
        super().__init__()
        self.entity_classifier = nn.Linear(4, entities + 1)


def frame(rows: list[dict]) -> pd.DataFrame:
    """A split frame in the shape `brenda_references` hands over."""
    records = []
    for row in rows:
        record = {
            "pubmed_id": row["pubmed_id"],
            "fulltext": "<p>body</p>",
            "relations": [],
        }
        entities = []
        for entity_type in SCHEMA.entity_types:
            ids = row.get(entity_type.name, [])
            record[entity_type.name] = ids
            entities += [entity_type.prefix + str(i) for i in ids]
        record["entities"] = entities
        records.append(record)
    return pd.DataFrame(records)


def dataset_over(rows: list[dict], tmp_path, vocabulary=None):
    return brenda.build_dataset(
        schema=SCHEMA,
        splits={"train": frame(rows), "test": frame(rows)},
        encodings=tmp_path / "encodings.hdf5",
        vocabulary=vocabulary,
    )


def test_save_and_load_round_trip_the_weights_and_the_vocabulary(tmp_path):
    path = tmp_path / "model.pt"
    trained = _Head(len(VOCABULARY))

    checkpoint.save(path, trained.state_dict(), VOCABULARY)
    loaded = checkpoint.load(path)

    assert loaded.vocabulary == VOCABULARY
    assert not loaded.is_legacy
    torch.testing.assert_close(
        loaded.state_dict["entity_classifier.weight"],
        trained.entity_classifier.weight,
    )


def test_the_checkpoint_reads_back_without_trusting_it(tmp_path):
    """`torch.load`'s `weights_only=True` default admits tensors and builtins
    only. The vocabulary goes in as lists and dicts precisely so a checkpoint
    stays readable without unpickling whatever it happens to contain."""
    path = tmp_path / "model.pt"
    checkpoint.save(path, _Head(len(VOCABULARY)).state_dict(), VOCABULARY)

    contents = torch.load(path, weights_only=True)

    assert contents[checkpoint.VOCABULARY_KEY] == VOCABULARY.to_payload()


def test_a_bare_state_dict_still_loads_and_reports_no_vocabulary(tmp_path):
    """Checkpoints written before the vocabulary was recorded are readable —
    they simply cannot say what their columns mean, which is what
    `vocabulary is None` tells the caller to warn about."""
    path = tmp_path / "legacy.pt"
    trained = _Head(len(VOCABULARY))
    torch.save(trained.state_dict(), path)

    loaded = checkpoint.load(path)

    assert loaded.is_legacy
    assert loaded.vocabulary is None
    torch.testing.assert_close(
        loaded.state_dict["entity_classifier.bias"],
        trained.entity_classifier.bias,
    )


def test_a_checkpoint_from_a_newer_format_is_refused(tmp_path):
    """Reading its `state_dict` and ignoring the rest is how a format change
    becomes a wrong-numbers bug instead of an error."""
    path = tmp_path / "future.pt"
    torch.save(
        {
            checkpoint.FORMAT_KEY: checkpoint.FORMAT + 1,
            checkpoint.STATE_DICT_KEY: {},
            checkpoint.VOCABULARY_KEY: VOCABULARY.to_payload(),
        },
        path,
    )

    with pytest.raises(ValueError, match="this d3text reads format"):
        checkpoint.load(path)


def test_a_checkpoint_declaring_the_format_but_missing_a_key_is_refused(
    tmp_path,
):
    path = tmp_path / "truncated.pt"
    torch.save({checkpoint.FORMAT_KEY: checkpoint.FORMAT}, path)

    with pytest.raises(ValueError, match="missing"):
        checkpoint.load(path)


def test_a_recorded_vocabulary_loads_against_a_corpus_that_has_grown(
    tmp_path, patch_base_model
):
    """The reported failure, end to end: a checkpoint trained on a truncated
    split, evaluated against the untruncated corpus. The recorded vocabulary
    is what keeps the head the width the weights expect."""
    small = dataset_over([{"pubmed_id": 10, "enzymes": [7]}], tmp_path)
    config = ModelConfig(
        model_class="BrendaClassificationModel",
        base_model="prajjwal1/bert-mini",
        hidden_layers=[8],
    )
    trained = factory.build_model(config, small, SCHEMA)
    path = tmp_path / "model.pt"
    checkpoint.save(
        path,
        trained.state_dict(),
        Vocabulary.from_index(small.entity_index, small.class_map),
    )

    grown = [
        {"pubmed_id": 10, "enzymes": [7]},
        {"pubmed_id": 20, "enzymes": [8], "bacteria": [42]},
    ]
    loaded = checkpoint.load(path)
    evaluated = factory.build_model(
        config,
        dataset_over(grown, tmp_path, vocabulary=loaded.vocabulary),
        SCHEMA,
    )
    evaluated.load_state_dict(loaded.state_dict)

    torch.testing.assert_close(
        evaluated.classifier.entity_classifier[-1].weight,
        trained.classifier.entity_classifier[-1].weight,
    )


def test_the_same_checkpoint_is_unloadable_without_its_vocabulary(
    tmp_path, patch_base_model
):
    """Proves the vocabulary above is doing the work rather than the two
    corpora happening to agree. This is the RuntimeError the operator saw."""
    small = dataset_over([{"pubmed_id": 10, "enzymes": [7]}], tmp_path)
    config = ModelConfig(
        model_class="BrendaClassificationModel",
        base_model="prajjwal1/bert-mini",
        hidden_layers=[8],
    )
    state_dict = factory.build_model(config, small, SCHEMA).state_dict()

    grown = [
        {"pubmed_id": 10, "enzymes": [7]},
        {"pubmed_id": 20, "enzymes": [8], "bacteria": [42]},
    ]
    rebuilt = factory.build_model(config, dataset_over(grown, tmp_path), SCHEMA)

    with pytest.raises(RuntimeError, match="size mismatch"):
        rebuilt.load_state_dict(state_dict)
