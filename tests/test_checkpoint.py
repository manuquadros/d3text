"""What a `.pt` file carries, and what it refuses to read.

The class head is positional and nothing in a `state_dict` says which class
owns which column, so the column order travels with the weights instead of
being rebuilt beside them. Format 2 dropped the entity-linking head, which is
why everything older is refused outright rather than read for the half that
still fits.
"""

import pickle

import pandas as pd
import pytest
import torch
from torch import nn

from d3text import checkpoint, factory, surface_forms
from d3text.datasets import brenda
from d3text.linking import DictionaryLinker
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

# A `surface_forms.index_digest`, which is a hex sha256 of the index.
DIGEST = "d3" * 32

# An `encodings_store.content_digest`, which is a hex sha256 of the store.
ENCODINGS_DIGEST = "e5" * 32


class _Head(nn.Module):
    """A stand-in for a class head: as wide as its vocabulary plus `OOS`, and
    nothing about it says which class owns which column."""

    def __init__(self, classes: int) -> None:
        super().__init__()
        self.class_classifier = nn.Linear(4, classes + 1)


def frame(rows: list[dict]) -> pd.DataFrame:
    """A split frame in the shape `brenda_references` hands over."""
    records = []
    for row in rows:
        record = {
            "pubmed_id": row["pubmed_id"],
            "fulltext": "<p>body</p>",
            "relations": [],
            "source": "training",
        }
        for entity_type in SCHEMA.entity_types:
            record[entity_type.name] = row.get(entity_type.name, [])
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
    torch.testing.assert_close(
        loaded.state_dict["class_classifier.weight"],
        trained.class_classifier.weight,
    )


def test_the_checkpoint_reads_back_without_trusting_it(tmp_path):
    """`torch.load`'s `weights_only=True` default admits tensors and builtins
    only. The vocabulary goes in as lists and dicts precisely so a checkpoint
    stays readable without unpickling whatever it happens to contain."""
    path = tmp_path / "model.pt"
    checkpoint.save(path, _Head(len(VOCABULARY)).state_dict(), VOCABULARY)

    contents = torch.load(path, weights_only=True)

    assert contents[checkpoint.VOCABULARY_KEY] == VOCABULARY.to_payload()


def test_load_refuses_a_non_allowlisted_pickle_even_under_the_env_var(
    tmp_path, monkeypatch
):
    """`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` flips `torch.load`'s default to
    `weights_only=False`, but only for a call site that left the argument
    unset. `checkpoint.load` must pass it explicitly, so the env var must not
    be able to make it unpickle arbitrary objects."""
    path = tmp_path / "untrusted.pt"
    torch.save(
        {
            checkpoint.FORMAT_KEY: checkpoint.FORMAT,
            checkpoint.STATE_DICT_KEY: {},
            checkpoint.VOCABULARY_KEY: VOCABULARY.to_payload(),
            "payload": _Head(1),
        },
        path,
    )
    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    with pytest.raises(pickle.UnpicklingError, match="Unsupported global"):
        checkpoint.load(path)


def test_the_written_format_is_2(tmp_path):
    """The version stamped on disk is what every older reader compares
    against, so it is pinned here rather than only through what `load`
    accepts."""
    path = tmp_path / "model.pt"
    checkpoint.save(path, _Head(len(VOCABULARY)).state_dict(), VOCABULARY)

    assert torch.load(path, weights_only=True)[checkpoint.FORMAT_KEY] == 2


def test_a_format_1_checkpoint_is_refused_and_says_why(tmp_path):
    """A format-1 file carries the entity-linking head's parameters and a
    vocabulary keyed by entity column. No model this code builds has a place
    for either, so loading it for the class head alone would silently score a
    half-restored model."""
    path = tmp_path / "format1.pt"
    torch.save(
        {
            checkpoint.FORMAT_KEY: 1,
            checkpoint.STATE_DICT_KEY: {
                "classifier.entity_classifier.3.weight": torch.zeros(3, 4)
            },
            checkpoint.VOCABULARY_KEY: {
                "entities": ["ec2", "ec7", "taxon42"],
                "class_map": {"enzymes": ["ec2", "ec7"]},
            },
        },
        path,
    )

    with pytest.raises(ValueError, match="entity-linking head"):
        checkpoint.load(path)


def test_a_bare_state_dict_is_refused_and_says_why(tmp_path):
    """The shape `save` wrote before the format key existed. It predates
    format 1, so it carries the entity head too and there is nothing to
    reconstruct its columns from either."""
    path = tmp_path / "legacy.pt"
    torch.save({"classifier.entity_classifier.3.bias": torch.zeros(3)}, path)

    with pytest.raises(ValueError, match="bare state dict"):
        checkpoint.load(path)


def test_the_label_store_the_targets_came_from_round_trips(tmp_path):
    """The vocabulary says which class owns which column; it says nothing
    about which strings the token-level targets were matched against. A
    checkpoint scored against a store rebuilt from another surface-form index
    is scored against a different count of gold spans, and both existing
    guards stay silent through it — the store's own because each store is
    self-consistent, the vocabulary's because the columns never moved."""
    path = tmp_path / "model.pt"

    checkpoint.save(
        path, _Head(len(VOCABULARY)).state_dict(), VOCABULARY, DIGEST
    )
    loaded = checkpoint.load(path)

    assert loaded.token_labels_digest == DIGEST
    # Plain builtins, like the vocabulary beside it, so the file stays
    # readable without unpickling what it contains.
    contents = torch.load(path, weights_only=True)
    assert contents[checkpoint.TOKEN_LABELS_DIGEST_KEY] == DIGEST


def test_the_labelling_rules_the_targets_came_from_round_trips(tmp_path):
    """The index digest says which strings the targets were matched against;
    it says nothing about what the sweep did with that answer. A rule change
    that touches no string in the index — `fd55b3a`'s guard on `fuzzy_ids` is
    the demonstration — relabels the corpus against a byte-identical index
    digest, so the two have to travel and be compared separately."""
    path = tmp_path / "model.pt"
    rules_digest = "f" * 64

    checkpoint.save(
        path,
        _Head(len(VOCABULARY)).state_dict(),
        VOCABULARY,
        token_labels_digest=DIGEST,
        labelling_rules_digest=rules_digest,
    )
    loaded = checkpoint.load(path)

    assert loaded.labelling_rules_digest == rules_digest
    # Plain builtins, like the vocabulary beside it, so the file stays
    # readable without unpickling what it contains.
    contents = torch.load(path, weights_only=True)
    assert contents[checkpoint.LABELLING_RULES_DIGEST_KEY] == rules_digest


def test_a_checkpoint_from_before_the_rules_digest_existed_still_loads(
    tmp_path,
):
    """`token_labels_digest` alone does not imply `labelling_rules_digest`:
    a checkpoint written after the index digest but before this field must
    read back `None` rather than erroring or, worse, comparing as a match it
    never recorded."""
    path = tmp_path / "model.pt"
    trained = _Head(len(VOCABULARY))
    torch.save(
        {
            checkpoint.FORMAT_KEY: checkpoint.FORMAT,
            checkpoint.STATE_DICT_KEY: trained.state_dict(),
            checkpoint.VOCABULARY_KEY: VOCABULARY.to_payload(),
            checkpoint.TOKEN_LABELS_DIGEST_KEY: DIGEST,
        },
        path,
    )

    loaded = checkpoint.load(path)

    assert loaded.token_labels_digest == DIGEST
    assert loaded.labelling_rules_digest is None


def test_the_tokenization_the_inputs_came_from_round_trips(tmp_path):
    """The label digest says which strings the targets were matched against;
    it says nothing about the ids the heads were shown. A store rebuilt under
    a newer tokenizer holds different ids for the same documents at the same
    window, and the geometry stamp the file carries is unchanged — so scoring
    a checkpoint against it compares two runs that never read the same
    input."""
    path = tmp_path / "model.pt"

    checkpoint.save(
        path,
        _Head(len(VOCABULARY)).state_dict(),
        VOCABULARY,
        encodings_digest=ENCODINGS_DIGEST,
    )
    loaded = checkpoint.load(path)

    assert loaded.encodings_digest == ENCODINGS_DIGEST
    # Plain builtins, like the vocabulary beside it, so the file stays
    # readable without unpickling what it contains.
    contents = torch.load(path, weights_only=True)
    assert contents[checkpoint.ENCODINGS_DIGEST_KEY] == ENCODINGS_DIGEST


def test_a_run_over_an_unstamped_encodings_store_records_no_digest(tmp_path):
    path = tmp_path / "model.pt"

    checkpoint.save(path, _Head(len(VOCABULARY)).state_dict(), VOCABULARY)

    assert checkpoint.load(path).encodings_digest is None


def test_a_run_that_read_no_label_store_records_no_digest(tmp_path):
    path = tmp_path / "model.pt"

    checkpoint.save(path, _Head(len(VOCABULARY)).state_dict(), VOCABULARY)

    assert checkpoint.load(path).token_labels_digest is None


def test_the_surface_form_index_round_trips_and_still_links(tmp_path):
    """`infer` reads this back instead of rebuilding an index from BRENDA's
    data, so what comes out has to link exactly as the index `train` built
    does -- not merely compare equal, since `SurfaceFormIndex` has no
    value-equality of its own (`eq=False` in `d3text.surface_forms`)."""
    path = tmp_path / "model.pt"
    index = surface_forms.build_index({"enz7": ["catalase"]})

    checkpoint.save(
        path,
        _Head(len(VOCABULARY)).state_dict(),
        VOCABULARY,
        surface_form_index=index,
    )
    loaded = checkpoint.load(path)

    assert loaded.surface_form_index is not None
    assert surface_forms.index_digest(
        loaded.surface_form_index
    ) == surface_forms.index_digest(index)
    assert DictionaryLinker(loaded.surface_form_index).link(
        "catalase", "enzymes"
    ) == {"enz7"}


def test_the_surface_form_index_is_plain_builtins_on_disk(tmp_path):
    """Like the vocabulary beside it, so the file stays readable without
    unpickling what it contains."""
    path = tmp_path / "model.pt"
    index = surface_forms.build_index({"enz7": ["catalase"]})

    checkpoint.save(
        path,
        _Head(len(VOCABULARY)).state_dict(),
        VOCABULARY,
        surface_form_index=index,
    )

    contents = torch.load(path, weights_only=True)
    assert contents[
        checkpoint.SURFACE_FORM_INDEX_KEY
    ] == surface_forms.index_to_payload(index)


def test_a_run_that_built_no_surface_form_index_records_none(tmp_path):
    """A training run that could not build one -- or, before this was
    recorded, any training run at all -- writes a checkpoint `infer` still
    loads; only linking is unavailable, not the model."""
    path = tmp_path / "model.pt"

    checkpoint.save(path, _Head(len(VOCABULARY)).state_dict(), VOCABULARY)

    assert checkpoint.load(path).surface_form_index is None


def test_a_format_2_checkpoint_without_the_digests_still_loads(tmp_path):
    """The digests are optional *within* the format: a reader that does not
    know a key must read exactly the checkpoint it read before, since these
    fields qualify a comparison rather than interpret a weight."""
    path = tmp_path / "before.pt"
    trained = _Head(len(VOCABULARY))
    torch.save(
        {
            checkpoint.FORMAT_KEY: checkpoint.FORMAT,
            checkpoint.STATE_DICT_KEY: trained.state_dict(),
            checkpoint.VOCABULARY_KEY: VOCABULARY.to_payload(),
        },
        path,
    )

    loaded = checkpoint.load(path)

    assert loaded.token_labels_digest is None
    assert loaded.labelling_rules_digest is None
    assert loaded.encodings_digest is None
    assert loaded.surface_form_index is None
    assert loaded.vocabulary == VOCABULARY
    torch.testing.assert_close(
        loaded.state_dict["class_classifier.weight"],
        trained.class_classifier.weight,
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


def test_a_recorded_vocabulary_indexes_a_corpus_that_has_grown(tmp_path):
    """A checkpoint trained on a truncated split, evaluated against the
    untruncated corpus: the recorded vocabulary is what the later splits are
    indexed under, so the class columns and their members stay the training
    run's rather than today's corpus's."""
    small = dataset_over([{"pubmed_id": 10, "enzymes": [7]}], tmp_path)
    path = tmp_path / "model.pt"
    checkpoint.save(
        path,
        _Head(len(small.class_map)).state_dict(),
        Vocabulary.from_class_map(small.class_map),
    )

    grown = [
        {"pubmed_id": 10, "enzymes": [7]},
        {"pubmed_id": 20, "enzymes": [8], "bacteria": [42]},
    ]
    loaded = checkpoint.load(path)
    evaluated = dataset_over(grown, tmp_path, vocabulary=loaded.vocabulary)

    assert evaluated.class_map == small.class_map
    assert loaded.vocabulary.entity_ids == frozenset({"ec7"})


def test_a_recorded_vocabulary_whose_classes_are_permuted_is_refused(tmp_path):
    """The dangerous case, and the one `check_fits` exists for: the class
    head's targets are built in schema order and its columns in vocabulary
    order, so equal sets in a different order scores every class against
    another class's logits at an unchanged width."""
    permuted = Vocabulary(
        class_map={"bacteria": ("taxon42",), "enzymes": ("ec7",)}
    )

    with pytest.raises(ValueError, match="do not match the schema"):
        dataset_over(
            [{"pubmed_id": 10, "enzymes": [7]}], tmp_path, vocabulary=permuted
        )


@pytest.mark.parametrize(
    "model_class",
    ["BrendaClassificationModel", "NERClassificationModel"],
)
def test_a_saved_checkpoint_carries_no_entity_head_parameter(
    tmp_path, patch_base_model, model_class
):
    """What "format 2" names on disk. A leftover `entity_classifier` weight,
    `class_matrix` or `entity_pos_weight` would be a format-1 checkpoint
    written under the new version number, which is the one thing `load`
    cannot detect."""
    config = ModelConfig(
        model_class=model_class,
        base_model="prajjwal1/bert-mini",
        hidden_layers=[8],
    )
    model = factory.build_model(config, SCHEMA)
    path = tmp_path / "model.pt"
    checkpoint.save(path, model.state_dict(), VOCABULARY)

    keys = set(checkpoint.load(path).state_dict)

    assert not [key for key in keys if "entity" in key], sorted(keys)
    assert not [key for key in keys if "class_matrix" in key], sorted(keys)


def test_a_real_model_round_trips_through_the_checkpoint(
    tmp_path, patch_base_model
):
    """End to end over a built model rather than the `_Head` stand-in: what
    `train` writes is what `evaluate` loads, `strict=True` and all."""
    config = ModelConfig(
        model_class="BrendaClassificationModel",
        base_model="prajjwal1/bert-mini",
        hidden_layers=[8],
    )
    trained = factory.build_model(config, SCHEMA)
    path = tmp_path / "model.pt"
    checkpoint.save(path, trained.state_dict(), VOCABULARY)

    loaded = checkpoint.load(path)
    rebuilt = factory.build_model(config, SCHEMA)
    rebuilt.register_load_state_dict_pre_hook(factory.fix_keys_hook)
    rebuilt.load_state_dict(loaded.state_dict)

    torch.testing.assert_close(
        rebuilt.classifier.class_classifier.weight,
        trained.classifier.class_classifier.weight,
    )
    assert loaded.vocabulary == VOCABULARY
