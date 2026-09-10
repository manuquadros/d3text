"""What `evaluate` scores a checkpoint against, and why.

`load_evaluation_dataset` decides the corpus: a checkpoint that records its
vocabulary is scored against *that*, and one that does not is scored against a
reconstruction, which is only as good as the operator's memory of the training
run's `--limit`. `token_labels_provenance` and `encodings_provenance` decide
nothing and report the other two halves — which dictionary the distant labels
the detection metrics count came from, and which tokenization produced the ids
the heads read. All three differences have to be visible, hence the warnings
pinned here.
"""

import argparse
import contextlib
import sys
import types
import warnings

import h5py
import numpy
import pytest
import torch

from d3text import encodings_store, linking_corpora
from d3text.checkpoint import Checkpoint
from d3text.cli import evaluate
from d3text.data.data import EntityRelationDataset
from d3text.datasets import brenda
from d3text.identifier_bridge import (
    NCBI_TAXID,
    BridgeRow,
    ExternalMention,
    IdentifierBridge,
)
from d3text.linking import DictionaryLinker
from d3text.linking_eval import score_linking
from d3text.models.config import ModelConfig
from d3text.surface_forms import build_index
from d3text.vocabulary import Vocabulary

VOCABULARY = Vocabulary.from_class_map(
    {"enzymes": {"enz7"}, "bacteria": {"bac42"}}
)

# Two `surface_forms.index_digest`s, which are hex sha256s of an index.
TRAINED_ON = "a" * 64
REBUILT = "b" * 64

# Two `encodings_store.content_digest`s, which are hex sha256s of a store.
TOKENIZED = "c" * 64
RETOKENIZED = "d" * 64

SENTINEL = EntityRelationDataset(
    data={},
    entity_index=VOCABULARY.entity_index,
    class_map=VOCABULARY.as_class_map(),
    class_matrix=torch.zeros(len(VOCABULARY), 2),
)


@pytest.fixture
def recorded_calls(monkeypatch):
    """Record what `brenda_dataset` is asked for instead of loading a corpus."""
    calls = []

    def brenda_dataset(**kwargs):
        calls.append(kwargs)
        return SENTINEL

    monkeypatch.setattr(evaluate, "brenda_dataset", brenda_dataset)
    return calls


def load(vocabulary, limit, base_model="prajjwal1/bert-mini"):
    return evaluate.load_evaluation_dataset(
        config_base_model=base_model, vocabulary=vocabulary, limit=limit
    )


def test_a_recorded_vocabulary_indexes_the_test_split_alone(recorded_calls):
    """The training split exists only to derive the entity columns. Once they
    are recorded, reading it is hundreds of MB spent on nothing."""
    load(VOCABULARY, limit=None)

    (call,) = recorded_calls
    assert call["vocabulary"] == VOCABULARY
    assert call["split_names"] == ("test",)
    assert "limit" not in call


def test_limit_is_ignored_and_said_to_be_ignored(recorded_calls):
    """It resized the entity head by resizing the split it was derived from.
    Silently honouring it against a recorded vocabulary would put the flag
    back in a position to matter."""
    with pytest.warns(RuntimeWarning, match="--limit is ignored"):
        load(VOCABULARY, limit=250)

    (call,) = recorded_calls
    assert call["vocabulary"] == VOCABULARY


def test_a_legacy_checkpoint_rebuilds_the_columns_and_warns(recorded_calls):
    """Nothing recovers the order such a checkpoint was trained on, so the
    reconstruction stands — but the operator has to be told it is one."""
    with pytest.warns(RuntimeWarning, match="records no entity vocabulary"):
        load(None, limit=250)

    (call,) = recorded_calls
    assert call["limit"] == 250
    assert "vocabulary" not in call


def test_a_legacy_checkpoint_without_a_limit_takes_the_whole_corpus(
    recorded_calls,
):
    with pytest.warns(RuntimeWarning, match="records no entity vocabulary"):
        load(None, limit=None)

    (call,) = recorded_calls
    # Exhaustive, and that is the point: `None` is what the loader takes for
    # "all of it", so the value is the contract -- but naming only the value
    # stopped pinning that `split_names` is absent, and this branch has to
    # load the training split, since rebuilding the entity columns from it is
    # the whole reason the branch exists. Passing `split_names=("test",)` here
    # is a mutation that an assertion on `limit` alone does not catch.
    assert set(call) == {"schema", "encodings", "limit", "base_model"}
    assert call["limit"] is None


def test_the_store_the_checkpoint_trained_on_is_recognised():
    """Matching digests are the whole point of recording one: the detection
    metrics then count the gold spans the training run counted."""
    assert evaluate.token_labels_provenance(TRAINED_ON, TRAINED_ON) == "matched"


def test_a_rebuilt_label_store_warns_and_is_still_scored():
    """The failure this exists to catch. `test/detection_*` is scored against
    the store's distant labels, so a dictionary naming more strings yields
    more gold spans and the same tagger scores differently — with the store's
    own guard silent, since each store is self-consistent, and the
    vocabulary's silent, since the columns never moved. Unlike
    `token_labels.check_index`, which refuses, this one warns: the numbers are
    the numbers, and a stale digest must cost an evaluation its silence rather
    than its hours."""
    with pytest.warns(RuntimeWarning, match="not comparable"):
        tag = evaluate.token_labels_provenance(TRAINED_ON, REBUILT)

    assert tag == "mismatched"


def test_a_checkpoint_recording_no_store_warns_where_one_is_read():
    """Nothing recovers which dictionary such a checkpoint was trained
    against, so the comparison cannot be made — which is exactly what the
    operator has to be told, as for a rebuilt vocabulary."""
    with pytest.warns(RuntimeWarning, match="records no token-label"):
        tag = evaluate.token_labels_provenance(None, REBUILT)

    assert tag == "unrecorded"


def test_an_evaluation_with_no_label_store_is_unchanged():
    """`token_labels_store` is empty by default. A model that never read one
    has no provenance to compare and must hear nothing about it."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert evaluate.token_labels_provenance(None, None) == "unused"


def test_the_encodings_the_checkpoint_trained_on_are_recognised():
    assert evaluate.encodings_provenance(TOKENIZED, TOKENIZED) == "matched"


def test_a_retokenized_corpus_warns_and_is_still_scored():
    """The failure this exists to catch. A store rebuilt under a newer
    tokenizer revision, or after `document_text` changed what it feeds the
    tokenizer, holds different ids for the same documents at the same window
    and stride — so the model is scored on inputs it never trained on, with
    the geometry stamp silent because it did not move and the vocabulary
    silent because the columns did not either."""
    with pytest.warns(RuntimeWarning, match="different token ids"):
        tag = evaluate.encodings_provenance(TOKENIZED, RETOKENIZED)

    assert tag == "mismatched"


def test_a_checkpoint_recording_no_tokenization_warns():
    with pytest.warns(RuntimeWarning, match="records no encodings digest"):
        tag = evaluate.encodings_provenance(None, TOKENIZED)

    assert tag == "unrecorded"


def test_two_absent_digests_are_not_reported_as_a_match():
    """`None == None` is not agreement. Reporting it as `matched` would be the
    stamp asserting something no file on disk says, which is worse than the
    silence it replaced."""
    with pytest.warns(RuntimeWarning, match="records no encodings digest"):
        assert evaluate.encodings_provenance(None, None) == "unrecorded"


def test_an_unstamped_store_cannot_confirm_a_checkpoints_inputs():
    """Every encodings file written before the digest existed is this case, so
    it warns and scores rather than refusing: rebuilding the store is hours,
    and the numbers are still the numbers."""
    with pytest.warns(RuntimeWarning, match="carries no digest of its own"):
        tag = evaluate.encodings_provenance(TOKENIZED, None)

    assert tag == "unstamped"


def test_no_corpus_root_logs_no_linking_metrics():
    """The linking block is an extra a machine may not have the corpora for.
    An evaluation that failed without them would make an optional measurement
    a dependency of every scored checkpoint."""
    assert evaluate.report_linking(None) == {}


def test_the_linking_metrics_reach_the_run(monkeypatch):
    """The block is assembled outside `evaluate_model` because it reads no
    checkpoint, which is exactly the seam that can be built and never wired
    up: the reports would print and the run would carry no linking key."""
    logged = {}
    block = linking_corpora.LinkingBlock(
        (
            score_linking(
                mentions=[
                    ExternalMention(
                        document="species001",
                        start=0,
                        end=16,
                        surface="Escherichia coli",
                        external_id="562",
                    )
                ],
                bridge=IdentifierBridge.from_rows(
                    NCBI_TAXID, [BridgeRow("bac1", "562", "lpsn_id")]
                ),
                linker=DictionaryLinker(
                    build_index({"bac1": ["Escherichia coli"]})
                ),
                entity_types=["bacteria"],
                namespace=NCBI_TAXID,
            ),
        ),
        index_digest="deadbeef",
    )
    monkeypatch.setattr(
        evaluate.linking_corpora, "linking_block", lambda root: block
    )
    monkeypatch.setattr(evaluate.tracking, "log_metrics", logged.update)
    monkeypatch.setattr(evaluate.tracking, "log_text", lambda *_: None)

    returned = evaluate.report_linking("/anywhere")

    assert logged == returned
    assert logged[f"test/linking_{NCBI_TAXID}_strict_accuracy"] == 1.0


class _StubModel:
    """A model `evaluate.main` can load a checkpoint into and score."""

    device = "cpu"

    def register_load_state_dict_pre_hook(self, _hook):
        pass

    def load_state_dict(self, _state):
        pass

    def to(self, _device):
        pass

    def evaluate_model(self, _data):
        pass


def _run_evaluate(tmp_path, monkeypatch, recorded_digest):
    """Drive `evaluate.main` with everything but the provenance report stubbed
    out, and return the `checkpoint_encodings` tag it opened its run with."""
    config = tmp_path / "config.toml"
    config.write_text("")
    tags: dict[str, str] = {}

    monkeypatch.setattr(evaluate.runtime, "configure", lambda: None)
    monkeypatch.setattr(
        evaluate,
        "command_line_args",
        lambda: argparse.Namespace(
            config=str(config),
            model_state_dict=str(tmp_path / "model.pt"),
            limit=None,
        ),
    )
    monkeypatch.setattr(
        evaluate,
        "load_model_config",
        lambda _path: ModelConfig(base_model="prajjwal1/bert-mini"),
    )
    monkeypatch.setattr(
        evaluate.checkpoint,
        "load",
        lambda _path: Checkpoint(
            state_dict={},
            vocabulary=VOCABULARY,
            encodings_digest=recorded_digest,
        ),
    )
    monkeypatch.setattr(
        evaluate,
        "load_evaluation_dataset",
        lambda **_kwargs: types.SimpleNamespace(data={"test": ()}),
    )
    monkeypatch.setattr(evaluate.data, "get_batch_loader", lambda **_k: ())
    monkeypatch.setattr(
        evaluate.factory, "build_model", lambda *_args: _StubModel()
    )
    monkeypatch.setattr(evaluate.factory, "dataset_metrics", lambda _d: {})
    monkeypatch.setattr(evaluate.factory, "model_metrics", lambda _m: {})
    monkeypatch.setattr(evaluate, "report_linking", lambda _root: {})
    monkeypatch.setattr(evaluate.tracking, "stamped", lambda name: name)
    monkeypatch.setattr(evaluate.tracking, "provenance_tags", lambda *_a: {})
    monkeypatch.setattr(evaluate.tracking, "environment_tags", lambda: {})
    monkeypatch.setattr(evaluate.tracking, "log_metrics", lambda *_a: None)
    monkeypatch.setattr(evaluate.tracking, "log_artifact", lambda *_a: None)
    monkeypatch.setattr(
        evaluate.tracking,
        "run",
        contextlib.contextmanager(
            lambda **kwargs: iter([tags.update(kwargs["tags"])])
        ),
    )

    evaluate.main()
    return tags["checkpoint_encodings"]


def test_the_run_records_the_store_it_actually_scored_against(
    tmp_path, monkeypatch
):
    """The helpers above compare two digests; this pins where the second one
    comes from. `encodings` names the store relative to the data directory, so
    a digest read from the bare config value finds no file, reports every
    checkpoint as scored against an unstamped store, and says so about a store
    that is stamped."""
    store = tmp_path / "store.hdf5"
    with h5py.File(store, "w") as handle:
        handle.create_group("10").create_dataset(
            "input_ids", data=numpy.zeros((1, 8), dtype="uint32")
        )
        digest = encodings_store.stamp_content_digest(handle)

    monkeypatch.setattr(brenda, "DATA_DIR", tmp_path)
    monkeypatch.setitem(evaluate.encodings, "prajjwal1/bert-mini", store.name)

    assert _run_evaluate(tmp_path, monkeypatch, digest) == "matched"


def test_a_checkpoint_from_before_the_digest_still_evaluates(
    tmp_path, monkeypatch
):
    """Every checkpoint on disk records none, and the tag is what separates
    them from a run that could be checked."""
    monkeypatch.setattr(brenda, "DATA_DIR", tmp_path)
    monkeypatch.setitem(
        evaluate.encodings, "prajjwal1/bert-mini", "absent.hdf5"
    )

    with pytest.warns(RuntimeWarning, match="records no encodings digest"):
        tag = _run_evaluate(tmp_path, monkeypatch, None)

    assert tag == "unrecorded"


def test_a_negative_limit_is_refused_at_the_command_line(monkeypatch, capsys):
    """On an old, vocabulary-less checkpoint `--limit -1` reaches
    `load_split` the same way `train --limit -1` does, so it must be refused
    at the same argparse boundary rather than surfacing from the data layer.
    """
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate", "config.toml", "model.pt", "--limit", "-1"],
    )

    with pytest.raises(SystemExit) as exc_info:
        evaluate.command_line_args()

    assert exc_info.value.code == 2
    assert "--limit" in capsys.readouterr().err
