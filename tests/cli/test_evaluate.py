"""What `evaluate` scores a checkpoint against, and why.

`load_evaluation_dataset` decides the corpus: every checkpoint this code reads
records its vocabulary, and it is scored against *that*.
`token_labels_provenance` decides nothing and reports which dictionary the
distant labels the detection metrics count came from; the difference has to
be visible, hence the warnings pinned here. `encodings_provenance`'s own
tests live beside it in `tests/test_encodings_store.py`; this file still
covers the tag it feeds into a run, through `evaluate.main`.
"""

import argparse
import contextlib
import logging
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
from d3text.datasets import s800
from d3text.identifier_bridge import (
    NCBI_TAXID,
    BridgeRow,
    ExternalMention,
    IdentifierBridge,
)
from d3text.linking import DictionaryLinker
from d3text.linking_eval import score_linking
from d3text.models.config import MachineConfig, ModelConfig
from d3text.surface_forms import build_index
from d3text.vocabulary import Vocabulary

VOCABULARY = Vocabulary.from_class_map(
    {"enzymes": {"enz7"}, "bacteria": {"bac42"}}
)

# Two `surface_forms.index_digest`s, which are hex sha256s of an index.
TRAINED_ON = "a" * 64
REBUILT = "b" * 64

# Two `token_labels._rules_digest`s, which are hex sha256s of the labelling
# rules — a separate axis from the index digests above: a rule change like
# `fd55b3a`'s guard on `fuzzy_ids` moves this without moving those.
RULES_TRAINED_ON = "e" * 64
RULES_MOVED = "f" * 64

# An `encodings_store.content_digest`, a hex sha256 of a store.
TOKENIZED = "c" * 64

SENTINEL = EntityRelationDataset(data={}, class_map=VOCABULARY.as_class_map())


@pytest.fixture(autouse=True)
def encodings_entry(machine_stores, tmp_path):
    """`evaluate` resolves its encodings from the machine config; this names
    one nothing here opens, so a test that cares configures its own."""
    machine_stores(
        encodings_store={"prajjwal1/bert-mini": tmp_path / "absent.hdf5"}
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


def load(vocabulary, base_model="prajjwal1/bert-mini"):
    return evaluate.load_evaluation_dataset(
        config_base_model=base_model, vocabulary=vocabulary
    )


def test_a_recorded_vocabulary_indexes_the_test_split_alone(recorded_calls):
    """The training split exists only to derive the class columns and their
    members. Once they are recorded, reading it is hundreds of MB spent on
    nothing."""
    load(VOCABULARY)

    (call,) = recorded_calls
    # Exhaustive, and that is the point: asserting on `vocabulary` alone
    # stopped pinning that no `limit` is passed and that the training split
    # is not asked for, either of which would put the corpus back in a
    # position to decide the columns.
    assert set(call) == {
        "schema",
        "encodings",
        "vocabulary",
        "split_names",
        "base_model",
    }
    assert call["vocabulary"] == VOCABULARY
    assert call["split_names"] == ("test",)


def test_evaluate_takes_no_limit_flag(monkeypatch, capsys):
    """`--limit` existed to reproduce the training split's entity columns.
    With none to reproduce, accepting it would be a flag that silently does
    nothing to the run it is passed to."""
    monkeypatch.setattr(
        sys, "argv", ["evaluate", "config.toml", "model.pt", "--limit", "250"]
    )

    with pytest.raises(SystemExit) as exc_info:
        evaluate.command_line_args()

    assert exc_info.value.code == 2
    assert "unrecognized arguments: --limit" in capsys.readouterr().err


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


def test_matching_index_and_rules_digests_are_matched():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tag = evaluate.token_labels_provenance(
            TRAINED_ON, TRAINED_ON, RULES_TRAINED_ON, RULES_TRAINED_ON
        )

    assert tag == "matched"


def test_a_rules_only_change_is_caught_though_the_index_matches():
    """The exact gap this exists to close: `fd55b3a` guards `fuzzy_ids`,
    which touches neither the `exact` nor the `folded` table, so a store
    rebuilt after it carries the *same* index digest as one built before
    while tens of thousands of tokens change label. The index digest alone
    reports `matched`; the rules digest is what catches it."""
    with pytest.warns(RuntimeWarning, match="rules that turned them into"):
        tag = evaluate.token_labels_provenance(
            TRAINED_ON, TRAINED_ON, RULES_TRAINED_ON, RULES_MOVED
        )

    assert tag == "mismatched"


def test_a_checkpoint_missing_the_rules_digest_is_not_a_spurious_mismatch():
    """A checkpoint saved before this field existed carries
    `labelling_rules_digest=None` even though it does carry a real
    `token_labels_digest`. That absence must read as nothing to compare,
    not as a difference — the same shape `checkpoint.load`'s other optional
    fields already take."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tag = evaluate.token_labels_provenance(
            TRAINED_ON, TRAINED_ON, None, RULES_MOVED
        )

    assert tag == "matched"


def test_an_evaluation_with_no_label_store_is_unchanged():
    """`token_supervision` is off by default. A model that never read a store
    has no provenance to compare and must hear nothing about it."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert evaluate.token_labels_provenance(None, None) == "unused"


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


class _ScoringModel(_StubModel):
    """A stub whose evaluation logs a metric, as `evaluate_model` does."""

    def evaluate_model(self, _data):
        evaluate.tracking.log_metrics({"test/class_micro_f1": 0.5})


class _NoveltyModel(_StubModel):
    """A stub that declares `training_entity_ids`, as `BrendaClassificationModel`
    and `ETEBrendaModel` do (`NERClassificationModel` does not)."""

    training_entity_ids: frozenset[str] | None = None


def _run_evaluate(tmp_path, monkeypatch, recorded_digest):
    """Drive `evaluate.main` with everything but the provenance report stubbed
    out, and return the `checkpoint_encodings` tag it opened its run with."""
    tags = _stub_main(tmp_path, monkeypatch, recorded_digest)
    monkeypatch.setattr(evaluate, "report_linking", lambda _root: {})
    evaluate.main()
    return tags["checkpoint_encodings"]


def _stub_main(tmp_path, monkeypatch, recorded_digest):
    """Stub out everything `evaluate.main` touches but the provenance report
    and the linking block, and return the dict its run's tags land in."""
    config = tmp_path / "config.toml"
    config.write_text("")
    tags: dict[str, str] = {}

    monkeypatch.setattr(evaluate.runtime, "configure", lambda **_: None)
    monkeypatch.setattr(
        evaluate,
        "command_line_args",
        lambda: argparse.Namespace(
            config=str(config),
            model_state_dict=str(tmp_path / "model.pt"),
        ),
    )
    monkeypatch.setattr(
        evaluate,
        "load_model_config",
        lambda _path: ModelConfig(
            model_class="NERClassificationModel",
            base_model="prajjwal1/bert-mini",
        ),
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
    monkeypatch.setattr(evaluate.factory, "dataset_metrics", lambda _d, _s: {})
    monkeypatch.setattr(evaluate.factory, "model_metrics", lambda _m: {})
    monkeypatch.setattr(evaluate.tracking, "stamped", lambda name: name)
    monkeypatch.setattr(evaluate.tracking, "provenance_tags", lambda *_a: {})
    monkeypatch.setattr(evaluate.tracking, "environment_tags", lambda *_a: {})
    monkeypatch.setattr(evaluate.tracking, "log_metrics", lambda *_a: None)
    monkeypatch.setattr(evaluate.tracking, "log_artifact", lambda *_a: None)

    # A generator rather than an iterator, so an exception raised inside the
    # run reaches the test as itself and not as a failed `throw` on the stub.
    @contextlib.contextmanager
    def run(**kwargs):
        tags.update(kwargs["tags"])
        yield

    monkeypatch.setattr(evaluate.tracking, "run", run)

    return tags


def test_a_missing_brenda_dump_skips_the_linking_block_not_the_run(
    tmp_path, monkeypatch, caplog
):
    """The block is scored last, after every other metric is logged, and with
    gold on disk it goes on to build its index from the BRENDA dump. A machine
    set up for the corpora but without the dump must lose the block, not exit
    a finished evaluation non-zero."""
    gold = tmp_path / "corpora" / linking_corpora.S800
    (gold / s800.ABSTRACTS).mkdir(parents=True)
    (gold / s800.ANNOTATIONS).write_text(
        "562\tspecies001:111\t10\t25\tEscherichia coli\n", encoding="utf8"
    )
    (gold / s800.ABSTRACTS / "species001.txt").write_text(
        "Growth of Escherichia coli was measured.", encoding="utf8"
    )
    brenda_data = tmp_path / "brenda"
    brenda_data.mkdir()
    for split in linking_corpora.SPLITS:
        (brenda_data / f"{split}_data.csv").write_text("id\n", encoding="utf8")
    monkeypatch.setattr(linking_corpora, "DATA_DIR", brenda_data)

    _stub_main(tmp_path, monkeypatch, TOKENIZED)
    monkeypatch.setattr(
        evaluate.encodings_store, "store_content_digest", lambda _p: TOKENIZED
    )
    monkeypatch.setattr(
        evaluate.factory, "build_model", lambda *_args: _ScoringModel()
    )
    monkeypatch.setattr(
        evaluate,
        "machine_config",
        lambda: MachineConfig(linking_corpora=str(tmp_path / "corpora")),
    )
    logged: dict[str, float] = {}
    monkeypatch.setattr(evaluate.tracking, "log_metrics", logged.update)

    with caplog.at_level(logging.WARNING, logger=linking_corpora.__name__):
        evaluate.main()

    assert logged == {"test/class_micro_f1": 0.5}
    messages = [record.getMessage() for record in caplog.records]
    (consequence,) = [
        message for message in messages if "linking block is skipped" in message
    ]
    (warning,) = [message for message in messages if message != consequence]
    assert str(brenda_data / "documents.json") in warning
    assert str(brenda_data / "documents.json") not in consequence


def test_the_run_records_the_store_it_actually_scored_against(
    tmp_path, monkeypatch, machine_stores
):
    """The helpers above compare two digests; this pins where the second one
    comes from: the store `[encodings_store]` names for the base model. A
    digest read from anywhere else finds no file, reports every checkpoint as
    scored against an unstamped store, and says so about a store that is
    stamped."""
    store = tmp_path / "store.hdf5"
    with h5py.File(store, "w") as handle:
        handle.create_group("10").create_dataset(
            "input_ids", data=numpy.zeros((1, 8), dtype="uint32")
        )
        digest = encodings_store.stamp_content_digest(handle)

    machine_stores(encodings_store={"prajjwal1/bert-mini": store})

    assert _run_evaluate(tmp_path, monkeypatch, digest) == "matched"


def test_a_checkpoint_from_before_the_digest_still_evaluates(
    tmp_path, monkeypatch
):
    """Every checkpoint on disk records none, and the tag is what separates
    them from a run that could be checked."""

    with pytest.warns(RuntimeWarning, match="records no encodings digest"):
        tag = _run_evaluate(tmp_path, monkeypatch, None)

    assert tag == "unrecorded"


def test_the_run_is_not_tagged_with_a_vocabulary_provenance(
    tmp_path, monkeypatch
):
    """`checkpoint_vocabulary` separated a recorded column order from a
    rebuilt one. Format 2 refuses everything it cannot read, so the tag has
    exactly one value left and would assert a distinction no run can make."""
    tags = _stub_main(tmp_path, monkeypatch, TOKENIZED)
    monkeypatch.setattr(evaluate, "report_linking", lambda _root: {})
    monkeypatch.setattr(
        evaluate.encodings_store, "store_content_digest", lambda _p: TOKENIZED
    )

    evaluate.main()

    assert "checkpoint_vocabulary" not in tags


def test_a_span_tagging_model_learns_the_training_entity_ids(
    tmp_path, monkeypatch
):
    """This is what lets `DetectionAccumulator`'s seen/unseen novelty split
    run on a real `evaluate` invocation. Every other test here uses a model
    that does not declare `training_entity_ids`, so none of them would catch
    a wrong attribute name, a flipped `hasattr` condition, or the wrong
    vocabulary field being read."""
    model = _NoveltyModel()
    _stub_main(tmp_path, monkeypatch, None)
    monkeypatch.setattr(evaluate.factory, "build_model", lambda *_a: model)
    monkeypatch.setattr(evaluate, "report_linking", lambda _root: {})

    with pytest.warns(RuntimeWarning, match="records no encodings digest"):
        evaluate.main()

    assert model.training_entity_ids == VOCABULARY.entity_ids


def test_a_model_with_no_span_tagger_is_left_alone(tmp_path, monkeypatch):
    """`NERClassificationModel` detects no spans and declares no
    `training_entity_ids` attribute at all; the `hasattr` guard must not
    give it one it never asked for."""
    model = _StubModel()
    _stub_main(tmp_path, monkeypatch, None)
    monkeypatch.setattr(evaluate.factory, "build_model", lambda *_a: model)
    monkeypatch.setattr(evaluate, "report_linking", lambda _root: {})

    with pytest.warns(RuntimeWarning, match="records no encodings digest"):
        evaluate.main()

    assert not hasattr(model, "training_entity_ids")


def _s800_gold_one_document(root):
    """An S800 gold corpus of one annotated document, `species001`."""
    gold = root / linking_corpora.S800
    (gold / s800.ABSTRACTS).mkdir(parents=True)
    (gold / s800.ANNOTATIONS).write_text(
        "562\tspecies001:111\t10\t25\tEscherichia coli\n", encoding="utf8"
    )
    (gold / s800.ABSTRACTS / "species001.txt").write_text(
        "Growth of Escherichia coli was measured.", encoding="utf8"
    )
    return root


def _s800_gold_two_documents(root):
    """The same corpus, with a second annotated document, `species002`."""
    root = _s800_gold_one_document(root)
    gold = root / linking_corpora.S800
    gold.joinpath(s800.ANNOTATIONS).write_text(
        "562\tspecies001:111\t10\t25\tEscherichia coli\n"
        "5833\tspecies002:222\t4\t24\tPlasmodium falciparum\n",
        encoding="utf8",
    )
    (gold / s800.ABSTRACTS / "species002.txt").write_text(
        "The Plasmodium falciparum genome.", encoding="utf8"
    )
    return root


def _write_finished_group(store, key, text_length):
    """A group `predicted_spans_from_store` can read to completion: one
    window covering `text_length` characters, CLS/SEP at the ends."""
    width = text_length + 2
    offset_mapping = numpy.zeros((1, width, 2), dtype=numpy.uint32)
    for token in range(1, width - 1):
        offset_mapping[0, token] = (token - 1, token)
    group = store.create_group(key)
    group.create_dataset(
        "input_ids", data=numpy.zeros((1, width), dtype=numpy.uint32)
    )
    group.create_dataset(
        "attention_mask", data=numpy.ones((1, width), dtype=numpy.int64)
    )
    group.create_dataset("offset_mapping", data=offset_mapping)
    encodings_store.mark_group_complete(group)


class _SpanModel:
    """A tagger that proposes no span for whatever it is handed -- what the
    two tests below drive is whether a document is read at all, not what the
    tagger says about it."""

    def token_tagger(self, hidden_output):
        return torch.zeros((*hidden_output.shape[:-1], 2))

    def get_token_embeddings(self, batch):
        length = batch[0]["sequence"]["input_ids"].shape[0]
        return torch.zeros(1, length, 1), torch.ones(1, length)

    def hidden(self, embeddings, _mask):
        return embeddings

    def autocast_context(self):
        return contextlib.nullcontext()


def test_predicted_linking_skips_a_store_with_no_s800_group(
    tmp_path, monkeypatch, caplog
):
    """A store built without `precompute-encodings --s800` holds no group at
    all for the corpus's document, so `predicted_spans_from_store` would
    return an empty list indistinguishable from a tagger that read the
    document and proposed nothing -- scored, that logs the gold mention as a
    missed detection instead of naming the precompute gap."""
    monkeypatch.setattr(
        evaluate.linking_corpora,
        "brenda_index",
        lambda: build_index({"bac1": ["Escherichia coli"]}),
    )
    root = _s800_gold_one_document(tmp_path / "corpora")
    store_path = tmp_path / "store.hdf5"
    with h5py.File(store_path, "w"):
        pass

    with caplog.at_level(logging.WARNING, logger=evaluate.__name__):
        metrics = evaluate.report_predicted_linking(
            str(root), store_path, _SpanModel()
        )

    assert metrics == {}
    (warning,) = [
        record.getMessage()
        for record in caplog.records
        if "s800" in record.getMessage()
    ]
    assert "0 of 1" in warning


def test_predicted_linking_refuses_a_partially_populated_store(
    tmp_path, monkeypatch, caplog
):
    """A store holding a finished group for only one of two S800 documents
    must not be scored against both documents' gold -- the unread one would
    be charged as a missed detection for a document the tagger never saw."""
    monkeypatch.setattr(
        evaluate.linking_corpora,
        "brenda_index",
        lambda: build_index({"bac1": ["Escherichia coli"]}),
    )
    root = _s800_gold_two_documents(tmp_path / "corpora")
    store_path = tmp_path / "store.hdf5"
    with h5py.File(store_path, "w") as store:
        _write_finished_group(
            store,
            encodings_store.external_key("s800", "species001"),
            text_length=len("Growth of Escherichia coli was measured."),
        )

    with caplog.at_level(logging.WARNING, logger=evaluate.__name__):
        metrics = evaluate.report_predicted_linking(
            str(root), store_path, _SpanModel()
        )

    assert metrics == {}
    (warning,) = [
        record.getMessage()
        for record in caplog.records
        if "s800" in record.getMessage()
    ]
    assert "1 of 2" in warning
