"""What `infer` keeps, and what it refuses to claim.

The command exists because every prediction the evaluation path builds is
consumed by a metric and dropped, so the pins here are about what survives
into the file: a span's offsets, its surface and its type beside the id the
linker chose for it, and the two fields whose `null` and whose empty list
say different things.
"""

import argparse
import ast
import json
import pathlib

import h5py
import numpy
import pytest
import torch
from d3text import encodings_store
from d3text.checkpoint import Checkpoint
from d3text.cli import evaluate, infer
from d3text.models.config import ModelConfig
from d3text.models.ete import PredictedRelation
from d3text.token_labels import BRENDA_LABELS
from d3text.vocabulary import Vocabulary

ENZYMES = BRENDA_LABELS.by_prefix["enz"]

TEXT = "ABCDEFGHIJKL"
PUBMED_ID = "12345"
VOCABULARY = Vocabulary.from_class_map({"enzymes": {"enz7"}})

# One enzyme mention over the third, fourth and fifth aggregated tokens.
CODES = [0, 0, ENZYMES, ENZYMES, ENZYMES, 0, 0, 0, 0, 0]


def _write_store(path: pathlib.Path) -> str:
    """One finished group for `PUBMED_ID`: a single 12-token window whose ten
    real tokens each cover one character of `TEXT`."""
    offset_mapping = numpy.zeros((1, 12, 2), dtype=numpy.uint32)
    for token in range(1, 11):
        offset_mapping[0, token] = (token - 1, token)

    with h5py.File(path, "w") as store:
        group = store.create_group(PUBMED_ID)
        group.create_dataset(
            "input_ids", data=numpy.zeros((1, 12), dtype=numpy.uint32)
        )
        group.create_dataset(
            "attention_mask", data=numpy.ones((1, 12), dtype=numpy.int64)
        )
        group.create_dataset("offset_mapping", data=offset_mapping)
        encodings_store.mark_group_complete(group)
        return encodings_store.stamp_content_digest(store)


def _write_corpus(path: pathlib.Path) -> None:
    path.write_text(
        f"pubmed_id,abstract,fulltext\n{PUBMED_ID},{TEXT},\n", encoding="utf8"
    )


class _SpanModel:
    """A checkpoint stand-in that tags `CODES` whatever it is shown, and
    declares no relation head — `BrendaClassificationModel`'s shape."""

    device = "cpu"

    def register_load_state_dict_pre_hook(self, _hook) -> None:
        pass

    def load_state_dict(self, _state) -> None:
        pass

    def to(self, _device) -> None:
        pass

    def eval(self) -> None:
        pass

    def get_token_embeddings(self, _batch):
        return torch.zeros(1, len(CODES), 1), torch.ones(1, len(CODES))

    def hidden(self, embeddings):
        return embeddings

    def token_tagger(self, _hidden_output):
        logits = torch.full((1, len(CODES), max(CODES) + 2), -10.0)
        for position, code in enumerate(CODES):
            logits[0, position, code] = 10.0
        return logits


class _RelationModel(_SpanModel):
    """A stand-in that also carries a relation head, as `ETEBrendaModel` does."""

    def predicted_relations(self, _batch) -> list[PredictedRelation]:
        return [
            PredictedRelation(
                predicate="produces",
                arguments=(frozenset({"bac1"}), frozenset({"enz7", "enz9"})),
            )
        ]


class _NoRelationFoundModel(_SpanModel):
    """A stand-in whose head scored this document's pairs and called every
    one of them null, which `predicted_relations` reports as an empty list."""

    def predicted_relations(self, _batch) -> list[PredictedRelation]:
        return []


class _FixedLinker:
    """Links every enzyme surface to one id and nothing else to anything."""

    def link(self, mention: str, entity_type: str) -> frozenset[str]:
        del mention
        return frozenset({"enz7"}) if entity_type == "enzymes" else frozenset()


@pytest.fixture
def run_infer(tmp_path, monkeypatch):
    """Drive `infer.main` over one stored document and return its records."""

    def _run(model, linker=_FixedLinker()):
        store = tmp_path / "encodings.hdf5"
        digest = _write_store(store)
        dataset = tmp_path / "corpus.csv"
        _write_corpus(dataset)
        output = tmp_path / "predictions.jsonl"

        monkeypatch.setattr(infer.runtime, "configure", lambda **_: None)
        monkeypatch.setattr(
            infer,
            "command_line_args",
            lambda: argparse.Namespace(
                config=str(tmp_path / "config.toml"),
                checkpoint=str(tmp_path / "model.pt"),
                output=str(output),
                datasets=[dataset],
            ),
        )
        monkeypatch.setattr(
            infer,
            "load_model_config",
            lambda _path: ModelConfig(
                model_class="NERClassificationModel",
                base_model="prajjwal1/bert-mini",
            ),
        )
        monkeypatch.setattr(
            infer.checkpoint,
            "load",
            lambda _path: Checkpoint(
                state_dict={},
                vocabulary=VOCABULARY,
                encodings_digest=digest,
            ),
        )
        monkeypatch.setattr(infer, "encodings_path", lambda _name: store)
        monkeypatch.setattr(infer.factory, "build_model", lambda *_a: model)
        monkeypatch.setattr(infer, "build_linker", lambda: linker)

        infer.main()

        return [
            json.loads(line)
            for line in output.read_text(encoding="utf8").splitlines()
        ]

    return _run


def test_a_predicted_span_is_written_with_its_offsets_surface_and_id(
    run_infer,
) -> None:
    """The whole point of the command: the tagger's span, the text it covers
    and the entity the linker chose for it all reach the file. Offsets index
    the assembled document text, and the surface is beside them because a
    consumer that assembles that text differently cannot trust them alone."""
    (record,) = run_infer(_RelationModel())

    assert record["document"] == PUBMED_ID
    assert record["spans"] == [
        {
            "start": 2,
            "end": 5,
            "surface": "CDE",
            "entity_type": "enzymes",
            "entity_ids": ["enz7"],
        }
    ]


def test_a_checkpoint_without_a_relation_head_says_so(run_infer) -> None:
    """A checkpoint with no relation head is never asked, so its records say
    `null`. The empty list is the narrower claim that a head scored this
    document's pairs and labelled every one of them null; a consumer reading
    the two the same way turns "this model cannot predict relations" into
    "this article has none"."""
    (record,) = run_infer(_SpanModel())

    assert record["relations"] is None


def test_a_relation_head_writes_the_predicate_and_both_argument_sets(
    run_infer,
) -> None:
    """The other side of the same pin: with a head, the list is real, so the
    `null` above reports the checkpoint rather than being hardcoded."""
    (record,) = run_infer(_RelationModel())

    assert record["relations"] == [
        {"predicate": "produces", "arguments": [["bac1"], ["enz7", "enz9"]]}
    ]


def test_a_head_that_labelled_every_pair_null_writes_an_empty_list(
    run_infer,
) -> None:
    """The record writer keeps `[]` as it found it. Coercing it to `null`
    here would erase the only difference between a head that scored this
    document's pairs and one that was put none."""
    (record,) = run_infer(_NoRelationFoundModel())

    assert record["relations"] == []


def test_no_linker_leaves_the_ids_unclaimed_rather_than_empty(
    run_infer,
) -> None:
    """A machine without the BRENDA dump can still detect spans; writing them
    with an empty `entity_ids` would claim the linker was asked and declined."""
    (record,) = run_infer(_RelationModel(), linker=None)

    (span,) = record["spans"]
    assert span["entity_ids"] is None
    assert span["surface"] == "CDE"


def test_a_checkpoint_with_no_span_tagger_is_refused(
    run_infer, monkeypatch
) -> None:
    """It proposes no mention and no relation argument, so every record it
    could write would be empty — a file of nothing reads as a model that
    found nothing."""
    model = _SpanModel()
    monkeypatch.setattr(_SpanModel, "token_tagger", None)

    with pytest.raises(SystemExit, match="no span tagger"):
        run_infer(model)


def _cli_modules_naming_the_tagger_attribute() -> list[str]:
    """Modules under `cli/` carrying `"token_tagger"` as a string constant."""
    root = pathlib.Path(__file__).resolve().parents[2]
    paths = sorted((root / "src/d3text/cli").glob("*.py"))
    assert len(paths) > 5, "the listing broke; the check below is vacuous"
    return [
        path.name
        for path in paths
        if any(
            isinstance(node, ast.Constant) and node.value == "token_tagger"
            for node in ast.walk(ast.parse(path.read_text(encoding="utf8")))
        )
    ]


def test_the_span_tagger_probe_has_one_home() -> None:
    """`evaluate.report_predicted_linking` resolved and narrowed
    `token_tagger` with its own `getattr` and its own cast, so what
    `nn.Module.__getattr__` hands back for a model shape had two places to
    be believed and only one of them would be edited. Every command goes
    through `token_supervision.resolve_token_tagger` now, so no module under
    `cli/` needs the attribute name at all — which is why this sweeps the
    package instead of naming the two functions that carried the probe: the
    regression it guards against is a command written later resolving it by
    hand, and such a module cannot be in a list written today. A caller
    still decides for itself what a checkpoint carrying no tagger means, so
    the None branches stay where they are."""
    assert _cli_modules_naming_the_tagger_attribute() == []


def test_a_report_over_a_model_with_no_tagger_is_skipped_not_refused() -> None:
    """`infer` exits on the same miss, but a model that detects no span is an
    ordinary thing to evaluate: the block it cannot fill drops out and the
    rest of the run is scored."""
    assert (
        evaluate.report_predicted_linking(None, "nonexistent.hdf5", object())
        == {}
    )
