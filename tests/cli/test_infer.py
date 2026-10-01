"""What `infer` keeps, and what it refuses to claim.

The command exists because every prediction the evaluation path builds is
consumed by a metric and dropped, so the pins here are about what survives
into the file: a span's offsets, its surface and its type beside the id the
linker chose for it, and the two fields whose `null` and whose empty list
say different things.
"""

import argparse
import ast
import contextlib
import json
import logging
import hashlib
import os
import pathlib
import string
import subprocess
import sys

import lmdb
import pytest
import torch
from d3text import (
    encodings_store,
    factory,
    surface_forms,
    token_labels,
    utils,
)
from d3text.checkpoint import Checkpoint
from d3text.cli import (
    evaluate,
    infer,
    precompute_encodings,
    precompute_token_labels,
)
from d3text.embeddings_store import (
    EmbeddingsStore,
    LayerBoundaryStore,
    sub_databases,
)
from d3text.models import base as model_base
from d3text.models import token_supervision
from d3text.models.config import MachineConfig, ModelConfig
from d3text.models.ete import ETEBrendaModel, PredictedRelation
from d3text.schema import BRENDA_SCHEMA
from d3text.token_labels import BRENDA_LABELS
from d3text.utils import aggregate_embeddings
from d3text.vocabulary import Vocabulary
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

ENZYMES = BRENDA_LABELS.by_prefix["enz"]
BACTERIA = BRENDA_LABELS.by_prefix["bac"]
BASE_MODEL = "prajjwal1/bert-mini"


def test_infer_does_not_import_evaluate_or_the_brenda_dataset(tmp_path):
    """Importing `infer` pulls in no training-data or evaluation module.

    The linker's index comes off the checkpoint, not BRENDA's data files.
    Checked in a subprocess: the session imports those modules itself, so
    an in-process `sys.modules` check could not tell whose import it was.
    """
    probe = (
        "import sys; import d3text.cli.infer; "
        "print(any(m in ("
        "'d3text.datasets.brenda', 'd3text.cli.evaluate', "
        "'d3text.linking_corpora', 'brenda_references'"
        ") for m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=True,
    )

    assert result.stdout.strip().endswith("False"), (
        "importing d3text.cli.infer pulled in d3text.datasets.brenda, "
        "d3text.cli.evaluate, d3text.linking_corpora or brenda_references: "
        f"{result.stdout!r} {result.stderr}"
    )


def test_build_linker_reads_the_checkpoints_own_index() -> None:
    """`infer` links against the index `train` shipped inside the checkpoint,
    not against a fresh read of the BRENDA data. `build_linker` returns
    straight off `saved.surface_form_index` when it is not `None`
    (`d3text/cli/infer.py`), never reaching `linking_corpora.brenda_index` or
    the data it reads -- proven by import alone in
    `test_infer_does_not_import_evaluate_or_the_brenda_dataset`."""
    index = surface_forms.build_index({"enz7": ["catalase"]})
    saved = Checkpoint(
        state_dict={},
        vocabulary=Vocabulary.from_class_map({"enzymes": {"enz7"}}),
        surface_form_index=index,
    )

    linker = infer.build_linker(saved)

    assert linker is not None
    assert linker.link("catalase", "enzymes") == {"enz7"}


def test_build_linker_without_a_recorded_index_warns_and_links_nothing(
    caplog,
) -> None:
    """A checkpoint written before this d3text recorded an index, or by a
    training run that could not build one, carries none. `infer` warns and
    links no span rather than falling back to rebuilding one from BRENDA
    data -- the fallback `build_linker` no longer has."""
    saved = Checkpoint(
        state_dict={},
        vocabulary=Vocabulary.from_class_map({"enzymes": {"enz7"}}),
    )

    with caplog.at_level(logging.WARNING, logger=infer.__name__):
        linker = infer.build_linker(saved)

    assert linker is None
    assert any(
        "no surface-form index" in record.getMessage()
        for record in caplog.records
    )


TEXT = "ABCDEFGHIJKL"
PUBMED_ID = "12345"
VOCABULARY = Vocabulary.from_class_map({"enzymes": {"enz7"}})
INDEX = surface_forms.build_index({"enz7": ["catalase"]})

# One enzyme mention over the third, fourth and fifth aggregated tokens.
CODES = [0, 0, ENZYMES, ENZYMES, ENZYMES, 0, 0, 0, 0, 0]


def _offline_tokenizer() -> PreTrainedTokenizerFast:
    """A real WordPiece tokenizer over an inline vocabulary: no download.

    Every ASCII letter is one token, so a word's tokens each cover one
    character of it. The same shape as the twin in
    `tests/cli/test_precompute_encodings.py`, widened to upper case, which
    `TEXT` is written in.
    """
    specials = ("[PAD]", "[UNK]", "[CLS]", "[SEP]")
    vocabulary = {token: index for index, token in enumerate(specials)}
    for character in string.ascii_letters:
        vocabulary.setdefault(character, len(vocabulary))
        vocabulary.setdefault("##" + character, len(vocabulary))

    backend = Tokenizer(models.WordPiece(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[
            ("[CLS]", vocabulary["[CLS]"]),
            ("[SEP]", vocabulary["[SEP]"]),
        ],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )


def _precompute_store(path: pathlib.Path, documents: dict[str, str]) -> str:
    """An encodings store holding `documents`, written the way
    `precompute-encodings` writes one."""
    with encodings_store.EncodingsStore(path, writable=True) as store:
        precompute_encodings._write_window(
            store, list(documents.items()), _offline_tokenizer(), False
        )
        return encodings_store.stamp_content_digest(store)


def _write_corpus(
    path: pathlib.Path, documents: dict[str, str] | None = None
) -> None:
    rows = "".join(
        f"{pubmed_id},{text},\n"
        for pubmed_id, text in (documents or {PUBMED_ID: TEXT}).items()
    )
    path.write_text(f"pubmed_id,abstract,fulltext\n{rows}", encoding="utf8")


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

    def hidden(self, embeddings, _mask):
        return embeddings

    def autocast_context(self):
        return contextlib.nullcontext()

    def uncached_embeddings(self):
        return contextlib.nullcontext()

    def token_tagger(self, _hidden_output):
        logits = torch.full((1, len(CODES), max(CODES) + 2), -10.0)
        for position, code in enumerate(CODES):
            logits[0, position, code] = 10.0
        return logits


class _RelationModel(_SpanModel):
    """A stand-in that also carries a relation head, as `ETEBrendaModel` does."""

    def predicted_relations(
        self, _batch, _mentions=None
    ) -> list[PredictedRelation]:
        return [
            PredictedRelation(
                predicate="produces",
                arguments=(frozenset({"bac1"}), frozenset({"enz7", "enz9"})),
            )
        ]


class _NoRelationFoundModel(_SpanModel):
    """A stand-in whose head scored this document's pairs and called every
    one of them null, which `predicted_relations` reports as an empty list."""

    def predicted_relations(
        self, _batch, _mentions=None
    ) -> list[PredictedRelation]:
        return []


class _FixedLinker:
    """Links every enzyme surface to one id and nothing else to anything."""

    def link(self, mention: str, entity_type: str) -> frozenset[str]:
        del mention
        return frozenset({"enz7"}) if entity_type == "enzymes" else frozenset()


def _patch_command_line(monkeypatch, tmp_path, dataset, output) -> None:
    """Name `dataset` and `output` on `infer`'s command line, and skip the
    process-global runtime setup."""
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
        "d3text.utils.load_fast_tokenizer", lambda _name: _offline_tokenizer()
    )


def _records(output: pathlib.Path) -> list[dict]:
    return [
        json.loads(line)
        for line in output.read_text(encoding="utf8").splitlines()
    ]


@pytest.fixture
def run_infer(tmp_path, monkeypatch):
    """Drive `infer.main` over the corpus and return its records."""

    def _run(model, linker=_FixedLinker(), documents=None, saved=None):
        dataset = tmp_path / "corpus.csv"
        _write_corpus(dataset, documents)
        output = tmp_path / "predictions.jsonl"

        _patch_command_line(monkeypatch, tmp_path, dataset, output)
        monkeypatch.setattr(
            infer,
            "load_model_config",
            lambda _path: ModelConfig(
                model_class="NERClassificationModel", base_model=BASE_MODEL
            ),
        )
        monkeypatch.setattr(
            infer.checkpoint,
            "load",
            lambda _path: (
                Checkpoint(
                    state_dict={},
                    vocabulary=VOCABULARY,
                    surface_form_index=INDEX,
                )
                if saved is None
                else saved
            ),
        )
        monkeypatch.setattr(infer.factory, "build_model", lambda *_a: model)
        monkeypatch.setattr(infer, "build_linker", lambda _saved: linker)

        infer.main()

        return _records(output)

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
    """No `cli/` module resolves `token_tagger` by hand.

    Every command goes through `token_supervision.resolve_token_tagger`.
    The whole package is swept, not a named list, because the regression is
    a command written later, which no list written today can name.
    """
    assert _cli_modules_naming_the_tagger_attribute() == []


def test_a_report_over_a_model_with_no_tagger_is_skipped_not_refused() -> None:
    """`infer` exits on the same miss, but a model that detects no span is an
    ordinary thing to evaluate: the block it cannot fill drops out and the
    rest of the run is scored."""
    assert (
        evaluate.report_predicted_linking(None, "nonexistent.hdf5", object())
        == {}
    )


def test_documents_in_no_encodings_store_are_tokenized_and_predicted(
    run_infer, machine_stores, tmp_path
) -> None:
    """`infer` runs the model on the text it is given, so a document no
    `precompute-encodings` run ever stored gets a record, and its offsets
    index that text. A configured store holding other documents is beside
    the point."""
    store = tmp_path / "encodings"
    _precompute_store(store, {"999": "unrelated"})
    machine_stores(encodings_store={BASE_MODEL: store})
    documents = {PUBMED_ID: TEXT, "67890": "XYZ ABCDEFG"}

    records = run_infer(_RelationModel(), documents=documents)

    assert [record["document"] for record in records] == list(documents)
    for record in records:
        text = documents[record["document"]]
        assert record["spans"]
        for span in record["spans"]:
            assert text[span["start"] : span["end"]] == span["surface"]
    assert records[0]["spans"][0]["surface"] == "CDE"


def test_no_dataset_is_a_usage_error_and_imports_no_training_data(
    tmp_path,
) -> None:
    """With nothing named, there is no input: falling back to the training
    corpus would import `brenda_references` at run time. Checked in a
    subprocess, for the reason the import test above gives."""
    probe = (
        "import sys\n"
        "sys.argv = ['infer', 'config.toml', 'model.pt', 'out.jsonl']\n"
        "from d3text.cli import infer\n"
        "try:\n"
        "    infer.command_line_args()\n"
        "except SystemExit as exit:\n"
        "    print('exit', exit.code)\n"
        "print('brenda_references' in sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=True,
    )

    assert "the following arguments are required: DATASET" in result.stderr
    assert result.stdout.split() == ["exit", "2", "False"], result.stderr


def _real_model(unfrozen_top_layers: int = 0) -> tuple[ModelConfig, object]:
    """A real `ETEBrendaModel` over the injected tiny trunk, identity hidden
    block, its tagger zeroed for the test to set; built before any embeddings
    store is configured, so building it opens none."""
    config = ModelConfig(
        base_model=BASE_MODEL,
        common_hidden_block=False,
        ramp_epochs=0,
        token_supervision=True,
        unfrozen_top_layers=unfrozen_top_layers,
    )
    model = factory.build_model(config, BRENDA_SCHEMA)
    model.to(model.device)
    model.eval()
    tagger = token_supervision.resolve_token_tagger(model)
    assert isinstance(tagger, torch.nn.Linear)
    with torch.no_grad():
        tagger.weight.zero_()
        tagger.bias.zero_()
    return config, model


def _run_real_infer(
    monkeypatch, tmp_path, config, reference, documents, **recorded
) -> list[dict]:
    """`infer.main` building its own model, loaded with `reference`'s
    weights."""
    dataset = tmp_path / "corpus.csv"
    _write_corpus(dataset, documents)
    output = tmp_path / "predictions.jsonl"
    _patch_command_line(monkeypatch, tmp_path, dataset, output)
    monkeypatch.setattr(infer, "load_model_config", lambda _path: config)
    state = reference.state_dict()
    monkeypatch.setattr(
        infer.checkpoint,
        "load",
        lambda _path: Checkpoint(
            state_dict=state, vocabulary=VOCABULARY, **recorded
        ),
    )

    infer.main()

    return _records(output)


@pytest.fixture
def fresh_store_openers():
    """Forget the stores `embeddings_store` and `layer_boundary_store` opened
    for earlier tests; both are cached per base model."""
    model_base.embeddings_store.cache_clear()
    model_base.layer_boundary_store.cache_clear()
    yield
    model_base.embeddings_store.cache_clear()
    model_base.layer_boundary_store.cache_clear()


def _store_files_digest(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(path.iterdir()):
        if file.name != "lock.mdb":
            digest.update(file.name.encode())
            digest.update(file.read_bytes())
    return digest.hexdigest()


RELATION_ID = "67890"
RELATION_TEXT = "catalase x subtilis"
RELATION_INDEX = surface_forms.build_index(
    {"enz7": ["catalase"], "bac1": ["subtilis"]}
)


def _live_hidden(model, text: str, document: str) -> torch.Tensor:
    """`text`'s hidden states on the aggregated token axis, as `infer`
    computes them."""
    (encoding,) = encodings_store.document_encodings(
        utils.split_and_tokenize(_offline_tokenizer(), [text]), 1
    )
    item = token_supervision.store_batch_item(encoding, int(document))
    with torch.no_grad(), model.uncached_embeddings():
        embeddings, mask = model.get_token_embeddings([item])
        with model.autocast_context():
            hidden = model.hidden(embeddings, mask)
    return hidden[0, : int(mask[0].sum())].float().cpu()


def _fit_tagger(model) -> None:
    """Set the tagger so that, over live features, `TEXT` is one enzyme span
    and `RELATION_TEXT`'s two mentions are tagged their own types, the word
    between them outside: two grounded arguments of a pair the schema
    admits. The bacteria logit also carries the first feature less 10, so an
    entry planted with a first feature of 1000 reads as bacteria.
    """
    (encoding,) = encodings_store.document_encodings(
        utils.split_and_tokenize(_offline_tokenizer(), [RELATION_TEXT]), 1
    )
    span = _live_hidden(model, TEXT, PUBMED_ID)
    related = _live_hidden(model, RELATION_TEXT, RELATION_ID)
    codes = torch.zeros(related.shape[0], dtype=torch.int64)
    for mention in token_supervision.live_mentions(
        RELATION_TEXT, encoding, RELATION_INDEX
    ):
        (entity_id,) = mention.entity_ids
        codes[mention.positions] = BRENDA_LABELS.by_prefix[entity_id[:3]]
    codes = torch.cat([torch.full((span.shape[0],), ENZYMES), codes])
    hidden = torch.cat([span, related])
    rows = (ENZYMES, BACTERIA)
    target = torch.stack(
        [torch.where(codes == row, 1.0, -1.0) for row in rows], dim=-1
    )
    target[:, 1] -= hidden[:, 0] - 10.0
    weight = torch.linalg.lstsq(hidden[:, 1:], target).solution
    tagger = token_supervision.resolve_token_tagger(model)
    with torch.no_grad():
        for column, row in enumerate(rows):
            tagger.weight[row, 1:] = weight[:, column]
        tagger.weight[BACTERIA, 0] = 1.0
        tagger.bias[BACTERIA] = -10.0


@pytest.mark.slow
@pytest.mark.parametrize(
    "planted", ["planted", "other", "absent", "empty"], ids=str
)
@pytest.mark.parametrize("unfrozen_top_layers", [0, 1])
def test_infer_reads_and_writes_no_embeddings_cache(
    patch_base_model,
    empty_token_label_store,
    machine_stores,
    fresh_store_openers,
    monkeypatch,
    tmp_path,
    planted,
    unfrozen_top_layers,
) -> None:
    """The embeddings caches key a document by its id alone, and the text a
    caller hands `infer` need not be what any of them was built from. So
    neither its span forward nor its relation forward reads or writes the
    CPU cache or an embeddings store, and building the model creates none,
    nor a sub-database in an env holding none. A writable LMDB open rewrites
    the env's lock file, so a store already there, of the kind the trunk
    reads or the other, keeps every file's bytes and modification time.

    The planted entry's first feature is 1000, which the tagger reads as
    bacteria: a span read through a cache comes out as bacteria. A second
    document puts two grounded arguments to the relation forward, so that
    forward's trunk runs too.
    """
    config, reference = _real_model(unfrozen_top_layers)
    _fit_tagger(reference)

    encodings = tmp_path / "encodings"
    _precompute_store(encodings, {PUBMED_ID: TEXT})
    machine_stores(encodings_store={BASE_MODEL: encodings})
    with encodings_store.EncodingsStore(encodings) as store:
        encoding = store.get(PUBMED_ID)
    item = token_supervision.store_batch_item(encoding, int(PUBMED_ID))
    hidden = reference.base_model.config.hidden_size

    embeddings = tmp_path / "embeddings"
    if planted == "empty":
        lmdb.open(str(embeddings)).close()
    if planted in ("planted", "other"):
        provenance = model_base._store_provenance(BASE_MODEL)
        if bool(unfrozen_top_layers) == (planted == "planted"):
            written = LayerBoundaryStore.create(embeddings, provenance, 1)
            value = torch.zeros(*encoding["input_ids"].shape, hidden)
        else:
            written = EmbeddingsStore.create(embeddings, provenance)
            value = torch.zeros(model_base.document_token_count(item), hidden)
        value[..., 0] = 1000.0
        written.put(int(PUBMED_ID), value)
        written.close()
        # Back-dated, so a write within the filesystem's timestamp tick shows.
        for file in embeddings.iterdir():
            os.utime(file, ns=(0, 0))
        before = _store_files_digest(embeddings)
    monkeypatch.setattr(
        model_base,
        "mconfig",
        MachineConfig(embeddings_store={BASE_MODEL: str(embeddings)}),
    )
    # Building `reference` cached each opener's answer from before the store
    # was configured; left there, it would hide any read below.
    model_base.embeddings_store.cache_clear()
    model_base.layer_boundary_store.cache_clear()
    cache = model_base.ByteBudgetCache(max_bytes=10**9)
    monkeypatch.setattr(model_base, "cpu_embeddings_cache", cache)
    # Linking is not under test, and the index is there to ground relations.
    monkeypatch.setattr(infer, "build_linker", lambda _saved: None)

    record, related = _run_real_infer(
        monkeypatch,
        tmp_path,
        config,
        reference,
        {PUBMED_ID: TEXT, RELATION_ID: RELATION_TEXT},
        surface_form_index=RELATION_INDEX,
    )

    assert [
        (span["surface"], span["entity_type"]) for span in related["spans"]
    ] == [("catalase", "enzymes"), ("subtilis", "bacteria")]
    assert related["relations"] is not None
    assert record["spans"] == [
        {
            "start": 0,
            "end": len(TEXT),
            "surface": TEXT,
            "entity_type": "enzymes",
            "entity_ids": None,
        }
    ]
    assert cache.size() == 0
    if planted in ("planted", "other"):
        assert _store_files_digest(embeddings) == before
        assert {
            file.name: file.stat().st_mtime_ns for file in embeddings.iterdir()
        } == {"data.mdb": 0, "lock.mdb": 0}
    elif planted == "empty":
        assert sub_databases(embeddings) == frozenset()
    else:
        assert not embeddings.exists()


@pytest.mark.slow
def test_live_spans_equal_the_store_backed_spans(
    patch_base_model,
    empty_token_label_store,
    machine_stores,
    monkeypatch,
    tmp_path,
) -> None:
    """Tokenizing in `infer` changes where the token ids come from, not what
    they are: for a document `precompute-encodings` stored, over more than
    one window, `infer`'s spans are the ones the store-backed path predicts.
    """
    monkeypatch.setattr(model_base, "mconfig", MachineConfig())
    monkeypatch.setattr(model_base, "cpu_embeddings_cache", None)
    config, reference = _real_model()
    tagger = token_supervision.resolve_token_tagger(reference)
    with torch.no_grad():
        tagger.weight[ENZYMES, 0] = 1.0
    text = " ".join(
        string.ascii_letters[index % 52] * (1 + index % 3)
        for index in range(300)
    )
    encodings = tmp_path / "encodings"
    _precompute_store(encodings, {PUBMED_ID: text})
    machine_stores(encodings_store={BASE_MODEL: encodings})

    (record,) = _run_real_infer(
        monkeypatch, tmp_path, config, reference, {PUBMED_ID: text}
    )

    with encodings_store.EncodingsStore(encodings) as store:
        assert store.get(PUBMED_ID)["input_ids"].shape[0] > 1
        expected = token_supervision.predicted_spans_from_store(
            store,
            None,
            {PUBMED_ID: text},
            reference.get_token_embeddings,
            reference.hidden,
            tagger,
            reference.autocast_context,
        )
    assert expected
    assert [
        (span["start"], span["end"], span["surface"], span["entity_type"])
        for span in record["spans"]
    ] == [
        (span.start, span.end, span.surface, span.entity_type)
        for span in expected
    ]


def test_infer_imports_no_cli_command_or_external_corpus_module(
    tmp_path,
) -> None:
    """`infer` tokenizes through a library module, so importing it loads no
    other command's module and no external corpus reader. Checked in a
    subprocess, for the reason the import test above gives."""
    probe = (
        "import sys; import d3text.cli.infer; "
        "print(sorted(m for m in ("
        "'d3text.cli.precompute_encodings', 'd3text.datasets.enzymener', "
        "'d3text.datasets.s800'"
        ") if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=True,
    )

    assert result.stdout.strip().endswith("[]"), result.stdout + result.stderr


STORED_TEXT = "catalase was found"
GIVEN_TEXT = "we then saw that the catalase acts"


def _label_store(path: pathlib.Path, documents: dict[str, str]) -> None:
    """A token-label store holding `documents`, each labelled by
    `precompute-token-labels`' own `label_document` against `INDEX`."""
    tokenizer = _offline_tokenizer()
    with token_labels.TokenLabelStore(path, writable=True) as store:
        token_labels.write_label_space(
            store,
            stamp=token_labels.IndexStamp.from_index(INDEX),
            tokenizer=token_labels.TokenizerStamp(
                base_model=BASE_MODEL,
                digest="test-tokenizer",
                window_length=utils.WINDOW_LENGTH,
                window_stride=utils.WINDOW_STRIDE,
            ),
        )
        for pubmed_id, text in documents.items():
            labels, _ = precompute_token_labels.label_document(
                text, frozenset({"enz7"}), INDEX, tokenizer
            )
            token_labels.store_token_labels(store, pubmed_id, labels)


def _surfaces(text: str, mentions) -> list[tuple[frozenset[str], str]]:
    """Each mention's candidates and the characters of `text` its positions
    cover, read through `text`'s own encoding merged as the model merges it.
    """
    encoding = utils.split_and_tokenize(_offline_tokenizer(), [text])
    offsets = aggregate_embeddings(
        torch.as_tensor(encoding["offset_mapping"]),
        torch.as_tensor(encoding["attention_mask"]),
    )
    return [
        (
            mention.entity_ids,
            text[
                int(offsets[mention.positions[0], 0]) : int(
                    offsets[mention.positions[-1], 1]
                )
            ],
        )
        for mention in mentions
    ]


@pytest.fixture
def grounding(patch_base_model, machine_stores, monkeypatch, tmp_path):
    """Run the real model through `infer.main` over `GIVEN_TEXT`-like text
    filed under the id the label store holds `STORED_TEXT` under; return the
    records and the mentions the relation forward was handed."""
    monkeypatch.setattr(model_base, "mconfig", MachineConfig())
    monkeypatch.setattr(model_base, "cpu_embeddings_cache", None)
    labels = tmp_path / "labels"
    _label_store(labels, {PUBMED_ID: STORED_TEXT})
    machine_stores(token_labels_store={BASE_MODEL: labels})
    config, reference = _real_model()
    handed: list[object] = []
    forward = ETEBrendaModel.forward

    def spy(self, *args, **kwargs):
        handed.append(kwargs.get("stored_mentions"))
        return forward(self, *args, **kwargs)

    monkeypatch.setattr(ETEBrendaModel, "forward", spy)

    def run(text: str) -> tuple[list[dict], list[object]]:
        records = _run_real_infer(
            monkeypatch,
            tmp_path,
            config,
            reference,
            {PUBMED_ID: text},
            surface_form_index=INDEX,
            token_labels_digest=surface_forms.index_digest(INDEX),
        )
        return records, handed

    return run


@pytest.mark.slow
def test_relations_ground_on_the_given_text_not_the_stored_entry(
    grounding,
) -> None:
    """The label store knows a document by its id alone, and the text a
    caller hands `infer` under that id need not be the text it was built
    from. The relation forward is handed the mentions matched in the given
    text, at that text's own token positions, not the stored entry's."""
    records, handed = grounding(GIVEN_TEXT)

    (record,) = records
    assert record["document"] == PUBMED_ID
    (mentions,) = handed
    assert set(mentions) == {0}
    assert _surfaces(GIVEN_TEXT, mentions[0]) == [
        (frozenset({"enz7"}), "catalase")
    ]


@pytest.mark.slow
def test_a_given_text_of_another_window_count_is_grounded_not_refused(
    grounding,
) -> None:
    """Given text spanning more windows than the stored entry under its id is
    the store's geometry check's `ValueError` were the entry read; grounding
    on the given text never meets it, and the document gets its record."""
    text = "x " * utils.WINDOW_LENGTH + "catalase"
    assert (
        len(utils.split_and_tokenize(_offline_tokenizer(), [text])["input_ids"])
        > 1
    )

    records, handed = grounding(text)

    (record,) = records
    assert record["document"] == PUBMED_ID
    (mentions,) = handed
    assert _surfaces(text, mentions[0]) == [(frozenset({"enz7"}), "catalase")]


@pytest.mark.slow
def test_infer_reads_no_token_label_entry(grounding, monkeypatch) -> None:
    """No entry of the label store is read on `infer`'s path, so none can
    stand in for the given text."""

    def refuse(self, pubmed_id):
        raise AssertionError(f"read the label store's entry {pubmed_id}")

    monkeypatch.setattr(token_supervision.TokenLabelReader, "_load", refuse)

    records, _ = grounding(GIVEN_TEXT)

    assert [record["document"] for record in records] == [PUBMED_ID]


def test_live_mentions_equal_the_stored_ones_for_the_stored_text(
    tmp_path,
) -> None:
    """For text the label store was built from, matching it live gives the
    mentions `exact_mentions` reads back: same candidates, same aggregated
    positions, across a window overlap. Gold plays no part in either."""
    text = "catalase " * 60 + "x " * utils.WINDOW_LENGTH + "catalase"
    path = tmp_path / "labels"
    _label_store(path, {PUBMED_ID: text})
    (encoding,) = encodings_store.document_encodings(
        utils.split_and_tokenize(_offline_tokenizer(), [text]), 1
    )
    assert encoding["input_ids"].shape[0] > 1

    live = token_supervision.live_mentions(text, encoding, INDEX)
    stored = token_supervision.TokenLabelReader(
        path, base_model=BASE_MODEL
    ).exact_mentions(PUBMED_ID, encoding["attention_mask"])

    assert stored
    assert [
        (mention.entity_ids, mention.positions.tolist()) for mention in live
    ] == [
        (mention.entity_ids, mention.positions.tolist()) for mention in stored
    ]


def _grounding_warnings(caplog) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == infer.__name__ and "ground" in record.getMessage()
    ]


def test_no_index_writes_null_relations_and_warns_once(
    run_infer, caplog
) -> None:
    """With no surface-form index there is nothing to match the given text
    against, so no relation argument can be grounded: every record says
    `null`, not a list, and the spans are still written."""
    saved = Checkpoint(state_dict={}, vocabulary=VOCABULARY)

    with caplog.at_level(logging.WARNING, logger=infer.__name__):
        records = run_infer(
            _RelationModel(),
            documents={PUBMED_ID: TEXT, "67890": TEXT},
            saved=saved,
        )

    assert [record["relations"] for record in records] == [None, None]
    assert all(record["spans"] for record in records)
    assert len(_grounding_warnings(caplog)) == 1


@pytest.mark.parametrize("moved", ["index", "rules", "unrecorded"])
def test_a_drifted_index_warns_once_and_still_writes_relations(
    run_infer, caplog, moved
) -> None:
    """An index or labelling rules other than the ones the training targets
    were placed by grounds relations on different mentions than the head
    learned from; `evaluate` warns on that and continues, and so does this.
    """
    index = surface_forms.index_digest(INDEX)
    recorded = {
        "index": lambda: {
            "token_labels_digest": "0" * 64,
            "labelling_rules_digest": token_labels.labelling_rules_digest(),
        },
        "rules": lambda: {
            "token_labels_digest": index,
            "labelling_rules_digest": "0" * 64,
        },
        "unrecorded": dict,
    }[moved]()
    saved = Checkpoint(
        state_dict={},
        vocabulary=VOCABULARY,
        surface_form_index=INDEX,
        **recorded,
    )

    with caplog.at_level(logging.WARNING, logger=infer.__name__):
        records = run_infer(
            _RelationModel(),
            documents={PUBMED_ID: TEXT, "67890": TEXT},
            saved=saved,
        )

    assert all(record["relations"] for record in records)
    assert len(_grounding_warnings(caplog)) == 1


def test_an_index_its_digests_tie_to_training_grounds_without_a_warning(
    caplog,
) -> None:
    """Where both recorded digests match the index and this build's labelling
    rules, relations ground on the mentions the head trained on, and nothing
    is warned: a warning there would bury the drifted case's."""
    saved = Checkpoint(
        state_dict={},
        vocabulary=VOCABULARY,
        surface_form_index=INDEX,
        token_labels_digest=surface_forms.index_digest(INDEX),
        labelling_rules_digest=token_labels.labelling_rules_digest(),
    )

    with caplog.at_level(logging.WARNING, logger=infer.__name__):
        index = infer.grounding_index(saved)

    assert index is INDEX
    assert [
        record for record in caplog.records if record.name == infer.__name__
    ] == []
