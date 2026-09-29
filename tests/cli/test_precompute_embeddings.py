"""`precompute-embeddings` embeds what it was asked to, and stores it all.

Everything here is about the bookkeeping around the embedding, which is stubbed
out: that every document reaches the LMDB, that every flag reaches the
embedder, that a run which stopped early cannot look finished, that the store
records what wrote it, and that a dead writer ends the run instead of hanging
on a bounded queue.
"""

import json
import mmap
import pathlib
import queue
import re
import shutil
import string
import threading
import types
from collections.abc import Callable
from typing import Any, NamedTuple, cast

import lmdb
import numpy as np
import pytest
import tokenizers
import torch
import tqdm
import transformers
from d3text import corpus, utils
from d3text.cli import precompute_embeddings
from d3text.embeddings_store import (
    EmbeddingsStore,
    LayerBoundaryStore,
    StoreProvenance,
    bytes_to_tensor,
    bytes_to_windowed_tensor,
    read_provenance,
    tensor_to_bytes,
)
from d3text.runtime import select_amp_dtype

_EMBEDDING_SHAPE = (2, 4)
_CONTEXT_WINDOW = 512

# A reservation past any 64-bit address space (2**62 bytes), so `lmdb.open`'s
# mmap fails on every host rather than only on a small one.
_UNMAPPABLE_GIB = float(2**32)

# What `transformers` reports for a tokenizer whose config declares no limit —
# which is true of the default base model, michiyasunaga/BioLinkBERT-base.
_NO_LIMIT_DECLARED = 1000000000000000019884624838656


class _RecordingEmbedder:
    """Stands in for `utils.embed_document`, recording how it was called.

    Each embedding is stamped with its own pubmed id, so a key/value mix-up
    between the row loop, the compression pool and the writer fails too.
    """

    def __init__(self, fill: float | None = None) -> None:
        self.calls: list[types.SimpleNamespace] = []
        self.loaded_tokenizers: list[str] = []
        self.loaded_base_models: list[str] = []
        self._fill = fill

    def __call__(self, doc: str, **kwargs: object) -> torch.Tensor:
        if not doc:
            # The real embedder's answer for an empty document. Recorded, not
            # refused, so a caller embedding one fails on its assertion here.
            self.calls.append(types.SimpleNamespace(pubmed_id=None, **kwargs))
            return torch.empty((0, _EMBEDDING_SHAPE[1]))

        pubmed_id = int(doc.split()[0])
        self.calls.append(types.SimpleNamespace(pubmed_id=pubmed_id, **kwargs))
        fill = pubmed_id if self._fill is None else self._fill
        return torch.full(_EMBEDDING_SHAPE, float(fill))

    @property
    def embedded_ids(self) -> list[int | None]:
        return [call.pubmed_id for call in self.calls]


_FAKE_CONFIG = transformers.BertConfig(
    max_position_embeddings=_CONTEXT_WINDOW,
    name_or_path="fake-base-model",
)


class _FakeBaseModel:
    """Stands in for the frozen transformer, which `_RecordingEmbedder` never
    calls. `main` moves it to a device and puts it in eval mode."""

    config = _FAKE_CONFIG

    def to(self, _device: torch.device) -> "_FakeBaseModel":
        return self

    def eval(self) -> "_FakeBaseModel":
        return self


@pytest.fixture
def embedder(monkeypatch: pytest.MonkeyPatch) -> _RecordingEmbedder:
    """Run `main` with no network, no config, no tokenizer, no transformer.

    The tokenizer and base model stubs record that they were loaded, so a test
    can assert a run ended before either. The config stub does not: reading a
    config.json is what a run may do before it commits to the weights.
    """
    recorder = _RecordingEmbedder()

    def load_fast_tokenizer(base_model: str) -> types.SimpleNamespace:
        recorder.loaded_tokenizers.append(base_model)
        return types.SimpleNamespace(model_max_length=_NO_LIMIT_DECLARED)

    def config_from_pretrained(
        *_args: object, **_kwargs: object
    ) -> transformers.PretrainedConfig:
        return _FAKE_CONFIG

    def from_pretrained(*args: object, **_kwargs: object) -> _FakeBaseModel:
        recorder.loaded_base_models.append(str(args[0]))
        return _FakeBaseModel()

    monkeypatch.setattr(utils, "load_fast_tokenizer", load_fast_tokenizer)
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", config_from_pretrained
    )
    monkeypatch.setattr(
        transformers.AutoModel, "from_pretrained", from_pretrained
    )
    monkeypatch.setattr(utils, "embed_document", recorder)
    # `main` setdefaults this; keep the mutation out of the wider test session.
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return recorder


def _write_dataset(path: pathlib.Path, pubmed_ids: list[int]) -> pathlib.Path:
    """Write the csv layout `stream_rows` expects: a leading unnamed index
    column (dropped by name), then pubmed_id/abstract/fulltext."""
    rows = "\n".join(
        f"{row},{pubmed_id},{pubmed_id} abstract,{pubmed_id} fulltext"
        for row, pubmed_id in enumerate(pubmed_ids)
    )
    path.write_text(f",pubmed_id,abstract,fulltext\n{rows}\n")
    return path


def _stored_embeddings(output_path: pathlib.Path) -> dict[bytes, np.ndarray]:
    """Read the store through its own codec, which is the only thing that
    knows the byte layout. Reaching for `blosc2` directly does not merely read
    the wrong thing — `unpack_array` **segfaults** on a blob it did not write,
    taking the whole session with it rather than failing one test."""
    return _sub_database_rows(output_path, "aggregated") or {}


def _run(
    monkeypatch: pytest.MonkeyPatch,
    output_path: pathlib.Path,
    datasets: list[pathlib.Path],
    *flags: str,
) -> dict[bytes, np.ndarray]:
    monkeypatch.setattr(
        "sys.argv",
        [
            "precompute-embeddings",
            "base-model",
            str(output_path),
            *(str(dataset) for dataset in datasets),
            *flags,
        ],
    )
    precompute_embeddings.main()
    return _stored_embeddings(output_path)


@pytest.mark.usefixtures("embedder")
def test_layer_boundary_store_accepts_all_encoder_layers_unfrozen(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All encoder layers trainable leaves a valid boundary at layer zero."""
    dataset = _write_dataset(tmp_path / "data.csv", [1])
    output_path = tmp_path / "embeddings"
    frozen_boundaries: list[list[int]] = []

    def fake_embed_document_and_prefix(
        _doc: str, *_args: object, frozen_layers: list[int], **_kwargs: object
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        frozen_boundaries.append(frozen_layers)
        return torch.zeros(_EMBEDDING_SHAPE), {0: torch.zeros(1, 2, 4)}

    monkeypatch.setattr(
        precompute_embeddings,
        "embed_document_and_prefix",
        fake_embed_document_and_prefix,
    )

    _run(
        monkeypatch,
        output_path,
        [dataset],
        "--unfrozen_top_layers",
        str(_FAKE_CONFIG.num_hidden_layers),
    )

    name = f"unfrozen_{_FAKE_CONFIG.num_hidden_layers}"
    assert set(_sub_database_rows(output_path, name) or {}) == {b"1"}
    assert frozen_boundaries == [[0]]


@pytest.mark.usefixtures("embedder")
@pytest.mark.parametrize("compress", [True, False])
def test_no_compress_stores_both_stores_raw(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, compress: bool
) -> None:
    """`--no_compress` reaches both stores' writers, not only the aggregated
    one. All-zero matrices compress to almost nothing, so a blob at least
    the raw bf16 size can only have been stored uncompressed; they are
    large enough that blosc2's frame header does not blur the two."""
    dataset = _write_dataset(tmp_path / "data.csv", [1])
    output_path = tmp_path / "embeddings"
    aggregated = torch.zeros(64, 64)
    prefix = torch.zeros(2, 64, 64)

    def fake_embed_document_and_prefix(
        *_args: object, frozen_layers: list[int], **_kwargs: object
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        return aggregated, {frozen: prefix for frozen in frozen_layers}

    monkeypatch.setattr(
        precompute_embeddings,
        "embed_document_and_prefix",
        fake_embed_document_and_prefix,
    )

    _run(
        monkeypatch,
        output_path,
        [dataset],
        "--unfrozen_top_layers",
        "1",
        *([] if compress else ["--no_compress"]),
    )

    for name, tensor, decode in (
        ("aggregated", aggregated, bytes_to_tensor),
        ("unfrozen_1", prefix, bytes_to_windowed_tensor),
    ):
        env = lmdb.open(str(output_path), readonly=True, lock=False, max_dbs=64)
        try:
            db = env.open_db(name.encode(), create=False)
            with env.begin(db=db) as txn:
                blob = txn.get(b"1")
        finally:
            env.close()
        raw_size = tensor.to(torch.bfloat16).nbytes
        assert (len(blob) < raw_size) is compress
        assert torch.equal(decode(blob), tensor.to(torch.bfloat16))


def _tiny_offline_tokenizer() -> transformers.PreTrainedTokenizerFast:
    """A real WordPiece tokenizer over an inline ASCII vocabulary.

    No network, no download: the same kind of in-process tokenizer
    `tests/test_utils.py`'s `_build_offline_fast_tokenizer` builds for
    `split_and_tokenize`'s own tests, so the tests below exercise a real
    tokenizer and a real (tiny) BERT rather than stand-ins for either.
    """
    specials = ("[PAD]", "[UNK]", "[CLS]", "[SEP]")
    vocabulary = {token: index for index, token in enumerate(specials)}
    for character in string.ascii_letters + string.digits:
        vocabulary.setdefault(character, len(vocabulary))
        vocabulary.setdefault("##" + character, len(vocabulary))
    backend = tokenizers.Tokenizer(
        tokenizers.models.WordPiece(vocabulary, unk_token="[UNK]")
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.BertPreTokenizer()
    backend.post_processor = tokenizers.processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[
            ("[CLS]", vocabulary["[CLS]"]),
            ("[SEP]", vocabulary["[SEP]"]),
        ],
    )
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )


def _write_tiny_dataset(path: pathlib.Path, texts: dict[int, str]) -> None:
    """One short line of text per pubmed id, so every document tokenizes to
    a single window under the patched `MAX_LENGTH` both stores share."""
    rows = "\n".join(
        f"{row},{pubmed_id},{text},"
        for row, (pubmed_id, text) in enumerate(texts.items())
    )
    path.write_text(f",pubmed_id,abstract,fulltext\n{rows}\n")


_TINY_HIDDEN = 8
_TINY_LAYERS = 4
_TINY_UNFROZEN = 1  # frozen_layers = _TINY_LAYERS - _TINY_UNFROZEN
# Must clear `utils.WINDOW_STRIDE` by at least 2 (the [CLS]/[SEP]
# special tokens), or the tokenizer refuses it, even for a one-window
# document that never exercises the overlap.
_TINY_MAX_LENGTH = 32


def _tiny_model_and_tokenizer() -> (
    tuple[
        transformers.BertConfig,
        transformers.BertModel,
        transformers.PreTrainedTokenizerFast,
    ]
):
    torch.manual_seed(0)
    config = transformers.BertConfig(
        vocab_size=64,
        hidden_size=_TINY_HIDDEN,
        num_hidden_layers=_TINY_LAYERS,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=32,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        name_or_path="tiny-bert",
    )
    model = transformers.BertModel(config).eval()
    return config, model, _tiny_offline_tokenizer()


def _patch_tiny_base_model(
    monkeypatch: pytest.MonkeyPatch,
    config: transformers.BertConfig,
    model: transformers.BertModel,
    tokenizer: transformers.PreTrainedTokenizerFast,
) -> None:
    # Pinned to the CPU, else a GPU box runs the reference and the real
    # pass on different devices, and a bf16 rounding boundary can flip on
    # their non-bit-identical reduction order.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", lambda *_a, **_k: config
    )
    monkeypatch.setattr(
        transformers.AutoModel, "from_pretrained", lambda *_a, **_k: model
    )
    monkeypatch.setattr(utils, "load_fast_tokenizer", lambda _m: tokenizer)
    monkeypatch.setattr(precompute_embeddings, "MAX_LENGTH", _TINY_MAX_LENGTH)


def _reference_layer_prefix(
    text: str,
    tokenizer: transformers.PreTrainedTokenizerFast,
    model: transformers.BertModel,
    frozen_layers: int,
    stride: int,
    max_len: int,
) -> torch.Tensor:
    """The layer-boundary prefix `embed_document_and_prefix` should produce.

    Independent of it: reads `BertModel.forward`'s own `hidden_states`
    tuple rather than replaying its encoder loop. `hidden_states[0]` is
    the embeddings output, `hidden_states[i]` (`i >= 1`) is after
    encoder layer i, so the frozen prefix is
    `hidden_states[frozen_layers]`.
    """
    encoding = utils.split_and_tokenize(
        tokenizer=tokenizer,
        inputs=text,
        stride=stride,
        max_length=max_len,
        return_offsets_mapping=False,
    )
    with (
        torch.inference_mode(),
        torch.amp.autocast(
            device_type=model.device.type,
            dtype=select_amp_dtype(model.device.type),
        ),
    ):
        outputs = model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            output_hidden_states=True,
        )
    assert outputs.hidden_states is not None  # asked for above
    return outputs.hidden_states[frozen_layers].detach().cpu()


def test_the_aggregated_and_layer_boundary_stores_are_built_in_one_walk(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--unfrozen_top_layers` must tokenize and forward each document once.

    Pins three things: the two stores hold exactly what two independent
    oracles predict for the same forward (`utils.embed_document`, and
    `BertModel`'s own `hidden_states`); each dataset streams once, not
    once per store; and the frozen layers forward each window once, not
    once per store either.
    """
    config, model, tokenizer = _tiny_model_and_tokenizer()
    _patch_tiny_base_model(monkeypatch, config, model, tokenizer)

    texts = {1: "a", 2: "b"}
    dataset = tmp_path / "tiny.csv"
    _write_tiny_dataset(dataset, texts)

    stream_calls: list[pathlib.Path] = []
    real_stream_rows = corpus.stream_rows

    def counting_stream_rows(path: pathlib.Path, batch_size: int) -> object:
        stream_calls.append(pathlib.Path(path))
        return real_stream_rows(path, batch_size)

    monkeypatch.setattr(
        precompute_embeddings.corpus, "stream_rows", counting_stream_rows
    )

    frozen_layers = _TINY_LAYERS - _TINY_UNFROZEN
    expected_full: dict[bytes, np.ndarray] = {}
    expected_prefix: dict[bytes, np.ndarray] = {}
    for pubmed_id, text in texts.items():
        key = str(pubmed_id).encode()
        expected_full[key] = (
            utils.embed_document(
                text,
                tokenizer=tokenizer,
                model=model,
                stride=precompute_embeddings.STRIDE,
                batch_size=50,
                max_len=_TINY_MAX_LENGTH,
            )
            .to(torch.bfloat16)
            .float()
            .numpy()
        )
        expected_prefix[key] = (
            _reference_layer_prefix(
                text,
                tokenizer=tokenizer,
                model=model,
                frozen_layers=frozen_layers,
                stride=precompute_embeddings.STRIDE,
                max_len=_TINY_MAX_LENGTH,
            )
            .to(torch.bfloat16)
            .float()
            .numpy()
        )

    # Installed only now, after the oracles above have made their own
    # forward calls: counts calls to the first frozen layer during `main`
    # alone, so a double forward per window inside one walk goes red.
    frozen_layer_0 = model.get_submodule("encoder.layer")[0]
    frozen_calls: list[None] = []
    real_layer_0_forward = frozen_layer_0.forward

    def counting_layer_0_forward(*args: object, **kwargs: object) -> object:
        frozen_calls.append(None)
        return real_layer_0_forward(*args, **kwargs)

    monkeypatch.setattr(frozen_layer_0, "forward", counting_layer_0_forward)

    output_path = tmp_path / "embeddings.lmdb"
    monkeypatch.setattr(
        "sys.argv",
        [
            "precompute-embeddings",
            "tiny-bert",
            str(output_path),
            str(dataset),
            "--unfrozen_top_layers",
            str(_TINY_UNFROZEN),
        ],
    )
    precompute_embeddings.main()

    stored_full = _stored_embeddings(output_path)
    stored_prefix = (
        _sub_database_rows(output_path, f"unfrozen_{_TINY_UNFROZEN}") or {}
    )

    assert stored_full.keys() == expected_full.keys()
    for key, expected in expected_full.items():
        np.testing.assert_array_equal(stored_full[key], expected)

    assert stored_prefix.keys() == expected_prefix.keys()
    for key, expected in expected_prefix.items():
        np.testing.assert_array_equal(stored_prefix[key], expected)

    assert stream_calls == [dataset], (
        f"the corpus must be streamed exactly once per dataset with "
        f"--unfrozen_top_layers set; streamed it {len(stream_calls)} "
        f"times: {stream_calls}"
    )

    assert len(frozen_calls) == len(texts), (
        f"the frozen layers must forward each window exactly once with "
        f"--unfrozen_top_layers set; forwarded them {len(frozen_calls)} "
        f"times for {len(texts)} one-window documents"
    )


def test_a_document_already_in_one_store_still_needs_the_other(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sub-databases' skip sets are independent.

    Seeds the aggregated sub-database with pubmed 1 and the boundary with
    pubmed 2, so a plain rerun must still forward 1 (for the boundary only),
    2 (for the aggregated rows only) and 3 (fresh, needs both), while 4 --
    already in both -- must not be forwarded at all.
    """
    config, model, tokenizer = _tiny_model_and_tokenizer()
    _patch_tiny_base_model(monkeypatch, config, model, tokenizer)
    frozen_layers = _TINY_LAYERS - _TINY_UNFROZEN

    texts = {1: "one", 2: "two", 3: "three", 4: "four"}
    dataset = tmp_path / "tiny.csv"
    _write_tiny_dataset(dataset, texts)

    output_path = tmp_path / "embeddings.lmdb"

    identity = StoreProvenance(
        base_model="tiny-bert",
        max_length=_TINY_MAX_LENGTH,
        stride=precompute_embeddings.STRIDE,
        forward_dtype=None,
    )
    aggregated = EmbeddingsStore.create(output_path, identity)
    for pubmed_id in (1, 4):
        aggregated.put(pubmed_id, torch.zeros(1, _TINY_HIDDEN))
    aggregated.close()
    boundary = LayerBoundaryStore.create(output_path, identity, _TINY_UNFROZEN)
    for pubmed_id in (2, 4):
        boundary.put(pubmed_id, torch.zeros(1, _TINY_MAX_LENGTH, _TINY_HIDDEN))
    boundary.close()

    calls: dict[str, list[tuple[list[int], bool]]] = {}
    real_fn = precompute_embeddings.embed_document_and_prefix
    real_embed_document = utils.embed_document

    def spy(
        text: str,
        *,
        frozen_layers: list[int],
        need_full: bool,
        **kwargs: object,
    ) -> tuple[torch.Tensor | None, dict[int, torch.Tensor]]:
        calls.setdefault(text, []).append((frozen_layers, need_full))
        return real_fn(
            text, frozen_layers=frozen_layers, need_full=need_full, **kwargs
        )

    def spy_full(text: str, **kwargs: Any) -> torch.Tensor:
        calls.setdefault(text, []).append(([], True))
        return real_embed_document(text, **kwargs)

    monkeypatch.setattr(precompute_embeddings, "embed_document_and_prefix", spy)
    monkeypatch.setattr(utils, "embed_document", spy_full)

    monkeypatch.setattr(
        "sys.argv",
        [
            "precompute-embeddings",
            "tiny-bert",
            str(output_path),
            str(dataset),
            "--unfrozen_top_layers",
            str(_TINY_UNFROZEN),
        ],
    )
    precompute_embeddings.main()

    assert calls["one"] == [([frozen_layers], False)]
    assert calls["two"] == [([], True)]
    assert calls["three"] == [([frozen_layers], True)]
    assert "four" not in calls


def _stamp(pubmed_id: int) -> float:
    """The stamp as the store can hold it.

    bf16 carries 8 significant bits, so 801 and 802 both come back as 800 —
    harmless for activations and fatal for a mix-up detector.
    """
    return torch.tensor(float(pubmed_id)).to(torch.bfloat16).float().item()


def _assert_holds_embeddings_for(
    stored: dict[bytes, np.ndarray], pubmed_ids: list[int]
) -> None:
    stamps = [_stamp(pubmed_id) for pubmed_id in pubmed_ids]
    assert len(set(stamps)) == len(stamps), (
        f"{pubmed_ids} do not stay distinct as bf16 stamps ({stamps}), so a "
        f"key/value mix-up would pass this assertion unnoticed"
    )

    assert sorted(stored) == sorted(str(p).encode() for p in pubmed_ids)
    for pubmed_id, stamp in zip(pubmed_ids, stamps):
        embedding = stored[str(pubmed_id).encode()]
        assert embedding.shape == _EMBEDDING_SHAPE
        assert (
            embedding == stamp
        ).all(), f"{pubmed_id} was stored under the wrong key"


@pytest.mark.usefixtures("embedder")
def test_writes_every_document_of_a_dataset_shorter_than_the_backlog(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The extreme case: with fewer rows than `MAX_BACKLOG` (at least 8), the
    in-loop flush never fires even once, so the drain is the only thing that
    writes anything."""
    pubmed_ids = [101, 102, 103]
    assert len(pubmed_ids) < precompute_embeddings.MAX_BACKLOG

    stored = _run(
        monkeypatch,
        tmp_path / "embeddings.lmdb",
        [_write_dataset(tmp_path / "small.csv", pubmed_ids)],
    )

    _assert_holds_embeddings_for(stored, pubmed_ids)


@pytest.mark.usefixtures("embedder")
def test_writes_the_documents_left_in_flight_when_the_rows_run_out(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The general case: the in-loop flush fires, but the rows run out with a
    tail of jobs still in flight. `MAX_BACKLOG` is pinned to a small value so
    the flush is exercised regardless of the box's core count."""
    monkeypatch.setattr(precompute_embeddings, "MAX_BACKLOG", 2)
    pubmed_ids = [201, 202, 203, 204, 205]

    stored = _run(
        monkeypatch,
        tmp_path / "embeddings.lmdb",
        [_write_dataset(tmp_path / "tail.csv", pubmed_ids)],
    )

    _assert_holds_embeddings_for(stored, pubmed_ids)


@pytest.mark.usefixtures("embedder")
def test_writes_each_dataset_in_full(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Datasets must not depend on each other to get flushed. When the backlog
    was shared across them, one dataset's leftovers were written only once a
    *later* dataset pushed the backlog over the threshold, and the final
    dataset's tail was lost for good."""
    first = [301, 302, 303]
    second = [401, 402, 403]

    stored = _run(
        monkeypatch,
        tmp_path / "embeddings.lmdb",
        [
            _write_dataset(tmp_path / "first.csv", first),
            _write_dataset(tmp_path / "second.csv", second),
        ],
    )

    _assert_holds_embeddings_for(stored, first + second)


def test_batch_size_reaches_the_embedder_as_given(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    embedder: _RecordingEmbedder,
) -> None:
    """The flag was accepted and dropped on the floor, so a run used
    `embed_document`'s own default no matter what was asked for."""
    _run(
        monkeypatch,
        tmp_path / "embeddings.lmdb",
        [_write_dataset(tmp_path / "flags.csv", [501, 502])],
        "--batch_size",
        "3",
    )

    assert embedder.embedded_ids == [501, 502]
    for call in embedder.calls:
        assert call.batch_size == 3
        assert call.max_len == precompute_embeddings.MAX_LENGTH


def test_the_window_is_pinned_regardless_of_the_tokenizer_sentinel(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    embedder: _RecordingEmbedder,
) -> None:
    """The window is `MAX_LENGTH`, the constant `precompute-encodings` uses.

    The default tokenizer's `model_max_length` is a huge sentinel, so the
    window cannot be asked of it.
    """
    _run(
        monkeypatch,
        tmp_path / "embeddings.lmdb",
        [_write_dataset(tmp_path / "default.csv", [601])],
    )

    (call,) = embedder.calls
    assert call.max_len == precompute_embeddings.MAX_LENGTH
    assert call.batch_size == 50


def test_a_base_model_narrower_than_the_pinned_window_is_rejected(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    embedder: _RecordingEmbedder,
) -> None:
    """A base model whose own context is narrower than the pinned window
    indexes past its position table if embedded anyway. The command must say
    so before anything loads, rather than dying inside the base model's
    forward."""
    narrow_config = transformers.BertConfig(
        max_position_embeddings=precompute_embeddings.MAX_LENGTH - 1,
        name_or_path="narrow-base-model",
    )
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *_a, **_k: narrow_config,
    )

    with pytest.raises(
        ValueError,
        match=(
            f"narrower than the {precompute_embeddings.MAX_LENGTH}-token "
            "window"
        ),
    ):
        _run(
            monkeypatch,
            tmp_path / "embeddings.lmdb",
            [_write_dataset(tmp_path / "narrow.csv", [701])],
        )

    assert embedder.calls == []
    assert embedder.loaded_tokenizers == []
    assert embedder.loaded_base_models == []


def test_a_base_model_wider_than_the_pinned_window_is_still_pinned(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    embedder: _RecordingEmbedder,
) -> None:
    """A base model with a wider context than the pinned window must still be
    embedded at the pin, not at its own, wider context — the encodings this
    store has to agree with are cut at the pin regardless of what the base
    model could support."""
    wide_config = transformers.BertConfig(
        max_position_embeddings=2 * utils.WINDOW_LENGTH,
        name_or_path="wide-base-model",
    )
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *_a, **_k: wide_config,
    )
    output_path = tmp_path / "embeddings.lmdb"

    _run(
        monkeypatch,
        output_path,
        [_write_dataset(tmp_path / "wide.csv", [702])],
    )

    (call,) = embedder.calls
    assert call.max_len == utils.WINDOW_LENGTH
    assert _provenance(output_path).max_length == utils.WINDOW_LENGTH


def test_documents_already_in_the_lmdb_are_not_re_embedded(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    embedder: _RecordingEmbedder,
) -> None:
    """Re-running over a dataset must resume, not redo. Embedding is the
    expensive half of this command; the LMDB already holds the answer."""
    output_path = tmp_path / "embeddings.lmdb"
    dataset = _write_dataset(tmp_path / "resume.csv", [801, 803])

    _run(monkeypatch, output_path, [dataset])
    assert embedder.embedded_ids == [801, 803]

    embedder.calls.clear()
    stored = _run(monkeypatch, output_path, [dataset])

    assert embedder.embedded_ids == []
    _assert_holds_embeddings_for(stored, [801, 803])


def test_force_regenerate_re_embeds_documents_already_in_the_lmdb(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    embedder: _RecordingEmbedder,
) -> None:
    """`-f` is the escape hatch from the skip above: it must re-embed, and
    overwrite the stored value rather than recompute and discard it."""
    output_path = tmp_path / "embeddings.lmdb"
    dataset = _write_dataset(tmp_path / "regen.csv", [901])

    _run(monkeypatch, output_path, [dataset])

    # A different fill proves the stored value was rewritten, not left behind.
    regenerated = _RecordingEmbedder(fill=7.0)
    monkeypatch.setattr(utils, "embed_document", regenerated)
    stored = _run(monkeypatch, output_path, [dataset], "-f")

    assert regenerated.embedded_ids == [901]
    assert (stored[b"901"] == 7.0).all()


def _write_dataset_with_empty_text(
    path: pathlib.Path, pubmed_ids: list[int]
) -> pathlib.Path:
    """The same layout as `_write_dataset`, with both text columns empty.

    Polars reads an empty cell as null, which `document_text` resolves to the
    empty string — the corpus's way of saying a document has neither an
    abstract nor a fulltext."""
    rows = "\n".join(
        f"{row},{pubmed_id},," for row, pubmed_id in enumerate(pubmed_ids)
    )
    path.write_text(f",pubmed_id,abstract,fulltext\n{rows}\n")
    return path


def test_a_document_with_no_text_is_not_embedded_or_stored(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    embedder: _RecordingEmbedder,
) -> None:
    """A row the corpus has no text for is described the same way by all three
    precompute commands: warned about and left out. Embedding it instead costs
    a forward pass and stores a 0-row matrix for a document the training splits
    drop, and that key then reads as already done on every resume."""
    output_path = tmp_path / "embeddings.lmdb"
    dataset = _write_dataset_with_empty_text(tmp_path / "empty.csv", [811])

    stored = _run(monkeypatch, output_path, [dataset])

    assert stored == {}
    assert embedder.embedded_ids == []


def test_force_regenerate_deletes_a_document_that_lost_its_text(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    embedder: _RecordingEmbedder,
) -> None:
    """`-f` makes the store agree with the corpus, in both directions: a
    document whose text is gone loses its stale entry rather than being
    re-embedded into one."""
    output_path = tmp_path / "embeddings.lmdb"
    dataset = _write_dataset(tmp_path / "lost.csv", [821, 823])
    _run(monkeypatch, output_path, [dataset])
    assert sorted(_stored_embeddings(output_path)) == [b"821", b"823"]

    embedder.calls.clear()
    _write_dataset_with_empty_text(dataset, [821])
    stored = _run(monkeypatch, output_path, [dataset], "-f")

    assert b"821" not in stored
    assert embedder.embedded_ids == []


class _BulkyEmbedder:
    """An embedding the store cannot shrink away.

    The codec turns a constant matrix into a few hundred bytes however large it
    is; bf16 noise compresses by about 1.4x and nothing more.
    """

    _COLUMNS = 768

    def __init__(self, rows: int) -> None:
        self._rows = rows
        self.embedded_ids: list[int] = []

    def __call__(self, doc: str, **_kwargs: object) -> torch.Tensor:
        self.embedded_ids.append(int(doc.split()[0]))
        generator = torch.Generator().manual_seed(len(self.embedded_ids))
        return torch.randn(self._rows, self._COLUMNS, generator=generator)


def _recorded_lmdb_open(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """The keyword arguments `main` opens the writable environment with.

    A read-only reopen brings its own reservation, so the requested one is
    visible only in the call itself.
    """
    opened: dict[str, object] = {}
    real_open = lmdb.open

    def recording_open(path: str, **kwargs: object) -> lmdb.Environment:
        opened.update(kwargs)
        return real_open(path, **kwargs)

    monkeypatch.setattr(precompute_embeddings.lmdb, "open", recording_open)
    return opened


@pytest.mark.usefixtures("embedder")
def test_the_reserved_map_size_covers_the_corpus_and_follows_the_flag(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full-corpus store has outgrown 256 GiB, so a default at or below that
    runs out partway through a pass — after the GPU time is already spent.
    The reservation is virtual on Linux, so the headroom is free."""
    opened = _recorded_lmdb_open(monkeypatch)
    dataset = _write_dataset(tmp_path / "budget.csv", [1001])

    _run(monkeypatch, tmp_path / "default.lmdb", [dataset])
    assert opened["map_size"] > 256 * 1024**3

    _run(monkeypatch, tmp_path / "flagged.lmdb", [dataset], "--map_size", "2")
    assert opened["map_size"] == 2 * 1024**3


@pytest.mark.parametrize(
    "map_size",
    ["0", "-1", "1e-12"],
    ids=["zero", "negative", "under-a-byte"],
)
def test_a_map_size_reserving_nothing_is_rejected_before_any_embedding(
    map_size: str,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    embedder: _RecordingEmbedder,
) -> None:
    """A reservation of zero bytes is not a small budget, it is no budget.

    The complaint LMDB does have comes too late or not at all, and the
    alternative is hours of GPU time ending at the very first write.
    """
    output_path = tmp_path / "nomap.lmdb"

    # As the message renders it, which is the float argparse parsed.
    named = re.escape(str(float(map_size)))
    with pytest.raises(ValueError, match=f"--map_size .*got {named}"):
        _run(
            monkeypatch,
            output_path,
            [_write_dataset(tmp_path / "nomap.csv", [1201])],
            "--map_size",
            map_size,
        )

    assert embedder.calls == []
    assert embedder.loaded_tokenizers == []
    assert embedder.loaded_base_models == []
    assert not output_path.exists()


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--batch_size", "0"),
        ("--batch_size", "-1"),
        ("--stream_batch", "0"),
        ("--stream_batch", "-1"),
        ("--commit_every", "0"),
        ("--commit_every", "-1"),
    ],
    ids=[
        "batch_size-zero",
        "batch_size-negative",
        "stream_batch-zero",
        "stream_batch-negative",
        "commit_every-zero",
        "commit_every-negative",
    ],
)
def test_a_non_positive_batch_flag_is_rejected_before_any_embedding(
    flag: str,
    value: str,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    embedder: _RecordingEmbedder,
) -> None:
    """A non-positive count is rejected before anything loads.

    Pins that `positive_int` raises for `--batch_size`, `--commit_every` and
    `--stream_batch` alike, and before the tokenizer or base model, so a bad
    value never reaches the weights.
    """
    output_path = tmp_path / "nonpositive.lmdb"

    with pytest.raises(ValueError, match=f"{flag[2:]} .*got {value}"):
        _run(
            monkeypatch,
            output_path,
            [_write_dataset(tmp_path / "nonpositive.csv", [1301])],
            flag,
            value,
        )

    assert embedder.calls == []
    assert embedder.loaded_tokenizers == []
    assert embedder.loaded_base_models == []
    assert not output_path.exists()


def test_a_map_size_lmdb_rejects_is_refused_before_anything_loads(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    embedder: _RecordingEmbedder,
) -> None:
    """A map past the address space is LMDB's to refuse, and cheaply.

    `lmdb.MemoryError` is the library's own class, not the builtin it shadows,
    so a bare `MemoryError` would match nothing.
    """
    with pytest.raises(lmdb.MemoryError):
        _run(
            monkeypatch,
            tmp_path / "unmappable.lmdb",
            [_write_dataset(tmp_path / "unmappable.csv", [1601])],
            "--map_size",
            str(_UNMAPPABLE_GIB),
        )

    assert embedder.loaded_tokenizers == []
    assert embedder.loaded_base_models == []


@pytest.mark.parametrize("map_size", [0.0, -1.0, 1e-12])
def test_the_map_size_rejection_describes_the_lmdb_it_stands_in_for(
    map_size: float,
) -> None:
    """The refusal's explanation holds for every value it refuses.

    It once claimed LMDB reads any non-positive `map_size` as the store's
    current size, which is true of zero and false of a negative value.
    """
    with pytest.raises(ValueError) as raised:
        precompute_embeddings.map_size_bytes(map_size)

    message = str(raised.value)
    assert "non-positive" not in message
    assert "base model" not in message
    assert "1 MiB" in message
    assert "negative" in message


def test_lmdb_reads_a_map_size_of_zero_as_the_store_default(
    tmp_path: pathlib.Path,
) -> None:
    """Zero means "keep whatever this store has", which for a new one is 1 MiB.

    Pinned so that a future LMDB raising instead reads as the validator's
    reasoning having changed, rather than standing on a claim nothing checks.
    """
    env = lmdb.open(str(tmp_path / "fresh.lmdb"), map_size=0)
    try:
        assert env.info()["map_size"] == 1024**2
    finally:
        env.close()


def test_lmdb_refuses_a_negative_map_size(tmp_path: pathlib.Path) -> None:
    """The other half of the same premise, and it does not behave like zero:
    the reservation reaches C as an unsigned size, so `lmdb.open` refuses it
    outright. That it refuses is not why the validator exists: the
    `OverflowError` names neither the flag nor the value that produced it, and
    the command that has both must be the one to say so."""
    with pytest.raises(OverflowError):
        lmdb.open(str(tmp_path / "negative.lmdb"), map_size=-1)


def test_lmdb_rounds_a_map_size_up_to_whole_pages(
    tmp_path: pathlib.Path,
) -> None:
    """Why the validator draws its line at one byte and claims nothing more.

    The page size is the host's — 8192 here, 32768 on a 16 KiB-page host — so a
    literal would fail with nothing wrong.
    """
    env = lmdb.open(str(tmp_path / "onebyte.lmdb"), map_size=1)
    try:
        assert env.info()["map_size"] == 2 * mmap.PAGESIZE
        with pytest.raises(lmdb.MapFullError), env.begin(write=True) as txn:
            txn.put(b"1301", b"0123456789")
    finally:
        env.close()


@pytest.mark.usefixtures("embedder")
def test_a_map_size_too_small_for_the_dataset_ends_the_run(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Running out of `map_size` used to be indistinguishable from finishing:
    the writer committed its prefix, the command logged `Done.`, and the next
    run read the missing documents as already embedded."""
    bulky = _BulkyEmbedder(rows=512)
    monkeypatch.setattr(utils, "embed_document", bulky)
    output_path = tmp_path / "toosmall.lmdb"
    dataset = _write_dataset(tmp_path / "bulky.csv", [1101, 1102, 1103])

    with pytest.raises(RuntimeError) as raised:
        _run(
            monkeypatch,
            output_path,
            [dataset],
            # 1.5 MiB against three embeddings of roughly half a MiB each,
            # committed one at a time so that two of them are already on disk
            # when the third runs out.
            "--map_size",
            str(1.5 / 1024),
            "--commit_every",
            "1",
        )

    message = str(raised.value)
    assert "map_size" in message
    named = [p for p in bulky.embedded_ids if str(p) in message]
    assert len(named) == 1, message

    # What is left is a prefix no reader can tell from a finished store, which
    # is what makes reporting success on it dangerous: the resume path reads
    # every key it holds as done and stops asking about the rest.
    stored = _stored_embeddings(output_path)
    assert stored
    assert str(named[0]).encode() not in stored


def test_a_map_size_too_small_for_one_document_is_refused_before_anything_loads(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    embedder: _RecordingEmbedder,
) -> None:
    """A map can clear `lmdb.open` and still be hopeless.

    The probe is one window of bf16 activations, well past what the provenance
    record's own small write would have caught.
    """
    output_path = tmp_path / "toosmall_for_any_document.lmdb"
    dataset = _write_dataset(tmp_path / "toosmall.csv", [1401])

    with pytest.raises(ValueError, match="map_size"):
        _run(
            monkeypatch,
            output_path,
            [dataset],
            "--map_size",
            str(0.25 / 1024),
        )

    assert embedder.calls == []
    assert embedder.loaded_tokenizers == []
    assert embedder.loaded_base_models == []


# Long enough that the embedding loop fills the queue it hands the writer and
# has to wait for room, which is where a dead writer turns into a hang.
_LONGER_THAN_THE_QUEUE = list(range(2001, 2201))


class _WriterDied(RuntimeError):
    """Injected: a failure inside the writer's loop it has no branch for."""


class _FailingProgressBar(tqdm.tqdm):  # type: ignore[type-arg]
    """A `Written` bar that fails the first time the writer counts a document.

    Subclassing the real bar keeps the writer's contract intact, so what is
    injected is a failure *inside* its loop and nothing else.
    """

    error: BaseException

    def update(self, n: float | None = 1) -> bool | None:
        super().update(n)
        raise self.error


class _Injection(NamedTuple):
    """What an injected writer failure promises the test.

    `expected` is the error `main` must raise; `writer_entered` is set once
    the writer was actually started. The second exists because an injection
    the guards ahead of the writer satisfy proves nothing about the writer.
    """

    expected: type[BaseException]
    writer_entered: threading.Event


def _watch_the_writer(
    monkeypatch: pytest.MonkeyPatch,
    env_for_writer: Callable[
        [lmdb.Environment], lmdb.Environment
    ] = lambda env: env,
) -> threading.Event:
    """Record that the writer was started, and hand it `env_for_writer(env)`.

    `main` looks the writer up by name when it starts the thread, so the
    wrapper is what runs there and the real writer runs inside it.
    """
    entered = threading.Event()
    real_writer = precompute_embeddings.writer_thread

    def writer(env: lmdb.Environment, *args: Any) -> None:
        entered.set()
        real_writer(env_for_writer(env), *args)

    monkeypatch.setattr(precompute_embeddings, "writer_thread", writer)
    return entered


def _writer_dies_mid_document(monkeypatch: pytest.MonkeyPatch) -> _Injection:
    """Kill the writer with a live transaction to unwind."""
    error = _WriterDied("the writer died holding an open transaction")
    real_bar = tqdm.tqdm

    def bar(**kwargs: Any) -> Any:
        if str(kwargs.get("desc", "")).strip() != "Written":
            return real_bar(**kwargs)
        failing = _FailingProgressBar(**kwargs)
        failing.error = error
        return failing

    monkeypatch.setattr(
        precompute_embeddings, "tqdm", types.SimpleNamespace(tqdm=bar)
    )
    return _Injection(_WriterDied, _watch_the_writer(monkeypatch))


class _ReadonlyView:
    """`env` as a read-only mount presents it: write transactions refused.

    The writer is type-checked at the call and takes an `lmdb.Environment`
    only — a C type that cannot be subclassed, and one py-lmdb refuses to open
    a second time in one process — so this passes as one the way a `spec`ed
    `unittest.mock` does, by reporting its class.
    """

    def __init__(self, env: lmdb.Environment) -> None:
        self._env = env

    @property
    def __class__(self) -> type[lmdb.Environment]:  # type: ignore[misc]
        return lmdb.Environment

    def begin(self, write: bool = False, **kwargs: Any) -> lmdb.Transaction:
        if write:
            raise lmdb.ReadonlyError("injected: the store is read-only")
        return self._env.begin(write=write, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)


def _writer_cannot_open_its_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> _Injection:
    """Kill the writer before it has a transaction at all.

    A read-only mount would be refused earlier by the guards, so only the
    writer gets a read-only view, making its own first `begin(write=True)`
    the call that raises.
    """

    def readonly_view(env: lmdb.Environment) -> lmdb.Environment:
        return cast(lmdb.Environment, _ReadonlyView(env))

    return _Injection(
        lmdb.ReadonlyError, _watch_the_writer(monkeypatch, readonly_view)
    )


def _run_with_deadline(
    monkeypatch: pytest.MonkeyPatch,
    output_path: pathlib.Path,
    datasets: list[pathlib.Path],
    *flags: str,
    seconds: float = 30.0,
) -> BaseException | None:
    """Run `main` off the test thread and return whatever it raised.

    Calling it directly would turn the regression this guards — a command that
    never returns — into a pytest run that never returns.
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "precompute-embeddings",
            "base-model",
            str(output_path),
            *(str(dataset) for dataset in datasets),
            *flags,
        ],
    )
    raised: list[BaseException] = []

    def run() -> None:
        try:
            precompute_embeddings.main()
        except BaseException as exc:
            raised.append(exc)

    runner = threading.Thread(target=run, daemon=True)
    runner.start()
    runner.join(timeout=seconds)
    assert not runner.is_alive(), (
        f"the command was still running after {seconds}s: the writer thread "
        f"is gone and the embedding loop is still waiting to hand it work"
    )
    return raised[0] if raised else None


@pytest.mark.parametrize(
    "inject",
    [_writer_dies_mid_document, _writer_cannot_open_its_transaction],
    ids=["mid-document", "before-the-first-transaction"],
)
@pytest.mark.usefixtures("embedder")
def test_a_writer_that_dies_ends_the_run_instead_of_hanging(
    inject: Callable[[pytest.MonkeyPatch], _Injection],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A writer ending silently leaves the producer waiting forever.

    Only `MapFullError` was ever announced; everything else was a hang, which
    is worse than a crash because nothing on the other end can tell.
    """
    monkeypatch.setattr(precompute_embeddings, "MAX_BACKLOG", 2)
    output_path = tmp_path / "dying.lmdb"
    expected, writer_entered = inject(monkeypatch)
    dataset = _write_dataset(tmp_path / "dying.csv", _LONGER_THAN_THE_QUEUE)

    raised = _run_with_deadline(monkeypatch, output_path, [dataset])

    assert writer_entered.is_set(), (
        f"the run ended before the writer was started, so nothing here "
        f"exercised it; got {raised!r}"
    )
    assert isinstance(raised, expected), (
        f"the writer's failure has to reach the caller rather than be "
        f"swallowed into a `Done.`; got {raised!r}"
    )


# One value, fixed rather than random, so the page layout the search below
# lands on is the same on every run.
_FILLER = b"\xa5" * 180


def _commits(path: pathlib.Path, documents: int) -> bool:
    shutil.rmtree(path, ignore_errors=True)
    env = lmdb.open(str(path), map_size=1 << 20)
    try:
        with env.begin(write=True) as txn:
            for document in range(documents):
                txn.put(f"{document:06d}".encode(), _FILLER)
    except lmdb.MapFullError:
        return False
    finally:
        env.close()
    return True


def _store_with_no_room_left(path: pathlib.Path) -> lmdb.Environment:
    """A real LMDB filled to its last page, where even a delete runs out.

    A store built by writing until a `put` fails is no use: the pages its
    earlier commits freed are exactly what the delete then spends.
    """
    low, high = 1, 1 << 15
    while low < high:
        middle = (low + high + 1) // 2
        if _commits(path, middle):
            low = middle
        else:
            high = middle - 1
    assert _commits(path, low), "no transaction at all fits in the map"
    return lmdb.open(str(path), map_size=1 << 20)


class _SnapshotBar(tqdm.tqdm):  # type: ignore[type-arg]
    """A `Written` bar that reads the store's committed snapshot per document.

    `commit_every` leaves no trace in the finished store, so mid-run is the
    only moment the writer's accounting can be observed.
    """

    env: lmdb.Environment
    watched: bytes
    seen: list[bool]

    def update(self, n: float | None = 1) -> bool | None:
        with self.env.begin() as txn:
            self.seen.append(txn.get(self.watched) is not None)
        return super().update(n)


def _snapshot_bar(env: lmdb.Environment, watched: bytes) -> _SnapshotBar:
    bar = _SnapshotBar(disable=True)
    bar.env = env
    bar.watched = watched
    bar.seen = []
    return bar


def _drain(
    env: lmdb.Environment,
    items: list[tuple[bytes, bytes | None]],
    commit_every: int,
    pbar: tqdm.tqdm | None = None,
) -> precompute_embeddings.WriterState:
    """Run the writer over `items` here, and return what it recorded.

    Everything it needs in order to stop is in the queue before it starts, so
    these assertions need not race a thread. Every item goes to the main
    database, which is what LMDB reads a `db` of None as.
    """
    in_q: queue.Queue[Any] = queue.Queue()
    for key, value in items:
        in_q.put((None, key, value))
    stop_evt = threading.Event()
    stop_evt.set()
    state = precompute_embeddings.WriterState()
    precompute_embeddings.writer_thread(
        env,
        in_q,
        stop_evt,
        commit_every,
        tqdm.tqdm(disable=True) if pbar is None else pbar,
        state,
    )
    return state


def test_a_map_full_on_a_delete_does_not_report_a_write(
    tmp_path: pathlib.Path,
) -> None:
    """The queue carries deletes as well as puts, and a delete has to grow the
    map the same way a put does, so it reaches the map-full branch just as
    readily. Reporting it as a write names an operation the run never
    attempted, in the one message someone reads while diagnosing a store that
    would not grow."""
    env = _store_with_no_room_left(tmp_path / "brim.lmdb")
    try:
        state = _drain(env, [(b"000000", None)], commit_every=100)
    finally:
        env.close()

    assert isinstance(state.failure, precompute_embeddings.StoreFullError)
    message = str(state.failure)
    assert "000000" in message
    assert "writing document" not in message, message
    assert "deleting" in message, message


def test_a_map_full_on_a_put_still_reports_a_write(
    tmp_path: pathlib.Path,
) -> None:
    env = _store_with_no_room_left(tmp_path / "brim.lmdb")
    try:
        state = _drain(env, [(b"000000", _FILLER)], commit_every=100)
    finally:
        env.close()

    assert "writing document 000000" in str(state.failure)


def test_a_delete_of_a_key_the_store_lacks_does_not_close_the_batch(
    tmp_path: pathlib.Path,
) -> None:
    """An item that wrote nothing must not spend one of `commit_every`'s slots.

    `-f` drops stale entries without reading the store first, so most of those
    deletes remove nothing at all.
    """
    env = lmdb.open(str(tmp_path / "absent.lmdb"), map_size=1 << 20)
    bar = _snapshot_bar(env, watched=b"4101")
    try:
        state = _drain(
            env,
            [
                (b"4101", _FILLER),
                (b"4102", None),
                (b"4103", None),
                (b"4104", _FILLER),
            ],
            commit_every=3,
            pbar=bar,
        )
    finally:
        env.close()

    assert state.failure is None
    assert bar.seen == [False, False], (
        f"nothing between the two stored documents changed the transaction, "
        f"so the batch of three should not have closed; got {bar.seen}"
    )


def test_a_delete_that_removes_a_key_closes_the_batch(
    tmp_path: pathlib.Path,
) -> None:
    """The other half, and the guard against over-correcting: a delete that
    actually removes something is work the transaction is holding, so it has
    to keep counting."""
    env = lmdb.open(str(tmp_path / "present.lmdb"), map_size=1 << 20)
    with env.begin(write=True) as txn:
        txn.put(b"4202", _FILLER)
    bar = _snapshot_bar(env, watched=b"4201")
    try:
        state = _drain(
            env,
            [(b"4201", _FILLER), (b"4202", None), (b"4203", _FILLER)],
            commit_every=2,
            pbar=bar,
        )
    finally:
        env.close()

    assert state.failure is None
    assert bar.seen == [False, True], (
        f"the delete removed a key, closing the batch of two and committing "
        f"the document before it; got {bar.seen}"
    )


def test_the_lmdb_is_closed_when_main_raises_mid_dataset(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    embedder: _RecordingEmbedder,
) -> None:
    """Anything raising out of the dataset loop still closes the environment.

    Cheap to leak for a one-shot CLI, but the suite calls `main` more than once
    in a process, so the lock file is checked directly rather than through
    exit.
    """
    output_path = tmp_path / "leaky.lmdb"
    opened: list[lmdb.Environment] = []
    real_open = lmdb.open

    def recording_open(path: str, **kwargs: Any) -> lmdb.Environment:
        env = real_open(path, **kwargs)
        opened.append(env)
        return env

    monkeypatch.setattr(precompute_embeddings.lmdb, "open", recording_open)

    class _EmbedBlewUp(RuntimeError):
        pass

    def failing_embed(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise _EmbedBlewUp("embed_document blew up")

    monkeypatch.setattr(utils, "embed_document", failing_embed)

    with pytest.raises(_EmbedBlewUp):
        _run(
            monkeypatch,
            output_path,
            [_write_dataset(tmp_path / "boom.csv", [1701])],
        )

    (env,) = opened
    with pytest.raises(lmdb.Error):
        env.stat()


def _provenance(output_path: pathlib.Path) -> StoreProvenance | None:
    env = lmdb.open(str(output_path), readonly=True, lock=False)
    try:
        return read_provenance(env)
    finally:
        env.close()


@pytest.mark.usefixtures("embedder")
def test_the_store_records_the_model_window_and_stride_that_wrote_it(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The store records what wrote it, which its blob shapes cannot tell.

    Encoders of one hidden size write identical shapes. `MAX_LENGTH` is
    patched off its real value so the assertion is not trivially true.
    """
    monkeypatch.setattr(precompute_embeddings, "MAX_LENGTH", 128)
    output_path = tmp_path / "embeddings.lmdb"

    _run(
        monkeypatch,
        output_path,
        [_write_dataset(tmp_path / "stamp.csv", [1301])],
    )

    assert _provenance(output_path).identity == (
        "base-model",
        128,
        precompute_embeddings.STRIDE,
    )


def test_the_store_is_stamped_with_the_window_and_stride_it_was_embedded_at(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    embedder: _RecordingEmbedder,
) -> None:
    """The stamp is read against the embedding it describes, not against the
    constant both are meant to be taken from: a literal at either site leaves
    the other one right. A mis-stamped store is worse than an unstamped one,
    since `record_provenance` refuses the second and resumes onto the first."""
    monkeypatch.setattr(precompute_embeddings, "MAX_LENGTH", 128)
    output_path = tmp_path / "embeddings.lmdb"

    _run(
        monkeypatch,
        output_path,
        [_write_dataset(tmp_path / "stride.csv", [1302, 1303])],
    )

    stamped = _provenance(output_path)
    assert stamped is not None
    assert embedder.embedded_ids == [1302, 1303]
    for call in embedder.calls:
        assert call.max_len == stamped.max_length
        assert call.stride == stamped.stride


def test_a_store_stamped_before_the_dtype_field_still_resumes(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    embedder: _RecordingEmbedder,
) -> None:
    """A stamp with no forward dtype still resumes, and is left unstamped.

    Identity is model, window and stride; the dtype decides nothing.
    Restamping would claim the older documents were computed the new way.
    """
    monkeypatch.setattr(precompute_embeddings, "MAX_LENGTH", 128)
    output_path = tmp_path / "embeddings.lmdb"
    with lmdb.open(str(output_path), map_size=2**20) as env:
        with env.begin(write=True) as transaction:
            transaction.put(
                b"\x00provenance",
                json.dumps(
                    {
                        "format": 2,
                        "base_model": "base-model",
                        "max_length": 128,
                        "stride": precompute_embeddings.STRIDE,
                    }
                ).encode(),
            )

    _run(
        monkeypatch,
        output_path,
        [_write_dataset(tmp_path / "resume.csv", [1501])],
    )

    assert embedder.embedded_ids == [1501]
    assert _provenance(output_path).forward_dtype is None


def test_a_fresh_store_records_the_precision_its_forward_ran_in(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    embedder: _RecordingEmbedder,
) -> None:
    """Derived from `select_amp_dtype` rather than spelled out, since the
    right answer is a property of the card the run picked: a literal here
    would pin this machine and pass on no other."""
    output_path = tmp_path / "embeddings.lmdb"

    _run(
        monkeypatch,
        output_path,
        [_write_dataset(tmp_path / "fresh.csv", [1502])],
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert _provenance(output_path).forward_dtype == str(
        select_amp_dtype(device)
    )


def test_adding_to_a_store_another_model_wrote_is_refused(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    embedder: _RecordingEmbedder,
) -> None:
    """A resume onto another model's store is the last chance to tell.

    Once written, the matrices are the same shape and carry no mark. The
    refusal is free: the store says who wrote it, so the run ends before the
    model loads.
    """
    output_path = tmp_path / "embeddings.lmdb"
    dataset = _write_dataset(tmp_path / "mixed.csv", [1401])
    _run(monkeypatch, output_path, [dataset])

    embedder.calls.clear()
    embedder.loaded_tokenizers.clear()
    embedder.loaded_base_models.clear()
    monkeypatch.setattr(
        "sys.argv",
        [
            "precompute-embeddings",
            "another-base-model",
            str(output_path),
            str(_write_dataset(tmp_path / "second.csv", [1402])),
        ],
    )
    with pytest.raises(ValueError, match="was written by base-model"):
        precompute_embeddings.main()

    assert embedder.calls == []
    assert embedder.loaded_tokenizers == []
    assert embedder.loaded_base_models == []
    assert sorted(_stored_embeddings(output_path)) == [b"1401"]


def test_adding_to_a_store_that_names_no_model_is_refused(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    embedder: _RecordingEmbedder,
) -> None:
    """A store from before the record existed cannot be shown to hold this
    run's activations, and a resume that assumed it did would produce exactly
    the mixture the record exists to prevent."""
    output_path = tmp_path / "unstamped.lmdb"
    with lmdb.open(str(output_path), map_size=2**20) as env:
        with env.begin(write=True) as transaction:
            transaction.put(b"1501", tensor_to_bytes(torch.rand(2, 4)))

    with pytest.raises(ValueError, match="does not record which model"):
        _run(
            monkeypatch,
            output_path,
            [_write_dataset(tmp_path / "unstamped.csv", [1502])],
        )

    assert embedder.calls == []
    assert embedder.loaded_tokenizers == []
    assert embedder.loaded_base_models == []


@pytest.mark.usefixtures("embedder")
def test_a_resume_by_the_model_that_wrote_the_store_carries_on(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The check must not cost the run it is meant to protect: the same model
    over the same store is the resume path this command is built around."""
    output_path = tmp_path / "embeddings.lmdb"
    _run(monkeypatch, output_path, [_write_dataset(tmp_path / "a.csv", [161])])

    stored = _run(
        monkeypatch,
        output_path,
        [_write_dataset(tmp_path / "b.csv", [162])],
    )

    _assert_holds_embeddings_for(stored, [161, 162])


def _sub_database_rows(
    path: pathlib.Path, name: str
) -> dict[bytes, np.ndarray] | None:
    """The rows one named sub-database holds, or None if the env has none."""
    decode = (
        bytes_to_tensor if name == "aggregated" else bytes_to_windowed_tensor
    )
    env = lmdb.open(str(path), readonly=True, lock=False, max_dbs=64)
    try:
        try:
            db = env.open_db(name.encode(), create=False)
        except lmdb.NotFoundError:
            return None
        with env.begin(db=db) as txn:
            return {
                key: decode(value).float().numpy()
                for key, value in txn.cursor().iternext()
            }
    finally:
        env.close()


def _bf16_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.to(torch.bfloat16).float().numpy()


def test_several_boundaries_are_written_into_one_env_in_one_walk(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every requested boundary is a sub-database of the one env, named by
    its unfrozen count, and each holds what `BertModel`'s own
    `hidden_states` hold there; one forward per window feeds them all and
    the aggregated rows besides."""
    config, model, tokenizer = _tiny_model_and_tokenizer()
    _patch_tiny_base_model(monkeypatch, config, model, tokenizer)
    texts = {1: "a", 2: "b"}
    dataset = tmp_path / "tiny.csv"
    _write_tiny_dataset(dataset, texts)

    expected: dict[str, dict[bytes, np.ndarray]] = {"aggregated": {}}
    for unfrozen in (1, 3):
        expected[f"unfrozen_{unfrozen}"] = {}
    for pubmed_id, text in texts.items():
        key = str(pubmed_id).encode()
        expected["aggregated"][key] = _bf16_numpy(
            utils.embed_document(
                text,
                tokenizer=tokenizer,
                model=model,
                stride=precompute_embeddings.STRIDE,
                batch_size=50,
                max_len=_TINY_MAX_LENGTH,
            )
        )
        for unfrozen in (1, 3):
            expected[f"unfrozen_{unfrozen}"][key] = _bf16_numpy(
                _reference_layer_prefix(
                    text,
                    tokenizer=tokenizer,
                    model=model,
                    frozen_layers=_TINY_LAYERS - unfrozen,
                    stride=precompute_embeddings.STRIDE,
                    max_len=_TINY_MAX_LENGTH,
                )
            )

    frozen_layer_0 = model.get_submodule("encoder.layer")[0]
    frozen_calls: list[None] = []
    real_layer_0_forward = frozen_layer_0.forward

    def counting_layer_0_forward(*args: object, **kwargs: object) -> object:
        frozen_calls.append(None)
        return real_layer_0_forward(*args, **kwargs)

    monkeypatch.setattr(frozen_layer_0, "forward", counting_layer_0_forward)

    output_path = tmp_path / "embeddings.lmdb"
    _run(monkeypatch, output_path, [dataset], "--unfrozen_top_layers", "1", "3")

    for name, rows in expected.items():
        stored = _sub_database_rows(output_path, name)
        assert stored is not None, f"no {name} sub-database"
        assert stored.keys() == rows.keys()
        for key, row in rows.items():
            np.testing.assert_array_equal(stored[key], row)
    assert len(frozen_calls) == len(texts)


def test_the_aggregated_sub_database_is_optional(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, model, tokenizer = _tiny_model_and_tokenizer()
    _patch_tiny_base_model(monkeypatch, config, model, tokenizer)
    dataset = tmp_path / "tiny.csv"
    _write_tiny_dataset(dataset, {1: "a"})
    output_path = tmp_path / "embeddings.lmdb"

    _run(
        monkeypatch,
        output_path,
        [dataset],
        "--unfrozen_top_layers",
        "2",
        "--no_aggregated",
    )

    assert _sub_database_rows(output_path, "aggregated") is None
    assert set(_sub_database_rows(output_path, "unfrozen_2") or {}) == {b"1"}


def test_a_later_run_adds_a_boundary_to_the_same_env(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new depth is one more sub-database under the same stamp, filled by
    a fresh forward, and leaves what the env already held alone."""
    config, model, tokenizer = _tiny_model_and_tokenizer()
    _patch_tiny_base_model(monkeypatch, config, model, tokenizer)
    dataset = tmp_path / "tiny.csv"
    _write_tiny_dataset(dataset, {1: "a", 2: "b"})
    output_path = tmp_path / "embeddings.lmdb"

    _run(monkeypatch, output_path, [dataset], "--unfrozen_top_layers", "1")
    before = {
        name: _sub_database_rows(output_path, name)
        for name in ("aggregated", "unfrozen_1")
    }
    _run(monkeypatch, output_path, [dataset], "--unfrozen_top_layers", "2")

    for name, rows in before.items():
        after = _sub_database_rows(output_path, name)
        assert rows is not None and after is not None
        assert after.keys() == rows.keys()
        for key, row in rows.items():
            np.testing.assert_array_equal(after[key], row)
    added = _sub_database_rows(output_path, "unfrozen_2")
    assert added is not None and set(added) == {b"1", b"2"}


def test_adding_to_an_env_in_the_old_layout_is_refused(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The old layout keeps its rows in the main database under a format-1
    stamp; resuming it would put new rows beside rows no reader opens."""
    from d3text.embeddings_store import ProvenanceError

    config, model, tokenizer = _tiny_model_and_tokenizer()
    _patch_tiny_base_model(monkeypatch, config, model, tokenizer)
    dataset = tmp_path / "tiny.csv"
    _write_tiny_dataset(dataset, {1: "a"})
    output_path = tmp_path / "embeddings.lmdb"
    record = {"format": 1, "base_model": "base-model"}
    record |= {
        "max_length": _TINY_MAX_LENGTH,
        "stride": precompute_embeddings.STRIDE,
    }
    with lmdb.open(str(output_path), map_size=2**20) as env:
        with env.begin(write=True) as txn:
            txn.put(b"\x00provenance", json.dumps(record).encode())
            txn.put(b"7", tensor_to_bytes(torch.zeros(1, _TINY_HIDDEN)))

    with pytest.raises(ProvenanceError, match="[Rr]ebuild"):
        _run(monkeypatch, output_path, [dataset])


STRIDE = precompute_embeddings.STRIDE


def test_no_aggregated_without_a_boundary_is_refused_before_loading(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    embedder: _RecordingEmbedder,
) -> None:
    """With nothing left to write, a run would load the weights, walk the
    corpus and report success over an env it never filled."""
    dataset = _write_dataset(tmp_path / "data.csv", [1])

    with pytest.raises(ValueError, match="--no_aggregated"):
        _run(monkeypatch, tmp_path / "embeddings", [dataset], "--no_aggregated")

    assert embedder.loaded_base_models == []
