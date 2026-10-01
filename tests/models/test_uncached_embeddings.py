"""`Model.uncached_embeddings`: a live forward that no cache answers or keeps.

Every embeddings cache keys a document by its id alone, so a forward over
text no store was built from has to bypass all of them: an entry under the
same id would otherwise be served in its place, and its activations written
back under that id.
"""

import logging
import os
import pathlib

import lmdb
import pytest
import torch
from d3text.embeddings_store import (
    AGGREGATED,
    EmbeddingsStore,
    LayerBoundaryStore,
    sub_databases,
)
from d3text.models.base import ByteBudgetCache, cpu_cache_key
from d3text.models.base import (
    _store_provenance,
    document_token_count,
    embeddings_store,
    layer_boundary_store,
)
from d3text.models.config import MachineConfig, ModelConfig
from d3text.models.ner import NERClassificationModel
from d3text.schema import EntityType, Schema
from transformers import BertConfig, BertModel

SCHEMA = Schema(entity_types=(EntityType(name="enzymes", prefix="enz"),))
HIDDEN_SIZE = 32
WINDOW_TOKENS = 16
DOCUMENT = 7


def _tiny_bert(*_args: object, **_kwargs: object) -> BertModel:
    return BertModel(
        BertConfig(
            vocab_size=1000,
            hidden_size=HIDDEN_SIZE,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=64,
        )
    )


class _RecordingStore:
    """Answers for `DOCUMENT` with `planted`; records every read and write."""

    writable = True

    def __init__(self, planted: torch.Tensor, unfrozen_top_layers: int):
        self.planted = planted
        self.unfrozen_top_layers = unfrozen_top_layers
        self.reads: list[int] = []
        self.puts: list[int] = []

    def get(self, pubmed_id: int, **_expected: int) -> torch.Tensor | None:
        self.reads.append(pubmed_id)
        return self.planted.clone() if pubmed_id == DOCUMENT else None

    def put(self, pubmed_id: int, _value: torch.Tensor) -> None:
        self.puts.append(pubmed_id)


class _RecordingBoundaryStore(_RecordingStore, LayerBoundaryStore):
    """The same, typed as the layer-boundary store its callers declare;
    `_RecordingStore.__init__` skips `LayerBoundaryStore`'s LMDB open."""


def _item() -> dict:
    return {
        "id": torch.tensor(DOCUMENT),
        "doc_id": torch.zeros(2, dtype=torch.uint8),
        "sequence": {
            "input_ids": torch.randint(5, 999, (2, WINDOW_TOKENS)),
            "attention_mask": torch.ones(2, WINDOW_TOKENS, dtype=torch.long),
        },
    }


@pytest.mark.parametrize("unfrozen_top_layers", [0, 2])
def test_an_uncached_forward_is_live_and_leaves_every_cache_alone(
    monkeypatch, unfrozen_top_layers
) -> None:
    """Planted under the document's id, in the shape each path serves, sit a
    CPU-cache entry and a store entry for other tokens; under the switch the
    forward is the one no cache could answer, and neither gains or changes
    an entry."""
    monkeypatch.setattr("d3text.models.base.load_base_model", _tiny_bert)
    monkeypatch.setattr("d3text.models.base.cpu_embeddings_cache", None)
    monkeypatch.setattr("d3text.models.base.embeddings_store", lambda _: None)
    monkeypatch.setattr(
        "d3text.models.base.layer_boundary_store", lambda *_a, **_k: None
    )
    model = NERClassificationModel(
        schema=SCHEMA,
        config=ModelConfig(
            model_class="NERClassificationModel",
            base_model="tiny",
            hidden_layers=[8],
            unfrozen_top_layers=unfrozen_top_layers,
        ),
        device="cpu",
    )
    model.eval()
    item = _item()
    with torch.no_grad():
        live, _ = model.get_token_embeddings([item])

    if unfrozen_top_layers:
        key = cpu_cache_key(
            "tiny", DOCUMENT, unfrozen_top_layers=unfrozen_top_layers
        )
        planted = torch.full((2, WINDOW_TOKENS, HIDDEN_SIZE), 50.0)
        store = _RecordingBoundaryStore(planted, unfrozen_top_layers)
        monkeypatch.setattr(
            "d3text.models.base.layer_boundary_store",
            lambda *_a, **_k: store,
        )
    else:
        key = cpu_cache_key("tiny", DOCUMENT)
        planted = torch.full((document_token_count(item), HIDDEN_SIZE), 50.0)
        store = _RecordingStore(planted, 0)
        monkeypatch.setattr(
            "d3text.models.base.embeddings_store", lambda _: store
        )
    cache = ByteBudgetCache(max_bytes=10**8)
    cache.set(key, planted.clone())
    used = cache.used_bytes
    monkeypatch.setattr("d3text.models.base.cpu_embeddings_cache", cache)

    with torch.no_grad(), model.uncached_embeddings():
        uncached, _ = model.get_token_embeddings([item])

    assert torch.equal(uncached, live)
    assert store.reads == []
    assert store.puts == []
    assert cache.size() == 1
    assert cache.used_bytes == used
    assert torch.equal(cache.get(key), planted)


def test_building_a_model_adds_no_sub_database_and_training_still_does(
    monkeypatch, tmp_path
) -> None:
    """An env holding no sub-database yet is what a frozen trunk's first
    lookup creates one in. Building the model only asks whether a store is
    there, so the env is left as it was; a forward outside
    `uncached_embeddings` then creates the aggregated store and fills it."""
    path = tmp_path / "embeddings"
    lmdb.open(str(path)).close()
    monkeypatch.setattr("d3text.models.base.load_base_model", _tiny_bert)
    monkeypatch.setattr("d3text.models.base.cpu_embeddings_cache", None)
    monkeypatch.setattr(
        "d3text.models.base.mconfig",
        MachineConfig(embeddings_store={"tiny": str(path)}),
    )
    embeddings_store.cache_clear()

    try:
        model = NERClassificationModel(
            schema=SCHEMA,
            config=ModelConfig(
                model_class="NERClassificationModel",
                base_model="tiny",
                hidden_layers=[8],
            ),
            device="cpu",
        )
        model.eval()
        assert sub_databases(path) == frozenset()

        with torch.no_grad():
            model.get_token_embeddings([_item()])

        store = embeddings_store("tiny")
        assert isinstance(store, EmbeddingsStore)
        assert store.get(DOCUMENT, document_token_count(_item())) is not None
        store.close()
        assert AGGREGATED in sub_databases(path)
    finally:
        embeddings_store.cache_clear()


def _files(path: pathlib.Path) -> dict[str, tuple[bytes, int]]:
    return {
        file.name: (file.read_bytes(), file.stat().st_mtime_ns)
        for file in sorted(path.iterdir())
    }


def _frozen_tiny_model() -> NERClassificationModel:
    return NERClassificationModel(
        schema=SCHEMA,
        config=ModelConfig(
            model_class="NERClassificationModel",
            base_model="tiny",
            hidden_layers=[8],
        ),
        device="cpu",
    )


@pytest.mark.parametrize("kind", ["aggregated", "boundary"])
def test_building_a_model_over_a_store_leaves_every_file_of_it_alone(
    monkeypatch, tmp_path, caplog, kind
) -> None:
    """A writable LMDB open rewrites the env's lock file even when nothing is
    put, and `infer` builds a frozen-trunk model without ever reading its
    store. So the build checks an existing store read-only: every file, the
    lock file included, keeps its bytes and its modification time, and no
    store-less warning fires. A training lookup still opens an aggregated
    store writable and tops it up."""
    path = tmp_path / "embeddings"
    provenance = _store_provenance("tiny")
    if kind == "aggregated":
        EmbeddingsStore.create(path, provenance).close()
    else:
        LayerBoundaryStore.create(path, provenance, 2).close()
    # Back-dated, so a write within the filesystem's timestamp tick shows.
    for file in path.iterdir():
        os.utime(file, ns=(0, 0))
    before = _files(path)
    monkeypatch.setattr("d3text.models.base.load_base_model", _tiny_bert)
    monkeypatch.setattr("d3text.models.base.cpu_embeddings_cache", None)
    monkeypatch.setattr(
        "d3text.models.base.mconfig",
        MachineConfig(embeddings_store={"tiny": str(path)}),
    )
    embeddings_store.cache_clear()
    layer_boundary_store.cache_clear()

    try:
        with caplog.at_level(logging.WARNING, logger="d3text.models.base"):
            model = _frozen_tiny_model()
        model.eval()
        assert _files(path) == before
        assert "no usable embeddings store" not in caplog.text

        if kind == "aggregated":
            with torch.no_grad():
                model.get_token_embeddings([_item()])
            store = embeddings_store("tiny")
            assert isinstance(store, EmbeddingsStore)
            assert store.writable
            assert (
                store.get(DOCUMENT, document_token_count(_item())) is not None
            )
    finally:
        embeddings_store.cache_clear()
        layer_boundary_store.cache_clear()


def test_building_a_model_warns_when_frozen_embeddings_stores_bars_creating(
    monkeypatch, tmp_path, caplog
) -> None:
    """An env holding no sub-database counts as a store only because the
    first lookup would create one there; `frozen_embeddings_stores` bars
    that, so the trunk runs storeless and the build says so."""
    path = tmp_path / "embeddings"
    lmdb.open(str(path)).close()
    monkeypatch.setattr("d3text.models.base.load_base_model", _tiny_bert)
    monkeypatch.setattr("d3text.models.base.cpu_embeddings_cache", None)
    monkeypatch.setattr(
        "d3text.models.base.mconfig",
        MachineConfig(
            embeddings_store={"tiny": str(path)},
            frozen_embeddings_stores=["tiny"],
        ),
    )
    embeddings_store.cache_clear()

    try:
        with caplog.at_level(logging.WARNING, logger="d3text.models.base"):
            _frozen_tiny_model()
        assert "no usable embeddings store" in caplog.text
        assert sub_databases(path) == frozenset()
    finally:
        embeddings_store.cache_clear()
