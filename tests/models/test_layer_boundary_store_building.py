"""A training run fills a configured layer-boundary store it finds missing.

The store is created on first use, a document whose prefix the trunk
computes goes in until a write is refused, and a later pass reads it back
instead of running the frozen layers again.
"""

import logging
import os
import shutil
import types

import pytest
import torch
from d3text import embeddings_store as store_module
from d3text.models import base
from d3text.models.config import MachineConfig, ModelConfig
from d3text.models.ner import NERClassificationModel
from d3text.schema import EntityType, Schema

SCHEMA = Schema(entity_types=(EntityType(name="enzymes", prefix="enz"),))
BASE_MODEL = "prajjwal1/bert-mini"
WINDOW_TOKENS = 8


def _item(doc_id: int, n_windows: int) -> dict:
    generator = torch.Generator().manual_seed(doc_id)
    return {
        "id": torch.tensor(doc_id),
        "doc_id": torch.zeros(n_windows, dtype=torch.uint8),
        "sequence": {
            "input_ids": torch.randint(
                1, 1000, (1, n_windows, WINDOW_TOKENS), generator=generator
            ),
            "attention_mask": torch.ones(
                1, n_windows, WINDOW_TOKENS, dtype=torch.long
            ),
        },
    }


@pytest.fixture
def store_path(monkeypatch, tmp_path):
    path = tmp_path / "layer-boundary"
    monkeypatch.setitem(base.mconfig.embeddings_store, BASE_MODEL, str(path))
    base.layer_boundary_store.cache_clear()
    yield path
    base.layer_boundary_store.cache_clear()


def test_the_building_pass_matches_the_pass_that_reads_it_back(
    patch_base_model, store_path, monkeypatch
):
    """The pass that builds the store sees each prefix rounded through the
    store's bf16, so its embeddings are the ones every later pass replays
    from the store. Items of different window counts pin that each gets its
    own windows, not a neighbour's."""
    # CPU autocast is bf16 already, where the rounding is a no-op; fp32 is
    # the precision it has to reconcile with the store.
    monkeypatch.setattr(base, "select_amp_dtype", lambda _device: torch.float32)
    model = NERClassificationModel(
        schema=SCHEMA,
        config=ModelConfig(
            model_class="NERClassificationModel",
            base_model=BASE_MODEL,
            hidden_layers=[8],
            unfrozen_top_layers=1,
        ),
        device="cpu",
    )
    model.eval()
    batch = [_item(111, 2), _item(222, 3)]

    with torch.no_grad():
        built, _ = model.get_token_embeddings(batch)
        store = model._layer_boundary_store()
        assert store is not None and store.writable
        assert store.written == 2
        replayed, _ = model.get_token_embeddings(batch)

    assert store.hits == 2
    assert store_path.exists()
    assert torch.equal(built, replayed)
    store.close()


def _model(unfrozen_top_layers: int = 1) -> NERClassificationModel:
    model = NERClassificationModel(
        schema=SCHEMA,
        config=ModelConfig(
            model_class="NERClassificationModel",
            base_model=BASE_MODEL,
            hidden_layers=[8],
            unfrozen_top_layers=unfrozen_top_layers,
        ),
        device="cpu",
    )
    return model.eval()


def _build_store_with(document: int) -> None:
    """A run stores `document`, then ends: the next run finds an existing
    store holding only that one."""
    model = _model()
    with torch.no_grad():
        model.get_token_embeddings([_item(document, 2)])
    store = model._layer_boundary_store()
    assert store is not None and store.written == 1
    store.close()
    base.layer_boundary_store.cache_clear()


def test_a_run_tops_up_an_existing_store_with_the_documents_it_lacks(
    patch_base_model, store_path
):
    """A document the store lacks is computed once and kept, not recomputed
    every epoch: the second pass over the same batch is all hits. Opened
    read-only, the store never grew and 222 missed on every pass."""
    _build_store_with(111)
    model = _model()
    batch = [_item(111, 2), _item(222, 3)]

    with torch.no_grad():
        model.get_token_embeddings(batch)
        store = model._layer_boundary_store()
        assert store is not None and store.writable
        assert store.written == 1
        model.get_token_embeddings(batch)

    assert (store.hits, store.misses, store.written) == (3, 1, 1)
    store.close()


def test_a_boundary_added_after_a_read_of_another_is_stored(
    patch_base_model, store_path
):
    """One process, one handle on the env: a later boundary has to be added
    through the handle an earlier existing boundary was opened with."""
    _build_store_with(111)

    existing = base.layer_boundary_store(BASE_MODEL, 1)
    added = base.layer_boundary_store(BASE_MODEL, 2)

    try:
        assert existing is not None
        assert added is not None and added.writable
    finally:
        for store in (existing, added):
            if store is not None:
                store.close()


def test_a_store_on_an_unwritable_path_is_read_and_warned_about(
    patch_base_model, store_path, caplog
):
    """The writable open is attempted, and when the path refuses it the run
    keeps reading what the store holds, with one warning that it will not
    grow."""
    _build_store_with(111)
    files = [store_path / name for name in ("data.mdb", "lock.mdb")]
    for path in (*files, store_path):
        path.chmod(0o500 if path.is_dir() else 0o400)
    try:
        if os.access(files[0], os.W_OK):
            pytest.skip("cannot make a file unwritable for this user")
        with caplog.at_level(logging.WARNING, logger="d3text.models.base"):
            store = base.layer_boundary_store(BASE_MODEL, 1)

        assert store is not None and not store.writable
        assert store.get(111, expected_windows=2) is not None
        assert "will not grow" in caplog.text
        # The refused writable open left no writable handle cached.
        env = store_module._envs[os.path.realpath(store_path)]
        assert env.flags()["readonly"]
        store.close()
    finally:
        store_path.chmod(0o700)
        for path in files:
            path.chmod(0o600)


def test_a_store_listed_as_frozen_is_read_and_never_written(
    patch_base_model, store_path
):
    """The machine decides, not the run: a store on a writable path that
    this machine lists as frozen is opened read-only, and a document it
    lacks is computed without being put into it."""
    _build_store_with(111)
    monkey = pytest.MonkeyPatch()
    monkey.setattr(
        base,
        "mconfig",
        MachineConfig(
            embeddings_store={BASE_MODEL: str(store_path)},
            frozen_embeddings_stores=[BASE_MODEL],
        ),
    )
    try:
        model = _model()
        with torch.no_grad():
            model.get_token_embeddings([_item(111, 2), _item(222, 3)])
        store = model._layer_boundary_store()

        assert store is not None
        assert not store.writable
        assert (store.hits, store.misses, store.written) == (1, 1, 0)
        store.close()
    finally:
        monkey.undo()


def test_a_floor_on_free_space_stops_a_building_run_writing(
    patch_base_model, store_path, monkeypatch
):
    """Below the machine's free-space floor the run stops adding documents
    and keeps going: nothing is written, the store stops being writable,
    and the batch still comes back."""
    monkeypatch.setattr(
        base,
        "mconfig",
        MachineConfig(
            embeddings_store={BASE_MODEL: str(store_path)},
            embeddings_store_min_free_gib=2.0,
        ),
    )
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _path: types.SimpleNamespace(free=1024**3),
    )
    model = _model()

    with torch.no_grad():
        embeddings, _ = model.get_token_embeddings([_item(111, 2)])
    store = model._layer_boundary_store()

    assert store is not None
    assert (store.written, store.writable) == (0, False)
    assert embeddings.shape[0] == 1
    store.close()


def test_a_frozen_store_is_not_created_where_none_exists(
    patch_base_model, store_path, monkeypatch
):
    """Frozen means the run does not write this model's store, so an absent
    store is not made either: the run computes live, as with none
    configured."""
    monkeypatch.setattr(
        base,
        "mconfig",
        MachineConfig(
            embeddings_store={BASE_MODEL: str(store_path)},
            frozen_embeddings_stores=[BASE_MODEL],
        ),
    )

    base.embeddings_store.cache_clear()
    try:
        assert base.layer_boundary_store(BASE_MODEL, 1) is None
        assert base.embeddings_store(BASE_MODEL) is None
    finally:
        base.embeddings_store.cache_clear()
    assert not store_path.exists()


def test_a_floor_on_free_space_reaches_the_aggregated_store(
    store_path, monkeypatch
):
    """The aggregated factory hands the store the machine's floor, as the
    layer-boundary one does: without it the store writes until the disk is
    full."""
    monkeypatch.setattr(
        base,
        "mconfig",
        MachineConfig(
            embeddings_store={BASE_MODEL: str(store_path)},
            embeddings_store_min_free_gib=2.0,
        ),
    )
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _path: types.SimpleNamespace(free=1024**3),
    )

    base.embeddings_store.cache_clear()
    try:
        store = base.embeddings_store(BASE_MODEL)
        assert isinstance(store, store_module.EmbeddingsStore)
        store.put(111, torch.zeros(2, 4))
        assert (store.written, store.writable) == (0, False)
        store.close()
    finally:
        base.embeddings_store.cache_clear()


def test_a_top_up_is_logged_where_the_run_writes_and_not_where_it_reads(
    patch_base_model, store_path, caplog
):
    """`embeddings_store` serves a frozen-trunk run, which puts nothing into
    a layer boundary, so opening one there must not log a top-up; the
    aggregated store it does write into must."""
    _build_store_with(111)
    base.embeddings_store.cache_clear()
    try:
        with caplog.at_level(logging.INFO, logger="d3text.models.base"):
            boundary = base.embeddings_store(BASE_MODEL)
        assert isinstance(boundary, store_module.LayerBoundaryStore)
        assert "topping up" not in caplog.text.lower()
        boundary.close()
        base.embeddings_store.cache_clear()
        base.layer_boundary_store.cache_clear()
        caplog.clear()

        store_module.EmbeddingsStore.create(
            store_path, base._store_provenance(BASE_MODEL)
        ).close()
        with caplog.at_level(logging.INFO, logger="d3text.models.base"):
            aggregated = base.embeddings_store(BASE_MODEL)
        assert isinstance(aggregated, store_module.EmbeddingsStore)
        assert aggregated.writable
        assert "topping up" in caplog.text.lower()
        aggregated.close()
    finally:
        base.embeddings_store.cache_clear()
