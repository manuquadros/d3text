"""Shared test fixtures.

The `stub` factory is what makes the model methods in models.py unit-testable
without constructing a full model (no base-model download, no GPU): it builds a
bare instance carrying only the handful of attributes the method under test
reads.

The `tiny_brenda` fixture builds a small on-disk HDF5 + matching DataFrame so
`BrendaDataset` can be exercised without the ~300 MB BRENDA files.
"""

import logging
import types

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from d3text import logs

from d3text.schema import EntityType, RelationType, Schema


# HDF5 groups present on disk: pubmed_id -> number of 512-token chunks.
_HDF5_CHUNKS = {"10": 2, "20": 5, "30": 1}

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# What a `data`-marked test needs on disk: the BRENDA splits (shipped with
# `brenda_references` as a separate archive) and the precomputed encodings.
# Neither is in the repo, so these tests can only run where the corpus is.
BRENDA_SPLITS = (
    REPO_ROOT
    / "brenda_references"
    / "src"
    / "brenda_references"
    / "data"
    / "train_data.csv"
)
ENCODINGS = REPO_ROOT / "data" / "biolinkbert-base-zstd-22-encodings.hdf5"


@functools.cache
def gpu_usable() -> bool:
    """Whether a GPU is present *and* can actually execute a kernel.

    ``torch.cuda.is_available()`` alone is not enough. A ROCm build reports a
    device for any AMD card the driver enumerates, including ones whose
    architecture the wheel ships no kernels for, and every op then dies with
    ``HIP error: invalid device function`` — which errors the gpu tests instead
    of skipping them. Only running something settles it.
    """
    if not torch.cuda.is_available():
        return False
    try:
        torch.zeros(1, device="cuda").add_(1)
    except Exception:
        return False
    return True


def corpus_available() -> bool:
    """Whether the BRENDA corpus and its encodings are on this machine."""
    return BRENDA_SPLITS.is_file() and ENCODINGS.is_file()


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests whose environment this machine cannot provide.

    ``gpu`` and ``data`` are opt-out-by-environment rather than excluded: on a
    box that has the device or the corpus they run, and everywhere else they
    skip with a reason instead of erroring. ``network`` is not handled here —
    it is excluded by the default marker expression (see pyproject) and opted
    into explicitly, because there is no way to test for a *working* Hugging
    Face Hub without hitting it.
    """
    skip_gpu = pytest.mark.skip(reason="no usable GPU device")
    skip_data = pytest.mark.skip(
        reason=f"BRENDA corpus not present (looked for {BRENDA_SPLITS} "
        f"and {ENCODINGS})"
    )
    have_gpu = gpu_usable()
    have_corpus = corpus_available()

    for item in items:
        # Markers, not `item.keywords`: keywords also carry the names of the
        # item's parents, so the unit tests under `tests/data/` would every one
        # of them match a "data" keyword and skip.
        markers = {marker.name for marker in item.iter_markers()}
        if "gpu" in markers and not have_gpu:
            item.add_marker(skip_gpu)
        if "data" in markers and not have_corpus:
            item.add_marker(skip_data)


@pytest.fixture(autouse=True)
def deterministic_rng():
    """Seed torch for every test.

    Nothing in the library seeds the global RNG any more (`runtime.configure()`
    does, and only the scripts call it), so the suite seeds it explicitly
    instead of inheriting the seed a module used to set on import.
    """
    torch.manual_seed(0)


@pytest.fixture
def brenda_encodings() -> pathlib.Path:
    """The precomputed encodings for the real corpus.

    Only meaningful to a `data`-marked test, which is skipped outright where
    the corpus is absent.
    """
    return ENCODINGS


@pytest.fixture(autouse=True)
def isolated_embeddings_cache(monkeypatch):
    """Switch the process-global CPU embeddings cache off for every test.

    `models.py` builds it at import time from `cpu_embeddings_cache_size` in
    the repo-root `config.toml` — machine-local, untracked, and absent in CI.
    Left alone it decides which branch of `get_token_embeddings` a test takes,
    so the same test passes on one machine and fails on another; and because
    the cache is a module global keyed by document id, whatever one test stores
    is visible to the next. Tests that want it opt in via `embeddings_cache`.
    """
    from d3text.models import models

    monkeypatch.setattr(models, "cpu_embeddings_cache", None)


@pytest.fixture
def embeddings_cache(monkeypatch):
    """Turn the CPU embeddings cache on, as a private instance."""
    from cacheout import Cache

    from d3text.models import models

    cache = Cache(maxsize=16)
    monkeypatch.setattr(models, "cpu_embeddings_cache", cache)
    return cache


@pytest.fixture(
    params=[
        "cpu",
        pytest.param("cuda", marks=pytest.mark.gpu),
    ]
)
def device(request):
    """Parametrize a test over CPU and, when available, CUDA.

    The ``cuda`` parameter carries the ``gpu`` marker, so that variant is
    collected as a ``gpu`` test and auto-skipped by
    ``pytest_collection_modifyitems`` when no CUDA device is present. On a GPU
    machine the default suite exercises both device placements.
    """
    return request.param


@pytest.fixture
def restore_package_logger():
    """Yield the `d3text` logger with its state restored afterwards.

    `logs.configure()` sets `propagate = False`, which is right for a command
    that owns its process and wrong for a pytest session: left in place it
    hides every later test's package records from anything watching the root
    logger, `caplog` included.
    """
    logger = logging.getLogger(logs.PACKAGE_LOGGER)
    handlers, level, propagate = (
        list(logger.handlers),
        logger.level,
        logger.propagate,
    )

    yield logger

    logger.handlers[:] = handlers
    logger.setLevel(level)
    logger.propagate = propagate


@pytest.fixture
def stub():
    """Return a factory that builds a bare `cls` instance with `attrs` set.

    Bypasses ``__init__`` (via ``cls.__new__``) and ``nn.Module.__setattr__``
    (via ``object.__setattr__``), so tensors, sub-modules, and plain values can
    be attached directly. Methods that aren't overridden resolve normally off
    the class, so a method can call its real collaborators (e.g.
    ``self._pool_logits``) while reading only the stubbed attributes.
    """

    def _make(cls, **attrs):
        obj = cls.__new__(cls)
        for key, value in attrs.items():
            object.__setattr__(obj, key, value)
        return obj

    return _make


@pytest.fixture
def patch_base_model(monkeypatch):
    """Make model construction offline: `load_base_model` returns a tiny random
    BERT instead of downloading one. Its hidden size matches
    ``embedding_dims["prajjwal1/bert-mini"]``, so configs naming that base model
    line up with the injected weights."""
    from transformers import BertConfig, BertModel

    def tiny_bert(*_args, **_kwargs):
        return BertModel(
            BertConfig(
                vocab_size=1000,
                hidden_size=256,
                num_hidden_layers=2,
                num_attention_heads=4,
                intermediate_size=512,
            )
        )

    monkeypatch.setattr("d3text.models.models.load_base_model", tiny_bert)


@pytest.fixture
def tiny_hdf5(tmp_path):
    """A small HDF5 encodings file: one group per pmid, with input_ids /
    attention_mask of shape [n_chunks, 8]. Uncompressed, so it reads without
    the Zstd filter."""
    path = tmp_path / "encodings.hdf5"
    with h5py.File(path, "w") as f:
        for pmid, n_chunks in _HDF5_CHUNKS.items():
            group = f.create_group(pmid)
            group.create_dataset(
                "input_ids", data=np.zeros((n_chunks, 8), dtype=np.int64)
            )
            group.create_dataset(
                "attention_mask", data=np.ones((n_chunks, 8), dtype=np.int64)
            )
    return path


@pytest.fixture
def tiny_dataframe():
    """Matching DataFrame. Row 3 (pmid 40) is deliberately absent from the
    HDF5 file; the `fulltext` column proves BrendaDataset keeps only the four
    columns it needs."""
    return pd.DataFrame(
        {
            "pubmed_id": [10, 20, 30, 40],
            "relations": pd.Series([[], [], [], []]),
            "entities": [np.array([1, 0, 1], dtype=np.uint8)] * 4,
            "classes": [np.array([1, 0], dtype=np.float32)] * 4,
            "fulltext": ["x"] * 4,
        }
    )


@pytest.fixture
def tiny_brenda(tiny_hdf5, tiny_dataframe):
    """Two BrendaDataset views over the tiny fixtures.

    ``present`` holds only the three rows backed by HDF5 (chunk counts
    ``[2, 5, 1]``); ``full`` also holds the row whose pmid is missing from the
    file.
    """
    from d3text.data.data import BrendaDataset

    return types.SimpleNamespace(
        present=BrendaDataset(
            tiny_dataframe.iloc[:3].copy(), encodings=tiny_hdf5
        ),
        full=BrendaDataset(tiny_dataframe.copy(), encodings=tiny_hdf5),
        chunks=[2, 5, 1],
        missing_index=3,
    )
