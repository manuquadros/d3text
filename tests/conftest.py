"""Shared test fixtures.

The `stub` factory makes the model methods unit-testable without constructing a
full model, and `tiny_brenda` builds a small encodings store and matching frame
so `BrendaDataset` can be exercised without the ~300 MB BRENDA files.
"""

import logging
import pathlib
import types

import h5py
import numpy as np
import pandas as pd
import pytest
import tomlkit
import torch
from d3text import logs
from d3text.models import config as model_config
from hypothesis import settings

# No `@given` property here measures timing, so the default deadline is a
# flake on a slow machine. Loaded at import, before any test module's
# `@settings` evaluates, so every one leaving `deadline` unset inherits it.
settings.register_profile("d3text", deadline=None)
settings.load_profile("d3text")


# Documents in the encodings store: pubmed_id -> number of 512-token chunks.
_ENCODING_CHUNKS = {"10": 2, "20": 5, "30": 1}

_CHECKOUT = pathlib.Path(__file__).resolve().parent.parent


def pytest_configure(config):
    """Refuse to test a d3text other than the one in this checkout.

    Both local packages install editable through `.pth` files naming the
    checkout the venv was synced from, and `site` appends those after
    `PYTHONPATH`. Run from a git worktree without a `PYTHONPATH` prefix, the
    suite imports the other checkout's code and passes having tested none of
    the edits in front of it.
    """
    import brenda_references
    import d3text

    wrong = [
        f"{module.__name__} from {module.__file__}"
        for module, source in (
            (d3text, _CHECKOUT / "src"),
            (brenda_references, _CHECKOUT / "brenda_references" / "src"),
        )
        if module.__file__ is None
        or not pathlib.Path(module.__file__).resolve().is_relative_to(source)
    ]
    if wrong:
        raise pytest.UsageError(
            f"testing {_CHECKOUT}, but imported {'; '.join(wrong)}. From a "
            "git worktree, run with "
            f'PYTHONPATH="{_CHECKOUT}/src:{_CHECKOUT}/brenda_references/src".'
        )


def pytest_collection_modifyitems(config, items):
    """Auto-skip `gpu`-marked tests when no CUDA device is available.

    On a GPU box they run; on CPU they skip rather than error, so the default
    suite stays green without excluding them.
    """
    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="no CUDA device available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


@pytest.fixture(autouse=True)
def no_machine_config(monkeypatch):
    """Read no repo-root `config.toml` unless a test writes one.

    The store tables in it are this machine's, so a test reading them would
    pass or fail by what the machine running it has on disk.
    `d3text.models.base.mconfig` is read at import and is not covered.
    """
    monkeypatch.setattr(
        model_config,
        "MACHINE_CONFIG_PATH",
        pathlib.Path(__file__).parent / "no-machine-config" / "config.toml",
    )


@pytest.fixture
def machine_stores(monkeypatch, tmp_path):
    """Point `machine_config()` at a `config.toml` naming these stores.

    Call it with one keyword per `MachineConfig` table, each a base-model to
    path mapping; later calls add to the tables earlier ones wrote.
    """
    tables: dict[str, dict[str, str]] = {}
    path = tmp_path / "machine" / "config.toml"

    def configure(**entries: dict[str, str]) -> None:
        for table, by_model in entries.items():
            tables.setdefault(table, {}).update(
                {model: str(store) for model, store in by_model.items()}
            )
        path.parent.mkdir(exist_ok=True)
        path.write_text(tomlkit.dumps(tables))
        monkeypatch.setattr(model_config, "MACHINE_CONFIG_PATH", path)

    return configure


@pytest.fixture(autouse=True)
def deterministic_rng():
    """Seed torch for every test.

    Nothing in the library seeds the global RNG any more —
    `runtime.configure()` does, and only the scripts call it.
    """
    torch.manual_seed(0)


@pytest.fixture(autouse=True)
def restore_backward_lowering_flag():
    """Reset `torch._functorch.config`'s backward-lowering flag per test.

    `runtime.compile_model` sets `force_non_lazy_backward_lowering`
    process-globally and never unsets it, so without this the first test
    to compile would leave it set for the rest, making outcomes depend on
    run order.
    """
    import torch._functorch.config as functorch_config

    original = functorch_config.force_non_lazy_backward_lowering
    yield
    functorch_config.force_non_lazy_backward_lowering = original


@pytest.fixture(autouse=True)
def restore_fallback_random_flag():
    """Reset `torch._inductor.config.fallback_random` per test.

    Same story as `restore_backward_lowering_flag`: `compile_model` sets it
    process-globally and deliberately never unsets it, so leaving it set
    would make the suite's outcome depend on run order.
    """
    import torch._inductor.config as inductor_config

    original = inductor_config.fallback_random
    yield
    inductor_config.fallback_random = original


@pytest.fixture(autouse=True)
def clear_cpu_embeddings_cache():
    """Reset `d3text.models.base`'s process-wide embeddings cache per test.

    It is module state keyed by base model and document id, and fixtures across
    the suite reuse small integer pmids for unrelated documents. A no-op on a
    machine whose config leaves the cache off.
    """
    from d3text.models import base

    if base.cpu_embeddings_cache is not None:
        base.cpu_embeddings_cache.clear()
    yield
    if base.cpu_embeddings_cache is not None:
        base.cpu_embeddings_cache.clear()


@pytest.fixture(
    params=[
        "cpu",
        pytest.param("cuda", marks=pytest.mark.gpu),
    ]
)
def device(request):
    """Parametrize a test over CPU and, when available, CUDA.

    The `cuda` parameter carries the `gpu` marker, so that variant is
    auto-skipped when no CUDA device is present.
    """
    return request.param


@pytest.fixture
def restore_package_logger():
    """Yield the `d3text` logger, restoring every routed logger afterwards.

    `logs.configure()` sets `propagate = False` on every logger in
    `logs.ROUTED_LOGGERS`; left in place, that hides later tests' records
    from `caplog`, so all of them are restored, not just `d3text`.
    """
    loggers = [logging.getLogger(name) for name in logs.ROUTED_LOGGERS]
    saved = [
        (logger, list(logger.handlers), logger.level, logger.propagate)
        for logger in loggers
    ]

    yield loggers[0]

    for logger, handlers, level, propagate in saved:
        logger.handlers[:] = handlers
        logger.setLevel(level)
        logger.propagate = propagate


@pytest.fixture
def refuses_the_backward_graph(monkeypatch):
    """Compile with a backend accepting the forward, refusing the backward.

    Aims the failure at the half AOTAutograd lowers lazily, without a GPU or
    a C++ toolchain. The eager-lowering knob starts at torch's default, so a
    test sees only what `compile_model` itself sets.
    """
    from torch._dynamo.backends.common import aot_autograd

    def refuse(graph, example_inputs):
        raise RuntimeError("could not lower the backward graph")

    backend = aot_autograd(
        fw_compiler=lambda graph, example_inputs: graph, bw_compiler=refuse
    )
    compile_ = torch.compile

    def compile_with_a_failing_backward(*args, **kwargs):
        return compile_(*args, **{**kwargs, "backend": backend})

    monkeypatch.setattr(torch, "compile", compile_with_a_failing_backward)
    monkeypatch.setattr("d3text.runtime.is_triton_compatible", lambda: True)
    monkeypatch.setenv("D3TEXT_COMPILE", "1")
    monkeypatch.setattr(
        "torch._functorch.config.force_non_lazy_backward_lowering", False
    )
    # `_call_impl` is one code object shared by every module, so a compilation
    # another test left cached is a compilation this one would hit.
    torch._dynamo.reset()


@pytest.fixture
def stub():
    """A factory building a bare `cls` instance with `attrs` set.

    Bypasses `__init__` and `nn.Module.__setattr__`, so tensors, sub-modules
    and plain values attach directly while un-overridden methods still resolve
    off the class and can call their real collaborators.
    """

    def _make(cls, **attrs):
        obj = cls.__new__(cls)
        for key, value in attrs.items():
            object.__setattr__(obj, key, value)
        return obj

    return _make


def _watch_device_moves(tensor, on_move):
    real_to = tensor.to

    def to(*args, **kwargs):
        result = real_to(*args, **kwargs)
        if "device" in kwargs or any(
            isinstance(arg, (str, torch.device)) for arg in args
        ):
            on_move()
        elif result is not tensor:
            _watch_device_moves(result, on_move)
        return result

    tensor.to = to
    return tensor


@pytest.fixture
def watch_device_moves():
    """A function making `tensor.to` call `on_move` whenever it names a device.

    On a CPU the model's device is the host, so where a tensor sits proves
    nothing and an ordering test has to watch the move itself. A cast naming
    no device passes the watch on to the copy it returns, so a test does not
    depend on where the code under test places its casts.
    """
    return _watch_device_moves


@pytest.fixture
def patch_base_model(monkeypatch):
    """Make model construction offline: `load_base_model` returns a tiny random
    BERT instead of downloading one, hidden size 256 — models read this size
    from the returned base model's own config, so any config naming any base
    model lines up with the injected weights."""
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

    monkeypatch.setattr("d3text.models.base.load_base_model", tiny_bert)


@pytest.fixture
def empty_token_label_store(tmp_path, machine_stores):
    """A label store stamped with the label space but holding no documents.

    Config validation refuses an `ETEBrendaModel` without
    `token_supervision`, so a test that only needs the model's shape still
    needs a store. Registered as `prajjwal1/bert-mini`'s entry, the base
    model every user of this fixture configures.
    """
    from d3text import token_labels, utils

    path = tmp_path / "empty_labels.hdf5"
    with h5py.File(path, "w") as store:
        token_labels.write_label_space(
            store,
            token_labels.BRENDA_LABELS,
            stamp=token_labels.IndexStamp(digest="empty-store"),
            tokenizer=token_labels.TokenizerStamp(
                base_model="prajjwal1/bert-mini",
                digest="test-tokenizer",
                window_length=utils.WINDOW_LENGTH,
                window_stride=utils.WINDOW_STRIDE,
            ),
        )
    machine_stores(token_labels_store={"prajjwal1/bert-mini": path})
    return path


def _write_encodings(path, documents):
    """An encodings store at `path` holding `documents`, unstamped.

    Keyed pmid -> `(input_ids, attention_mask)`, each `[windows, tokens]`;
    the offsets are zeros, since no reader of these fixtures maps a token
    back to text.
    """
    from d3text.encodings_store import EncodingsStore

    with EncodingsStore(path, writable=True) as store:
        for pmid, (ids, mask) in documents.items():
            ids = np.asarray(ids)
            store.put(
                pmid,
                {
                    "input_ids": ids,
                    "attention_mask": mask,
                    "offset_mapping": np.zeros((*ids.shape, 2)),
                },
            )
    return path


@pytest.fixture
def write_encodings():
    """`_write_encodings`, for a test building a store of its own."""
    return _write_encodings


@pytest.fixture
def tiny_encodings(tmp_path):
    """A small encodings store: one document per pmid, with input_ids /
    attention_mask of shape [n_chunks, 8]."""
    return _write_encodings(
        tmp_path / "encodings",
        {
            pmid: (np.zeros((n_chunks, 8)), np.ones((n_chunks, 8)))
            for pmid, n_chunks in _ENCODING_CHUNKS.items()
        },
    )


@pytest.fixture
def tiny_dataframe():
    """Matching DataFrame. Row 3 (pmid 40) is deliberately absent from the
    encodings store; the `fulltext` column proves BrendaDataset keeps only
    the three columns it needs."""
    return pd.DataFrame(
        {
            "pubmed_id": [10, 20, 30, 40],
            "relations": pd.Series([[], [], [], []]),
            "classes": [np.array([1, 0], dtype=np.float32)] * 4,
            "fulltext": ["x"] * 4,
        }
    )


@pytest.fixture
def tiny_brenda(tiny_encodings, tiny_dataframe):
    """Two `BrendaDataset` views over the tiny fixtures.

    `present` holds only the rows backed by the store; `full` also holds the
    row whose pmid is missing from it.
    """
    from d3text.data.data import BrendaDataset

    return types.SimpleNamespace(
        present=BrendaDataset(
            tiny_dataframe.iloc[:3].copy(), encodings=tiny_encodings
        ),
        full=BrendaDataset(tiny_dataframe.copy(), encodings=tiny_encodings),
        chunks=[2, 5, 1],
        missing_index=3,
    )
