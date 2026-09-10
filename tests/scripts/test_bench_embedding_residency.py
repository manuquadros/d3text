"""The residency benchmark's source-regime selection is only worth reading if
a machine-configured cache or embeddings store cannot silently turn both arms
into reads of the same source. These pin `select_source_regime` against a
stub `d3text.models.base`-shaped module, so no GPU or real embeddings are
needed.
"""

import importlib.util
import pathlib
import types

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts"
    / "benchmarks"
    / "bench_embedding_residency.py"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "bench_embedding_residency", _SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = _load()


def _stub_module(cache_on: bool, store: str | None) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        cpu_embeddings_cache=object() if cache_on else None,
        mconfig=types.SimpleNamespace(embeddings_store=store),
    )


def test_off_forces_the_cache_and_store_off_regardless_of_config() -> None:
    """A machine's `config.toml` can set either knob; `off` must win over
    both so the benchmark measures the base-model forward, not a cache or
    store read."""
    stub = _stub_module(cache_on=True, store="/mnt/embeddings")

    info = bench.select_source_regime(bench.SOURCE_OFF, module=stub)

    assert stub.cpu_embeddings_cache is None
    assert stub.mconfig.embeddings_store is None
    assert info == {
        "regime": bench.SOURCE_OFF,
        "cpu_arm_cache": False,
        "cpu_arm_store": False,
        "gpu_arm_cache": False,
        "gpu_arm_store": False,
    }


def test_configured_leaves_the_machine_config_untouched() -> None:
    """`configured` measures the hit path on purpose, so it must not force
    either knob off."""
    stub = _stub_module(cache_on=True, store="/mnt/embeddings")

    info = bench.select_source_regime(bench.SOURCE_CONFIGURED, module=stub)

    assert stub.cpu_embeddings_cache is not None
    assert stub.mconfig.embeddings_store == "/mnt/embeddings"
    assert info["regime"] == bench.SOURCE_CONFIGURED


def test_a_configured_store_is_recorded_as_asymmetric_between_the_arms() -> (
    None
):
    """Only `cpu_impl` ever queries the store — `gpu_impl` never does — so a
    `configured` run with a store set is not a symmetric comparison. Before
    this fix nothing recorded that, and a store left on by the machine's
    `config.toml` made the timings compare a store read against a full
    forward (or, once the on-device arm's own cache warmed, a cache read)
    while the emitted JSON read like a clean, symmetric result."""
    stub = _stub_module(cache_on=False, store="/mnt/embeddings")

    info = bench.select_source_regime(bench.SOURCE_CONFIGURED, module=stub)

    assert info["cpu_arm_store"] is True
    assert info["gpu_arm_store"] is False


def test_no_store_configured_is_symmetric() -> None:
    """With no store set, both arms read from the same sources — the case
    that must not be flagged as an asymmetry."""
    stub = _stub_module(cache_on=True, store=None)

    info = bench.select_source_regime(bench.SOURCE_CONFIGURED, module=stub)

    assert info["cpu_arm_store"] is False
    assert info["gpu_arm_store"] is False
    assert info["cpu_arm_cache"] == info["gpu_arm_cache"] is True
