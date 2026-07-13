"""Tests for `d3text.runtime`.

Two halves. The first pins the *negative* guarantee — importing the library
leaves the process exactly as it found it — which can only be checked in a
fresh interpreter, since pytest has long since imported these modules. The
second pins what `configure()` applies when a script does opt in.
"""

import json
import os
import pathlib
import subprocess
import sys

import pytest
import torch
from d3text import runtime
from d3text.models.config import MachineConfig

REPO_ROOT = pathlib.Path(__file__).parent.parent

# Env vars the library used to set on the way in.
_ALLOC_VARS = ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF")
_RUNTIME_VARS = ("TOKENIZERS_PARALLELISM", *_ALLOC_VARS)

# Read torch's globals, import the library, read them again. `initial_seed()`
# forces the default generator to seed itself first, so a library-side
# `manual_seed` shows up as a changed seed rather than as a lazy first read.
_IMPORT_PROBE = """
import json, os
import torch

def snapshot():
    return {
        "matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "seed": torch.initial_seed(),
    }

before = snapshot()

import d3text.data
import d3text.models
import d3text.runtime

print("@@" + json.dumps({
    "before": before,
    "after": snapshot(),
    "env": {var: os.environ.get(var) for var in %(vars)r},
}))
""" % {"vars": _RUNTIME_VARS}


@pytest.fixture
def clean_env(monkeypatch):
    """Drop the runtime env vars, and restore them afterwards.

    `monkeypatch.delenv` records each variable's original value, so its teardown
    also undoes whatever `configure()` writes during the test.
    """
    for var in _RUNTIME_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def restore_torch_globals():
    """Snapshot and restore the process-global torch settings `configure()`
    writes, so a test that calls it cannot leak into the rest of the suite."""
    saved = (
        torch.get_float32_matmul_precision(),
        torch.backends.cudnn.allow_tf32,
        torch.get_rng_state(),
    )
    yield
    precision, cudnn_tf32, rng_state = saved
    torch.set_float32_matmul_precision(precision)
    torch.backends.cudnn.allow_tf32 = cudnn_tf32
    torch.set_rng_state(rng_state)


@pytest.fixture
def configured(clean_env, restore_torch_globals):
    """Call `runtime.configure()` with the process state restored afterwards."""
    return runtime.configure


def _machine_config(**overrides) -> MachineConfig:
    return MachineConfig(cpu_embeddings_cache_size=0, **overrides)


@pytest.mark.slow
def test_importing_the_library_does_not_reconfigure_the_process():
    """Importing `d3text` must leave torch's globals and the environment alone:
    these settings are sticky and process-wide, so they belong to the script
    that owns the process, not to whichever module got imported first."""
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in _RUNTIME_VARS
    }
    probe = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
    )
    assert probe.returncode == 0, probe.stderr

    # d3text/__init__ prints on a missing optional dep, hence the sentinel.
    payload = json.loads(
        next(
            line[2:]
            for line in probe.stdout.splitlines()
            if line.startswith("@@")
        )
    )
    assert payload["after"] == payload["before"]
    assert payload["env"] == dict.fromkeys(_RUNTIME_VARS, None)


def test_configure_applies_the_machine_settings(configured):
    configured(
        _machine_config(
            float32_matmul_precision="high",
            cudnn_allow_tf32=False,
            tokenizers_parallelism=False,
        ),
        seed=None,
    )

    assert torch.get_float32_matmul_precision() == "high"
    assert torch.backends.cudnn.allow_tf32 is False
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"


@pytest.mark.parametrize(
    ("precision", "cublas_tf32"),
    [("highest", False), ("high", True), ("medium", True)],
)
def test_matmul_precision_subsumes_the_cublas_tf32_flag(
    configured, precision, cublas_tf32
):
    """`torch.backends.cuda.matmul.allow_tf32` is a view of the matmul
    precision, not an independent setting: a second knob writing it would fight
    this one, depending on which ran last."""
    configured(_machine_config(float32_matmul_precision=precision), seed=None)

    assert torch.backends.cuda.matmul.allow_tf32 is cublas_tf32


def test_configure_defaults_to_the_repo_config(configured, monkeypatch):
    """With no argument, the settings come from `config.toml` — so the matmul
    precision is a knob, not a literal buried in a script."""
    monkeypatch.setattr(
        runtime,
        "machine_config",
        lambda: _machine_config(float32_matmul_precision="highest"),
    )
    configured(seed=None)

    assert torch.get_float32_matmul_precision() == "highest"


@pytest.mark.parametrize(
    ("hip", "cuda", "expected"),
    [
        (None, "12.8", "PYTORCH_CUDA_ALLOC_CONF"),
        ("6.3.0", None, "PYTORCH_HIP_ALLOC_CONF"),
    ],
    ids=["cuda-build", "hip-build"],
)
def test_allocator_variable_matches_the_torch_build(
    configured, monkeypatch, hip, cuda, expected
):
    """A build reads only its own allocator variable; setting the other one is
    what made `expandable_segments` a silent no-op on CUDA."""
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(torch.version, "cuda", cuda)

    configured(_machine_config(expandable_segments=True), seed=None)

    unset = set(_ALLOC_VARS) - {expected}
    assert os.environ[expected] == "expandable_segments:True"
    assert not unset & set(os.environ)


def test_configure_keeps_an_allocator_conf_set_in_the_environment(
    configured, monkeypatch
):
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

    configured(_machine_config(expandable_segments=True), seed=None)

    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:128"


def test_expandable_segments_off_sets_no_allocator_variable(configured):
    configured(_machine_config(expandable_segments=False), seed=None)

    assert not set(_ALLOC_VARS) & set(os.environ)


def test_configure_seeds_the_generator_the_samplers_draw_from(configured):
    """The samplers hold `data.g`; seeding has to reach *that* generator, which
    it only does because `g` is torch's global one."""
    from d3text.data import data

    configured(_machine_config(), seed=123)

    assert data.g is torch.default_generator
    assert data.g.initial_seed() == 123
    assert torch.initial_seed() == 123


def test_configure_leaves_the_rng_alone_when_seed_is_none(configured):
    torch.manual_seed(7)

    configured(_machine_config(), seed=None)

    assert torch.initial_seed() == 7


class TritonProbe:
    """Stands in for `torch.cuda`: a GPU of a given compute capability, or
    none at all."""

    def __init__(self, capability: tuple[int, int] | None) -> None:
        self._capability = capability

    def is_available(self) -> bool:
        return self._capability is not None

    def get_device_capability(self) -> tuple[int, int]:
        assert self._capability is not None
        return self._capability


@pytest.mark.parametrize(
    ("capability", "compatible"),
    [
        ((7, 0), True),  # Volta, the oldest Triton supports
        ((8, 9), True),  # Ada
        ((6, 1), False),  # Pascal — the P100 VM
        ((6, 0), False),
    ],
)
def test_triton_needs_compute_capability_7(monkeypatch, capability, compatible):
    """`torch.compile` is lazy, so an unsupported GPU is not reported by the
    call it is asked for — it fails at the first forward pass instead, past the
    try/except the call site wraps it in. The capability has to be checked up
    front."""
    monkeypatch.setattr(torch, "cuda", TritonProbe(capability))

    assert runtime.is_triton_compatible() is compatible


def test_triton_is_unavailable_without_a_gpu(monkeypatch):
    """Must not reach for the device capability at all: asking a CPU-only build
    for one raises."""
    monkeypatch.setattr(torch, "cuda", TritonProbe(None))

    assert runtime.is_triton_compatible() is False
