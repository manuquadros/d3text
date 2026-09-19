# Install a development environment

Goal: a virtual environment with the package, its data layer and the torch
build that matches the machine.

Prerequisites: Python 3.12, [pdm](https://pdm-project.org) and git.

## 1. Pick the lockfile for the machine

| Machine | Lockfile |
| --- | --- |
| NVIDIA GPU, recent driver | `locks/cu128.lock` |
| NVIDIA GPU, CUDA 11.8 driver | `locks/cu118.lock` |
| AMD GPU (ROCm) | `locks/rocm.lock` |
| No GPU, or CI | `locks/cpu.lock` |

The lockfiles differ only in which torch build they pin. Installing consults
no package index: every wheel URL is written into the lockfile.

## 2. Install

```bash
TMPDIR=~/.cache/pdm-tmp pdm install -L locks/cu128.lock --frozen-lockfile
```

Set `TMPDIR` to a directory on the same filesystem as the virtual
environment whenever torch is being installed or replaced. pdm moves the
outgoing package into a staging directory under `TMPDIR` to allow rollback;
a torch build does not fit in a small `/tmp`, and the failure is partial —
small packages install, torch does not, and the command exits non-zero
having already changed the environment. If an install dies with
`UninstallError: [Errno 122] Disk quota exceeded` or `Invalid cross-device
link`, re-run with `TMPDIR` set before anything else.

`pdm.toml` pins `use_uv = "false"`. Leave it: with pdm's uv backend the torch
source is ignored and torch resolves from PyPI in the wrong build, with no
error.

## 3. Verify

```bash
pdm run python -c "import torch, d3text; print(torch.__version__, torch.cuda.is_available())"
```

The version suffix (`+cu128`, `+rocm…`, none for CPU) is the build the
lockfile selected.

## 4. Machine settings (optional)

Copy `config.toml.example` to `config.toml` at the repository root and edit
what differs. Every key is optional and so is the file; the keys are listed
in the [configuration reference](../reference/configuration.md#machine-settings-configtoml).

## AMD cards the wheel has no kernels for

A ROCm wheel carries object code for a fixed list of GPU architectures and
nothing to compile for others. A card outside that list passes
`torch.cuda.is_available()` and dies at the first allocation with
`HIP error: invalid device function`. For an RDNA2 card of a sibling
architecture (for example a gfx1032 card against a gfx1030 build), present it
as the built one:

```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
```

`train`, `tuning` and `evaluate` log a warning naming this variable when the
device's architecture is not in the installed build's list. The override is
safe only between architectures sharing an ISA; it does nothing on NVIDIA
machines.

## Next

- [Fetch the data](fetch-the-data.md)
