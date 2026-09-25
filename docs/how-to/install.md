# Install a development environment

Goal: a virtual environment with the package, its data layer and the torch
build that matches the machine.

Prerequisites: Python 3.13, [pdm](https://pdm-project.org) and git.

## 1. Pick the lockfile for the machine

| Machine | Lockfile |
| --- | --- |
| NVIDIA Blackwell (compute capability 12.x) | `locks/cu130.lock` |
| NVIDIA GPU, recent driver | `locks/cu128.lock` |
| NVIDIA GPU, CUDA 11.8 driver | `locks/cu118.lock` |
| No GPU, or CI | `locks/cpu.lock` |

The lockfiles differ only in which torch build they pin. Installing consults
no package index: every wheel URL is written into the lockfile.

They do not pin the same torch release, and are not meant to: each CUDA
index carries its own set of builds, so `cu128` stops at the last release
PyTorch built against it while `cpu` and `cu130` track the current one.

A Blackwell card runs the `cu128` build too — a cubin is forward compatible
within a major architecture version, so the `sm_120` kernels that build
ships cover an `sm_121` device. `cu130` is what has kernels compiled for it.

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

The version suffix (`+cu130`, `+cu128`, none for CPU) is the build the
lockfile selected.

## 4. Machine settings (optional)

Copy `config.toml.example` to `config.toml` at the repository root and edit
what differs. Every key is optional and so is the file; the keys are listed
in the [configuration reference](../reference/configuration.md#machine-settings-configtoml).

## Next

- [Fetch the data](fetch-the-data.md)
