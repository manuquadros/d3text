# Change a dependency

Goal: a dependency added, removed or re-pinned, with all four lockfiles
regenerated so every machine installs the same resolution.

## Why four lockfiles

The torch build is a property of the machine, not the project, and pdm
cannot bind a package index to a dependency extra — only to a package name.
The torch source in `pyproject.toml` therefore takes its flavour from
`${TORCH_FLAVOUR}` at **lock time**, which makes the whole resolution
flavour-specific: one lockfile per flavour, and `TORCH_FLAVOUR` is never
read at install or run time.

## 1. Edit the declaration

In `pyproject.toml`, or in `brenda_references/pyproject.toml` for the data
layer. Both count: `brenda_references` is a path dependency resolved into
every lockfile.

## 2. Regenerate every lockfile

```bash
pdm run lock-all
```

Runs `lock-cpu`, `lock-cu126`, `lock-cu128` and `lock-cu130` in turn, each
with its own `TORCH_FLAVOUR`. Budget about ten minutes. Locking is chatty —
pdm queries the torch index for every package and falls back to PyPI — and
that is harmless.

Regenerate every one even for a change that touches no torch-related
package: the lockfile records a hash of the whole `pyproject.toml`, so any
edit to it makes them all stale, and a forgotten one is a red CI build
(`pdm lock --check` against `locks/cpu.lock`), not a silent re-resolution.

## 3. Reinstall and test

```bash
TMPDIR=~/.cache/pdm-tmp pdm install -L locks/<flavour>.lock --frozen-lockfile
pdm run check-locks
pdm run pytest tests/
```

`TMPDIR` matters whenever torch is replaced; see [Install](install.md#2-install).

## Git dependencies with two spellings

`[tool.pdm.resolution.overrides]` pins one URL spelling per git dependency
because two sub-packages require the same package under different URLs
(`git@github.com` against `https://github.com`), and pdm treats a direct
URL as the package's identity. Drop an override and locking fails with
`Unable to find a resolution for <pkg>`. The real fix is making the
spellings agree upstream, then dropping the override.

## What does not need a relock

Editing `[tool.pdm.scripts]` — scripts are not part of the lock-input hash.
