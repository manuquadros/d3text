# D3Text

Entity recognition, entity linking and relation extraction over the BRENDA
enzyme literature, with distant supervision from BRENDA's own entity tables.

## Install

Python 3.12 and [pdm](https://pdm-project.org). Pick the lockfile matching
the machine's torch build (`cpu`, `cu118`, `cu128`, `cu130`):

```bash
TMPDIR=~/.cache/pdm-tmp pdm install -L locks/cu128.lock --frozen-lockfile
pdm run python brenda_references/scripts/pull_data.py    # ~1.85 GB of corpus
```

## Documentation

Published at <https://manuquadros.github.io/d3text/>; `pdm run mkdocs serve`
serves the same site from `docs/`. It is organised by what the reader is
doing:

- **How-to guides** — install, fetch the data, run a first training,
  precompute the stores, train, evaluate, tune, track runs, run the checks.
- **Reference** — every command and flag, configuration key and environment
  variable, the checkpoint and store formats, the metric keys, the API.
- **Explanation** — why the pipeline is built the way it is.

Start at `docs/how-to/first-run.md`.
