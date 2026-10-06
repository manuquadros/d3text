# BRENDA reference data

These files are **not in git** and **not in the wheel**. They total ~1.85 GB
and are distributed through the Hugging Face Hub dataset repo
`manuquadros/brenda-references-data`.

```bash
pdm run python brenda_references/scripts/pull_data.py          # fetch + verify
pdm run python brenda_references/scripts/pull_data.py --check  # verify only
```

`SHA256SUMS` beside this file pins the content the current model numbers were
produced from, and `HUB_REVISION` the Hub commit holding that content; both
ship with the package so that a reader can always name what it expects.

**The files themselves need not be in this directory.**
`brenda_references.data_paths.resolve_data_dir()` decides where they live, and
everything that reads or writes them asks it: `BRENDA_DATA_DIR` if set, else
this directory when it already holds `documents.json`, else
`brenda-references` under `XDG_DATA_HOME`. Deriving the path any other way is
what made a non-editable install unusable — the fetch landed in the checkout
while every read went to `site-packages`. `pull_data.py` prints the directory
it used; `sha256sum -c` against this manifest, run from there, checks it
without any Python.

## Contents

| File | Size | Origin |
|---|---|---|
| `documents.json` | 1072 MB | The TinyDB corpus: BRENDA references joined with article full texts. Built by `sync_doc_db` from the BRENDA MySQL database plus NCBI/PMC retrieval (`scripts/retrieve_text.py`). |
| `training_data.csv` | 537 MB | Training split. This revision predates `generate_splits.py`: drawn by greedy maximum-entropy sampling over papers with relations, then extended with every paper the sampler left out. |
| `validation_data.csv` | 80 MB | Validation split. |
| `test_data.csv` | 75 MB | Test split. |
| `pmc_linguistics_articles.json` | 73 MB | Off-domain linguistics articles; the noise pool the splits draw from (`NOISE_BLOCKS` in `brenda_references.py`). |
| `enzyme_negative_pool.json` | 43 MB | PMC OA microbiology articles naming no enzyme under the guarded surface-form index (literal reading) — a hard negative for the enzyme head, same register and vocabulary as the positives, enzyme absent. Built under `negative_screen.LITERAL` by a since-removed study script. Not yet wired into `NOISE_BLOCKS` or excluded from the splits. |

`documents.json` is the primary artifact — it is the only one that cannot be
derived from anything else in the repo, and rebuilding it means re-running the
BRENDA sync and re-fetching every full text over the network.

## Why the splits ship as data rather than as a script

d3text's `scripts/generate_splits.py` derives the three `*_data.csv` files from
`documents.json`, deterministically for one seed. They still ship as data:
the draw also depends on the surface-form dictionary, which is code that
changes, so re-running the generator later can produce a different partition,
which would silently invalidate every comparison against previously recorded
model numbers without any error surfacing.

Treat the CSVs as experiment-pinning artifacts: fetch them, do not regenerate
them, and if a split genuinely has to change, publish a new Hub revision and
update `SHA256SUMS` and `HUB_REVISION` in the same commit that reports the new
numbers.

## Publishing a new revision

```bash
pdm run python brenda_references/scripts/publish_data.py --dry-run   # what changed
pdm run python brenda_references/scripts/publish_data.py -m "<what changed and why>"
```

It hashes every file `SHA256SUMS` lists, uploads the changed ones as one Hub
commit on top of `HUB_REVISION`, writes the new digests to `SHA256SUMS` and
the id that commit returned to `HUB_REVISION`. It refuses to upload if the
repo's head is no longer `HUB_REVISION`, and nothing is written locally
unless the upload succeeded. When every changed file is already identical on
the Hub, no commit is made and only `SHA256SUMS` changes. It exits 1 when no
file changed. It reads the blobs from the directory `pull_data.py` reports
and takes `--repo` or `BRENDA_DATA_REPO` like `pull_data.py`.

Commit `SHA256SUMS` and `HUB_REVISION` together. `HUB_REVISION` names the Hub
commit whose files match the manifest; `pull_data.py` downloads that commit
rather than the repo's `main`, so a checkout keeps fetching the files its
manifest pins after a newer revision is published. `--revision <sha>`
overrides it.

`pull_data.py` downloads only the names listed in `SHA256SUMS`, so the Hub
repo's own `README.md` never overwrites this one.
