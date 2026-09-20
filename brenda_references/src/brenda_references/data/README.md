# BRENDA reference data

These files are **not in git** and **not in the wheel**. They total ~1.85 GB
and are distributed through the Hugging Face Hub dataset repo
`manuquadros/brenda-references-data`.

```bash
pdm run python brenda_references/scripts/pull_data.py          # fetch + verify
pdm run python brenda_references/scripts/pull_data.py --check  # verify only
```

`SHA256SUMS` beside this file pins the exact revision the current model numbers
were produced from, and ships with the package so that a reader can always name
what it expects.

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
| `training_data.csv` | 537 MB | Training split, then extended with the unsampled remainder by `scripts/augment_training_data.py`. |
| `validation_data.csv` | 80 MB | Validation split. |
| `test_data.csv` | 75 MB | Test split. |
| `pmc_linguistics_articles.json` | 73 MB | Off-domain linguistics articles; the noise pool the splits draw from (`NOISE_BLOCKS` in `brenda_references.py`). |
| `enzyme_negative_pool.json` | 43 MB | PMC OA microbiology articles naming no enzyme under the guarded surface-form index (literal reading) — a hard negative for the enzyme head, same register and vocabulary as the positives, enzyme absent. Built by `scripts/build_enzyme_negative_pool.py`. Not yet wired into `NOISE_BLOCKS` or excluded from the splits. |

`documents.json` is the primary artifact — it is the only one that cannot be
derived from anything else in the repo, and rebuilding it means re-running the
BRENDA sync and re-fetching every full text over the network.

## Why the splits ship as data rather than as a script

`scripts/generate_dataset.py` derives the three `*_data.csv` files from
`documents.json`, so they look regenerable. They are not, in the sense that
matters: `GMESampler` passes **no seed** to `GreedyMaximumEntropySampler`, and
`augment_training_data.py` rewrites `training_data.csv` in place. Re-running the
generator can therefore produce a different train/test partition, which would
silently invalidate every comparison against previously recorded model numbers
without any error surfacing.

Treat the CSVs as experiment-pinning artifacts: fetch them, do not regenerate
them, and if a split genuinely has to change, publish a new Hub revision and
update `SHA256SUMS` in the same commit that reports the new numbers.

## Publishing a new revision

Run these from the directory `pull_data.py` reports, which is where the blobs
actually are:

```bash
hf upload manuquadros/brenda-references-data . . \
  --repo-type dataset \
  --include '*.json' --include '*.csv' \
  --commit-message "<what changed and why>"

# then re-pin, writing the manifest back into the package:
sha256sum documents.json pmc_linguistics_articles.json test_data.csv \
  training_data.csv validation_data.csv enzyme_negative_pool.json \
  > <checkout>/brenda_references/src/brenda_references/data/SHA256SUMS
```

`pull_data.py` downloads only the names listed in `SHA256SUMS`, so the Hub
repo's own `README.md` never overwrites this one.
