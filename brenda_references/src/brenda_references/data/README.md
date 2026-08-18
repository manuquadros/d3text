# BRENDA reference data

These files are **not in git**. They total ~1.8 GB and are distributed through
the Hugging Face Hub dataset repo `manuquadros/brenda-references-data`.

```bash
pdm run python brenda_references/scripts/pull_data.py          # fetch + verify
pdm run python brenda_references/scripts/pull_data.py --check  # verify only
```

`SHA256SUMS` beside this file pins the exact revision the current model numbers
were produced from. `sha256sum -c SHA256SUMS` from this directory checks it
without any Python.

## Contents

| File | Size | Origin |
|---|---|---|
| `documents.json` | 1072 MB | The TinyDB corpus: BRENDA references joined with article full texts. Built by `sync_doc_db` from the BRENDA MySQL database plus NCBI/PMC retrieval (`scripts/retrieve_text.py`). |
| `training_data.csv` | 537 MB | Training split, then extended with the unsampled remainder by `scripts/augment_training_data.py`. |
| `validation_data.csv` | 80 MB | Validation split. |
| `test_data.csv` | 75 MB | Test split. |
| `pmc_linguistics_articles.json` | 73 MB | Off-domain linguistics articles; the noise pool the splits draw from (`NOISE_BLOCKS` in `brenda_references.py`). |

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

```bash
hf upload manuquadros/brenda-references-data \
  brenda_references/src/brenda_references/data . \
  --repo-type dataset \
  --include '*.json' --include '*.csv' \
  --commit-message "<what changed and why>"

# then re-pin, from this directory:
sha256sum documents.json pmc_linguistics_articles.json test_data.csv \
  training_data.csv validation_data.csv > SHA256SUMS
```

`pull_data.py` downloads only the five names listed in `SHA256SUMS`, so the Hub
repo's own `README.md` never overwrites this one.
