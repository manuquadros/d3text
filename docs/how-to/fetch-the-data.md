# Fetch the data

Goal: the BRENDA corpus and its three splits on disk, verified against the
manifest the model numbers were produced from.

The files are not in git. They total about 1.85 GB and are published as the
Hugging Face dataset repository named in
`brenda_references/scripts/pull_data.py`.

Install the project before fetching: the download and every reader take the
destination from the installed `brenda_references.data_paths`, so that a
fetch cannot land somewhere the readers do not look.

## Download and verify

```bash
pdm run python brenda_references/scripts/pull_data.py
```

This downloads every file listed in the manifest
`brenda_references/src/brenda_references/data/SHA256SUMS` and checks each
digest. It prints the directory it used; see *Where the files land* below.
To check files already on disk without downloading:

```bash
pdm run python brenda_references/scripts/pull_data.py --check
```

## Where the files land

The manifest ships inside the package, but the blobs it pins do not — a
1.85 GB payload has no business in a wheel, and two of the six are in no git
at all. The destination is therefore resolved separately, in this order:

1. `BRENDA_DATA_DIR`, if set. Point it at a shared volume to have several
   checkouts read one copy.
2. `brenda_references/src/brenda_references/data/` in the checkout, when it
   already holds `documents.json`. An installation that fetched the data
   before keeps reading the copy it has.
3. Otherwise `brenda-references` under `XDG_DATA_HOME`, or
   `~/.local/share/brenda-references` when that is unset.

With no Python at all, verify from whichever directory holds the files:
`sha256sum -c <path to SHA256SUMS>`.

## What each file is

| File | Role |
| --- | --- |
| `documents.json` | The TinyDB corpus: BRENDA references joined with article full texts, followed by the entity tables. The input to `precompute-token-labels` |
| `training_data.csv`, `validation_data.csv`, `test_data.csv` | The three splits; what `train` and `evaluate` read |
| `pmc_linguistics_articles.json` | Off-domain noise documents the splits draw from |
| `enzyme_negative_pool.json` | In-domain documents naming no enzyme, likewise drawn into the splits |

Every `precompute-*` command takes the three splits and both pools: each
split is loaded with a block of each pool appended, so documents a store
was not given are documents the run cannot read.

## Do not regenerate the splits

`brenda_references/scripts/generate_dataset.py` derives the CSVs from
`documents.json`, but its sampler is unseeded and the training split is
rewritten in place by a second script, so a re-run can partition the corpus
differently and silently invalidate every recorded comparison. Treat the
CSVs as pinned artifacts. If a split must change, publish a new revision and
update `SHA256SUMS` in the same commit that reports the new numbers; the
publishing steps are in `brenda_references/src/brenda_references/data/README.md`.

## Vocabulary files

`data/bacteria.txt`, `data/enzymes.txt` and `data/strains.txt` at the
repository root are the plain wordlists the surface-form matcher reads. They
are tracked in git; nothing to fetch.

## Next

- [Run a first training](first-run.md)
- [Score the linker against external corpora](evaluate-linking.md) — a
  separate set of downloads, needed only for that block of `evaluate`
