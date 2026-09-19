# Fetch the data

Goal: the BRENDA corpus and its three splits on disk, verified against the
manifest the model numbers were produced from.

The files are not in git. They total about 1.85 GB and are published as the
Hugging Face dataset repository named in
`brenda_references/scripts/pull_data.py`.

## Download and verify

```bash
pdm run python brenda_references/scripts/pull_data.py
```

This downloads every file listed in
`brenda_references/src/brenda_references/data/SHA256SUMS` into that
directory and checks each digest. To check files already on disk without
downloading:

```bash
pdm run python brenda_references/scripts/pull_data.py --check
```

Or, from that directory and with no Python: `sha256sum -c SHA256SUMS`.

## What lands where

| File | Role |
| --- | --- |
| `documents.json` | The TinyDB corpus: BRENDA references joined with article full texts, followed by the entity tables. The input to `precompute-token-labels` |
| `training_data.csv`, `validation_data.csv`, `test_data.csv` | The three splits; the inputs to every `precompute-*` command and what `train` and `evaluate` read |
| `pmc_linguistics_articles.json` | Off-domain noise documents the splits draw from |
| `enzyme_negative_pool.json` | In-domain documents naming no enzyme |

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
