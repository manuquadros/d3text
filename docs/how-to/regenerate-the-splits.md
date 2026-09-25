# Regenerate the splits

Goal: new `training_data.csv`, `validation_data.csv` and `test_data.csv`
drawn from the corpus. Why they are drawn the way they are:
[How the splits are drawn](../explanation/splits.md).

The splits pin every recorded model number. A new draw makes those numbers
incomparable with anything trained or evaluated on it, and the stores built
over the old splits have to be rebuilt. Regenerate only to change the splits
on purpose.

## Draw

The corpus has to be on disk first ([Fetch the data](fetch-the-data.md)).

```bash
pdm run python scripts/generate_splits.py <output_dir>
```

It reads `documents.json` from the data directory, prints each split's size
and the entities validation and test hold that training does not, per entity
type, and writes the three CSVs into `<output_dir>` under their configured
names. Write into a scratch directory, not the data directory: the files there
are the ones `SHA256SUMS` pins.

The last number on each line counts entities absent from training that share a
surface form with it. It has to be 0; anything else is a bug in the splitter.

The draw is deterministic: the same corpus, dictionary and `--seed` give the
same splits. `--evaluation-share`, `--held-share` and `--max-held-documents`
change the split sizes and how much of validation and test is held out; `--help`
lists them with their defaults.

## Adopt

1. Copy the three CSVs over the ones in the data directory.
2. Publish them as a new revision of the data repository and re-pin
   `SHA256SUMS`, as the README beside that manifest
   (`brenda_references/src/brenda_references/data/README.md`) describes.
3. Rebuild every store over the splits
   ([Precompute the stores](precompute-stores.md)).
4. Commit the new `SHA256SUMS`, with that README's file table saying how the
   new revision was drawn, together with the numbers measured on the new
   splits.
