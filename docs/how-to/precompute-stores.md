# Precompute the stores

Goal: the on-disk artifacts a training run reads — the encodings every run
needs, the embeddings store every machine should hold, and the token-label
store the span tagger needs.

`D` below stands for `brenda_references/src/brenda_references/data`, where
[the splits](fetch-the-data.md) land. Flags are listed in the
[command-line reference](../reference/cli.md); on-disk layouts in the
[store reference](../reference/stores.md).

## Encodings (required)

```bash
pdm run precompute-encodings <base_model> data/<name>.hdf5 \
    $D/training_data.csv $D/validation_data.csv $D/test_data.csv
```

`<name>` must be what `d3text.models.config.encodings` maps `<base_model>`
to; `train` and `evaluate` look the file up by that name under `data/`. A
base model with no entry there needs one added before it can be used.

The command skips documents already in the store, so re-running after an
interruption continues. `-f` re-encodes everything. A store built for one
base model, window or stride refuses another; build a new file instead.

To encode an external corpus into the same store for
[linking evaluation](evaluate-linking.md), add `--s800 <root>` or
`--enzymener <root>`.

## Token labels (required by `ETEBrendaModel`, optional otherwise)

```bash
pdm run precompute-token-labels <base_model> $D/documents.json data/token-labels.hdf5 \
    $D/training_data.csv $D/validation_data.csv $D/test_data.csv
```

Name every split you will train or evaluate on in **one** invocation. The
store's other-organism dictionary is pooled from the files given, and a
later run with a different set is refused. The command uses every CPU by
default; `-j 1` labels serially.

Point a training configuration at the result with
`token_labels_store = "data/token-labels.hdf5"`.

### When the store must be rebuilt

The store records the surface-form index and the labelling rules it was
built under. After any of these, delete the store and run the command
again — a plain re-run refuses to extend it:

- a change to the entity tables or the split files (the index moves);
- a change to `d3text.surface_forms` or `d3text.token_labels` that alters
  which strings match or how a match is labelled (the rules fingerprint
  moves);
- a store-format bump.

`train` and `evaluate` only read the store, so they warn rather than refuse:
`train` when the store's rules predate the running code, `evaluate` when the
store's digests differ from the ones the checkpoint recorded at training.

## Embeddings

Build this one too. A run with `unfrozen_top_layers = 0` freezes the whole
trunk, so the base model's output for a document cannot change between
epochs; without the store, that output is recomputed for every document the
CPU embeddings cache cannot hold, on every epoch and every validation pass,
and the run says so in a warning at start-up. The store replaces that
forward with a disk read.

Budget for the size rather than skipping the store over it: about 100 GiB
for the whole corpus at a 768-wide model.

```bash
pdm run precompute-embeddings <base_model> /data/d3text-embeddings \
    $D/training_data.csv $D/validation_data.csv $D/test_data.csv
```

Lower `--batch_size` if the base model runs out of GPU memory. The command
resumes; `-f` re-embeds. The store refuses a second base model or window
outright, and `-f` is not a way past that — build a new store.

Point the machine at it in `config.toml` — every machine that trains
sets this:

```toml
embeddings_store = "/data/d3text-embeddings"
```

A store the run cannot open, or one whose rows do not match the encodings,
is disabled with one warning and the run recomputes the embeddings. Each
training and validation pass logs what the store has served so far, so a
run that opened one but is not being answered by it shows up mid-flight
rather than at process exit.
