# Tune hyperparameters

Goal: a CSV of trial results over a random sample of a hyperparameter grid.

Before starting: the same stores as for [training](train-and-evaluate.md).

## 1. Write a sweep configuration

Every key is a `ModelConfig` field holding a **list** of candidate values;
`tuning_config.toml` at the repository root is a working example:

```toml
optimizer = ["adamw", "adam", "nadam"]
lr = [0.0003, 0.003, 0.01]
hidden_layers = [64, 64, 32]
ramp_epochs = [4, 8, 0]
model_class = ["ETEBrendaModel"]
token_labels_store = ["data/token-labels.hdf5"]
```

The grid is the product of the lists. `hidden_layers` is the exception: its
list is a pool of widths, and each trial draws one width from it. Fields not
named keep their `ModelConfig` defaults.

## 2. Run the sweep

```bash
pdm run tuning <sweep.toml> <results.csv> [--limit N]
```

The command samples `d3text.models.config.SWEEP_SIZE` configurations from
the grid (the whole grid, when it is smaller) and trains each one in turn.
No checkpoint is written. After every trial one row is appended to
`<results.csv>`: the configuration's fields plus `val_loss`, the best
validation loss the trial reached. A header is written when the file is new
or empty, so a sweep can be resumed by re-running into the same file.

A trial that raises stops the sweep; its row is not written.

`--limit` applies to every trial, as for `train`.

## 3. Read the results

Sort the CSV by `val_loss`. With `MLFLOW_TRACKING_URI` set, each trial is
also an MLflow run tagged `sweep=<sweep.toml>` and `trial=<n>`, with the
full per-epoch curves; see [Track a run with MLflow](track-with-mlflow.md).
`tests/best_config_so_far.toml` records the best configuration found so
far.

Successive sweeps are independent draws. To replay a sweep exactly, call
`load_tuning_config(path, rng=random.Random(seed))` from Python instead.
