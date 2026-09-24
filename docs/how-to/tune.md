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
list is a pool of widths. It generates every one-, two-, and three-layer
architecture whose widths stay equal or decrease toward the output. For
example, `[64, 32]` generates `[64]`, `[32]`, `[64, 64]`, `[64, 32]`,
`[32, 32]`, and the corresponding three-layer architectures. Fields not named
keep their `ModelConfig` defaults.

## 2. Run the sweep

```bash
pdm run tuning <sweep.toml> <results.csv> [--limit N]
```

The command draws up to `d3text.models.config.SWEEP_SIZE` unique
configurations from the grid and builds each configuration immediately before
its trial. It does not hold the Cartesian product in memory. Existing rows in
`<results.csv>` are excluded, so resuming a sweep spends every trial on a new
configuration. No checkpoint is written. After every trial one row is appended
to `<results.csv>`: the configuration's fields plus `selection_score`, the
best validation selection score the trial reached (higher is better — see
[the training loop](../explanation/cli-and-training.md#the-training-loop)).
A header is written when the file is new or empty.

A trial that raises — building its dataset or model, or during training —
does not stop the sweep. Its row is still written, with `selection_score`
`NaN` marking it as failed, and the next trial runs; with
`MLFLOW_TRACKING_URI` set, its run is closed `FAILED` rather than left open
or missing. A sweep in which every trial failed exits with a nonzero status
instead of ending like one that produced results.

`--limit` applies to every trial, as for `train`.

## 3. Read the results

Sort the CSV by `selection_score`, descending. With `MLFLOW_TRACKING_URI`
set, each trial is
also an MLflow run tagged `sweep=<sweep.toml>` and `trial=<n>`, with the
full per-epoch curves; see [Track a run with MLflow](track-with-mlflow.md).
`tests/best_config_so_far.toml` records the best configuration found so
far.

Successive sweeps are independent draws. To replay a sweep exactly, call
`load_tuning_config(path, rng=random.Random(seed))` from Python instead.
