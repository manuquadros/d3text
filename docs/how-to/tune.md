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
token_supervision = [true]
```

The grid is the product of the lists. `hidden_layers` is the exception: its
list is a pool of widths. It generates every one-, two-, and three-layer
architecture whose widths stay equal or decrease toward the output. For
example, `[64, 32]` generates `[64]`, `[32]`, `[64, 64]`, `[64, 32]`,
`[32, 32]`, and the corresponding three-layer architectures. Fields not named
keep their `ModelConfig` defaults.

A combination whose `biaffine_hidden_size` exceeds the last hidden layer's
width is never drawn: the relation head would project its input up, adding
parameters and no information. Such a configuration is still valid for
`train`. Without a common hidden block the check is skipped, since the head
then reads the base model's own width.

## 2. Run the sweep

```bash
pdm run tuning <sweep.toml> <results.csv> [--limit N]
```

The command draws up to `d3text.models.config.SWEEP_SIZE` unique
configurations from the grid and builds each configuration immediately before
its trial. It does not hold the Cartesian product in memory. Existing rows in
`<results.csv>` other than failed trials' (below) are excluded, so resuming
a sweep never reruns a configuration that already has a score. No checkpoint
is written. After every trial one row is appended to `<results.csv>`: the
configuration's fields plus `selection_score`, the best validation selection
score the trial reached (higher is better — see
[the training loop](../explanation/cli-and-training.md#the-training-loop)).
A header is written when the file is new or empty.

After every trial that did not fail, the best-scoring configuration in
`<results.csv>` — earlier sessions' rows included — is written to the TOML
file of the same name (`results.csv` → `results.toml`), ready to pass to
`train`. Ties keep the earlier row. The sweep refuses to start if that file
would be its own sweep configuration, as with `tuning sweep.toml sweep.csv`.

A trial that raises — building its dataset or model, or during training —
does not stop the sweep. Its row is still written, with `selection_score`
`NaN` marking it as failed, and the next trial runs; with
`MLFLOW_TRACKING_URI` set, its run is closed `FAILED` rather than left open
or missing. A sweep in which every trial failed exits with a nonzero status
instead of ending like one that produced results. A resume does not exclude
a failed trial's configuration: it may be drawn and run again, so a sweep
whose trials all failed for a reason outside the grid (a missing encodings
file, a GPU held by another process) can be rerun once that is fixed.

`--limit` applies to every trial, as for `train`.

## 3. Read the results

The best configuration so far is already in `<results>.toml`. For the
rest, sort the CSV by `selection_score`, descending. With `MLFLOW_TRACKING_URI`
set, each trial is
also an MLflow run tagged `sweep=<sweep.toml>` and `trial=<n>`, with the
full per-epoch curves; see [Track a run with MLflow](track-with-mlflow.md).
`tests/best_config_so_far.toml` records the best configuration found so
far.

Successive sweeps are independent draws. To replay a sweep exactly, call
`load_tuning_config(path, rng=random.Random(seed))` from Python instead.
