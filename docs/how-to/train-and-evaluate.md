# Train and evaluate a model

Goal: a checkpoint from a training configuration, and its scores on the test
split. Commands run from the repository root.

Before starting: the [encodings](precompute-stores.md#encodings-required)
for the configuration's base model, and the
[token-label store](precompute-stores.md#token-labels-required-by-etebrendamodel-optional-otherwise)
if the configuration names one.

## Train

```bash
pdm run train <config.toml> <output.pt>
```

Writes a [checkpoint](../reference/checkpoint.md) to `<output.pt>` from the
best validation epoch, and stops early once `patience` epochs pass without
improvement.

| Need | Do |
| --- | --- |
| Train on a subset | `--limit N`. The first `N` documents of every split, so validation shortens with training; also truncates the training split the vocabulary is derived from, so two runs at different limits are different models |
| Profile a run | `-prof`, the only profiler switch |
| Keep the checkpoint with the MLflow run | `--log-checkpoint`; off by default because the state dict carries the frozen base model |
| Quieter console | `export D3TEXT_LOG_LEVEL=WARNING` before the command |
| Record the run | Set `MLFLOW_TRACKING_URI`; see [Track a run with MLflow](track-with-mlflow.md) |

The training configuration's keys are the [`ModelConfig`
fields](../reference/configuration.md#training-configuration);
`tests/best_config_so_far.toml` is a known-good starting point.

### Memory

Peak GPU memory follows the padded token count of a batch, not its document
count. Set `batch_max_chunks` (padded 512-token chunks per batch) rather than
`batch_size` to bound it; a document longer than the budget is batched
alone. `cpu_embeddings_cache_mb` in `config.toml` trades host memory for
the base model's forward pass on documents seen again; size it against
`free --si`, reckoning about 15 MB per cached document.

### Compilation

Training runs eager unless `D3TEXT_COMPILE` is set. Set to any non-empty
value, it compiles the model with `torch.compile` on GPUs Triton supports
(compute capability 7.0 and up):

```bash
D3TEXT_COMPILE=1 pdm run train <config.toml> <output.pt>
```

Why it is opt-in is in [runtime and
tracking](../explanation/runtime-and-tracking.md). A compile failure drops the
run back to eager execution, and the MLflow tag `compiled` records what the
epochs actually ran under.

## Evaluate

```bash
pdm run evaluate <config.toml> <output.pt>
```

`<config.toml>` is the configuration the checkpoint was trained with.
Prints one block per head: class scores, relation scores with `none`
excluded, span detection, and — when [external corpora are
configured](evaluate-linking.md) — the linking blocks. Every printed number
is also logged to MLflow when tracking is on; the keys are in the [metric
reference](../reference/metrics.md#evaluation).

There is no `--limit`: the checkpoint already records the vocabulary and the
training split is never read, so an evaluation machine needs only the test
split.

A warning that the token-label store's or the encodings store's digest
differs from the checkpoint's means the store was rebuilt since training;
the scores are still computed, but are not comparable to the training run's.

## Tune instead

For a sweep over several configurations, see [Tune hyperparameters](tune.md).
