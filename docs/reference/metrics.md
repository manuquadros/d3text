# Metric reference

Every metric a command logs to MLflow, with what one point measures and its
unit. The tables below are rendered from `d3text.metric_docs` at build time,
the same glossary a run posts as its MLflow description, so they cannot
disagree with what a run logs. `<split>` is `training`, `validation` or
`test`; `<objective>` and `<task>` are `entity`, `class`, `relation` or
`token`; `<type>` is an entity type name.

## Training and tuning

Logged by `train` (one run per training) and `tuning` (one run per trial).
Params hold the training configuration; the configuration file is attached
as an artifact.

Runs are named `<output file stem>@<short commit>` and tagged `stage`, `model`,
`base_model`, `git_commit` (suffixed `-dirty` when tracked files were
modified), `git_describe` (the nearest release tag and the commits since it),
`host`, `torch`, `accelerator`, `compiled`, and the `config.toml` settings the
run was launched under (`float32_matmul_precision`, `cudnn_allow_tf32`,
`expandable_segments`, `tokenizers_parallelism`, `cpu_embeddings_cache_mb`,
and `embeddings_store` and `linking_corpora` as whether each was set).
`tuning` adds `sweep=<config path>` and `trial=<n>`.

<!-- metric-glossary: train -->

## Evaluation

Logged by `evaluate`, in its own run tagged `stage=eval` and
`checkpoint=<path>`. Per-class tables are attached as text artifacts under
`test/`.

<!-- metric-glossary: eval -->

## Related

- [Track a run with MLflow](../how-to/track-with-mlflow.md)
- Why the keys are spelled the way they are:
  [Runtime and tracking](../explanation/runtime-and-tracking.md#experiment-tracking)
- API: [`d3text.tracking`, `d3text.metric_docs`](api/runtime-and-tracking.md)
