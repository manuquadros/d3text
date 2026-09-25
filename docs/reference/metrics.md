# Metric reference

Every metric a command logs to MLflow, with what one point measures and its
unit. The tables below are rendered from `d3text.metric_docs` at build time,
the same glossary a run posts as its MLflow description, so they cannot
disagree with what a run logs. `<split>` is `training`, `validation` or
`test`; `<objective>` is `class`, `relation` or `token`; `<task>` is
`class`; `<type>` is an entity type name.

## Training and tuning

Logged by `train` (one run per training) and `tuning` (one run per trial).
Params hold the training configuration; the configuration file is attached
as an artifact.

A training run is named `<output file stem>@<short commit>`, a tuning trial
`<model_class>-<trial>@<short commit>` with the trial zero-padded to three
digits. Both are tagged `stage`, `model`, `base_model`, `git_commit`
(suffixed `-dirty` when tracked files were modified), `git_describe` (the
nearest release tag and the commits since it), `host`, `torch`,
`accelerator`, `accelerator_count` (on a GPU only), `compiled`, and the `config.toml` settings the
run was launched under (`float32_matmul_precision`, `cudnn_allow_tf32`,
`expandable_segments`, `tokenizers_parallelism`, `cpu_embeddings_cache_mb`,
and `embeddings_store` and `linking_corpora` as whether each was set). As
the run closes it is also tagged `embeddings_store_lookups` (documents looked
up in an embeddings store) and `embeddings_store_coverage` (the share of them
the store answered, `0.0000` for a run that computed every embedding itself).
`tuning` adds `sweep=<config path>` and `trial=<n>`.

<!-- metric-glossary: train -->

## Evaluation

Logged by `evaluate`, in its own run tagged `stage=eval`,
`checkpoint=<path>`, and `checkpoint_token_labels` and `checkpoint_encodings`:
whether the label store and encodings store this run reads are the ones the
checkpoint recorded (`matched`, `mismatched` or `unrecorded`;
`checkpoint_token_labels` is `unused` when neither side has a label store).
It carries the same provenance and machine tags as a training run. Per-class tables are attached as text artifacts under
`test/`.

<!-- metric-glossary: eval -->

## Related

- [Track a run with MLflow](../how-to/track-with-mlflow.md)
- Why the keys are spelled the way they are:
  [Runtime and tracking](../explanation/runtime-and-tracking.md#experiment-tracking)
- API: [`d3text.tracking`, `d3text.metric_docs`](api/runtime-and-tracking.md)
