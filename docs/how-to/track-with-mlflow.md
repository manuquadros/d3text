# Track a run with MLflow

Goal: every training, tuning and evaluation run recorded on an MLflow
server, with its configuration, metrics and provenance.

Tracking is off unless asked for. With the variable below unset, `train`,
`tuning` and `evaluate` neither import mlflow nor change their behaviour.

## 1. Point the run at a server

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export MLFLOW_EXPERIMENT_NAME=my-sweep      # optional
```

The URI must be `http://` or `https://`. The dependency is `mlflow-skinny`,
which ships no local store: `file:` and `sqlite:` URIs are refused, with a
warning, and the run continues untracked. Setting up the server itself is
outside this project; any MLflow tracking server reachable over HTTP works.

Without `MLFLOW_EXPERIMENT_NAME`, runs from different commits land in
different experiments (`d3text_<short commit>`). Set it to gather a sweep
spread over several commits into one experiment.

## 2. Run as usual

```bash
pdm run train config.toml model.pt
```

Nothing else changes. What lands on the run — params, metrics, tags, the
configuration file as an artifact, the metric glossary as the run
description — is listed in the [metric reference](../reference/metrics.md).
`--log-checkpoint` also uploads the checkpoint; it is hundreds of MB.

## What to expect

- **A dead server does not kill the run.** A connection failure, an expired
  token or an incompatible client disables tracking for the rest of the
  process behind one warning. A multi-hour run never stops over a metric
  that could not be posted.
- **A run that raises is closed as `FAILED`**, so it is distinguishable in
  the UI from one that stopped early; the exception propagates unchanged.
- **`git_commit` ending in `-dirty`** means tracked files were modified when
  the run started; the run is not reproducible from that hash. No stamp at
  all means no repository was found (a non-editable install).
- **`git_describe` is the citable form of the same fact** — the nearest
  release tag, plus the commits since it and the hash when the run was not on
  the tag itself.
- **Renamed metric keys do not back-fill.** A chart spanning runs from
  before and after a rename needs both names.

## Add a metric

A key nobody documented cannot land: `tests/training/test_trainer.py` drives
a real `fit` and fails on any logged key that `metric_docs.describe` cannot
resolve, and `tests/test_metric_docs.py` does the same for the evaluation
helpers.

1. Log it through `d3text.tracking.log_metrics` where the number is
   computed. Name it `<pass>/<kind>_<what>` so the key says what quantity it
   is: `training/loss_relation`, `test/detection_recall`.
2. Add an `Entry` to the matching tuple in `d3text.metric_docs` —
   `_PER_EPOCH`, `_SUMMARY`, `_CONTEXT` or `_TEST` — with the regex the key
   matches, how the family is written in the table, what one point is, and
   its unit.
3. Run `pdm run pytest tests/training/test_trainer.py tests/test_metric_docs.py`.

The [metric reference](../reference/metrics.md) is rendered from those
entries at documentation build time, so it needs no separate edit.
