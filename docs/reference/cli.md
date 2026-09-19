# Command-line reference

Six console scripts ship with the package (`[project.scripts]` in
`pyproject.toml`). Each is run as `pdm run <name> …` from a checkout, or as
`<name> …` from an environment the wheel is installed into.

`train`, `tuning` and `evaluate` need a **writable working directory**: they
import the BRENDA data layer, which opens `lpsn.log` in the current directory
at import time. The three `precompute-*` commands do not.

`train`, `tuning` and `evaluate` also read the machine settings in
[`config.toml`](configuration.md#machine-settings-configtoml) and apply them
to the process before anything else runs.

## `precompute-encodings`

```
precompute-encodings BASE_MODEL OUTPUT_PATH [DATASET …] [-f] [--s800 ROOT] [--enzymener ROOT]
```

Tokenizes each document into overlapping 512-token windows (stride 20) and
writes one HDF5 group per document. Resumes: a document already in the store
is skipped.

| Argument | Meaning |
| --- | --- |
| `BASE_MODEL` | Hugging Face model id whose tokenizer is used |
| `OUTPUT_PATH` | HDF5 file to write; created if absent |
| `DATASET …` | Split files (`.csv` or `.json`) to encode |
| `-f`, `--force-regenerate` | Re-encode documents the store already holds |
| `--s800 ROOT` | Also encode the S800 corpus found under `ROOT` |
| `--enzymener ROOT` | Also encode the enzymeNER corpus found under `ROOT` |

At least one of `DATASET`, `--s800` or `--enzymener` is required.

## `precompute-embeddings`

```
precompute-embeddings BASE_MODEL OUTPUT_PATH DATASET … [-f] [--batch_size N]
                      [--max_length N] [--commit_every N] [--map_size GIB]
                      [--stream_batch N]
```

Runs the frozen base model over every document and writes one compressed
token-embedding matrix per document into an LMDB. Resumes: a document already
keyed is skipped.

| Argument | Default | Meaning |
| --- | --- | --- |
| `BASE_MODEL` | | Hugging Face model id to embed with |
| `OUTPUT_PATH` | | LMDB directory to write |
| `DATASET …` | | Split files to embed |
| `-f`, `--force-regenerate` | off | Re-embed documents already stored |
| `--batch_size` | 50 | Token windows per forward pass |
| `--max_length` | the model's `max_position_embeddings` | Tokens per window; rejected above the model's limit |
| `--commit_every` | 100 | Documents per LMDB commit |
| `--map_size` | 256 | GiB of address space to reserve for the LMDB |
| `--stream_batch` | 1000 | Corpus rows read per Polars slice |

`--batch_size`, `--commit_every` and `--stream_batch` must be positive.
`--map_size` must round to at least one byte.

## `precompute-token-labels`

```
precompute-token-labels BASE_MODEL ENTITY_TABLES OUTPUT_PATH DATASET … [-f] [-j N]
```

Places per-token distant-supervision targets for every document by matching
BRENDA's surface forms, and writes them to an HDF5 store keyed like the
encodings. Resumes: a document the store holds completely is skipped; a
partially written one is relabelled.

| Argument | Default | Meaning |
| --- | --- | --- |
| `BASE_MODEL` | | Model whose tokenizer the encodings were built with |
| `ENTITY_TABLES` | | BRENDA's TinyDB dump (`documents.json`) |
| `OUTPUT_PATH` | | HDF5 store to write; its directory must exist |
| `DATASET …` | | Split files to label; every file is scanned for organism names before any is labelled |
| `-f`, `--force-regenerate` | off | Re-label documents the store already holds |
| `-j`, `--workers` | every logical CPU | Worker processes; `0` or `1` labels serially |

A store built under a different surface-form index, label space or labelling
rules is refused rather than extended.

## `train`

```
train CONFIG OUTPUT [--limit N] [--log-checkpoint] [-prof]
```

Trains the model `CONFIG` describes and writes a
[checkpoint](checkpoint.md) to `OUTPUT`.

| Argument | Meaning |
| --- | --- |
| `CONFIG` | Training configuration (TOML; see [`ModelConfig`](configuration.md#training-configuration)) |
| `OUTPUT` | Path of the checkpoint to write |
| `--limit N` | Train on the first `N` documents of the training split; `0` or omitted means all. Also determines the entity vocabulary the heads are sized to |
| `--log-checkpoint` | Upload the checkpoint to the MLflow run (hundreds of MB; off by default) |
| `-prof` | Run under the PyTorch profiler |

Negative `--limit` is rejected.

## `tuning`

```
tuning CONFIG OUTPUT [--limit N]
```

Random search over the grid in `CONFIG`; every trial trains a model and
appends one row to the CSV at `OUTPUT`.

| Argument | Meaning |
| --- | --- |
| `CONFIG` | Sweep configuration (TOML; every key a list of `ModelConfig` values) |
| `OUTPUT` | CSV to append trial results to |
| `--limit N` | As for `train`, applied to every trial |

## `evaluate`

```
evaluate CONFIG CHECKPOINT
```

Scores the checkpoint on the test split and prints one block of metrics per
head, plus the linking blocks when [external corpora](../how-to/evaluate-linking.md)
are configured.

| Argument | Meaning |
| --- | --- |
| `CONFIG` | The training configuration the checkpoint was produced with |
| `CHECKPOINT` | Checkpoint written by `train` |

There is no `--limit`: the checkpoint records the vocabulary its heads were
sized to, and the training split is not read.

## Environment variables read by the commands

See [environment variables](configuration.md#environment-variables).
