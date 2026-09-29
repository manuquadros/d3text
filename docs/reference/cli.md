# Command-line reference

The console scripts listed under `[project.scripts]` in `pyproject.toml`
ship with the package. Each is run as `pdm run <name> …` from a checkout, or
as `<name> …` from an environment the wheel is installed into.

`train`, `tuning`, `evaluate` and `infer` read the machine settings in
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
| `DATASET …` | Corpus files (`.csv` or `.json`) to encode; defaults to [the configured corpus](configuration.md#the-corpus-files) |
| `-f`, `--force-regenerate` | Re-encode documents the store already holds |
| `--s800 ROOT` | Also encode the S800 corpus found under `ROOT` |
| `--enzymener ROOT` | Also encode the enzymeNER corpus found under `ROOT` |

Naming no `DATASET` encodes the configured corpus — unless `--s800` or
`--enzymener` is given, which encodes that corpus alone. To encode both,
name the corpus files as well.

## `precompute-embeddings`

```
precompute-embeddings BASE_MODEL OUTPUT_PATH [DATASET …] [-f] [--batch_size N]
                      [--commit_every N] [--no_compress] [--map_size GIB]
                      [--stream_batch N] [--unfrozen_top_layers N [N …]]
                      [--no_aggregated]
```

Runs the frozen base model over every document and writes its rows into the
base model's [embeddings LMDB](stores.md#embeddings-precompute-embeddings-lmdb):
the aggregated last-layer rows, and one sub-database per boundary asked for,
all from one forward per window. Resumes: a document already in a
sub-database is skipped for that sub-database. The window is pinned to `utils.WINDOW_LENGTH`, the same
constant `precompute-encodings` cuts at, not a flag — a base model whose own
context is narrower is refused before the weights load.

| Argument | Default | Meaning |
| --- | --- | --- |
| `BASE_MODEL` | | Hugging Face model id to embed with |
| `OUTPUT_PATH` | | LMDB directory to write |
| `DATASET …` | [the configured corpus](configuration.md#the-corpus-files) | Corpus files to embed |
| `-f`, `--force-regenerate` | off | Re-embed documents already stored |
| `--batch_size` | 50 | Token windows per forward pass |
| `--commit_every` | 100 | Rows per LMDB commit, a document writing one row per sub-database it lacks |
| `--no_compress` | off | Store every sub-database's matrices uncompressed; see [the embeddings codec](../explanation/data.md#the-embeddings-codec) |
| `--map_size` | 1024 | GiB of address space to reserve for the LMDB |
| `--stream_batch` | [`corpus.STREAM_BATCH`][d3text.corpus.STREAM_BATCH] | Corpus rows read per Polars slice |
| `--unfrozen_top_layers` | none | One or more training runs' `unfrozen_top_layers`; each is a boundary written into its own sub-database. Each must lie between 0 and the model's encoder layers |
| `--no_aggregated` | off | Skip the aggregated sub-database; needs `--unfrozen_top_layers` |

`--batch_size`, `--commit_every` and `--stream_batch` must be positive.
`--map_size` must round to at least one byte.

## `inspect-embeddings`

```
inspect-embeddings PATH
```

Prints what an embeddings LMDB records and holds, writing nothing: the
env's provenance, then one line per sub-database with its document count,
how many frames are stored compressed and how many raw, the compression
ratio, and its size decompressed and on disk. An env in an older layout is
described rather than refused — its rows are listed as the main database,
beside the reason a store would refuse it.

A frame does not record its compression level. Raw frames are those written
with `--no_compress`, plus any whose data would not shrink at the writer's
level (`zstd` level 1, see
[the embeddings codec](../explanation/data.md#the-embeddings-codec)).
Every document's frame header is read, so the command takes about as long
as touching one page per document.

## `precompute-token-labels`

```
precompute-token-labels BASE_MODEL OUTPUT_PATH [DATASET …] [-e PATH] [-f] [-j N]
```

Places per-token distant-supervision targets for every document by matching
BRENDA's surface forms, and writes them to an HDF5 store keyed like the
encodings. Resumes: a document the store holds complete and still matching
its text and gold set is skipped; a partially written one, or one whose text
or gold set has since changed, is relabelled.

| Argument | Default | Meaning |
| --- | --- | --- |
| `BASE_MODEL` | | Model whose tokenizer the encodings were built with |
| `OUTPUT_PATH` | | HDF5 store to write; its directory must exist |
| `DATASET …` | [the configured corpus](configuration.md#the-corpus-files) | Corpus files to label; every file is scanned for organism names before any is labelled |
| `-e`, `--entity-tables` | the `documents.json` `brenda_references` is configured with | BRENDA's TinyDB dump, holding the entity tables |
| `-f`, `--force-regenerate` | off | Re-label documents the store already holds, and replace a store that would be refused (below) with a fresh one |
| `-j`, `--workers` | every logical CPU | Worker processes; `0` or `1` labels serially |

A store built under a different surface-form index, label space or labelling
rules, or under a different tokenizer, window geometry or layout version, is
refused rather than extended; `-f` discards it and labels afresh.

## `train`

```
train CONFIG OUTPUT [--limit N] [--log-checkpoint] [-prof]
```

Trains the model `CONFIG` describes and writes a
[checkpoint](checkpoint.md) to `OUTPUT`.

| Argument | Meaning |
| --- | --- |
| `CONFIG` | Training configuration (TOML; see [`ModelConfig`](configuration.md#training-configuration)) |
| `OUTPUT` | Path of the checkpoint to write, or, under `-prof`, the chrome trace to write instead (no checkpoint is written then) |
| `--limit N` | Use the first `N` documents of each split, the synthetic documents each one appends scaled by the same fraction; `0` or omitted means all. Also truncates the training split the vocabulary is derived from, so two runs at different limits build different vocabularies |
| `--log-checkpoint` | Upload the checkpoint to the MLflow run (hundreds of MB; off by default) |
| `-prof` | Run under the PyTorch profiler over real training steps, logging a `key_averages` table and exporting a chrome trace to `OUTPUT` (loadable in chrome://tracing or https://ui.perfetto.dev) |

Negative `--limit` is rejected.

## `tuning`

```
tuning CONFIG OUTPUT [--limit N]
```

Random search over the grid in `CONFIG`; every trial trains a model and
appends one row to the CSV at `OUTPUT`. The best configuration in `OUTPUT` is
kept in the TOML file of the same stem, which must not be `CONFIG`.

| Argument | Meaning |
| --- | --- |
| `CONFIG` | Sweep configuration (TOML; every key a list of `ModelConfig` values, except that `hidden_layers` lists widths to combine; see [sweep configuration](configuration.md#training-configuration)) |
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

There is no `--limit`: the checkpoint already records the vocabulary, and the
training split is not read.

## `infer`

```
infer CONFIG CHECKPOINT OUTPUT [DATASET …]
```

Runs the checkpoint over each named article and writes what it predicted,
one JSON object per line. Scores nothing and opens no MLflow run.

| Argument | Meaning |
| --- | --- |
| `CONFIG` | The training configuration the checkpoint was produced with |
| `CHECKPOINT` | Checkpoint written by `train` |
| `OUTPUT` | JSON Lines file to write |
| `DATASET …` | Corpus files (`.csv` or `.json`) to predict over; defaults to [the configured corpus](configuration.md#the-corpus-files) |

Every document is read from the encodings store `CONFIG`'s base model names,
so [`precompute-encodings`](#precompute-encodings) must have run over these
files first; a document the store holds no finished group for is not run, and
the count of those is logged at the end. A checkpoint with no span tagger
proposes no mention and no relation argument, and is refused.

Each record carries the document's pubmed id, its predicted spans, and its
predicted relations:

```json
{"document": "12345",
 "spans": [{"start": 10, "end": 26, "surface": "Escherichia coli",
            "entity_type": "bacteria", "entity_ids": ["bac1"]}],
 "relations": [{"predicate": "produces",
                "arguments": [["bac1"], ["enz1"]]}]}
```

`start` and `end` are character offsets into the document's assembled text —
the abstract and the body with their markup stripped, which is what the
tagger read. The `surface` is written beside them so a consumer that
assembles the text differently can find the span again.

`entity_ids` is every id the linker chose for the span, which is an empty
list where it chose none and `null` where the checkpoint carries no
surface-form index to link against — written before `train` recorded one, or
by a training run that could not build one (`linking_corpora.brenda_index`'s
warning names why). `relations` is
`null` wherever the relation head made
no claim about the document — the checkpoint carries none, or nothing in the
document grounded to a pair to put to it — which is not the same as the
empty list written for a document whose pairs the head labelled and called
every one of them null.

A relation's two `arguments` are the candidate id sets of the pair the head
labelled, in the head's own order, which carries no subject/object role.

## Environment variables read by the commands

See [environment variables](configuration.md#environment-variables).
