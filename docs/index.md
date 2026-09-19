# D3Text

Entity recognition, linking and relation extraction over the BRENDA corpus.

These pages carry the **reasoning** behind the code — the traps, the measured
trade-offs, and the decisions that a reader cannot recover from a diff. The
docstrings in the source stay short on purpose: a one-line summary, the sphinx
parameter fields, and at most a sentence where a real invariant would otherwise
be invisible. Everything longer lives here.

Each page ends with the mkdocstrings API reference for the modules it explains.

## Where to start

| Page | What it covers |
| --- | --- |
| [Data path](data.md) | Reading the corpus, the encodings and embeddings stores, provenance, batching |
| [Distant supervision](distant-supervision.md) | The three-way token targets, the label space, mention spans, the label store |
| [Surface forms](surface-forms.md) | The BRENDA dictionary: which forms carry an ID, the case policy, the fuzzy layer |
| [Schema and checkpoints](schema-and-checkpoints.md) | `Schema`, `Vocabulary`, the checkpoint format, the linker seam |
| [Dataset and factory](dataset-and-factory.md) | Indexing the splits, building a model from a config, the dictionary tagger |
| [Models](models.md) | Pooling, the loss divisors, mixed precision, the three concrete models |
| [Evaluation](evaluation.md) | Detection, linking, and why the ignore set is masked |
| [CLI and training](cli-and-training.md) | The precompute commands, the epoch schedule, the weight update |
| [Runtime and tracking](runtime-and-tracking.md) | Process-global torch state, console logging, MLflow |

## The pipeline

```
corpus (csv/json)
  ├─ precompute-encodings    → HDF5 of token ids
  ├─ precompute-embeddings   → LMDB of frozen activations   (optional)
  └─ precompute-token-labels → HDF5 of per-token targets    (optional)
                                    ↓
                        train → checkpoint (.pt + vocabulary)
                                    ↓
                                 evaluate
```

`tune` runs the same training loop over a sampled hyperparameter grid.

## The seams around the CLI

CLI modules are glue; everything reusable they used to hold now lives in a
few small library modules, placed so the import graph stays honest:

| Module | Holds | Imports |
|---|---|---|
| `corpus.py` | `document_text`, `stream_rows` — the corpus reader | leaf (polars, xmlparser) |
| `embeddings_store.py` | `tensor_to_bytes` / `bytes_to_tensor` — the LMDB codec | leaf |
| `factory.py` | `build_model`, `fix_keys_hook`, `model_size_mb` | `d3text.models` **and** `d3text.data` |
| `progress.py` | `batch_progress` — the epoch/eval bar | leaf (tqdm, torch) |
| `vocabulary.py` | `Vocabulary` — the heads' column order | leaf (torch, schema) |
| `checkpoint.py` | `save()`, `load()` — the on-disk contract | `d3text.vocabulary` |
| `logs.py` | `configure()`, `TqdmLoggingHandler` — the console handler | leaf (tqdm) |
| `runtime.py` | `configure()`, `is_triton_compatible()` | `d3text.logs`, `d3text.models.config` |
| `metric_docs.py` | `Entry`, `describe()`, `glossary()` — what each metric's y-axis measures | leaf |
| `tracking.py` | `run()`, `log_params`, `log_metrics`, `log_artifact`, `set_description` | `d3text.metric_docs` (mlflow lazily) |
| `training/trainer.py` | `Trainer` — `fit`, `_setup`, `_early_stop`, `_validate` | `d3text.models`, `d3text.tracking` |
| `training/update.py` | `BatchUpdate`, `GRAD_CLIP_NORM` | leaf |

`document_text` is the one place a corpus row becomes a string — nulls
dropped (a missing cell is `None`/`nan`, never `""`; `str(nan)` is the
*truthy* `"nan"`, how the word "nan" got tokenized into 3% of the training
encodings), halves joined with a newline, JATS tags stripped. Both precompute
commands go through it; when each had its own copy they disagreed on both
decisions.

`factory.py` sits above `d3text.models`, not inside it, because resolving a
dataset into constructor arguments needs `d3text.data`, and `d3text.models`
must stay importable without the BRENDA layer (and its `lpsn.log` import-time
write) coming along.

## Two rules that recur

**A store must say what produced it.** A mismatched tokenizer, base model or
window produces artifacts of exactly the right shape and dtype over the wrong
vocabulary or representation space, so every store stamps its provenance and
every reader checks it. The same argument puts the label vocabulary inside the
checkpoint.

**The divisor is the weight sum, not the element count.** Every masked loss
divides by what it actually read. Dividing by the whole population instead
scales each real element's loss by the share of the batch that happened to be
masked — the dilution the mask exists to remove, reintroduced by the reduction.
