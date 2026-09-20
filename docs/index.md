# D3Text

Entity recognition, entity linking and relation extraction over the BRENDA
enzyme literature, with distant supervision from BRENDA's own entity tables.
A frozen transformer feeds three document-level heads — entity class,
entity ID, relation — plus a span tagger trained on dictionary-placed token
targets.

```
corpus (csv/json)
  ├─ precompute-encodings    → HDF5 of token ids                  (required)
  ├─ precompute-token-labels → HDF5 of per-token targets          (needed for the tagger)
  └─ precompute-embeddings   → LMDB of frozen activations         (frozen-trunk runs)
                                    ↓
                        train → checkpoint (weights + vocabulary)
                                    ↓
                                 evaluate
```

`tuning` runs the same training loop over a sampled hyperparameter grid.

## Where to go

**To get something done** — [How-to guides](how-to/install.md): install,
fetch the data, run a first training, precompute the stores, train and
evaluate, tune, track runs with MLflow, score the linker on external
corpora, run the checks, change a dependency. Each page is a task; start at
[Run a first training](how-to/first-run.md) if the pipeline is new to you.

**To look something up** — [Reference](reference/cli.md): every command and
flag, every configuration key and environment variable, the checkpoint and
store formats, the metric keys, the module map, and the API rendered from
the source.

**To understand why** — [Explanation](explanation/recurring-rules.md): the
reasoning behind the code — the traps, the measured trade-offs, and the
decisions a reader cannot recover from a diff. One page per area of the
pipeline. The docstrings in the source stay short on purpose; everything
longer lives here.

## Requirements

Python 3.12, [pdm](https://pdm-project.org), and about 2 GB of disk for the
corpus. A GPU is needed for a full training run; a `--limit` slice trains
on a CPU.
