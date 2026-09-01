# D3Text

Document-level entity linking and relation extraction over the BRENDA corpus.

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
