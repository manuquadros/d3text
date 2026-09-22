# Configuration reference

Three things configure a run: the **training configuration** passed to
`train`, `tuning` and `evaluate` (per run); the **machine settings** in
`config.toml` at the repository root (per machine); and a handful of
**environment variables** (per invocation).

## Training configuration

A TOML file whose keys are the fields of `ModelConfig` (rendered [below](#api)
with their types and defaults). Unknown keys are rejected. Every field has a
default; the file may name only the ones that differ.

| Key | Meaning |
| --- | --- |
| `model_class` | `NERClassificationModel`, `BrendaClassificationModel` or `ETEBrendaModel` |
| `base_model` | Hugging Face id of the transformer. Must have an entry in `d3text.models.config.encodings`, which names its encodings file under `data/` |
| `seed` | Global RNG seed the run applies, recorded with its params. Default `42`; `tuning` reseeds from it per trial |
| `optimizer` | A key of `d3text.models.config.optimizers` |
| `lr` | Learning rate for the heads |
| `lr_scheduler` | A key of `d3text.models.config.schedulers`, or `""` for none |
| `dropout` | Dropout probability in the hidden block |
| `hidden_layers` | Widths of the hidden layers between the transformer and the heads |
| `normalization` | Normalization applied in the hidden block |
| `common_hidden_block` | Share one hidden block across heads rather than one per head |
| `separate_predicate_layer` | Give the relation head its own layer (`ETEBrendaModel`) |
| `biaffine_hidden_size` | Width of the biaffine relation scorer |
| `entity_logits_pooling` | How per-token class logits pool to one per document; see [pooling](../explanation/models.md#document-level-pooling) |
| `batch_size` | Documents per batch when `batch_max_chunks` is `0` |
| `batch_max_chunks` | Padded 512-token chunks per batch; `0` batches by document count instead |
| `num_epochs` | Maximum epochs |
| `patience` | Epochs without validation improvement before early stopping |
| `gradient_checkpointing` | Recompute the hidden block's activations in backward instead of keeping them |
| `unfrozen_top_layers` | Number of top transformer layers left trainable; `0` freezes the whole base model |
| `base_model_lr` | Learning rate for unfrozen transformer layers; `0` means the same as `lr` |
| `token_labels_store` | Path to a `precompute-token-labels` store. Non-empty adds the span tagger head. Required by `ETEBrendaModel` |
| `token_loss_weighting` | Tagger loss weighting scheme |
| `token_focal_gamma` | Focal exponent for `token_loss_weighting = "focal"` |
| `token_ambiguous_downweight` | Fraction of the tagger loss kept on a token the store flags `ambiguous`; `0` excludes it. Multiplies whatever `token_loss_weighting` assigns |
| `class_negative_abstention` | Abstain a document-level class negative wherever the token-label store's dictionary matched that type in the text. Requires `token_labels_store` |
| `class_negative_abstention_min_chars` | Minimum match length for the abstention above |
| `class_negative_abstention_min_chars_by_class` | Per-class override of the cutoff above, e.g. `{ bacteria = 20 }` |
| `class_negative_downweight` | Fraction of the class loss an abstained pair keeps; `0` drops it |
| `relation_loss_weighting` | Relation loss weighting scheme |
| `relation_focal_gamma` | Focal exponent for `relation_loss_weighting = "focal"` |
| `relation_label_smoothing` | Label smoothing on the relation head |
| `ramp_epochs` | Epochs over which `ETEBrendaModel` ramps the relation loss up to full weight; `0` means no ramp |

Two constraints are checked at load time: `class_negative_abstention` and
`model_class = "ETEBrendaModel"` each require a non-empty
`token_labels_store`.

A sweep configuration for `tuning` has the same keys, each holding a **list**
of values to sample from.

## Machine settings (`config.toml`)

Read by `machine_config()` from the repository root. The file is optional and
every key is optional; `config.toml.example` at the root documents each one.
Unknown keys are rejected.

| Key | Meaning |
| --- | --- |
| `cpu_embeddings_cache_mb` | Megabytes of token embeddings to cache in host memory; `0` disables the cache |
| `embeddings_store` | `precompute-embeddings` LMDB paths, keyed by the base model each was built from |
| `linking_corpora` | Directory holding the external corpora `evaluate` scores the dictionary linker against |
| `float32_matmul_precision` | As `torch.set_float32_matmul_precision` takes it |
| `cudnn_allow_tf32` | Let cuDNN use TF32 in convolutions |
| `expandable_segments` | Set the caching allocator's `expandable_segments:True` for the installed torch build |
| `tokenizers_parallelism` | Value written to `TOKENIZERS_PARALLELISM` |

The last four are process-global torch state, applied by
`d3text.runtime.configure()` when `train`, `tuning` or `evaluate` starts.
Importing the library applies none of them.

## The corpus files

Which files make up the corpus is `brenda_references`' own
`config.toml`, shipped inside the package:

```toml
[datasets.splits]
training = "training_data.csv"
validation = "validation_data.csv"
test = "test_data.csv"

[datasets.noise_pools]
psycholinguistics = "pmc_linguistics_articles.json"
enzyme_negative = "enzyme_negative_pool.json"
```

The names are resolved against the data directory `BRENDA_DATA_DIR` names,
not against the package. Both the split loaders and every `precompute-*`
command read this one table: the loaders take a path by key
(`split_path("validation")`, `noise_pool_path("enzyme_negative")`), and a
command given no `DATASET` argument reads `corpus_files()`, which is all
five in the order above. Naming files on the command line overrides the
default for that run; it does not extend it.

Both pools are in the default because every split is loaded with a block of
each appended, so a store built from the three CSVs alone holds none of
those documents.

## Environment variables

| Variable | Read by | Meaning |
| --- | --- | --- |
| `MLFLOW_TRACKING_URI` | `train`, `tuning`, `evaluate` | `http(s)://` address of an MLflow tracking server. Unset disables tracking entirely |
| `MLFLOW_EXPERIMENT_NAME` | same | Experiment to log runs under; unset derives one from the commit |
| `D3TEXT_LOG_LEVEL` | every command | Console verbosity, a `logging` level name; unparseable values fall back to `INFO` |
| `PYTORCH_CUDA_ALLOC_CONF` / `PYTORCH_HIP_ALLOC_CONF` | `runtime.configure()` | Allocator settings. A value already set wins over `expandable_segments` in `config.toml` |
| `TOKENIZERS_PARALLELISM` | `runtime.configure()` | Overwritten from `tokenizers_parallelism` in `config.toml` |
| `HSA_OVERRIDE_GFX_VERSION` | ROCm runtime | Present the GPU as another architecture when the installed torch ships no kernels for it; `runtime.configure()` warns when this is needed |
| `TORCH_FLAVOUR` | `pdm lock` only | Selects which torch index the lockfile is resolved against. Not read at install or run time |
| `BRENDA_DATA_REPO` | `brenda_references/scripts/pull_data.py` | Hugging Face dataset repository to fetch the corpus from |
| `BRENDA_DATA_DIR` | `brenda_references.data_paths` | Directory the corpus and splits are downloaded into and read from. Unset, an existing checkout that already holds them keeps being used, and anything else resolves to `brenda-references` under `XDG_DATA_HOME` (`~/.local/share` when that too is unset) |
| `BRENDA_HOST`, `BRENDA_USER`, `BRENDA_PASSWORD` | `brenda_references.db.get_engine()` | BRENDA MySQL mirror credentials. Data-collection scripts only; training and evaluation never open the database |

## API

::: d3text.models.config
