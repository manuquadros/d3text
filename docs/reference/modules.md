# Module map

Where each piece of the library lives and what it imports. "Leaf" means the
module imports nothing else from `d3text` beyond what the row names, so it
can be imported without the BRENDA data layer (and its import-time
`lpsn.log` write) coming along. `d3text.constraints` and `d3text.schema`
are leaves imported almost everywhere and are not repeated in the column.

## Library

| Module | Holds | Imports |
| --- | --- | --- |
| `d3text.schema` | `Schema`, `BRENDA_SCHEMA` — entity types, ID prefixes, relation types | leaf |
| `d3text.corpus` | `document_text`, `document_fields`, `stream_rows`, `stream_documents` — the corpus reader | leaf (polars, xmlparser) |
| `d3text.surface_forms` | `build_index`, `SurfaceFormIndex`, `index_digest` — the dictionary | `d3text.schema` |
| `d3text.token_labels` | `find_mentions`, `document_token_labels`, the label store | `d3text.schema`, `d3text.surface_forms` |
| `d3text.negative_screen` | Screening a candidate negative document | `d3text.corpus`, `d3text.surface_forms`, `d3text.token_labels` |
| `d3text.encodings_store` | Provenance and content digest of the encodings HDF5 | leaf |
| `d3text.embeddings_store` | `tensor_to_bytes` / `bytes_to_tensor` — the LMDB codec | leaf |
| `d3text.vocabulary` | `Vocabulary` — the class head's column order | `d3text.schema` |
| `d3text.checkpoint` | `save`, `load` — the on-disk checkpoint contract | `d3text.vocabulary` |
| `d3text.linking` | `Linker`, `DictionaryLinker` — span to entity IDs | `d3text.surface_forms`, `d3text.token_labels` |
| `d3text.models` | `Model` and the three concrete models, heads, losses | torch, transformers; no data layer |
| `d3text.models.config` | `ModelConfig`, `MachineConfig`, `machine_config` | leaf |
| `d3text.datasets.brenda` | `brenda_dataset` — the split adapter | `brenda_references` (data layer) |
| `d3text.datasets.{s800,enzymener,expasy,nlp4pheno,culture_numbers}` | External corpus loaders | `d3text.surface_forms` |
| `d3text.data.data` | `BrendaDataset`, `TokenBudgetBatchSampler`, `collate_documents` | `brenda_references` (data layer) |
| `d3text.factory` | `build_model`, `fix_keys_hook`, `model_size_mb` | `d3text.models` **and** `d3text.data` |
| `d3text.training.trainer` | `Trainer` — `fit`, early stopping, best-epoch snapshot | `d3text.models`, `d3text.tracking` |
| `d3text.training.update` | `BatchUpdate`, `GRAD_CLIP_NORM` | leaf |
| `d3text.mention_metrics` | Detection and linking scores over mentions | `d3text.token_labels` |
| `d3text.identifier_bridge` | BRENDA entity to outside identifier tables | leaf |
| `d3text.linking_eval` | Scoring the linker against an external corpus | `d3text.identifier_bridge`, `d3text.linking`, `d3text.mention_metrics` |
| `d3text.annotation_hub` | Predictions mapped onto annotation-hub's `POST /save/` object | `d3text.corpus`, `d3text.identifier_bridge` |
| `d3text.linking_corpora` | Finding and scoring every configured corpus | `d3text.linking_eval`, `d3text.datasets`, `d3text.corpus` |
| `d3text.progress` | `batch_progress` — the epoch and evaluation bar | leaf (tqdm, torch) |
| `d3text.logs` | `configure`, `TqdmLoggingHandler` — the console handler | leaf (tqdm) |
| `d3text.runtime` | `configure`, `compile_model`, GPU checks | `d3text.logs`, `d3text.models.config` |
| `d3text.metric_docs` | `Entry`, `describe`, `glossary` — the metric glossary | leaf |
| `d3text.tracking` | `run`, `log_params`, `log_metrics`, `log_artifact` | `d3text.metric_docs` (mlflow lazily) |
| `d3text.excepthook` | Console-script exception hook that keeps `add_note` notes | leaf |
| `d3text.constraints` | `Annotated` range aliases | leaf |
| `d3text.utils` | Window aggregation, tokenizer loading | torch, transformers |

## Commands

`d3text.cli.*` holds one module per console script; see the
[command-line reference](cli.md). `scripts/` at the repository root holds
ad-hoc scripts that are not part of the package: embedding generation, the
gold-data bridge builders, the linking scorers, and frozen experiment
directories.

## Sub-packages

| Package | Source | Role |
| --- | --- | --- |
| `brenda_references` | `brenda_references/` in this repository (editable) | Training, validation and test DataFrames; relation preprocessing; the data downloader |
| `xmlparser` | git dependency | PubMed XML parsing |
| `d3types` | git dependency (transitive) | Shared Pydantic types |
| `lpsn_interface` | git dependency (transitive) | LPSN taxonomy adapter; opens `lpsn.log` in the working directory at import |
| `apiadapters` | git dependency (transitive) | NCBI/Entrez adapters |
