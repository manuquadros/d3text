# Run a first training

Goal: a checkpoint trained on a slice of the corpus, evaluated on the test
split, in one sitting. Every command runs from the repository root.

Before starting: [install](install.md) and [fetch the data](fetch-the-data.md).

## 1. Tokenize the corpus

```bash
pdm run precompute-encodings michiyasunaga/BioLinkBERT-base \
    data/biolinkbert-base-zstd-22-encodings.hdf5
```

With no file named, the command reads
[the configured corpus](../reference/configuration.md#the-corpus-files) —
the three splits and both noise pools, which is what a training run loads.
The output name is not free: `train` finds the encodings of a base model
under `data/` by the name `d3text.models.config.encodings` maps it to. The
command resumes, so an interrupted run can be repeated as is.

## 2. Place the token targets

```bash
pdm run precompute-token-labels michiyasunaga/BioLinkBERT-base \
    data/token-labels.hdf5
```

Same default, and everything on one invocation: the other-organism names
are pooled from every file read, and a store resumed over a different set
is refused. A document left out of the store is masked out of the tagger
loss, with one warning naming it.

## 3. Write a training configuration

Save as `first.toml`:

```toml
model_class = "ETEBrendaModel"
base_model = "michiyasunaga/BioLinkBERT-base"
token_labels_store = "data/token-labels.hdf5"
num_epochs = 3
batch_max_chunks = 64
```

Every other field keeps its default; the full list is in the
[configuration reference](../reference/configuration.md#training-configuration).
`batch_max_chunks` bounds a batch by padded 512-token chunks rather than
by document count, which is what keeps peak memory predictable on a corpus
whose documents span a thirtyfold length range.

## 4. Train on a slice

```bash
pdm run train first.toml first.pt --limit 250
```

`--limit 250` trains on the first 250 training documents, and validates
on the first 250 of the validation split. It also sets the
entity vocabulary the heads are sized to, so it is part of the run's
identity and is recorded in the checkpoint. The command writes
`first.pt` from the best validation epoch.

## 5. Evaluate

```bash
pdm run evaluate first.toml first.pt
```

Prints one block per head. There is no `--limit`: the checkpoint carries
the vocabulary, and the training split is not read.

## What to read next

- [Train and evaluate](train-and-evaluate.md) — every flag, and what a full
  run needs.
- [Track a run with MLflow](track-with-mlflow.md) — to keep the numbers.
- [The pipeline, explained](../explanation/data.md) — why the stores exist.
