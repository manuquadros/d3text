# Precompute the stores

Goal: the on-disk artifacts a training run reads — the encodings every run
needs, the embeddings store every machine should hold, and the token-label
store the span tagger needs.

`D` below stands for `brenda_references/src/brenda_references/data`, where
[the splits](fetch-the-data.md) land. Flags are listed in the
[command-line reference](../reference/cli.md); on-disk layouts in the
[store reference](../reference/stores.md).

## Encodings (required)

```bash
pdm run precompute-encodings <base_model> data/<name>.hdf5
```

Naming no file encodes [the configured corpus](../reference/configuration.md#the-corpus-files):
the three splits and both noise pools. The pools are in that set because
every split is loaded with noise appended — `d3text.datasets.brenda` draws
450 off-domain and 150 enzyme-negative articles into training, 100 and 30
into validation, 50 and 15 into test — and a document the store does not
hold is logged as `No data for pmid <id>` and dropped from its batch. Name
files explicitly only to encode something other than that set; `--s800` or
`--enzymener` on its own encodes that corpus alone.

`<name>` must be what `d3text.models.config.encodings` maps `<base_model>`
to; `train` and `evaluate` look the file up by that name under `data/`. A
base model with no entry there needs one added before it can be used.

The command skips documents already in the store, so re-running after an
interruption continues. `-f` re-encodes everything. A store built for one
base model, window or stride refuses another; build a new file instead.

To encode an external corpus into the same store for
[linking evaluation](evaluate-linking.md), add `--s800 <root>` or
`--enzymener <root>`.

## Token labels (required by `ETEBrendaModel`, optional otherwise)

```bash
pdm run precompute-token-labels <base_model> data/token-labels.hdf5
```

The same corpus default applies, and it matters more here: a noise document
links to no entity, so its targets are `OUTSIDE` everywhere a surface form
does not match — exactly the negative evidence that holds the tagger's
false-positive rate down on off-domain text. Left out of the store, it is
masked out of the tagger loss instead, counted into one summary line per
training pass rather than a warning per document.

The entity tables default too, to the `documents.json` `brenda_references`
is configured with (`$D/documents.json`); name a different dump with
`-e <path>`.

Everything labelled in **one** invocation: the store's other-organism
dictionary is pooled from the files read, and a later run over a different
set is refused. A store built before the tokenizer stamp was recorded —
which includes every store built before the pools were the default —
carries no such stamp, so it is refused outright; delete it and run the
command again to build a fresh one. The command uses every CPU by default;
`-j 1` labels serially.

Point a training configuration at the result with
`token_labels_store = "data/token-labels.hdf5"`.

### When the store must be rebuilt

The store records the surface-form index and the labelling rules it was
built under. After any of these, delete the store and run the command
again — a plain re-run refuses to extend it:

- a change to the entity tables or the split files (the index moves);
- a change to `d3text.surface_forms` or `d3text.token_labels` that alters
  which strings match or how a match is labelled (the rules fingerprint
  moves);
- a store-format bump.

`train` and `evaluate` only read the store, so they warn rather than refuse:
`train` when the store's rules predate the running code, `evaluate` when the
store's digests differ from the ones the checkpoint recorded at training.

## Embeddings

Build this one too. A run with `unfrozen_top_layers = 0` freezes the whole
trunk, so the base model's output for a document cannot change between
epochs; without the store, that output is recomputed for every document the
CPU embeddings cache cannot hold, on every epoch and every validation pass,
and the run says so in a warning at start-up. The store replaces that
forward with a disk read.

Budget for the size rather than skipping the store over it: about 100 GiB
for the whole corpus at a 768-wide model.

```bash
pdm run precompute-embeddings <base_model> /data/d3text-embeddings
```

Same default again, though a miss costs only speed here: an unstored
document is embedded again on every pass rather than dropped, and the
served/missed counts logged at the end of each pass name the shortfall.

Lower `--batch_size` if the base model runs out of GPU memory. The command
resumes; `-f` re-embeds. The store refuses a second base model or window
outright, and `-f` is not a way past that — build a new store.

Point the machine at it in `config.toml`, keyed by the base model the store
was built from — every machine that trains sets this:

```toml
[embeddings_store]
"<base_model>" = "/data/d3text-embeddings"
```

A path with nothing there yet is built by the first run that needs it:
the run creates and stamps the store, and each document goes in the first
time the base model embeds it, so the first epoch pays the forwards and
later epochs, and later runs, read them back. The heads are fed each
document already rounded to the store's bf16, so the epoch that builds the
store trains on the same values the ones after it read. A run interrupted
part-way leaves a partial store that later runs only read;
`precompute-embeddings` completes it, skipping what it holds.

A base model with no entry, a store the run cannot open, or one whose rows
do not match the encodings, is disabled with one warning and the run
recomputes the embeddings. Each training and validation pass logs what the
store has served so far, so a run that opened one but is not being answered
by it shows up mid-flight rather than at process exit.

### Turning a store on is a re-baselining

A run that reads the store and a run that recomputes do not produce the same
numbers, and no configuration makes them. Two frozen runs differing in
nothing else were 0.25% apart on epoch 0's training loss and 22% apart by
epoch 1: a stored embedding and a live one differ slightly, and a training
trajectory amplifies that. [Why, and why it cannot be configured
away](../explanation/data.md#a-stored-embedding-and-a-live-one-are-not-the-same-number).

So adopt the store before the runs you mean to compare, not between them.
Results from before it are not a baseline for results after it, and two
machines that disagree about whether they have one cannot compare numbers
with each other.

The difference is the size of a change of seed, so **a store you already
have does not need rebuilding** — not for this, and not for the precompute
having since changed which dtype it computes in. Each store records the
precision it was built at, and reports it in the line it logs at the end of
a run.
