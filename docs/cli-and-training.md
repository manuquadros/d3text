# The command-line stages and the training loop

The pipeline CLI lives in `src/d3text/cli/`, not in `scripts/`. Every stage is a
`[project.scripts]` console script, so all six ship with the wheel: an entry
point *must* resolve inside the installed package, because a console script runs
with the venv's `bin/` as `sys.path[0]`, never the repo root.

## `precompute-token-labels`

One HDF5 store of [token targets](distant-supervision.md), keyed by pubmed id
and shaped like the encodings the tagger reads. Producing them needs the entity
tables, the corpus, and the tokenizer the encodings were built with — and
nothing else. In particular it needs **no encodings file**: re-tokenizing
`corpus.document_text` reproduces the stored `input_ids` element for element,
which is what makes the offsets addressable against them.

**A leaf, like the other two precompute commands.** It reads the corpus through
`d3text.corpus` and takes each document's gold entity set from the split frame's
own columns, rather than through `brenda_references.preprocess_labels`, which
would drag the BRENDA data layer — and its import-time write of an `lpsn.log`
into the working directory — into a command that only reads files it was handed.
`tests/cli/test_precompute_token_labels.py` pins that in a subprocess.

**Two passes over each corpus file.** The other-organism namespace has no table
in the BRENDA dump; the only place those names exist is inline in each
document's own `other_organisms` column, so the index cannot be built until
every file has been scanned for them. Pooling is the point rather than an
accident: a document naming an organism it was *not* annotated with is exactly
the case the abstain target exists for, and that mention is only recognizable
from some other document's naming of it.

Every path is validated before the entity tables and the tokenizer are read: the
tables are 1.1 GB and the index build scans every corpus file, so a mistyped
output directory must not be discovered after all of that.

A resumed store has its label space **checked rather than re-stamped**: its
existing targets were written under whatever space it records, and continuing
under a different one would leave a file whose halves mean different things. The
same argument refuses a store of an older layout, which holds codes with no
mention spans beside them. The answer to either is a regeneration.

## `precompute-embeddings`

### Validating the flags before the weights load

`window_size` cannot ask the tokenizer. `model_max_length` is a ~1e30 sentinel
whenever the tokenizer config declares no limit — which is the case for the
default base model — and `split_and_tokenize` pads *to* `max_length`, so that
sentinel asks for an impossible tensor. The position embeddings are the real
cap: a longer window indexes past the table.

`map_size_bytes` refuses a reservation that does not come out as at least one
byte, because neither of LMDB's two ways of dealing with one is any use. A
`map_size` of zero — what any reservation smaller than a byte truncates to,
either sign — reads as "keep the size this store already has", which for a new
store is LMDB's own 1 MiB default, so the run dies at the first write against a
budget nobody asked for. A negative one does raise, but with an `OverflowError`
naming neither the flag nor the value.

There is no floor above one byte there: a reservation is rounded up to whole
pages, and a floor at that point would have to guess at a document's embedded
size from a hidden width and a token count the flag parser never sees.
`check_map_size_for_one_document` closes that gap once the base model's config
is known, probing the map right after `lmdb.open` with a write sized at
`max_len * hidden_size` bf16 values — the uncompressed footprint of one full
window, a lower bound on what one document costs. The write lands in a
transaction that is aborted either way. A map that passes both checks and still
runs out mid-corpus stops and names the budget it hit, so nothing between "not
enough for one document" and "not enough for the corpus" fails silently.

`lmdb.open` accepts any reservation LMDB itself can mmap, however small —
`--map_size 1e-9` rounds to two pages and opens without complaint — so a map
merely too small for the data was previously caught nowhere until the first real
`put`, hours of GPU time later. The provenance record's own write catches the
smallest of these for free, but it is a few hundred bytes and most too-small
maps clear it easily.

`positive_int` rejects `--batch_size`, `--commit_every` and `--stream_batch`
before the tokenizer and base model load. Each fails differently, and only one
of them loudly:

- `--stream_batch` reaches `corpus.stream_rows`' `range(0, total, batch_size)`.
  Zero raises from `range` itself, but a **negative step yields nothing at
  all**, so the command loads the base model, iterates zero rows, writes zero
  documents, and reports `Done.` — a run that looks resume-safe and is actually
  empty.
- `--batch_size <= 0` reaches `embed_document`'s own batching.
- `--commit_every <= 0` makes the commit test true on every write, so the writer
  commits once per document instead of once per batch — a silent throughput
  cliff, not a wrong result.

None of the three is any use to catch after the weights are already on the
device.

### The store refuses a mixture outright

A pass that appends to a store built by another model, or with another window,
produces one LMDB holding two kinds of matrix that nothing downstream can
separate: the widths agree between encoders of the same hidden size, so the
heads simply train on both. An unstamped store that already holds documents is
refused for the same reason and not a weaker one — what wrote them is unknown,
so they cannot be shown to be this run's.

**`-f` is not a way past either**: it re-embeds the documents *these datasets*
name, and the ones they do not name would stay behind under the new stamp. A
rebuild is a new store.

`stored_keys` reads keys only: the values are the compressed embeddings, and
pulling those in just to test for presence would defeat the point of skipping
them. The provenance record rides along harmlessly, keyed on bytes no pubmed id
can spell.

### The writer thread

The writer is the queue's only consumer, so a producer waiting for room in a
full queue is really waiting for that thread; if it dies without saying so, the
wait never ends. Every exit therefore goes through `stop_evt`, and every failure
is recorded in `WriterState` for `main` to raise after the join — a thread has
no return value and an exception raised inside one is invisible to the caller.
Whether the writer can carry on draining its queue (a full map) or cannot
(anything else), it is `stop_evt` that tells a producer to stop waiting.

`put_or_stop` breaks the wait into timeouts, because a plain `put` on a full
queue waits for a consumer that may already be gone and setting an event does
not wake it.

**A delete travels the queue too.** A `None` value asks for the key to be
deleted rather than stored. Removing a stale entry has to go through the writer
rather than be done where it is decided, because LMDB allows one writer at a
time: a second write transaction opened while the writer holds its own would
wait for a commit that only arrives after `commit_every` further documents,
which only the producer that is now waiting can supply. A delete also has to
grow the map the same way a put does — LMDB rewrites the pages it touches rather
than editing them in place — so `store_full` names the operation that ran out.

## `evaluate`

A recorded vocabulary is authoritative and the training split is **not loaded at
all**: it existed only to derive the entity columns, and those are already
known. That is also what makes `--limit` irrelevant — the flag resized the
entity head by resizing the split it was derived from, which is how a checkpoint
came to be unloadable against the very corpus it was trained on.

Without one there is no recovering the order the run used, so the old behaviour
stands: rebuild it from the training split and warn that the result is a
reconstruction, valid only if `--limit`, `noise=` and the corpus itself all
match the training run.

## The training loop

`Model` computes losses; `Trainer` decides what is done with them. The split is
what lets a model be constructed, loaded and evaluated without carrying an
optimizer, a best-epoch snapshot and a stop counter around with it.

`d3text.training` deliberately re-exports nothing. `d3text.models.base` imports
`d3text.training.update` — `Model.run_epoch` stays on the model and is handed
the update to apply — so a re-export of `.trainer`, which imports
`d3text.models.base` in turn, would close that cycle at import time. The same
argument is why `BatchUpdate` is its own module rather than part of `trainer`.

**A `Trainer` is single-use.** The optimizer, scheduler and gradient scaler are
built once in `__init__` and never rebuilt, so a second `fit()` would resume
their state — including the LR schedule — rather than start a fresh run.

`fit` **returns** the parameters a checkpoint should be written from — the best
epoch's, copied while that epoch was current — or `None` when the run kept no
snapshot: `save_checkpoint` off, or no validation data to choose a best epoch
by. Handing them over frees the caller from knowing that `fit` also loads the
snapshot into the model on its way out; a caller that saved the model instead
was relying on that mutation without naming it, and nothing at the call site
would have noticed it stop happening. The best validation loss is on
`best_val_loss`, where `tune` reads it.

`_early_stop` carries the epoch rather than letting `fit` track it, so the epoch
and the loss it belongs to are written by the same comparison; two comparisons
in two places is how `best_epoch` came to disagree with `best_val_loss`.

`_cpu_state_dict` exists because `deepcopy(state_dict())` preserved each
tensor's device, so on CUDA the best-epoch snapshot was a second resident copy
of the whole model — the frozen base model included, 0.4 GiB of it — pinned for
the rest of the run and briefly doubled at every improving epoch, since the new
copy is built before the old one is dropped. Nothing ever reads it on-device: it
is `torch.save`d, or loaded back once at convergence, and `load_state_dict`
copies each tensor to its parameter's own device either way. `copy=True` is
load-bearing, and only on CPU runs: `.to("cpu")` on a tensor already there
returns *self*, which would leave the snapshot aliasing the live parameters.

### The weight update

Loss scaling exists only to keep fp16 gradients out of the subnormal range, so
it is enabled for float16 alone: bfloat16 has float32's exponent range and
nothing to rescue, while the scaler still costs a scale multiply, an `unscale_`
division over every gradient, and a `.item()` inside `step` that synchronises
the host against the device on every optimizer step. A disabled scaler passes
`scale`, `unscale_`, `step` and `update` straight through, so the call is the
same code either way. The default is the conservative one: a caller that does
not say which dtype it autocasts to gets the scaling.

`clip_grad_norm_` returns the norm it measured *before* clipping, which is the
only informative one — after clipping it is `GRAD_CLIP_NORM` by construction on
every step that clipped at all. The sum is kept on the accelerator and read once
per epoch: an `.item()` per optimizer step would serialise the training loop
against the device.

`grad_norm_metrics` is empty when no optimizer step ran — a validation-only
pass, or a model whose `run_epoch` never applies the update — so nothing logs a
gradient statistic for an epoch that computed no gradients. **A clipping rate
pinned at 1.0 is the signal that `GRAD_CLIP_NORM` is doing the optimising rather
than the learning rate.**

## Machine configuration

`MachineConfig` holds per-machine settings read from the repo-root
`config.toml`. The runtime fields are process-global torch/allocator settings,
applied by `runtime.configure()` at script start-up rather than at import.
`machine_config` falls back to a zero-cache default when the file is absent — a
fresh checkout, or CI — so importing `d3text.models` never fails on a missing,
uncommitted config.

`load_tuning_config`'s `rng` is injectable so a sweep can be replayed exactly;
the default draws from a fresh `Random`, which leaves successive sweeps
independent of each other without reading or advancing the process-global
`random` state.

## Window aggregation

`aggregate_embeddings` stitches overlapping windows back into one token
sequence. Within the overlap between two regions, a token at position *n* (with
*n* = 0 at the first token of the overlap) goes to the first sequence while
*n* < stride/2 and to the following one otherwise, which selects the embedding
that saw the most balanced context — preceding and following — for each token.

`load_fast_tokenizer` rejects a slow tokenizer where the base model is named
rather than deeper in the pipeline: `AutoTokenizer.from_pretrained` may return a
SentencePiece-backed one, and both `split_and_tokenize` and `embed_document`
depend on fast-only features (`return_overflowing_tokens`, `offset_mapping`).

`Token.candidate_labels` is filled only when a span was matched equally well by
more than one wordlist, in which case `prediction` carries
`dict_tagger.AMBIGUOUS`. A span two entity types fit is not evidence for either,
so a consumer building training targets has to be able to recognise it and drop
it — which recording no match at all would not allow. An unambiguous match
leaves the set empty: the label is in `prediction`, and a label stored twice is
a label that can disagree with itself.

::: d3text.cli.precompute_token_labels

::: d3text.cli.precompute_embeddings

::: d3text.cli.evaluate

::: d3text.training.trainer

::: d3text.training.update

::: d3text.models.config
