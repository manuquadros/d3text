# The data path: corpus, stores, dataset

Three artifacts stand between the raw corpus and a training batch: the
csv/json corpus itself, the `precompute-encodings` HDF5 of token ids, and the
optional `precompute-embeddings` LMDB of frozen activations. Each is read by a
module that also owns the rule for reading it, so the stages cannot describe the
same document differently.

## Reading the corpus

`d3text.corpus` is **deliberately a leaf**. Importing `d3text.data` drags in the
whole BRENDA stack (`brenda_references` → `d3types` → `lpsn_interface`, and
their database and API dependencies) to read csv and json rows, which need none
of it. The precompute commands are the only d3text commands that do not already
pay that import cost; reading the corpus must not be what makes them.

**`document_text` is the one place a corpus row becomes a string.** The
precompute commands are the only readers of the corpus, and the first two each
got half of it right, which is worse than either being wrong: the encodings path
stripped the XML tags but turned a missing abstract into the literal string
`"nan"`; the embeddings path handled the missing abstract but fed raw JATS
markup to the transformer.

Two traps sit inside it, and both are the same trap:

- A missing cell is `None` from polars and `float("nan")` from pandas. The
  obvious spelling — `str(value) or ""` — is a bug, because `str(nan)` is
  `"nan"`, a *truthy* string, so the fallback never fires and the word "nan"
  gets tokenized into the document.
- Whitespace is not content. A body that is markup wrapping nothing but
  newlines strips to a *truthy* string of indentation, so every caller's `if not
  text` check waves it through and the tokenizer returns a window holding
  `[CLS]` and `[SEP]` and no token of the document at all.

`stream_documents` reads a row's **gold entity set** off the same file, rather
than borrowing `brenda_references.preprocess_labels`: that function is in the
trunk, and a labelling command that reached it would pay the whole BRENDA stack
to read four columns of a csv it is already streaming. What it does is small —
parse the cell, prefix each numeric ID with its type's tag — and the tags come
off the schema, so the two spellings cannot drift. `stream_rows` is the same
without the annotations, which the two encoding commands do not need and would
pay four `literal_eval`s per row for.

The split frames store Python `repr`s, not JSON — `{'2785': 'Jaculus
orientalis'}` — which is what `brenda_references` reads them back with. The
element type is not knowable statically: `enzymes` and `strains` are lists of
numeric IDs, `bacteria` and `other_organisms` mappings from an ID to the name
this document gave it, and iterating a mapping yields its keys, which covers
both shapes.

`CorpusDocument` carries `other_organisms` separately from `entity_ids` because
it is the one namespace whose *names* exist nowhere else: the BRENDA dump has no
other-organisms table, so a surface-form index over that namespace can only be
built by pooling this column across the whole corpus. `other_organism_names` is
a separate pass for that reason — the index has to exist before any document can
be labelled, and `stream_documents` would strip a gigabyte of JATS markup to
hand back a column already in the file.

The PMC noise dump carries none of the schema's entity columns — it is
unannotated text — and a document with no gold entities is exactly what that
means, so a missing column contributes an empty set rather than raising.

### Streaming and the ReDoS guard

`_slices` reads lazily on purpose: the corpus is ~1 GB of json and every command
consumes it one document at a time. `collect_batches` is what keeps it lazy
without re-scanning — unlike `lazy.slice(start, batch_size).collect()`, which
parses the file from the top for every batch it produces, since CSV and NDJSON
have no random access and a scan cannot seek to `start`.

nltk 3.10 runs every tokenizer pattern under a *wall-clock* timeout
(`nltk.redos`, five seconds by default, read off the module global at match
time). `remove_tags`' pattern is `xmlparser`'s own hardcoded constant and strips
linearly — no input reaches the bound by matching — so a guard that fires there
is timing the host, and a few seconds of write-back stall during an 80 GiB
precompute pass is enough to end a multi-hour run on a match costing five
milliseconds of CPU. `_remove_tags` therefore grants nltk's documented exemption
for a trusted pattern, per call and restored on the way out: importing the
module changes nothing, and every caller-supplied pattern elsewhere — the
tagger, the chunk rules, `tgrep`, which are what the five seconds exist for —
keeps its guard. Assigning the global at import is what got `0062e89` reverted
in `23f1503`. Its lock is for the restore, not the match: two overlapping calls
would interleave their save/restore and could leave the exemption behind for the
whole process.

When the guard does fire, `_text_or_drop` drops the row rather than the pass:
both precompute commands read the whole corpus in a single multi-hour run, and a
document that cannot be stripped is one the consumer already knows how to be
without — absent, which is what a document the corpus never had looks like too.
The guard's exception is the *builtin* `TimeoutError` (nltk subclasses nothing),
and `TimeoutError` is an `OSError`, so the catch is kept around the one call,
which does no I/O; a future I/O timeout raised anywhere else still ends the
stream loudly instead of shrinking it silently.

`CorpusStream` counts the rows it dropped. The streaming functions return a row
count and then an iterator that may drop rows, so the count and the stream can
disagree; without `dropped` the stream shrinks silently and the only signal is
one warning per drop in the middle of a multi-hour pass's log.

## Provenance: what a store cannot tell you from its shapes

Both stores record what produced them, and for the same reason: **the only
mistake the geometry cannot catch is the one worth catching.**

A mismatched tokenizer produces an `input_ids` array of exactly the same shape
and dtype as the right one, only over the wrong vocabulary. 768 dimensions are
768 dimensions whichever encoder emitted them, so a store built with one base
model and read by another hands the heads a second representation space with no
shape to fail on. And the aggregated row count is `sum(L_i) - stride*(N-1)`
while `sum(L_i)` is `T + stride*(N-1)`, so it comes to `T` for any
`max_length` — measured identical at 512, 384, 256, 128 and 64 — which means a
document split at one window and resumed at another leaves no trace a shape
check can catch either.

So both stores stamp the base model, the window and the stride, and both check
them on the read side.

### The two stores answer an unstamped file differently

`d3text.encodings_store.record_provenance` **warns and stamps**. An unstamped
store that already holds documents predates the stamp existing at all, and every
encodings file `precompute-encodings` had ever written is exactly that on this
build's first run against it; refusing them would turn every one of them
unresumable in one release. The groups already there stay unattributed, but the
run proceeds and every group from here on is attributed.

`precompute_embeddings.record_provenance` **refuses**, because the LMDB is two
orders of magnitude larger to rebuild.

Both refuse a store that already recorded *another* geometry: a pass that
appends produces one file holding two kinds of record that nothing downstream
can separate, and the mixture, once written, is indistinguishable from a store
that agrees with itself.

On the read side, `None` means a store written before provenance was recorded —
not the same as a store written by the wrong model, and not distinguishable from
one either. What it means is that nothing on disk attributes those records to
anything. A record that is *present but unreadable* — a future format, or a
damaged one — raises instead: reading it as though it were unstamped would hide
that behind the friendlier of the two diagnoses.

`ProvenanceError` is raised rather than warned about because the reader has no
safe answer to give; the caller decides whether a store it cannot attribute is
worth running without (`models.base.embeddings_store` disables it and
recomputes) or worth stopping for.

`BrendaDataset._check_encodings_provenance` deliberately does **not** compare
the stamped `max_length`. It is the one field of the geometry the aggregation
never consults: windows are stitched off the attention mask, so a store built at
a shorter window still reconstructs each document token-for-token, and one built
past the base model's position count fails loudly in the embedding layer rather
than quietly.

## The embeddings codec

`precompute-embeddings` stores one compressed token-embedding matrix per pubmed
id. `tensor_to_bytes` and `bytes_to_tensor` are the two halves of that store's
contract; keeping them in one place is what makes it a contract rather than two
independent guesses at a byte layout.

Nothing else may reach for `blosc2` directly — `blosc2.unpack_array` segfaults
on a blob it did not write rather than raising, so the magic-number check in
`bytes_to_tensor` is what stands between a stale store and a downed process. A
blob written by the previous fp16 `pack_array` format has the same itemsize as
this one, so without a magic to reject it, it would decode into a plausible
matrix of garbage.

**The stored dtype is bf16, and the codec is zstd level 5 behind a byte
shuffle.** Both were measured (`scripts/benchmarks/bench_codecs.py`, tabulated
in `design/perf_baseline.md`). Two results drive them:

- These activations are very nearly incompressible losslessly. Every lossless
  combination of codec, filter and level lands between 1.00× and 1.17×, because
  the low mantissa bits are noise no entropy coder can model. Storing bf16
  instead of fp16 spends two of those bits and gets 1.42×, which is 100.8 GiB
  rather than 121.9 for the whole corpus. It is the only near-lossless lever
  there is; the codec knobs are not one.
- `blosc2.pack_array` is 3.8× slower than `compress2` at identical settings, and
  pack_array-at-zstd9 was 72× slower than what is used here.

bf16 costs precision, not range: it keeps fp32's exponent and drops mantissa
bits, so a value fp16 would have overflowed to infinity now survives, while a
value fp16 held exactly may come back rounded. For frozen base-model activations
— read once, never trained further — that is the right side of the trade, and
`test_embeddings_store.py` pins both halves of it. `bytes_to_tensor` therefore
round-trips the *stored* values exactly, but only approximates the fp32 tensor
`tensor_to_bytes` was handed.

The blob is a 13-byte header followed by a blosc2 frame: `compress2` stores no
shape or dtype of its own, and numpy has no bfloat16, so the matrix travels as
its int16 bit pattern and the header says how to read it back.

`bytes_to_tensor` takes a `memoryview` as well as `bytes` so a reader under
LMDB's `buffers=True` need not copy the mapped page in just to be allowed to
pass it: at ~11 MiB a document that memcpy was a fifth of the read. What keeps
the returned tensor valid once the transaction that lent the memory has closed
is `decompress2`, which allocates its output, so the mapped page leaves the
lifetime chain before `frombuffer` is reached. The `.copy()` that follows is
there because torch will not share memory with a read-only view — not for
lifetime.

## The embeddings reader

`EmbeddingsStore` is opened once per process and consulted per document, with
`readonly` and without a lock: the store is written by a separate command that
has long since exited, and a training run must not take a writer lock on a
100 GiB file it only reads. `readahead=False` matters at that size — the store
is far larger than RAM and the documents are visited in a shuffled order, so
letting the kernel read ahead evicts pages that will be wanted again for pages
that will not.

A `get` verifies the stored matrix against the token count the batch item
implies and returns `None` when they disagree, because the store and the
encodings are two recordings of the same text made at different times and
nothing else compares them: training reads the encodings, the store is built
from the corpus, and a corpus reader fixed in between leaves the two describing
different documents. That cannot raise on its own — both row counts are
plausible — so it is checked and the document falls back to the live forward.

It does **not** catch a store built with a different token window, and a window
mismatch would misalign nothing anyway: `aggregate_embeddings` stitches the
windows back into the document's own token order, so row *i* is token *i*
regardless. What changes is how much context each token saw, which is a quality
drift no row count can see.

`close` logs the store's summary because there is no call site that could:
`embeddings_store()` caches the reader for the life of the process and nothing
owns it, so `close` — registered with `atexit` — is the only moment that sees
the totals. A hit rate well under 1.0 is the difference between a run that reads
the store and one that merely opened it, and it costs the whole speed-up without
failing.

## Batching

**A batch *is* a list of documents**, one `BatchItem` each, holding exactly the
per-document tensors the dataset holds. There is no batch dimension anywhere,
and there cannot be one: two documents in a batch hold different numbers of
512-token chunks, so their `sequence` tensors do not stack. Torch's
`default_collate` adds one regardless, giving every field a phantom leading
singleton dim that the model methods then had to read around, which is why
`collate_documents` exists. A field the row does not carry is passed over rather
than invented, which is what `BatchItem`'s `total=False` already says.

`TokenBudgetBatchSampler` batches by padded chunk count instead of by document
count. Peak VRAM in a training step is linear in a batch's **padded** token
count — measured at ~0.05 GiB per 1000 tokens for the entity head — and a batch
pads to its longest document. `BatchSampler` fixes the document count instead,
so with documents spanning 6 to 182 chunks the peak is a lottery over which ones
the sampler happened to draw: a run trains for a while and then dies on an
unlucky batch.

It closes a batch when `(documents + 1) * longest` would exceed the budget,
which is the padded size the batch will actually allocate, not the sum of its
documents' lengths. Batch size therefore varies: many short documents ride
together, and a long one travels with few or no companions. A document longer
than the budget on its own is yielded **alone** rather than dropped or truncated
— the least destructive reading, and the only one that trains on the same corpus
as before.

It has no `__len__`: the number of batches depends on the order the inner
sampler draws, which is not known until the epoch runs. Nothing asks a loader
for its length; the training bars go through `d3text.progress.batch_progress`,
which totals the split's documents instead of its batches for exactly this
reason.

`get_batch_loader` treats `0` and `None` alike for `max_chunks`, because
`ModelConfig` carries the off state as `0` (TOML has no null) while the
parameter itself is naturally optional.

## Empty and missing documents

A document whose text was whitespace tokenizes to one window holding `[CLS]`
and `[SEP]` and nothing else, and `aggregate_embeddings` slices both away — so
the model is handed a document of zero tokens, which the supported poolings
variously score as a confident negative, turn into `NaN`, or refuse.
`_drop_empty_documents` drops such a row from the split before any sampler can
draw it; dropping it in `__getitems__` instead would leave `evaluate`'s
`batch_size=1` loader yielding an empty batch.

The encodings already on disk hold such documents, so it reads the file rather
than trusting the reader that wrote it. Only a one-window document can be empty
— a second window exists only because the first one filled up — so all but a
handful of rows cost a shape lookup and no read at all. A row whose pmid is
absent from the file is left in place: it is `__getitems__`' to skip. So is
every row when there is no file at all — a split built for its labels alone
indexes fine without one.

`_h5` caches this process's read handle keyed on the pid rather than installing
it by a `DataLoader`'s `worker_init_fn`: a loader with `num_workers=0` never
runs one, and an HDF5 handle inherited across a fork shares the parent's file
offset, so reading through it yields wrong bytes instead of raising. It is not
opened with `swmr=True` — nothing writes the file while a run reads it, and SWMR
reads are only legal on a file the writer created for them.

`sequence_lengths` is read from the HDF5 metadata in a single pass, so a
length-filtering sampler never has to materialise a document to learn its
length. It is computed on first access rather than in `__init__` because almost
no run asks: every run builds all three splits, and only a
`LengthLimitedRandomSampler` needs the lengths.

## Frequencies

`compute_frequencies` sums the rows one at a time rather than stacking them into
an `[n_documents, n_labels]` tensor, which would hold the whole column in
float32 to produce a result one row wide. The returned values are bitwise those
of the stacked mean: the column is multi-hot, so every column sum is a small
integer, exact in float32 at any document count below 2²⁴ and therefore
independent of summation order, and the final `/ len(data)` is the same division
`Tensor.mean` applies — `* (1 / n)` is *not*, and disagrees in the last place
for most n.

::: d3text.corpus

::: d3text.encodings_store

::: d3text.embeddings_store

::: d3text.data.data
