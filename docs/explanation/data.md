# The data path: corpus, stores, dataset

Three artifacts stand between the raw corpus and a training batch: the
csv/json corpus itself, the `precompute-encodings` HDF5 of token ids, and the
`precompute-embeddings` LMDB of frozen activations. Each is read by a
module that also owns the rule for reading it, so the stages cannot describe the
same document differently.

## Reading the corpus

`d3text.corpus` is **deliberately a leaf**. Importing the BRENDA dataset
adapter drags in the whole BRENDA stack (`brenda_references` → `d3types` → `lpsn_interface`, and
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

## Screening a candidate negative

`d3text.negative_screen` answers whether a document names no entity of one
type, against the same surface-form index that builds the positives. A negative
the loss can be consistent with is one the *labelling* index calls empty, not
one a topic filter calls off-topic, and the psycholinguistics noise pool is the
second kind: separating an enzyme paper from a linguistics paper needs no
localization at all.

**The screen reports two readings of the same matches, and the difference
between them is the measurement.** `Matches` splits what the index found into
three kinds. `fuzzy` is a near-miss, which `Mention` may withhold a type on but
never assert one from, so it disqualifies nothing unless asked. The exact hits
split again on `negative_screen.is_descriptive`, which reads a form by its
words joined. A form no longer than `SYMBOL_MAX_LENGTH` joined is `symbolic`
however the text spaces or punctuates it, so `PP1`, `PP-1`, `PP 1` and
`PP = 1` are one symbol. Past that, a form of more than one word is a
`descriptive` name whatever its case, and a single word is one only when it
carries no capital after its first character. A form holding no letter at all
is `symbolic`, because `find_mentions` splits `5.3.2.1` into the words
`5 3 2 1`, and a section number or a confidence interval read that way names
nothing.

**Why joined.** BRENDA registers numbered symbols with a hyphen — `PP-1`,
`SP-1`, `IF-2`, `or-5` — and the index keys a form by its words, so it finds
`PP-1` across the `PP = 1` of a statistic, `SP-1` in a survey's item label
`(SP 1)` and `or-5` in "4 or 5". Read as written, each of those spans was two
words and so a name, which let a pool that names no enzyme reject on
statistics. Joined, each is the symbol it was registered as. Length alone
would misfile more than one organism-name shape this way: an abbreviated
binomial (`E. coli`), a genus with a digit-bearing epithet (`C. phi6`), and
a genus with the `sp.`/`spp.` placeholder, which may carry one further
strain number (`Mus sp.`, `B. sp. A3`). None of those is a shape notation
produces, so a genus followed by a species or by a placeholder stays
`descriptive`.

What joining cannot reach is a collision as long as the name it hits: "very
complex, I think" reads as the registered `complex I`. The comma is no evidence
against it, since BRENDA's own names carry one (`pyruvate, orthophosphate
dikinase`), so only the sentence separates the two.

`surface_forms.is_symbol_like` is deliberately not what makes that split,
though it is one question away from it and was what the first screen used. It
answers whether *case* is load-bearing, which is true of every form carrying a
capital after its first character, so it holds 40,694 of the index's 76,737
multi-word enzyme forms longer than the cutoff to be symbols — `RNA
polymerase`, `ATP synthase` and `cytochrome P450 monooxygenase` among them. A
screen splitting on it certifies a document naming `RNA polymerase` throughout
as enzyme-free. Case decides nothing for a name whose words already collide
with no English word.

The split is not tidiness, and the numbers behind it are a measurement rather
than a property of the code. Over 600 documents of each of three corpora — a
PMC microbiology sample drawn per publication year over 2005-2024 by
`scripts/collect_microbiology_sample.py`, the psycholinguistics noise pool and
BRENDA's test split — under the index built from the shipped entity tables and
all three splits' inline organism names, digest `55545dbf`, 161,501 forms over
28,423 entities:

| screen | microbiology candidates | psycholinguistics pool | BRENDA positives |
|---|---|---|---|
| `LITERAL` | 1.5% | 12.0% | 0.0% |
| `DESCRIPTIVE` | 20.0% | 86.5% | 0.0% |

The psycholinguistics pool names no enzyme by construction, so its column is
the control, and `LITERAL` rejecting seven documents in eight of it means that
reading cannot certify a negative. The cause is visible in the form table the
screen printed beside the rate: the enzyme index fires on `PCR` 2,959 times,
`PBS` 830 and `LPS` 541 in the microbiology sample, and on `DLD` 2,431, `Yes`
465 and `But` 363 in the psycholinguistics one. `COMMON_WORD_ZIPF` was asked
only of the folding branch and reached none of them; none of the six carries an
ID today, so the table above measures the index at digest `55545dbf` rather
than the one built now. `DESCRIPTIVE` is not free either: it ignores every
acronym and every short form, so a document whose only enzyme is `renin`,
`NADH` or `LasI` passes it.

The 81 psycholinguistics documents `DESCRIPTIVE` rejected at that digest were
mostly not enzyme mentions. Two of the forms behind them are genuine,
`botulinum toxin` and `hemoglobin`; the rest were hyphenated symbols read
across the notation around them — `M = 2`, `or 5`, `PP−1`, `IF=2` — which the
joined reading above now files as the symbols they are.

Two consequences for how the numbers are read. A yield is uninterpretable
alone — it took the two controls, not the candidate column, to show what the
literal reading was measuring, so the study screened several corpora under
both readings in one pass and put their yields in one table. And the
survivors are characterised against the sample they were drawn from, by
journal and by length, because a screen that has selected a genre has
produced a free negative in a new costume. `LITERAL`'s nine survivors run to a
median 4,387 characters against the sample's 31,871 — short dispatches and
letters, not papers that name no enzyme. `DESCRIPTIVE`'s are close to the
sample at 26,477, but skew by journal: thirteen of the twenty-one
environmental-health papers survive against eleven of the seventy-six in *PLoS
ONE*.

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
keeps its guard. Assigning the global at import is what got `9d1af4c` reverted
in `942bf53`. Its lock is for the restore, not the match: two overlapping calls
would interleave their save/restore and could leave the exemption behind for the
whole process.

The exemption means the guard cannot fire on this call at all: `xmlparser`'s
`RegexpTokenizer` compiles through `redos.compile` with no per-pattern timeout,
so `TimedPattern._resolve` reads `nltk.redos.DEFAULT_TIMEOUT` at match time and
finds the `None` `_remove_tags` just set, and `regex` never raises
`TimeoutError` under `timeout=None`. `stream_rows` and `stream_documents` call
`document_text` directly rather than guarding it: there is no row-level catch
to drop and tally a `TimeoutError` this path cannot produce, so an exception
document_text does raise — a real I/O stall raises the same builtin
`TimeoutError`, which is an `OSError` — ends the pass loudly instead of
shrinking the stream silently.

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

### The geometry does not identify the ids

The stamp above answers *how* a store was built, not *what is in it*. A store
rebuilt at the same base model, window and stride — a newer tokenizer revision,
a corrected `document_text`, a corpus refresh — carries a stamp identical to
the one it replaced, so a checkpoint trained on the old ids is scored against
the new ones with nothing to say so. `encodings_store.content_digest` closes
that: the hex SHA-256 of every document key, its window shape and its token ids
at a fixed byte order, taken over the keys in sorted order so one file digests
the same in any process on any machine. The shape is in there because
`sum(L_i)` comes to the document's token count under any window, so the ids
alone cannot separate one cut from another.

**The writer pays for it, once.** Digesting a store decompresses every id in
it — one pass over the whole file, affordable at the end of a tokenization run
that took hours and not affordable per read. `precompute-encodings` therefore
computes it after the last group is written and stamps it on the root, and
`store_content_digest` opens the file for that one attribute. What it restates
covers the whole file rather than the documents one pass wrote, so a resume
that adds ten documents re-fingerprints all of them.

**A stamp is dropped before it can go stale.** Because the digest describes the
file's own contents, the first group a pass writes falsifies it — and an
interrupt propagates out of the enclosing `with h5py.File(...)`, which closes
the file *cleanly*. Stamping only at the end would therefore leave a killed
re-tokenization carrying the previous pass's fingerprint over ids it has
already replaced, which `evaluate` would read as agreement. So the writing pass
deletes the attribute on the way in and restates it only if it reaches the end,
and a store nobody finished writing reads as unstamped. The geometry stamp is
written before that pass rather than inside it: a store that refuses this run's
window has had no group written, and must keep the digest it still answers for.

**One pass at a time, and nesting is refused.** A nested pair that both run to
completion restamps fine — the outer's own exit restates the digest over the
whole file, covering whatever the inner pass wrote too. The risk needs an
interrupt just as the unnested case does: between the inner's exit and the
outer's, the inner has already restated the digest, the outer then writes
more and is killed before its own restate, and what is left on disk is the
inner's stamp describing ids that have since moved — the same stale-stamp
shape as an unguarded interrupt, one level up. The guard is keyed to the
store's `(st_dev, st_ino)`, not a path string, so a hard link or a
differently-spelled path onto the same file is still caught; it fires before
the inner pass writes anything, so the outer pass it aborts leaves the store
unstamped rather than falsely stamped.

**The guard is per-process, not per-file.** Two `precompute-encodings`
processes opened on the same store reproduce the interrupted-pass failure
above between themselves — one exits and restamps while the other is still
writing — and nothing in `writing_pass` catches it, since the re-entry guard
is a module-level set. What stops it is HDF5's own file lock refusing the
second process's `r+` open. That lock is routinely disabled with
`HDF5_USE_FILE_LOCKING=FALSE`, standard advice on a network filesystem, which
is exactly where a shared corpus store is likely to live; with it disabled,
two writers on one store can both succeed, and the stamp left on disk is
whichever one exited last, over ids the other also touched.

**A group holding no ids does not count as content.** An interrupt between
`create_group` and the `create_dataset` that follows it leaves one, and it stays
on disk until a resume reaches that key and redoes it, as it redoes every group
`is_finished_group` rejects. Among readers, `stored_ids` is the one place that
case is recognised — `content_digest` passes such a group over and
`BrendaDataset.sequence_lengths` omits it, so the digest tolerates exactly what
the reader tolerates. A store carrying one therefore digests as the same file
without it does: no reader can serve that document either way, so separating
the two would report a difference that changes no number, and it would make the
store unstampable, since digesting it at all used to raise.

Like the checkpoint's own provenance fields, it is optional. Every encodings
file already written carries none, and `read_content_digest` reports that as
`None` rather than refusing the file — the same call `record_provenance` makes
about an unstamped geometry, for the same reason. `train` records the digest
beside the label store's and `evaluate` warns on a mismatch; neither refuses,
because a re-tokenized corpus makes two runs incomparable rather than making
either of them wrong.

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

**The stored dtype is bf16, and the codec is zstd level 1 behind a byte
shuffle.** Both were measured with `scripts/benchmarks/bench_codecs.py`. Three
results drive them:

- These activations are very nearly incompressible losslessly. Every lossless
  combination of codec, filter and level lands between 1.00× and 1.17×, because
  the low mantissa bits are noise no entropy coder can model. Storing bf16
  instead of fp16 spends two of those bits and gets 1.42×, which is 100.8 GiB
  rather than 121.9 for the whole corpus. It is the only near-lossless lever
  there is; the codec knobs are not one.
- On bf16, a higher zstd level buys nothing: level 1 compresses slightly
  *better* than levels 3 and 5 (1.45× against 1.41× and 1.42×) and packs about
  3× faster, and it decompresses faster too. A blosc2 frame records its own
  codec, so a store written at another level still reads.
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

### A stored embedding and a live one are not the same number

`precompute-embeddings` and the training loop's fallback forward both take
their autocast dtype from `select_amp_dtype`, so one machine runs one
precision on both paths. What is not aligned, and cannot be, is the shape of
the forward: the two sides put a different number of windows through each one
— the precompute takes a document at a time at `--batch_size` windows, the
fallback takes every missing document of a training batch together — and
cuBLAS dispatches a different kernel per batch shape. Measured on one
26-window document at a fixed bf16, changing only that split moved 76.6% of
elements, mean absolute 0.0020, the same magnitude as a whole change of dtype.
Each path is deterministic in itself: the same dtype at the same batch shape
twice is bit-identical.

Aligning is free — bf16 measured 2.13 s against fp16's 2.20 s over eight
documents, at identical peak VRAM — and it removes a dtype that was hardcoded
in the precompute, so a CPU precompute no longer runs the one dtype
`select_amp_dtype` exists to refuse. Running the precompute in fp32 instead
was measured and rejected: it costs 2.6× the wall clock to buy a mean
absolute 0.002079 against bf16, which is the same size as the batch-shape
difference that remains either way, so the gain sits inside the band it would
have to clear to matter. Nothing on disk changes under any of the three:
`tensor_to_bytes` narrows to bf16 whatever it is handed.

Which precision a store was built at is recorded in its provenance and
repeated in the line it logs at the end of a run, because `select_amp_dtype`
names a machine rather than a dtype — two stores agreeing on model, window
and stride can have been built on cards that answer it differently, and one
written by a build predating the alignment holds fp16. Nothing reads that
field to decide anything: a store recording a precision this machine would
not have chosen is still read, and **an existing store does not need
rebuilding**, the difference being seed-sized either way.

**A run that reads the store and a run that recomputes are not numerically
comparable, and no configuration makes them so.** The gap is small per
document, but it is an input perturbation to a training trajectory, so it
compounds: two frozen runs differing in nothing but the store were 0.25% apart
on epoch 0's training loss and 22% apart by epoch 1. That is a divergence of
trajectories rather than of what the run learns — it is the size of a change
of seed, and final metrics sit inside the spread seeds already produce. Which
makes turning a store on, or off, a re-baselining rather than a speed-up with
the numbers held fixed —
results from before it are not a baseline for results after it, and two
machines that disagree about whether they have one cannot compare numbers with
each other. This is structural, not a defect awaiting a fix: bit-exactness
would need a fixed window count per forward on both sides *and* deterministic
kernels, for a value whose whole purpose is to be computed once.

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
The count is a cheap proxy and not an identity: two documents of one token
count stored under one key would pass it, which is why the key is a document's
own pubmed id rather than its position in a split.

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

`BrendaDataset.__getitem__` routes both index types through `__getitems__`:
`dataset[int]` returns a single dict and `dataset[list[int]]` a list of them,
the path `DataLoader` actually uses, and both carry `doc_id`. `doc_id` is
a **Tensor** whose last-dim size counts how many HDF5 sequences belong to
that item (read by `get_token_embeddings`), not a scalar; the PubMed ID
lives separately, in `item["id"]`.

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
count — measured at ~0.05 GiB per 1000 tokens back when the model carried a
per-entity output head — and a batch pads to its longest document.
`BatchSampler` fixes the document count instead, so with documents spanning 6
to 182 chunks the peak is a lottery over which ones the sampler happened to
draw: a run trains for a while and then dies on an unlucky batch.

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
