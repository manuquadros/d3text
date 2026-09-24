# Why the models pool, mask and compose the way they do

`d3text.models.base` holds `Model` — base transformer loading, AMP and
gradient checkpointing, token-embedding lookup, logit pooling — plus the loss
and metric helpers every concrete model shares. The three concrete models
(`NERClassificationModel`, `BrendaClassificationModel`, `ETEBrendaModel`) each
live in their own module (`ner.py`, `entity_linking.py`, `ete.py`) and inherit
directly from `Model`; none inherits from another. Shared submodules live in
`heads.py` and `token_supervision.py`.

The epoch *schedule* around `Model.run_epoch` — optimizer, LR scheduler, early
stopping, the best-epoch snapshot — belongs to `d3text.training.trainer.Trainer`
rather than to the model.

## Document-level pooling

All three models are document-level multi-label classifiers, not token-level
taggers. Per-token logits are pooled to one logit vector per document, and the
mode comes from `ModelConfig.entity_logits_pooling`:

| Mode | Behaviour |
| --- | --- |
| `logmeanexp` (default) | `logsumexp − log T`; length-invariant smooth-mean, but dilutes a lone mention in a long document |
| `logsumexp` | smooth-max — one strong token can carry the document; adds up to `+log T` for diffuse classes, so it is length-biased |
| `max` | hard max; length-invariant |
| `mean` | arithmetic mean |

**The attention mask has to reach the pooling.** Without it, `logmeanexp` and
`mean` normalise by the *padded* length, so a document's pooled logits depend on
how long its batch companions were — a short document batched with a long one is
shifted by `−log(T_pad / T_doc)` on every column. With it, both modes normalise
by each document's real token count, and `mean` also keeps the padding fill
values out of its numerator. `logsumexp` and `max` never needed it: the fills
vanish under both reductions.

`token_counts` floors each document's count at one, which keeps an all-padding
document finite — dividing by a true count of zero would make the mean a `NaN`
and `logmeanexp`'s `log(counts)` a `-inf` — while leaving every real document
untouched.

`reject_empty_token_dim` refuses to pool a document with no tokens at all. The
four modes disagree completely on an empty reduction: `logsumexp` returns `-inf`
(a confidently correct negative), `mean` returns a `NaN` that propagates into
the epoch's loss with nothing in the log to attribute it, `logmeanexp` dies
inside `math.log`, and only `max` names the dimension. None of them can answer
what a document with no text predicts, so the answer is given once.

`_pool_logits_segments` is the segmented counterpart of `_pool_logits(rows,
dim=0)`: segment `g` gets exactly what pooling its own rows in isolation would
give, in a fixed number of kernels instead of one launch per segment.

### Why the pooling is chunked

`torch.logsumexp(logits.float(), dim=1)` first materialises a float32 copy of
the per-token logits — the largest tensor in the step, and twice the size of the
bfloat16 original — and autograd holds it, plus a gradient of the same shape,
until backward has run. Together those two are about half the peak of a training
step.

`_ChunkedLogSumExp` walks the token dimension in slices, performing the same
two-pass shift-and-sum `torch.logsumexp` performs. Only the order in which the
exponentials are summed differs, and that difference does not survive the cast
back to bfloat16: the pooled logits are bitwise unchanged. Backward needs no
float32 copy either — the gradient of a logsumexp is `grad * exp(x - out)`,
which it recomputes slice by slice from the input and the tiny saved output.

`_ChunkedMean` is the same bargain and simpler: a mean spreads its gradient
evenly, so backward reads none of the input at all.

The float32 itself is right — pooling thousands of tokens in bfloat16 is where
the precision actually matters — it just does not have to exist all at once.
Every mode routes through `pool_token_dim` so the pooled values cannot depend on
which path ran. `pool_chunk_tokens` picks narrower slices for a wider batch so
`documents * tokens * width` stays put, with a floor of one token, since a batch
wide enough to exceed the budget on a single token would otherwise not advance.

## The losses

### The divisor is the weight sum, not the element count

This is the trap all three masked losses encode, and it is the same trap each
time. Summing the kept terms and dividing by the *whole* population scales every
real element's loss by the share of the batch that happened to be masked, so a
document with more uncurated entities in it teaches less about the ones it does
have. That is the dilution the mask exists to remove, reintroduced by the
reduction.

`masked_token_cross_entropy` divides by the unmasked token count.
`masked_bce_with_logits` divides by the weight sum over `(document, class)`
pairs. `focal_cross_entropy` divides by the modulation mass: under a plain
`.mean()` an easy pair still divides the denominator, so proposing more of them
shrinks the loss on the rare positives; dividing by the mass keeps an easy pair
out of *both* sides. Its clamp guards the degenerate batch in which every pair
is already scored confidently — the numerator vanishes with the mass, so the
loss decays to zero instead of exploding.

### Token loss

The distant-supervision targets in `d3text.token_labels` carry a third value for
tokens matching a surface form of an entity the document was not annotated with.
Those are the tokens nothing knows the answer for, and they are ~2.8% of the
document.

`torch.nn.functional.cross_entropy(..., ignore_index=...)` is the other spelling
of the unweighted case and divides the same way; `masked_token_cross_entropy`
exists so the divisor is visible at the call site rather than inherited from a
default, and `tests/models/test_masked_loss.py` pins the two against each other.
That equivalence holds only for `weighting="unweighted"` — the other two schemes
have no single-call `nn.functional` spelling.

Its `weighting` mirrors `relation_loss_weighting`'s three-way choice on the
relation head, aimed at the same shape of imbalance: `OUTSIDE` is ~91% of kept
tokens, so a plain average lets the majority class dominate the gradient.
`balanced` reweights by per-batch inverse frequency over the kept tokens, and
the reduction is then `nn.CrossEntropyLoss`'s own weighted mean, dividing by the
summed sample weights rather than the kept count. `focal` down-weights
confidently-correct tokens instead.

An all-masked batch returns a differentiable zero rather than a `NaN`: it is
reachable from a short document whose every match is uncurated, and losing a
training run to it would be absurd.

### Class loss

`masked_bce_with_logits`'s `abstain` marks a negative target this run has
decided not to fully enforce: a document the class head is told carries none of
a type, but whose text a dictionary match says otherwise. `None` reduces to a
plain `BCEWithLogitsLoss(reduction="mean")`.

`downweight` sets the weight an abstained pair keeps instead of being dropped
outright. `0.0` is a hard abstain, excluded from both the numerator and the
divisor and byte-identical to the function before the parameter existed. A value
in `(0, 1]` keeps that fraction of the negative pressure rather than removing
it.

### Relation loss

Candidate pairs are proposed per batch by the span tagger's own detections, so
the `none` share is a property of the current tagger rather than of the corpus:
there is no dataset frequency to precompute, and `balanced_class_weights` has to
re-derive the weights every batch. A class absent from a batch's targets would
divide by zero; its weight is never read, since `cross_entropy` gathers weights
by target value, so clamping the count is enough to keep the tensor finite.

`focal_cross_entropy` suppresses the loss from pairs the model already scores
confidently, which is most of what the pairing proposes. Unlike a fixed class
weight it tracks the tagger: as detection sharpens and stops emitting junk
spans, the down-weighting relaxes on its own. `gamma == 0` is plain
cross-entropy.

## Mixed precision

`has_bf16_hardware` asks whether the GPU runs bfloat16 in silicon rather than by
emulation. `torch.cuda.is_bf16_supported()` answers a different question: it
defaults to `including_emulation=True` and so returns True on cards with no bf16
units at all, which is how a Pascal card came to train under bf16 autocast.
Measured on a P100 that costs about 27% of the throughput of fp16 or fp32 and
close to three times the peak memory — 10.4 GiB against 3.5 GiB over 256 windows
— on a card whose configured training run already peaked at 99.2% of its 16 GiB.
It is asked by compute capability: bf16 units arrive with Ampere (8.0), and the
capability is readable on every torch version while the `including_emulation`
keyword is not.

`select_amp_dtype` asks each backend independently rather than ANDing one
backend's veto into the other's question. Compute capability is meaningless
under HIP — `get_device_capability` there returns gfx-derived numbers that would
answer True even for a card with no bf16 units — so `has_bf16_hardware` is gated
to CUDA and the device-name allowlist is the sole authority for ROCm. "MI300" is
absent from that allowlist as redundant: it is a strict substring of "MI3", kept
as a deliberate prefix match meant to catch future MI3xx parts without naming
each one.

A model placed on `"cpu"` takes bf16 outright, no hardware question asked: CPU
bf16 is software-emulated on every build, and it is what PyTorch's own CPU
autocast defaults to. fp16's narrow exponent range is a GPU-silicon trade-off,
not a CPU one, and genuinely overflows CPU-scale activations that bf16 —
sharing fp32's exponent range — does not.

**A frozen trunk holds its linear weights in `amp_dtype`, not fp32.**
Autocast caches a weight cast only for a leaf with `requires_grad=True`, so
every frozen `nn.Linear` copied its fp32 weight down again on every forward,
one `aten::_to_copy` per linear per batch. `freeze_base_model` stores the cast
result once instead, which hands the matmul the identical tensor autocast
produced before — the same cast, applied once rather than once per forward.
That equivalence is what carries the fp16 cards: a weight that overflows or
flushes to zero on the way down did so in autocast's own copy already.
Trainable layers keep fp32 master weights, which is what the optimizer steps.
`LayerNorm` and `Embedding` stay fp32 whatever their `requires_grad`: autocast
runs `layer_norm` in fp32 regardless, so a narrowed weight would only be
copied back up again, and it never casts `Embedding` at all, so a narrowed
table would change the values the first `LayerNorm` sees rather than just
where the cast happens.

The dtype of a frozen trunk weight in a checkpoint is therefore the writing
machine's — bf16 from an Ampere-or-later card, fp16 from a P100 or a T4 — and
none of it needs normalising on the way in or out. `load_state_dict` copies
into each parameter's own dtype, so a file from the other kind of card, and
the fp32 of every checkpoint written before the cast, both land in the dtype
the loading machine will compute in.

Gradient checkpointing skips the base model: it is frozen, and only ever runs
under `no_grad` in `get_token_embeddings`, so there is no activation graph to
trade against recomputation.

## Where token embeddings come from

`Model.get_token_embeddings` has three sources, cheapest first: the in-process
cache, the precomputed embeddings store, and the frozen base model. The base
model is a pure function of the input ids only because it is held in eval mode:
`no_grad` does not disable dropout, so a base model carried into train mode by
`model.train()` redrew a document's activations on every forward while the
cache and the store held one draw forever — which is how switching the cache on
moved the training loss. `Model.freeze_base_model` pins it at construction and
`Model.train` pins it again after every epoch's `model.train()`, since
`nn.Module.train` recurses into every submodule.

What is left between the three sources is arithmetic alone. The store's
matrices were computed under fp16 autocast and rounded to bf16, while the live
forward runs under `amp_dtype`, so a run that reads the store gets slightly
different activations from one that does not.

`ByteBudgetCache` budgets that first source in **bytes**, and
`cpu_embeddings_cache_mb` is what a machine sets. An entry is one row per token
of a whole paper: 14.5 MB on average over this corpus, 56 MB at the tail, a
tenfold spread. Counted in documents — as the predecessor key
`cpu_embeddings_cache_size` did — 4000 reads as a modest number and is 58 GB,
which is how a 30 GB machine had a run SIGKILLed mid-validation with nothing in
the log naming the cache. A non-zero value under the old key is therefore
refused at load rather than reinterpreted; `0` means the same thing in either
unit and is migrated with a warning. The ceiling is enforced in `set` rather
than at the call site, so a document too large for what is left is declined
while the cache stays open for the next, smaller one — `full` is a
short-circuit, not the enforcement.

Both of the other two sources fill the budget, and an entry records which
one did. A store hit is cached on first read — its bytes on disk cannot
change, so without that a stored document pays an LMDB read and a blosc2
decompress on every epoch and every validation pass. Only an admission from
the base model evicts, and only entries the store served, oldest admitted
first: a forward-only entry is evicted for nothing, and a store hit that
does not fit in what is free is declined rather than displacing another
store hit. That last is what keeps a warm prefix under a working set larger
than the budget — a pass reads its split once and in order, so store hits
taking each other's places would evict exactly the entries the next pass
wants, and the promotion would buy nothing. The budget is therefore still
first-come-first-served within each source, and since `fit` runs the
training pass before `_validate` every epoch, a budget smaller than the
whole working set is still claimed by training documents first. The cache is
worth configuring only where the working set fits: at the 14.5 MB mean
above, 2,000 documents want ~29 GB.

`cpu_cache_key` keys a cached activation by the base model that produced it.
The cache is process-wide and one process holds more than one base model: `tune`
builds a fresh model per trial and `base_model` is a sweepable field, so a
document id alone names an activation only while every consumer happens to share
a base model. Two base models of equal hidden width would otherwise serve one
trial's activations to the next. The read also checks the entry's row count
against the one the batch item implies and treats a disagreement as a miss —
the same check the store's read makes — because two different documents handed
one id would reach the heads as each other's activations, tagged and grounded
against the wrong text, without anything failing. That count is a cheap proxy
for the entry being this document's and not a proof of it: two documents of one
token count pass it. What keeps them apart is the id itself, which every
producer mints to be unique — a pubmed id for an article, and for a document of
a corpus that issues none, `encodings_store.external_document_id`'s negative
id, minted once per store key for the life of the process.

Every source lands its tensor on the model's own device, and only the live
forward's windows are aggregated there: a cache or store hit is already one
matrix per document and needs none. Hits stay on the host until the forward's
hidden states have been released, and move to the card only for
`pad_sequence`; any earlier, they would share it with the forward, whose
allocation the peak claim below rests on. What comes back to the host is what
the cache is offered — every freshly aggregated document while `full()` is
False, including one that `set` then declines for want of room — since a
device tensor in a cache budgeted in host RAM would pin VRAM for as long as
the process lives.

The alternative is to aggregate on the host: copy the hidden states down, pad
there, and move only the finished buffer to the card. Against that, keeping
the work on the card adds to what the card holds at two points in the call;
through the base model's forward itself the two hold the same. While the
hidden states are live the card also carries the forward's device attention
mask and the aggregation's working set: the current document's stacked windows
and masks, plus the windows' `amp_dtype` copy when that differs from the
hidden states' dtype; the previous document's, still bound while the next is
stacked; the copies `aggregate_embeddings` concatenates; and every aggregate
built so far. At `pad_sequence` it carries every document's
own tensor — the aggregates, and the cache and store hits, which arrive only
once the hidden states are released — beside the padded buffer built from
them, together with what outlives that release: the last document's stacked
windows and masks, and the forward's device attention mask. The host
alternative would deliver the finished buffer alone. Whether either point
moves the step's high-water mark depends on what else in the step allocates
more, and a batch served wholly or mostly from the cache or the store has
little or no forward in the call to cover them. The step's peak has been
measured unchanged on batches that run the base model's forward; a batch
carrying cache or store hits is unmeasured, as is any batch larger than those
measured. Where the work runs does not change the output: aggregation and
padding only index, copy and cast, so the padded batch matches the host's bit
for bit in every value but NaN, whose payload a host and a device cast may
set differently. The hidden states' residency ends at an explicit release
before the padding, not at a copy to the host.

`embeddings_store` is opened lazily, for the reason the rest of the library
defers its machine state: importing `d3text.models` must not touch the
filesystem. A store that cannot be opened — a path that has moved, a half-written
LMDB — disables itself and the run recomputes the embeddings, which is exactly
what it would have done with no store configured. Losing the speed-up is not
worth losing the run. The same route is taken by a store the wrong base model
wrote: the run pays the base model's speed rather than training on somebody
else's activations, and says so once. `base_model` is an argument rather than a
machine-config field because the store belongs to the machine and the model
belongs to the run, and it is the *pair* that has to agree.

`document_token_count` measures how many rows `aggregate_embeddings` produces
by running the aggregation over a zero-width tensor rather than by
reimplementing its overlap arithmetic: the number exists to catch a store whose
rows do not line up with the encodings, so a second, drifting copy of that
arithmetic would be a hole in the very check it serves. The zero-width feature
dimension is what makes it free.

`batch_input_tensors` flattens every dimension but the last, because the same
item reaches it under two shapes: `BrendaDataset[[...]]` yields a 2-D
`[n_chunks, token]`, while the `DataLoader` collates that through
`default_collate` and hands over a 3-D `[1, n_chunks, token]` — the leading 1
is an artefact of batching a one-element list, not a document axis.
Concatenating the 3-D form on dim 0 stacks documents along the *chunk* axis
instead of extending it, and raises as soon as two documents differ in chunk
count, which is every real batch.

`load_base_model` tolerates legacy configs that lack a `model_type` key (e.g.
`prajjwal1/bert-mini`). `AutoModel.from_pretrained` delegates to
`AutoConfig.from_pretrained`, which reads `model_type` from `config.json` to
choose the architecture; old-format repos omit it and raise `ValueError`, so an
explicit BERT config is the fallback — every supported base model (the keys of
`d3text.models.config.encodings`) is BERT-based.

## Column conventions

The last column of the class logits is always `OOS`. `label_columns` locates the
sentinel *by name* and lists every other column, which keeps loss and evaluation
correct if the sentinel ever stops being last. The registered column tensor is
non-persistent: it is derived from `self.classes`, so it must not enter a
checkpoint, where an older file would then be missing the key.

The entity head follows the same convention: its **last column** is always
`UNK`. Entity IDs are strings like `enz26836`, `bac42`, `str1234`, `oth567` —
a three-letter type prefix plus the numeric ID from the BRENDA database —
and `entity_index: dict[str, int]` maps them to logit positions. Loss
computation and evaluation rely on both sentinels' position via `[..., :-1]`
slicing; do not reorder either head.

## What gets reported

`Model.compute_losses` returns one batch's losses keyed by objective name. Every
key is one `update` sums and optimizes and one `run_epoch` accumulates under the
same name, so **a key present in one batch of an epoch must be present in every
batch of that epoch**. `NERClassificationModel` reports only `class`,
`BrendaClassificationModel` adds `token` when a token-label store is
configured, and `ETEBrendaModel` adds `relation`, already scaled by that
epoch's ramp weight. `step` is what lets the ramped model score validation under
its final weight while training still follows the schedule; a model with no ramp
ignores both `step` and `epoch`.

`epoch_loss_weights` reports the multiplier applied to each named loss. Its keys
match `run_epoch`'s, so a logged `loss_weight/relation` sits beside the
`training/relation` it scaled — without which a loss curve that bends because
the ramp moved is indistinguishable from one that bends because the model
changed.

`print_epoch_stats` returns what it prints, so `Trainer.fit` logs that dict to
MLflow rather than re-deriving the averages and the console and the tracking
server cannot disagree about an epoch's numbers.

`epoch_rate_metrics` records wall-clock, which is what makes two runs' loss
curves comparable as *choices*: a configuration that reaches the same validation
loss in half the epochs has not won anything if each epoch costs twice as much.
Rate is in batches rather than documents because `TokenBudgetBatchSampler` makes
the document count per batch a function of document length.

`relation_metrics` holds `none` separate. A macro-F1 across all three labels is
dominated by it, since it is both the majority class and the one nobody asked
about; what ranks runs is the score over the typed labels alone. `none_share`
is logged beside it because the candidate set is proposed by the *current span
tagger* rather than by the corpus — the same checkpoint can face a different
pair distribution from one run to the next, and this is the only record of
which one it met.

`support_metrics` is what tells one micro-F1 of zero from another: a head
predicting nothing at all and a head predicting the wrong labels score
identically, and only the predicted-positive count separates them.
`labels_predicted` counts the *columns* ever used rather than the positives,
which is how a head collapsed onto one frequent label shows up.

`coverage_metrics` reports how many of the split's documents the pass actually
scored. `dataset/test_documents` is what the split frame *planned* to hold and
is logged at run setup, before anything has been read; every `test/*` score is
computed over the documents that reached the model instead. The two come apart
whenever the frame and the encodings file disagree: `BrendaDataset._getitems`
drops a row whose pmid the HDF5 does not hold, and `batch_progress` drops a
batch left empty by those drops, which shrinks the denominator of every metric
without shrinking the number a run list shows beside them. The keys sit under
`dataset/` rather than `test/` so the three appear together in a run table;
`_missing` is 0 for a healthy split rather than absent, since an absent key
cannot be told from a run of a version that did not log one, and is omitted
altogether when the split size is unknown.

## The three concrete models

### `NERClassificationModel`

Entity class detection with no span tagger and no relation head: it predicts
entity types and pools them to the document. It has one objective and no
schedule rides it.

### `BrendaClassificationModel`

Entity class detection over the pooled logits, plus the span tagger when a
token-label store is configured. Neither of its losses is ramped, so `step` and
`epoch` are taken only to match the shared signature.

#### The span tagger

`compute_token_loss` is **additive to the document-level losses, never a
replacement**: the pooled terms carry the gold links that are never named in the
text, which no distant supervision reaches, and this term supplies the
localization the pooled loss cannot. The mask covers the tokens matching
entities BRENDA did not link to the document, the padding, and any document the
store has no targets for.

`token_targets` gives a document the store does not hold an all-`IGNORE_INDEX`
row — skipped by the loss, counted into one summary warning per pass —
because a split wider than the labelling run is a data gap, not a modelling
error. A document
whose stored row *disagrees in length* with its embeddings raises instead: that
store was built against other encodings, and every one of its codes would land
on the wrong token.

#### Abstaining on document-level class negatives

`class_negative_abstain_mask` returns `None` when the feature is off — the
ordinary case, where the class loss reduces to a plain masked-nowhere BCE.
Otherwise it marks `(document, class)` where the document is a gold negative for
that class yet the token-label store's dictionary matched a surface form of that
class's type somewhere in the document, at least that class's own length cutoff,
gold-linked or not.

The length gate is what keeps this from abstaining on a one- or two-character
incidental match. A *uniform* cutoff still collapses `bacteria` toward
predicting positive on nearly every document while rescuing `strains` and
`other_organisms`, which is why the cutoff is overridable per class rather than
one number for all four.

It reuses the tagger's own matches rather than running a second dictionary pass,
so it is exactly the mask `token_targets` already abstains at the token level,
one level up. The class-head column order is `schema.class_names`, the same
declaration order `token_labels.LabelSpace` assigns its codes 1..n from, so
column `j` is type code `j + 1` with no lookup needed.

### `ETEBrendaModel`

Entity class detection + relation extraction.

**It composes a `BrendaClassificationModel` rather than subclassing it.** The
two used to be related by inheritance, with this class overriding almost every
method of the parent at a wider arity — exactly the shape that widens a return
type and trips mypy's `[override]` check. Composition removes the subtype
relationship instead of suppressing the check: `ground_truth`,
`get_batch_logits`, `compute_batch_losses` and `forward` all return the *same*
typed container as `BrendaClassificationModel`'s, just with the relation-related
field populated instead of `None`.

Its `__getattr__` reaches through to the composed model for the class-head and
span-tagger attributes this class does not declare, so callers read `model.X` rather than
`model.two_head.X`. It is read-only by construction — nothing is ever assigned
through it — so a value that must reach the inner model on a write needs its own
property.

#### The relation loss ramp

The relation loss is the one objective in this package that rides a schedule:
`relation_loss_weight` ramps it linearly from `w0` (0.1) to 1.0 over
`ramp_epochs`, which at 0 means no ramp at all. The schedule holds the relation
head back until the span tagger proposes usable pairs to classify. No other
objective rides it, here or in any other model.

It is scaled inside `compute_losses`, before `run_epoch` ever sees it, so the
generic accumulation stays oblivious to the ramp. **Validation totals are scored
under the ramp's final (t = 1) weight**, the objective the run is ramping
toward, so `validation/loss_total` reads as one comparable series across
epochs on the chart; only the training gradient follows the schedule.
Neither `reduce_on_plateau` nor best-epoch selection reads the ramp or the
loss — see [the training loop](cli-and-training.md#the-training-loop).

`epoch_loss_weights` reports the unscheduled objectives at the full weight they
train under, so every objective has a curve.

#### What a relation argument is

An argument is a **candidate set**: every entity the label store's mentions
leave a tagged span able to name, narrowed within the document but never chosen
between (see `resolve_mentions` below). Two spans carrying the same set are one
argument, so a document proposes one pair per unordered pair of distinct sets
whose types some relation admits.

`ArgumentGroups` interns each distinct set to an integer, and that integer is
what `arg_pred_i` / `arg_pred_j` carry. The interning is what keeps the pair keys an
integer tensor: `align_relation_predictions` groups duplicate rows with one
`torch.unique` and joins gold with one `searchsorted`, neither of which a
frozenset can be packed into. It happens host-side, in the code building the
rows, where the store's data already is. An id means nothing outside the batch
that interned it — exactly as a `sequence` index does not — so `forward`
publishes the table it built (`_argument_sets`, `_argument_groups`) for the
loss and the metrics to read back, and rewrites both on every call.

**Detected pairs come first and gold is the fallback.** A detected pair whose
sets *cover* a gold pair — one set holds the subject, the other the object —
takes that pair's gold label, and only a gold pair no detected pair covers gets
a row of its own, pooled from its arguments' own stored mention positions. The
merge used to run the other way, keeping the gold row and dropping the
overlapping detected one, which trained a representation the evaluation never
builds and left the one it does score supervised as `none`.

Coverage is not key equality. A gold pair names two single entities while a
detected argument may carry several candidates, so the two sides share no key
even when the detected pair is exactly the gold one; `_covering_row_keys`
expands a gold pair into every row key that could be it, which is one key per
pair of groups holding its two arguments. Ordering is still worth stating: a
row's two argument ids ascend, while gold arguments arrive in whatever order
preprocessing stored — lexicographic on the entity-ID strings — so the
expansion sorts each key. Sorting loses no direction, since the string sort
already discarded argument order and the relation label is directional by
argument *type* instead.

`unscored_gold_relations` reports the gold that no scored row can account for.
`align_relation_predictions` builds its rows out of the pairs the tagger's
groundings were paired into, and gold only ever labels a row that already
exists; gold no row covers therefore cannot show up in any metric computed over
those rows. It is not a false negative, it is absent, and the denominator
becomes whatever detection chose to propose. **A caller computing metrics must
add these back as misses.** It is deliberately not folded into the aligner: the
loss path consumes that function, and these relations carry no logits to
backpropagate.

The two buckets it splits them into say whose problem the miss is. A relation
has *no anchor* when the store places no mention of one argument anywhere in
that document — a document the store does not cover at all included — so
nothing grounded in the store could ever have proposed it and charging it to
span recall would be charging a gap in the dictionary. The rest were simply not
proposed. A gold triple repeated across a document's pair-dicts yields one
entry either way.

#### Two scores, because an argument is a set

Relations are scored by intersection, the rule `LinkingRule.INTERSECTION`
already applies to linking: a row counts for a gold relation as soon as one
argument's set holds the subject and the other's the object. Beside it,
`test/relation_{macro,micro}_f1_typed_strict` scores the same rows under the
strict rule — each argument the one gold entity and nothing else — and
`test/relation_argument_set_size` reports the mean set size over those rows'
arguments. The gap between the two scores is then readable as what the
grounding left undisambiguated rather than as anything the relation head did,
which a single number cannot distinguish.

## Token targets in the model's geometry

`precompute-token-labels` writes per-window codes shaped like the stored
encodings; the model scores the *aggregated* document — the 512-token windows
merged along their 20-token overlaps by `aggregate_embeddings`.
`TokenLabelReader` carries the codes across that same merge by running them
through `aggregate_embeddings` itself rather than by restating its overlap
arithmetic: the targets exist to sit element-for-element beside the embeddings,
so a second, drifting copy of the selection rule would be a hole in exactly the
alignment being provided. The int8 codes ride through the float pass losslessly
— every value, `IGNORE_INDEX` included, is a small integer float32 represents
exactly.

**The label space is verified at open, not assumed.** A store written under a
permuted schema holds codes whose integers mean different types, and nothing in
the arrays says so. The store's recorded space must equal the space the tagger
head was sized to, or nothing is read at all — checked once at open so a
mismatch costs a file open rather than an epoch, and again on every read by
`load_token_labels`, which covers a reader that never comes through the class.

A `None` from `document_codes` or `mentioned_types` means the store holds
nothing for that document — outside what the store covers, not a document that
mentions nothing. It is the caller's to skip or to mask, since only the caller
knows whether that is a truncated split or a stale store.

`exact_mentions` is the read a detected span is linked through: every
exact mention's candidate IDs and its aggregated-axis positions, read off the
store's [anchors](distant-supervision.md#every-exact-mentions-candidates). It
never reads `entity_positions`' masks, which are gold-only — a proposer built
on them would propose gold entities alone.

`resolve_mentions` is the join over that read, and `ETEBrendaModel.forward`
runs it over every span the tagger proposes: a tagged span takes the candidates
of every stored mention it overlaps, keeping the IDs of its own tagged type,
since the store records each mention's whole candidate set and leaves the type
filter to whatever links. It is called with the *reader's* label space, not this
function's default, because the reader is what verified that space against the
store it came from. An empty result is NIL rather than a failure — a typed span
the dictionary grounds in nothing is exactly what the tagger exists to find, and
it proposes no relation argument, since relations train on grounded arguments
only.

**The answer is a narrowed set, not a chosen entity.** A candidate set shrinks
to the IDs the same document also names through a single-candidate mention
wherever that intersection is non-empty, and stays whole where it is empty.
Narrowing that far and no further is what the ambiguity is shaped like: a
strain designation standing for several BRENDA records is mostly the database
holding one strain under several records, not two strains the sentence
distinguishes, so admitting only unambiguous mentions would leave a large share
of strains and bacteria ungroundable by construction. Nothing in the rule reads
the gold entity set, so a span may ground in an entity this document is not
linked to — which is the capability, not a leak.

`padded_targets` pads with `ignore_index` rather than a class: the padded
positions have no token under them, and a pad contributing to the loss would be
the divisor bug `masked_token_cross_entropy` exists to avoid.

## Batch types

`BatchItem` holds **one document's** tensors, with no batch dimension: a batch
is the `Sequence[BatchItem]` that `data.collate_documents` builds, not a stack.
Nothing in it could be stacked anyway — documents differ in how many chunks they
hold — so a model wanting a `[batch, …]` target builds it itself out of the
per-document rows. It is `total=False` because the model methods are also called
with hand-built items carrying only the fields the method under test reads.

`GroundTruth` and `BatchLogits` are one shape for every model that carries a
class head: `relations` is `None` for a model with no relation head and
populated for one that has it. Composition rather than inheritance means
both models return exactly this type instead of two different tuple arities, so
a caller no longer has to know which model it holds before it can unpack the
result.

In `BatchLosses`, `relation` is `None` for a model with no relation head and
`token` is `None` for a model with no configured token-label store. Both are
trailing so a caller reading only the tail (`*_, token = ...`) still gets the
token loss regardless of which model produced the tuple.

## Head initialisation

`initialize_classifier_bias` seeds a classifier's bias from label frequencies as
log odds. `freqs` covers the supervised labels only, in column order;
`sentinel_index` names the head's one unsupervised column — `OOS` on the class
head — which has no frequency and is seeded from `sentinel_prior` instead. It
defaults to the last column, where the models put it; pass `None` for a head
with no sentinel column.
