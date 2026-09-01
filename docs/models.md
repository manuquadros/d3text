# Models: pooling, losses and the shared base

`d3text.models.base` holds `Model` — base transformer loading, AMP and
gradient checkpointing, token-embedding lookup, logit pooling — plus the loss
and metric helpers every concrete model shares. The three concrete models
(`NERClassificationModel`, `BrendaClassificationModel`, `ETEBrendaModel`) each
live in their own module and inherit directly from `Model`; none inherits from
another.

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
the entity logits — the largest tensor in the step, and twice the size of the
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

Candidate pairs are proposed per batch by the entity hard mask, so the `none`
share is a property of the current entity head rather than of the corpus: there
is no dataset frequency to precompute, and `balanced_class_weights` has to
re-derive the weights every batch. A class absent from a batch's targets would
divide by zero; its weight is never read, since `cross_entropy` gathers weights
by target value, so clamping the count is enough to keep the tensor finite.

`focal_cross_entropy` suppresses the loss from pairs the model already scores
confidently, which is most of what the hard mask proposes. Unlike a fixed class
weight it tracks the entity head: as the mask sharpens and stops emitting junk
pairs, the down-weighting relaxes on its own. `gamma == 0` is plain
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

Gradient checkpointing skips the base model: it is frozen, and only ever runs
under `no_grad` in `get_token_embeddings`, so there is no activation graph to
trade against recomputation.

## Where token embeddings come from

`Model.get_token_embeddings` has three sources, cheapest first: the in-process
cache, the precomputed embeddings store, and the frozen base model. The base
model is a pure function of the input ids and is never trained here, so the
first two are not approximations of the third in kind — only in arithmetic. The
store's matrices were computed under fp16 autocast and rounded to bf16, while
the live forward runs under `amp_dtype`, so a run that reads the store gets
slightly different activations from one that does not. It gets the *same* ones
every epoch, which the live path cannot promise either.

`cpu_cache_key` keys a cached activation by the base model that produced it.
The cache is process-wide and one process holds more than one base model: `tune`
builds a fresh model per trial and `base_model` is a sweepable field, so a
document id alone names an activation only while every consumer happens to share
a base model. Two base models of equal hidden width would otherwise serve one
trial's activations to the next.

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
explicit BERT config is the fallback — every model in `embedding_dims` is
BERT-based.

## Column conventions

The last column of the entity logits is always `UNK`, and the last column of the
class logits is always `OOS`. `label_columns` locates the sentinel *by name* and
lists every other column, which keeps loss and evaluation correct if the
sentinel ever stops being last. The registered column tensors are
non-persistent: they are derived from `self.entities` and `self.classes`, so
they must not enter a checkpoint, where an older file would then be missing the
key.

`ordered_entities` requires the entity indices to be exactly `0..N-1`: the model
treats an entity's index as a *position* in the logit vector, so anything else
would make `entities[i]` name a different entity than column `i` scores.

## What gets reported

`Model.compute_losses` returns one batch's losses keyed by objective name. Every
key is one `update` sums and optimizes and one `run_epoch` accumulates under the
same name, so **a key present in one batch of an epoch must be present in every
batch of that epoch**. `NERClassificationModel` reports only `class`,
`BrendaClassificationModel` adds `entity` (and `token` when a token-label store
is configured), and `ETEBrendaModel` adds `relation`, already scaled by that
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
about; what ranks runs is the score over the typed labels alone. `none_share` is
logged beside it because the candidate set is proposed by the *current entity
head* rather than by the corpus — the same checkpoint can face a different pair
distribution from one run to the next, and this is the only record of which one
it met.

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

Entity class detection without entity linking: it predicts entity types and
pools them to the document, but never maps a mention to a specific entity ID. It
has one objective and no schedule rides it.

### `BrendaClassificationModel`

Entity ID **and** class detection. Its `_consistency_loss` penalises predicting
an entity whose class the class head does not agree with, using only the proper
columns — UNK and OOS dropped — through the `[E-1, C-1]` class matrix.

Neither of its losses is ramped, so `step` and `epoch` are taken only to match
the shared signature.

#### The span tagger

`compute_token_loss` is **additive to the document-level losses, never a
replacement**: the pooled terms carry the gold links that are never named in the
text, which no distant supervision reaches, and this term supplies the
localization the pooled loss cannot. The mask covers the tokens matching
entities BRENDA did not link to the document, the padding, and any document the
store has no targets for.

`token_targets` gives a document the store does not hold an all-`IGNORE_INDEX`
row — skipped by the loss, warned about once per document — because a split
wider than the labelling run is a data gap, not a modelling error. A document
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

Entity ID + class detection + relation extraction.

**It composes a `BrendaClassificationModel` rather than subclassing it.** The
two used to be related by inheritance, with this class overriding almost every
method of the parent at a wider arity — exactly the shape that widens a return
type and trips mypy's `[override]` check. Composition removes the subtype
relationship instead of suppressing the check: `ground_truth`,
`get_batch_logits`, `compute_batch_losses` and `forward` all return the *same*
typed container as `BrendaClassificationModel`'s, just with the relation-related
field populated instead of `None`.

Its `__getattr__` reaches through to the composed model for the entity and class
attributes this class does not declare, so callers read `model.X` rather than
`model.two_head.X`. It is read-only by construction — nothing is ever assigned
through it — so a value that must reach the inner model on a write needs its own
property.

#### The relation loss ramp

The relation loss is the one objective in this package that rides a schedule:
`relation_loss_weight` ramps it linearly from `w0` (0.1) to 1.0 over
`ramp_epochs`, which at 0 means no ramp at all. The schedule holds the relation
head back until the entity head proposes usable pairs to classify. No other
objective rides it, here or in any other model.

It is scaled inside `compute_losses`, before `run_epoch` ever sees it, so the
generic accumulation stays oblivious to the ramp. **Validation totals are scored
under the ramp's final (t = 1) weight**, the objective the run is ramping
toward, because the trainer's early-stopping comparison reads them as one series
across epochs; only the training gradient follows the schedule.

`epoch_loss_weights` reports the unscheduled objectives at the full weight they
train under, so every objective has a curve.

#### Joining gold relations to candidate pairs

`_gold_relation_key` sorts the two argument columns ascending. Candidate pairs
come out of `torch.combinations` over sorted unique predictions, so their
columns always arrive in ascending order, while gold arguments arrive in
whatever order preprocessing stored — lexicographic on the entity-ID strings.
Joined on the raw gold order, every pair whose string order reverses its column
order (every `HasSpecies` gold, for one) could never match a candidate. Sorting
loses no direction: the string sort already discarded argument order, and the
relation label is directional by argument *type* instead.

`unscored_gold_relations` reports the gold that no scored row can account for.
`align_relation_predictions` builds its rows out of the *candidate* pairs the
entity head proposed, and gold only ever relabels a row that already exists;
gold whose triple was never proposed therefore leaves no row at all and cannot
show up in any metric computed over those rows. It is not a false negative, it
is absent, and the denominator becomes whatever the entity head chose to
propose. **A caller computing metrics must add these back as misses.**

It is deliberately not folded into `align_relation_predictions`: the loss path
consumes that function, and these relations carry no logits to backpropagate. A
relation is *out of vocabulary* when either argument is absent from
`entity_to_index`, which no relation head can fix; the rest were simply never
proposed. A gold triple repeated across a document's pair-dicts yields one
entry, since one candidate row is all it could ever have matched.

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

`GroundTruth` and `BatchLogits` are one shape for every model that carries an
entity and a class head: `relations` is `None` for a model with no relation head
and populated for one that has it. Composition rather than inheritance means
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
`sentinel_index` names the head's one unsupervised column — UNK for an entity
head, OOS for a class head — which has no frequency and is seeded from
`sentinel_prior` instead. It defaults to the last column, where both models put
it; pass `None` for a head with no sentinel column.

::: d3text.models.base

::: d3text.models.ner

::: d3text.models.entity_linking

::: d3text.models.ete

::: d3text.models.heads

::: d3text.models.token_supervision

::: d3text.models.model_types
