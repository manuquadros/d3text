# Mention-level evaluation

Scoring a span pipeline against BRENDA is three separate questions, not one
(relations, the third, keep their existing metrics in `d3text.models`):

1. **Detection** — did a predicted span's boundary and type match an annotated
   mention? Precision/recall/F1 over spans; there is no true-negative count
   because the negative space (every span nobody predicted and nobody
   annotated) is unbounded.
2. **Linking**, conditional on a correct detection — the right ID, a wrong ID,
   or NIL. NIL is a *right* answer when the mention has no BRENDA entity, and a
   wrong one when it has.
3. **The ignore set is applied, and reported as what it is.**

## Why the ignore set is masked rather than counted

The gold mentions come from distant supervision, so a mention of an entity
BRENDA did not link to the document is exactly the population
[`d3text.token_labels`](distant-supervision.md) refuses to label. Counting a hit
on one as a false positive would rebuild the very distortion the abstain target
exists to remove, so such predictions are *masked* — neither TP nor FP — and
surfaced under their own count.

The masked scores are honest for **curated** entities and blind to novel ones by
construction: the new-entity capability lives entirely inside the masked set,
and the standing proxy for it is the firing rate on that set, which
`ignore_firing` measures. Ignored mentions are known-plausible mentions
deliberately excluded from training, so the rate at which the tagger fires on
them measures generalization past the gold set — with no hand annotation
anywhere.

`DetectionScores.ignored` is kept beside the real counts because a score with a
growing masked share means something different from the same score with none,
and only the count says which was measured. `DetectionAccumulator.metrics` omits
the firing rate when the split held no ignore regions: 0/0 is not a measurement,
and an absent key cannot be mistaken for a tagger that never fired.

## The scoring rules

A predicted span is a **TP** when its `(start, end, type_code)` equals an
assertable gold mention's; an **FP** when it matches none *and* touches no
ignored mention; **masked** when it misses but overlaps the ignore set, since
nothing knows whether that hit was right. **FN** counts the assertable mentions
no prediction matched — ignored mentions are asserted by nobody, so missing one
costs nothing. Mentions are keyed by their span, which `find_mentions`'
non-overlapping sweep keeps unique on each side.

Linking judges only predictions whose span and type match an assertable gold
mention. A non-empty answer is correct when it **intersects** the mention's gold
IDs — intersection, not equality, because an ambiguous form carries every entity
it names and asserting the curated one among them is the most any linker can be
asked. NIL is correct exactly when the mention has no BRENDA entity, and the NIL
answer is split by what it met: `nil_correct` where the mention has no entity,
`nil_missed` where an ID existed and the linker declined it.

A mention the detector missed never reaches the linker, so it appears in no
linking column. That asymmetry — a detection FN is unrecoverable, a detection FP
is cheap — is the reason detection and linking are scored separately at all.

`gold_mentions` reads the type and the assertable flag off
`token_labels.mention_spans` rather than re-deriving them, so the evaluation
cannot disagree with the training targets about which mentions are gold and
which are ignored — the two are one computation. What it adds is the linking
side: the entity IDs a mention keeps are its candidates *narrowed to the gold
set*, because those are the only IDs BRENDA asserts for this document.

A `PredictedMention` with empty `entity_ids` is a NIL mention — a typed span the
linker could not ground — not a mention that skipped linking.

## Coordinates

Everything here is coordinate-agnostic: a mention is `(start, end, type)` in
whatever axis both sides share — character offsets from
`token_labels.mention_spans`, or the aggregated-token axis the model scores,
which `token_gold_mentions` / `token_predicted_mentions` decode from flat code
arrays.

**The token-axis reading inherits the projection's one loss**: two same-type
mentions with no token between them merge into one span (the reason the label
store keeps character spans at all), so token-level scores are a floor on
boundary accuracy, not the last word. Entity IDs are not recoverable from codes
either, so linking cannot be scored in that geometry — only detection and the
firing rate.

`spans_from_codes` returns `IGNORE_INDEX` runs like any other code: whether a
run is a mention or an ignore region is the caller's reading, and
`token_gold_mentions` is the caller that makes it.

`DetectionAccumulator` sums a split's scores one document at a time — overall,
per entity type, and the ignore-set diagnostics — so the metric assembly is
testable without a model anywhere near it.

::: d3text.mention_metrics
