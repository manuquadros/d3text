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

## Scoring linking against outside identifiers

The linking score above has a gold problem, and the obvious source for one is
rigged. Reconstructing a span's gold entity by looking its surface form up in
[`d3text.surface_forms`](surface-forms.md)' index and keeping whatever the
document's gold set confirms asks the very dictionary `DictionaryLinker`
queries, so the answer agrees by construction: measured that way the linker
scores 1.000 over ten thousand spans and has demonstrated nothing. Two
dictionaries agreeing is evidence; one dictionary agreeing with itself is not.

So the gold identifier has to come from a resource that never saw BRENDA's
synonym lists. `d3text.identifier_bridge` holds the join that makes that
possible — a table pairing a BRENDA entity ID with an identifier from an
outside authority: an NCBI taxid for a bacterium or another organism, an EC
number for an enzyme, a strain registry number later. There is one table per
namespace, and each records its own inside the file.

**Nothing in the bridge resolves anything.** Building the table needs the
outside resource (a 176 MB NCBI dump, or a registry's API) and belongs to a
script run on a machine that has it; the evaluation path and CI read the small
table that script emitted and import neither. That separation is why the table
exists at all rather than the resolver being called inline. The namespace is
recorded inside the file, for the same reason
[`d3text.checkpoint`](schema-and-checkpoints.md) records a vocabulary beside a
state dict: an EC table and a taxid table are both `entity_id -> string`, so
loading one where the other was meant raises nothing at all and scores every
span against an identifier that means something else.

### The subset is selected on the gold side only

A mention is judged when the outside authority's identifier pairs with exactly
one BRENDA entity — never because the linker returned one candidate. That
distinction is the whole point: keeping the spans BRENDA's index resolves
uniquely would keep exactly the spans where the linker's answer is a
singleton, and that answer would then be the gold. Selecting on the bridge
instead makes disagreement possible, which is what makes agreement mean
something.

`LinkingRule` is what the subset licenses. `INTERSECTION` is the rule the
corpus forces and stays the default, and its cost is that a linker which never
disambiguates still scores well — asserting any of nine candidates counts as
correct. `STRICT` is exact-match top-1, and is only meaningful where the gold
is known to be **one** entity on outside evidence; against a document-level
gold it would penalise a linker for ambiguity BRENDA's own tables carry. The
rule is a parameter rather than a second function because the two differ in
one predicate and share all of the NIL bookkeeping, which is the half that is
easy to get subtly wrong — a `nil_correct` that should have been a
`nil_missed` is a silent point of accuracy either way.

### One table over both organism types

BRENDA curates species in two places. The `bacteria` table is one; the other
is `other_organisms`, which has no table at all — each document carries an
inline `id -> name` column, and pooling those columns over the splits is the
only way to learn what an `oth` ID is called. S800 annotates both populations
and marks the difference nowhere, so both halves of the join live in one
table, under the one namespace they share.

That is not filing convenience. `IdentifierBridge.from_rows` refuses an entity
carrying two identifiers and `sole_entity` refuses an identifier two entities
carry, and both checks see only the rows that were loaded — so half a table
answers "exactly one" for a taxid that in fact names two entities. One taxid
in the current tables is carried by a bacterium *and* by an other organism;
kept in separate files it would be gold twice and deserve to be gold neither
time.

The type is therefore chosen on the gold side too. `score_linking` asks the
linker for the type of the entity the bridge named, and a run naming several
types judges a span only where exactly one entity across all of them carries
its identifier. Restricting a run to one type is what makes a per-type score
readable — an identifier only another type's entity carries is outside *that*
bridge, so it is counted rather than scored wrong — which is why the combined
score is a fourth run and not the sum of the others.

### The names NCBI's bacterial indexes cannot resolve

`ncbitax.resolve_tax_id` consults three indexes and every one of them is built
with `division_id == 0`, so a plant, a fungus or a vertebrate resolves to
nothing — which is the whole of what `other_organisms` holds.
`scripts/build_organism_taxid_bridge.py` builds the same normalized
name -> taxid mapping over every division from the same dump instead, and
caches it beside ncbitax's own pickles, since reading all 4.4 million names to
answer 1,762 questions is worth doing once.

Two of its rules matter to what the gold means. A normalized name two taxa
share is **dropped** rather than resolved to whichever row the dump lists last
— the genus `Oenanthe` is a bird and a plant, and a bridge row chosen by row
order is a gold nobody can check — with a scientific name beating the synonyms
it collides with, since NCBI keeps those unique per taxon. And the bacteria
half still calls `resolve_tax_id`, so its rows remain the ones already
measured rather than a second resolver's answer to the same question.

### The score is unreportable without its denominator

The filter keeps the mentions that resolve cleanly and drops the ones that do
not, and the bias runs the wrong way — symbols and abbreviations, where
linking is hard, are what get dropped. So `LinkingReport` carries no bare
accuracy: the accuracies live on `LinkingScores`, whose `total` is their own
denominator, and every rendering states the coverage beside them. "Linking
accuracy X on the Y% of mentions that resolve unambiguously" is an honest
claim; "linking accuracy X" is not.

**Detection is not being measured here.** The spans are the annotator's, so
the linker is asked the question it would face after a perfect tagger. That
isolates stage 2 deliberately: a linking score computed over a real tagger's
spans moves when the tagger moves, and there would be no way to read which
half changed. Scores are out-of-domain — the gold corpora are general
biomedical text and this project's is BRENDA's enzyme literature — so relative
comparisons between linkers transfer and absolute values do not.

### S800, and its inclusive offsets

`d3text.datasets.s800` reads 800 abstracts whose every species span carries a
taxid assigned by a human against the NCBI taxonomy, so it knows nothing about
which organisms BRENDA curates or what BRENDA calls them. It also annotates
every species mention, curated or not, which is the one property no
BRENDA-derived artifact has: a mention whose taxid pairs with no BRENDA
bacterium is either an organism outside BRENDA's curation or one the bridge
could not resolve, and `d3text.linking_eval` counts that population rather
than scoring it.

**`end` is inclusive**, and that is the trap the loader exists to absorb. For
`5833  species001:21183147  32  52  Plasmodium falciparum`, `text[32:52]` is
`'Plasmodium falciparu'` — a span one character short, which tokenizes,
matches nothing, and lowers a linking score by a few points with nothing
anywhere disagreeing. The offsets are converted to this package's half-open
convention on the way in, and `load_s800` then checks every mention against
the text it addresses rather than trusting the conversion: over all 3,708
annotations the inclusive reading is exact and the half-open one matches none,
so a disagreement means the corpus on disk is not this one.

### enzymeNER, and a gold that is itself a dictionary

The enzyme evaluation is the weakest of the three and has to be read as such.
`d3text.datasets.enzymener` reads PMC *sentences* whose enzyme mentions are
marked and **not named** — the corpus assigns no identifiers at all — so the
gold EC number comes from `d3text.datasets.expasy`, the ENZYME nomenclature,
by looking the span's surface form up. That the dictionary is Expasy's rather
than BRENDA's is the whole of what makes the score evidence; but a name Expasy
resolves wrongly is charged to the linker with nothing anywhere to separate
the two. S800's taxids are a human's answer per mention. These are a lookup.
Silver, and reported as silver wherever the number is.

**The anti-circularity rule bites differently here, and the usual guard does
not apply.** BRENDA's `ec_class` is a curated column on all 7,252 enzymes and
is perfectly 1:1 with them, so the bridge side is a pure identifier join —
and `IdentifierBridge.sole_entity`, which chose the judged subset for both
organism types, excludes nothing whatsoever. The entire subset is therefore
chosen by Expasy: a span is judged when its surface form denotes exactly one
EC number there. Selecting instead on the spans BRENDA's own index resolves
uniquely would keep exactly the spans whose gold is the linker's answer, which
is the same circularity in a more rigorous-looking dress.

**`external_id` is therefore optional.** A span whose name the nomenclature
does not hold keeps a `None` identifier and counts against `outside_bridge`
rather than leaving the corpus: dropping it would report the coverage as a
share of the names Expasy knows, which is not the population the score is
about. A span denoting several EC numbers is emitted once per number, so it
lands in `ambiguous_gold` by the same route a taxon BRENDA curates twice does.

**Its offsets are half-open** — the opposite of S800's, so the `+ 1` that
corpus needs is a one-character error here, and the two loaders cannot be
shared. Three of the 2,274 annotations, all in one sentence, address neither
reading; they are dropped and *counted* on the loaded corpus rather than
refused, since refusing over three known-bad rows delivers no measurement at
all. What the drop must not cost is the loud failure S800's check exists for,
so a corpus where more than `MISPLACED_LIMIT` of the rows miss their own
surface form — which is what reading the wrong convention looks like — is
still refused outright.

**Normalization is folding, not repair.** `expasy.normalize` folds case and
Unicode, transliterates the Greek letters Expasy spells out in Latin, and
turns hyphens into spaces; that last rule is what buys the coverage, taking
the judged population from 795 spans to 883. Two rules that looked promising
buy nothing measurable and are deliberately absent: `coenzyme A` -> `CoA` and
depluralization each add zero judged spans while collapsing keys that were
distinct. A key two EC numbers collide under stays ambiguous rather than
resolving to whichever record was read last — normalization may cost coverage
by surfacing ambiguity, and must never buy coverage by hiding it.

::: d3text.mention_metrics

::: d3text.identifier_bridge

::: d3text.linking_eval

::: d3text.datasets.s800

::: d3text.datasets.enzymener

::: d3text.datasets.expasy
