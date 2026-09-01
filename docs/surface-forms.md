# The surface-form dictionary

Distant supervision needs the inverse of BRENDA's entity tables: not "what is
this entity called" but "which entities could this string be". `build_index` is
that inverse, and [`d3text.token_labels`](distant-supervision.md) is its only
intended reader.

**Exact lookup, not fuzzy scoring.** `models.dict_tagger.Vocab` already matches
surface forms, and it is the wrong tool at this scale: it scores a query against
every term in a length band, which is ~50 s per fulltext over the ~160k forms
BRENDA carries, and its cutoff was calibrated against a scorer that no longer
exists. A false hit here is not a wrong prediction but a *silently mislabelled
training token*, so the trade this module wants is the opposite one — cheap and
literal. What it keeps from `dict_tagger` is the part that is a decision rather
than an algorithm: `is_symbol_like`, which lives here because the case policy is
a property of the dictionary and both readers must not drift apart on it.

The index is keyed by the *words* of a form rather than by the form itself, so
`D-3-hydroxybutyrate dehydrogenase` and `D 3 hydroxybutyrate dehydrogenase`
reach the same entry and no hyphenation convention has to be modelled.

**Deliberately a leaf**: the only `d3text` module it imports is `d3text.schema`,
which is itself a leaf, and it imports nothing from `brenda_references`, so
building an index costs neither the BRENDA data layer nor torch. The entity
tables arrive as plain mappings, which is what the TinyDB dump already is on
disk.

`BRENDA_PREFIXES` is read off the schema rather than restated, because a prefix
that disagrees with the corpus's spelling does not fail — it produces an index
whose keys no gold set can ever match, and every mention it finds is then
labelled as belonging to no annotated entity.

## Which forms carry an ID

`index_key` drops a form for five reasons: it is too short, it tokenizes to
nothing, it is longer than the sweep's widest window, it is a bare
`PLACEHOLDER_FORMS` entry, or it is one ordinary English word.

### Length

`MIN_FORM_LENGTH` is 3. One- and two-character forms are almost all element
symbols, figure labels and units; `CO` names cholesterol oxidase in BRENDA and
carbon monoxide everywhere else, and no amount of case sensitivity separates
those. `MAX_FORM_WORDS` is 8, which is also the widest window the sweep tries.

### Case is per form, not per index

`is_symbol_like` decides whether case is load-bearing. Case is the only feature
separating the enzyme symbol `FOR` from the English word `for`, `ARE` from
`are`, `HAS` from `has`; all three are real BRENDA entities, so folding case
away over the whole vocabulary trades a handful of recovered variants for a
match in nearly every sentence. Two shapes carry that risk: a short form
(`SYMBOL_MAX_LENGTH` or under), and one with a capital past its first character
(`MMP-3`, `HerE`, `CelL`) — the initial capital alone is just a sentence or a
genus and says nothing. Descriptive names (`catalase`, `cytochrome c oxidase`)
collide with no English word, so they are the population that can afford to
fold.

`SurfaceFormIndex` therefore keeps two tables, `exact` and `folded`, and
`lookup` reads both and unions the answers: a window can legitimately be a
symbol of one entity and a descriptive name of another, and choosing between
them at match time would be a guess.

### Placeholders

`PLACEHOLDER_FORMS` drops single-word forms that name no particular entity.
`More` is BRENDA's curation marker for "this enzyme has further entries
elsewhere". It is registered as a synonym of 1,123 separate enzymes and it is an
ordinary English word, so every occurrence of it in running text would resolve
to a thousand entities at once. The rest are category nouns: a mention of
"plants" links to no organism, and `protease` is the one that survives the
symbol/descriptive split, since it is long and lowercase and so folds
legitimately.

Only the *bare* form goes. A form is dropped when it is one word and that word
is in the set, so `alkaline protease` and `Bacillus strain 168` keep their IDs —
the "require a modifier" reading of the same rule.

### Ordinary English

`COMMON_WORD_ZIPF` is 3.0: the Zipf frequency above which a one-word
case-folded form names nothing. BRENDA registers ordinary English as strain
designations — `sensitive`, `original`, `yielding`, `hybrid`, `aerobic` — and as
place and surnames: `california`, `shanghai`, `berlin`, `johnson`. Each is long
enough to clear `MIN_FORM_LENGTH` and lowercase enough to fold, so neither the
length bar nor the case policy sees them, and `sensitive` alone then claims a
strain mention in a quarter of the corpus.

Frequency is the discriminating feature because the two populations barely
overlap: of 4,190 one-word folded keys in the full index only 431 register in
general English at all, the other 90% being technical names general text has no
use for. 3.0 is where the two bands meet — the bacterial genera sit just under
it (`escherichia` 2.63, `pseudomonas` 2.59, `bacillus` 2.70) and the ordinary
words just over (`aerobic` 3.19, `yielding` 3.40, `hybrid` 4.11). Measured over
the whole dictionary this drops 90 keys of 160,109 and removes 1.8 spurious
document-firings per document.

The one taxonomic casualty is `salmonella` (3.09), and it is a cheap one: the
bare genus fires on the same documents its binomials do, so the entity is still
found by `Salmonella enterica` and the genus-alone key was double counting a
single mention.

**Not a replacement for `PLACEHOLDER_FORMS`.** General frequency cannot see a
noun that is common only in this literature: `plasmid` (2.68), `protease` (2.78)
and `constitutive` (2.66) all pass this guard and name no particular entity. The
two rules cover different populations and both are needed.

`is_common_word` is asked only of forms that are a single word *and* have
already been judged descriptive enough to fold case, which is what keeps it
safe. A symbol keeps its case and is therefore never compared against the
English word it shares letters with — `FOR` the enzyme survives this while `for`
was never a key to begin with — and a multi-word form is exempt because the
modifier is what makes it specific. In `index_key` the guard is asked last, and
only of the folding branch, because it is that branch's own premise that decides
whether the question is meaningful.

It is memoized: `zipf_frequency` depends on nothing but its argument, and both
callers ask it of the same running-prose words over and over across a corpus.

### Reachability

`SurfaceFormIndex.entity_ids` is what `PLACEHOLDER_FORMS` is judged against:
dropping `More` is only safe because each of the 1,123 enzymes it stood in for
keeps a real name.

`COMMON_WORD_ZIPF` is deliberately **not** judged against it, and the difference
is the point. It costs 56 entities their last key, 52 of them strains registered
under nothing but an ordinary English word. Keeping such a key to preserve
reachability is the trade run backwards: the entity is not thereby findable,
since every occurrence of `sensitive` in the literature would answer to it, and
the mentions it manufactures are spread across the whole corpus rather than
confined to the one entity lost. A name that names everything names nothing.

## The fuzzy layer

`fuzzy_ids` is asked only of a word `lookup` already found nothing for, so it is
the layer that turns an unlisted inflection or a typo into an abstention rather
than a silent negative. Multi-word forms are out of scope: the exact index
already tolerates their internal punctuation and hyphenation via `form_words`,
and a genus already gets its abbreviated variant generated rather than
fuzzy-matched.

The two populations are searched the same way `lookup` reads them, case intact
against the symbol population and case-folded against the descriptive one, and a
hit in either contributes its entity IDs — a word can be a near-miss of a symbol
and a real word at once, and both are equally reasons to abstain.

**`FUZZY_CUTOFF` is loose by design, not calibrated.** A fuzzy hit can only ever
turn a token into `IGNORE_INDEX`, never assert a label, so the cost of a wrong
hit is one token of lost negative signal rather than a mislabelled positive.
That is what lets this cutoff be picked by inspection instead of swept against a
gold sample the way `Vocab`'s cannot be. 80 catches a single inflectional edit
on words of ordinary length — `oxidase` → `oxidases` scores 87.5,
`hydrogenase` → `hydrogenases` 91.7 — while still requiring most of the word's
characters to agree.

**`fuzz.ratio`, not `fuzz.QRatio` or `partial_ratio`.** Both alternatives
`DictTagger.match` uses for a different job are the wrong shape here. `QRatio`
applies its own case-folding and punctuation-stripping before scoring, which
duplicates and can disagree with the case policy this module already applies per
population; `ratio` is scored on exactly the string handed to it, so the symbol
population keeps its case and the folded population is compared already-folded.
`partial_ratio` scores the best-aligned *substring* of the longer string against
the shorter one, which suits a query embedded in a longer span — the wrong model
for one whole word compared against one whole candidate form, and it would let a
short candidate match as a substring of an unrelated long word (`or` scoring
high against `chlorophyll`) with no length penalty to stop it.

**`is_common_word` gates the query, not just the candidates.** A form this
common is already excluded from *being* a key, but nothing stops an ordinary
English word from scoring within the cutoff of an unrelated technical one at
this loose a threshold — `protein` reaches 80 against `prorenin` on
`fuzz.ratio` alone. Filtering the query is what keeps a cutoff loose enough to
catch `oxidases` from also catching every `protein` in the corpus.

`FUZZY_MIN_LENGTH` is 4. Below it, `fuzz.ratio`'s own length-normalization
already refuses almost everything a loose cutoff would otherwise admit (a
3-character word one edit away from a 3-character key scores at most 67), so the
floor exists to avoid the wasted lookups, not to change the outcome.

`FUZZY_CANDIDATE_MAX_TERMS` caps a first-letter bucket. The bucket is already
narrowed by first character, but a handful of letters concentrate a large share
of a 100k+ term wordlist (`s` alone holds a fifth of `strains.txt`).
`process.extractOne` is linear in the candidate count, so an unbounded bucket
turns one common initial letter into the `O(terms)` cost this module exists to
avoid; skipping the lookup on an oversized bucket costs a few missed abstentions
on the words that start with it, which is cheap next to scanning the bucket on
every word that does.

Results are memoized on the index, keyed by `(word, cutoff)`: word occurrence in
running text is Zipfian, so the same word reaches the method thousands of times
per corpus, and the answer is a pure function of that pair against the index's
immutable tables. Mutating the cache dict's *contents* does not need
`object.__setattr__` on the frozen dataclass; only reassigning the attribute
would.

`may_start` is asked once per sweep position so that the overwhelming majority
of tokens — ordinary prose — cost two set lookups rather than `MAX_FORM_WORDS`
window joins. `_singles_by_first_letter` is what keeps `fuzzy_ids` from scoring
a word against the whole population.

## Building the forms out of BRENDA

`build_index` takes a mapping from a *prefixed* ID to its forms. Prefixed
because that is the spelling the corpus uses: an entity is `enz3494` in a split
frame's `entities` column and `"3494"` in `documents.json`, and a label that has
to be compared against a document's gold set is only useful in the former.

**Genus abbreviation.** Only 37% of BRENDA's bacteria carry any synonym at all
(median 0), so the form running text actually uses — `E. coli`, `B. subtilis` —
is usually absent while the full binomial is present. `with_abbreviated_genus`
closes that gap without waiting on LPSN; without it, a measurement of the linker
measures the dictionary instead. Genus initials collide across genera, which the
index absorbs the way it absorbs every shared form: the key reaches both entity
sets.

**All three name-bearing extractors apply it**, and that uniformity is the
point: `bacteria_forms`, `strain_forms` and `other_organism_forms` index the
same species under the same two spellings, so distant supervision does not
label `E. coli` and leave `C. albicans` outside for no reason a reader could
state. On S800's hand-assigned taxids the other-organism half of the linker
answered NIL on 1017 of 1366 judged spans without it and 836 with it, at 176
newly correct answers against 5 newly wrong.

The collisions it buys are real and small. Over the whole corpus the expansion
adds 1483 keys, 48 of which newly reach more than one entity type, against 448
such keys already present: *Aliivibrio fischeri* and *Aspergillus fischeri*
share `A. fischeri`, *Hyphomicrobium vulgare* and barley share `H. vulgare`. A
third of the newly ambiguous keys are not ambiguous at all, but BRENDA holding
one organism under two records — `A. thaliana`, `H. sapiens`. None of it can
mislabel a token: a mention whose candidates disagree about the type resolves
to `IGNORE_INDEX`, so the expansion turns negatives into abstentions and
abstentions into labels, never one label into another.

`abbreviated_genus` restates the genus → initial-plus-dot convention of
`abbreviate_bacteria` in `brenda_references.utils` rather than importing it,
because this module is a leaf and that one is not — and it is guarded, where
that one is not, to forms actually opening with a binomial, so a
culture-collection number never comes back mangled. `_BINOMIAL_GENUS`'s
lookahead is that guard: `DSM 20745` and `ATCC 25922` open with no lowercase
epithet, `Candidatus Foo` capitalizes its second word, and an already
abbreviated `E. coli` has no lowercase run after its initial.

**Strains leave out `taxon`.** It names the *species*, so counting it as a
strain mention would label bacterium mentions as strain evidence. A designation
that itself opens with the binomial (`Escherichia coli K-12`) also contributes
its genus-abbreviated variant, which is the strain-qualified form running text
uses.

**Other organisms are pooled across every document on purpose.**
`documents.json` has no `other_organisms` table — the four it carries are
`documents`, `enzymes`, `bacteria` and `strains` — so the only place these names
exist is inline, one document at a time. A document that mentions an organism it
was not annotated with is the case the abstain target exists for, and that
mention can only be recognized from some *other* document's naming of it. It
is also the one namespace whose names come out of running prose rather than a
curated table, which is where an abbreviated genus is likeliest to be what the
text actually says.

`brenda_surface_forms` lets a table absent from `tables` contribute nothing
rather than raising: the tail-parse route in `load_entity_tables` cannot reach
`documents`, and a caller that only wants enzymes should not have to fabricate
the rest.

## Reading the TinyDB dump

The shipped `documents.json` is 1.1 GB of document records followed by the three
entity tables, which are its last ~8 MB. A dump larger than the tail-search
window is therefore parsed off its tail — which yields `enzymes`, `bacteria` and
`strains` but **not** `documents`, so an `oth` namespace built from that route
has to get its names from the split CSVs instead. Anything smaller is read
whole, which is what the tracked test fixture and any hand-built dump take.

`form_words` excludes underscore from its character class deliberately: `\w`
admits it, and a gene name written `pyr_C` should tokenize the way `pyr-C` does.

::: d3text.surface_forms
