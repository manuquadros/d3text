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

`index_keys` drops a form for five reasons: it is too short, it tokenizes to
nothing, it is longer than the sweep's widest window, it is a bare
`PLACEHOLDER_FORMS` entry, or it is one ordinary English word.

### Length

`MIN_FORM_LENGTH` is 4. One- and two-character forms are almost all element
symbols, figure labels and units; `CO` names cholesterol oxidase in BRENDA and
carbon monoxide everywhere else, and no amount of case sensitivity separates
those. Three characters is the same argument one step out, and it is where
the bar earns its keep: `PCR`, `PBS`, `LPS` and `MDR` are registered enzyme
symbols naming a method, a buffer, a polysaccharide and a resistance phenotype
in running text, and case tells none of them apart because the competing sense
is an acronym too. Three-character forms as a class were the commonest enzyme
"mentions" in every corpus measured, but under a different cast each time:
`PCR` and `PBS` headed both the microbiology sample and the BRENDA split,
`DLD`, `Yes` and `But` the psycholinguistics pool that names no enzyme by
construction. That is the argument for a bar on length rather than a list of
forms — no list drawn from one corpus would have named the next one's. The bar
costs 4,032 keys and 626 entities their last form, every one of those a strain
registered under nothing longer; no enzyme and no bacterium loses one.
`MAX_FORM_WORDS` is 8, which is also the widest window the sweep tries.

### Case is per form, not per index

`is_symbol_like` decides whether case is load-bearing. Case is the only feature
separating the enzyme symbol `CAMP` from the English word `camp`, `ChAT` from
`chat`, `CelL` from `cell`; all three are real BRENDA entities, so folding case
away over the whole vocabulary trades a handful of recovered variants for a
match in nearly every sentence. Two shapes carry that risk: a short form
(`SYMBOL_MAX_LENGTH` or under), and one with a capital past its first character
(`MMP-3`, `HerE`, `PseA`) — the initial capital alone is just a sentence or a
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
"plants" links to no organism. `protease`, `plasmid` and `archaeon` are the
ones only this set catches: each is long and lowercase, so it folds, and each
is too rare in general English for the frequency guard below. BRENDA files the
last two each as a bacterium of its own, as it does `bacterium`, and kept as a
key `plasmid` made every plasmid in the literature a mention of that one
record — abstained on wherever the record was not gold, labelled a bacterium
wherever it was. Dropping the key is not the whole fix, since the fuzzy layer
would pick the word back up; see [the fuzzy layer](#the-fuzzy-layer).

Only the *bare* form goes. A form is dropped when it is one word and that word
is in the set, so `alkaline protease` and `Bacillus strain 168` keep their IDs —
the "require a modifier" reading of the same rule.

### Ordinary English

`COMMON_WORD_ZIPF` is 3.0: the Zipf frequency above which a one-word form
names nothing. BRENDA registers ordinary English as strain designations —
`sensitive`, `original`, `yielding`, `hybrid`, `aerobic` — and as place and
surnames: `california`, `shanghai`, `berlin`, `johnson`. Each is long enough to
clear `MIN_FORM_LENGTH`, so the length bar does not see them, and `sensitive`
alone then claims a strain mention in a quarter of the corpus.

Frequency is the discriminating feature because the two populations barely
overlap: of 4,101 one-word folded keys in the full index only 341 register in
general English at all, the other 92% being technical names general text has no
use for. 3.0 is where the two bands meet — the bacterial genera sit just under
it (`escherichia` 2.63, `pseudomonas` 2.59, `bacillus` 2.70) and the ordinary
words just over (`aerobic` 3.19, `yielding` 3.40, `hybrid` 4.11). Measured over
the whole dictionary this drops 182 keys of 177,975.

The one taxonomic casualty is `salmonella` (3.09), and it is a cheap one: the
bare genus fires on the same documents its binomials do, so the entity is still
found by `Salmonella enterica` and the genus-alone key was double counting a
single mention.

**Not a replacement for `PLACEHOLDER_FORMS`.** General frequency cannot see a
noun that is common only in this literature: `plasmid` (2.68), `protease` (2.78)
and `constitutive` (2.66) all pass this guard and name no particular entity. The
two rules cover different populations and both are needed.

`is_common_word` is asked of every single-word form, whichever table it is
headed for, and `is_english_spelling` is what decides whether the question is
meaningful. `wordfreq` folds case, so its answer describes the word rather than
this spelling of it: where running text also produces the spelling — `Yes`
opening a sentence, `alpha`, `Name`, the strain designation `2019` — the
frequency is that form's own and the guard applies, and where running text
never produces it, `CAMP` or `ChAT`, the frequency is the ordinary word's and
applying it would delete the enzyme. Routing that question by table instead was
the defect: a form of `SYMBOL_MAX_LENGTH` characters or fewer is symbol-like
whatever its case, so `alpha` and `oral` were filed case-sensitively and never
asked. A multi-word form is exempt either way, because the modifier is what
makes it specific.

It is memoized: `zipf_frequency` depends on nothing but its argument, and both
callers ask it of the same running-prose words over and over across a corpus.

### Reachability

`SurfaceFormIndex.entity_ids` is what `PLACEHOLDER_FORMS` is judged against:
dropping `More` is only safe because each of the 1,123 enzymes it stood in for
keeps a real name. The exception is a record whose only name *is* a
placeholder — the bacteria BRENDA calls `plasmid`, `archaeon` and `bacterium`
— which has no real name to keep, and is unreachable for the reason the next
paragraph gives.

`COMMON_WORD_ZIPF` is deliberately **not** judged against it, and the difference
is the point. It costs 91 entities their last key, 87 of them strains registered
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

**A word carrying no letter is refused outright.** `fuzz.ratio` scores
character overlap and digits are interchangeable under it, so any number of
ordinary length reaches the cutoff of some numeric strain designation: `10000`
scores exactly 80.0 against the registered `10008`. Neither guard above covers
that — `zipf_frequency("10000")` is 0.35, and a thousands separator read into
the number carries it over the length floor, `10,000` having split into `10`
and `000` before `word_spans` joined it. A number one digit from a deposit
number is a different deposit rather than a misspelling of one, so there is
nothing there to withhold, and the literature's centrifugation speeds and
molecular weights stay the trained negatives they should be. The guard sits on
the query alone: it moves no key, so `index_digest` does not move with it and a
token-label store built before it is silently accepted, carrying every one of
the abstentions it removes. Rebuild by hand.

**A quantity is refused too, by the sweep rather than here.** A letter carries a
number past that guard, and a unit glued to it is letter enough: `3,000g` is one
word once `THOUSANDS` reads the separator into it, and scores 88.9 against a
registered `3000`. The letter cannot be what decides, since `20074T`, a deposit
number wearing its type-strain marker, has the same shape and scores 90.9
against `20074`. `is_quantity` decides by what the word is written as instead:
a number glued to one of the `UNIT_SYMBOLS` (`128bp`, `110aa`, `22min`), or a
number written with a thousands separator whatever follows it (`3,000g`,
`21,100x`). The unit list is multi-letter on purpose. One letter after a number
is how a designation is suffixed — `168T`, `10403S`, `14028s` and `210x` are
all keys, and `10403s` in running text is the *Listeria* strain rather than a
duration — while no key ends in a multi-letter unit symbol. A separator can be
read the same way because only a deposit number groups its digits and still
names something, so a word an `ACCESSION` reads into its number, as in
`DSM 22,228T` or `NRRL B-14,911T`, is never a quantity. That takes the text
around the word, which `fuzzy_ids` is not handed and could not memoize on, so
`find_mentions` drops the hit rather than `fuzzy_ids` refusing the query. A
one-letter unit written without a separator, `4500g`, keeps whatever near-hit
it has: of the two errors that is the cheap one, a lost negative rather than a
strain trained as one. The rule moves no key, so it is the labelling-rules
fingerprint and not `index_digest` that refuses a store built before it.

**A placeholder is refused as well.** Dropping its key leaves the word matching
nothing exactly, which is what sends it here, and a placeholder sits within the
cutoff of some unrelated key as readily as any word: `plasmid` scores 85.7
against the enzyme `plasmin` and `plasmids` 80.0, `archaeon` 80.0 against the
other-organism name `Archaea`, `protease` 88.9 against `proteasome`. Each hit
hands every occurrence of the word back its abstention, now as a near-miss of
some other entity. `fuzzy_ids` therefore refuses a `PLACEHOLDER_FORMS` entry in
any casing, and the entry with an `s`, since `plasmids` names no more than
`plasmid`. The gate is that narrow on purpose. Refusing every word that scores
nearer a placeholder than any key would also refuse `Bacteroidia`, a class of
bacteria, which scores 84.2 against `bacteria` and 81.8 against `Bacteroides`:
an abstention on an organism name traded for a trained negative on it, the
costly direction. A misspelt placeholder is not refused, so it is abstained on
only where it happens to sit near another key: `protase` scores 93.3 against
the enzyme `proctase`, but `plamid`, one edit from `plasmid`, reaches no key
once that one is gone, its nearest being `plasmin` at 76.9, and so is a
trained negative.

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

**Only the lengths that can reach the cutoff are scored.** `fuzz.ratio` scores
`200 * M / (q + t)` for a word of length `q` against a key of length `t`, with
`M` at most the shorter of the two, so at `FUZZY_CUTOFF` a key outside `2q/3 <=
t <= 3q/2` cannot clear the cutoff however its characters line up. Each
first-letter bucket is therefore split by length, and only the lengths
`length_band_ratios` admits are scored, the bounds rounded outwards — the band
`DictTagger` prunes by. Two things keep that from moving a single hit. The cap
is measured on the whole first-letter bucket, before the band narrows it, so a
letter skipped before is skipped still. And `process.extractOne` breaks a tie
by position, so the answer must be the key one sorted scan of the whole bucket
would reach first: each length is scored on its own, and a tie between lengths
goes to the smaller key, which is exactly that one. Which order the lengths are
visited in therefore cannot matter.

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

**EC numbers are indexed qualified.** The index is keyed by a form's words, so
a bare `5.3.2.1` enters it as the four-word key `5 3 2 1`, and every section
number, corpus size and confidence interval of that shape then names an enzyme
— in the training corpus as much as in a pool that contains no enzyme at all.
It is worse than the short symbol-like hits `MIN_FORM_LENGTH` answers, because
a multi-word key survives any filter that ignores symbol-like matches and is
then indistinguishable from a match on a written-out name. `enzyme_forms`
registers the number the way the literature writes it, `EC 5.3.2.1` or the
older `E.C. 5.3.2.1` — keyed `E C 5 3 2 1`, which ordinary text produces no
more than it does the first — and the qualifier is the whole difference.
Dropping the number instead would cost more than it saves: an unmatched span
is painted `OUTSIDE`, not withheld, so the spellings that name an enzyme
unambiguously would become trained negatives. The qualified keys replace the
bare one, so no enzyme loses reachability — but the digest moves, so a label
store built before the change is refused rather than silently mixed with
targets built after it.

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
abbreviated `E. coli` has no lowercase run after its initial. A bare
placeholder — `Agaricus sp.`, `Bacillus spp.`, `Firmicutes bacterium`, nothing
after it — is refused as well: its first word is the only one identifying it,
and `A. sp.` would be one key for every unnamed species of an `A` genus. A
designation after the placeholder keeps the form identified, so
`Paracoccus sp. N81106` still gains `P. sp. N81106`.

**Strains leave out `taxon`.** It names the *species*, so counting it as a
strain mention would label bacterium mentions as strain evidence. A designation
that itself opens with the binomial (`Escherichia coli K-12`) also contributes
its genus-abbreviated variant, which is the strain-qualified form running text
uses.

**Strains leave out letterless forms.** A bare `3577` or a lot number like
`9005-74` never reaches the index. Unlike an EC number, a strain designation
carries no `EC`-style qualifier to separate a genuine mention from a page-range
fragment or a lot number written the same way, so indexing it would train the
entity head on whatever running text happens to spell one. It costs the rare
strain whose *only* form is such a bare designation —
`negative_screen.is_descriptive` distrusts the same shape for the same reason —
but a strain in practice carries several designations and culture-collection
numbers, and it is the letter-bearing ones a document names it by.

**Strains leave out designations that describe rather than name.** BRENDA's
strain field holds protein names and phenotypes as well as strain names, and
the dump files each one as a separate strain record under every organism it
was written for, with neither a taxon nor a culture number: `CuZn-SOD` 21
times, `Mn-SOD` 19, `Fe-SOD`, `DsbA homologous` and `type S` 13 each. Such an
*anonymous* record is known by its designation alone, so the designation became
a key reaching a score of strains, none of which a text could mean by it. The
damage is worse than ambiguity. The SOD forms are enzyme synonyms too, and a
key naming two types resolves to neither. `type S` also hid a real mention: in
"wild-type S. pyogenes" the sweep splits the hyphen and, longest match first,
`type S` consumed the `S.`, so `S. pyogenes` was never found and `pyogenes` was
painted `OUTSIDE`, a trained negative on a bacterium name.

`strain_forms` therefore drops a designation from the anonymous records that
share it once `DESCRIPTOR_MIN_RECORDS` (5) of them do. Over the shipped dump
the count separates the two populations without overlap: no real strain is
shared by more than four anonymous records (`IL1403`, and `BL21(DE3)` spelled
four ways), and every designation shared by five or more is a
descriptor — the five above, `aiiA` (a gene) at 6, and `MHOM` (the host
segment of a WHO strain code) at 5. The rule empties 90 records, and no gold
strain link in the three splits lands on any of them; a bar of 4 would already
cost gold links, since `IL1403` and `BL21(DE3)` are linked in training and
validation. Counting is by index key, so `BL21(DE3)` and `BL21-DE3` are one
designation, as they are to the index. A record with a taxon or a culture
number keeps its designation however many share it: `Marburg` is seventeen
records across three species, every one with a deposit, and names a real
strain. The count cannot reach the small groups — `type A`, `GCN5` and `TruB`
share three anonymous records each, beside `ADP1` and `NEM316` — and nothing
in the table separates those. The swallowing is stopped at match time instead,
where no form may end on a genus initial (see
[Matching](distant-supervision.md#matching)), which is what keeps `type M` and
`type A` off *M. oryzae* and *A. thaliana*. The dropped keys move
`index_digest`, so a label store built before the rule must be regenerated:
`precompute-token-labels`
refuses to extend it, but `train` reads it and only records its digest, and
`evaluate` only warns where the store's digest differs from the checkpoint's.

**Strains leave out species epithets.** The dump also files the epithet of an
organism's name as a strain of its own, with neither a taxon nor a culture
number: `typhimurium`, `indica`, `mrakii`, `Glauca`, and `Japonica` twice. Each
is too few records for the count above, so each word became a key, and a bare
"Typhimurium" in running text — the serovar, not a strain — was painted as a
mention of str16702. `strain_forms` therefore drops a one-word designation
that is, case-folded, the species epithet of a binomial the dump names a
bacterium or a strain's taxon by: `typhimurium` from the synonym `Salmonella
typhimurium`, `mrakii` from the taxon `Cyberlindnera mrakii`. The epithet is
read off the dump rather than off the word's shape, because the shape does not
separate the two: BRENDA writes cultivar names lowercase too, and `gantai` (a
soybean cultivar) and `azul` (an agave variety) are gold-linked in training.

The rule reads the word, not the record, so a record with a taxon or a deposit
loses the word too: running text writes it as the epithet however the record
is filed. Over the shipped dump it drops eight designations, the six above and
two on such records. `aquatilis` sits beside `MBIC10216` on a record whose
taxon is the bare genus `Cyanobacterium`; it is left over from the strain's
former name, *Synechocystis aquatilis*, and all 109 of its bare hits in the
splits are the epithet of *Rahnella* or *Sphingomonas aquatilis*, which a rule
sparing named records would have kept as strain mentions. `Album` on str14092
is the cost: it names that *Pseudarthrobacter oxydans* strain, and goes only
because it equals the epithet of `Methylomicrobium album`. It loses nothing
today, since the common-word guard already refuses `album`, none of the 92
`album`s in the splits names the strain, and it keeps `Album ATCC14359` and
eight deposits. None of the eight records carries a gold link. An epithet the
dump names nothing by (`heidelbergensis`, `natto`) is out of reach, and a
designation of more than one word is left alone, so `serovar Typhimurium`
still names str11012.

**Other organisms are pooled across every document on purpose.**
`documents.json` has no `other_organisms` table — the four it carries are
`documents`, `enzymes`, `bacteria` and `strains` — so the only place these names
exist is inline, one document at a time. A document that mentions an organism it
was not annotated with is the case the abstain target exists for, and that
mention can only be recognized from some *other* document's naming of it. It
is also the one namespace whose names come out of running prose rather than a
curated table, which is where an abbreviated genus is likeliest to be what the
text actually says.

The pooling and the expansion are separate calls — `pooled_other_organism_names`
and `other_organism_forms` — because the expansion is only right for a caller
matching running text. Resolving `S. argus` against a nomenclature that lists
that abbreviation under some other taxon makes an entity whose binomial
resolved cleanly look contested, and there is nothing an abbreviation can add
once the full name has answered.

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

A comma is a boundary too, except between the digits of one number. `DSM 22,228`
is a single deposit number, and `DSM 22` — the window splitting it offers the
index — is a deposit BRENDA really records, held by a different strain, so the
split resolves confidently to the wrong strain rather than failing. `THOUSANDS`
therefore joins the digits around a comma only where exactly three follow it and
nothing else does: `ATCC 35984, 35983` and `NBRC 15308, 100` are lists of two
deposits, and gluing those invents an accession no collection issued. The join
happens in the *word*, not in the text — `word_spans` returns `22228` spanning
the original characters, comma included — because those offsets are what the
distant-supervision path paints labels with, and deleting the comma from the
string would shift every span after it. `form_words` is `word_spans` with the
offsets dropped, so the index and the document text cannot read a separator
differently.

## Both spellings of a deposit number

BRENDA records a culture-collection accession as `ATCC 14990` and the
literature writes `ATCC14990` in about a tenth of its mentions. A form is keyed
by its *words*, so those are two keys, and only the one BRENDA recorded was
held — a mention writing the other reached nothing at all, no wrong answer and
no partial one. `accession_spellings` therefore respells every accession a form
carries both ways and `index_keys` adds whichever second key that yields: over
the shipped dump, 20,944 more exact keys and none removed. Both directions are
needed, since BRENDA writes both — 1,436 of the accessions in its forms carry
no separator, and the text that names those writes the space.

The acronym decides, not the shape. `ACCESSION` and `COLLECTIONS` hold the same
closed list the strain evaluation reads spans with: `PAO1`, `IP 32953` and
`ST 131` are designations written exactly like deposits, and respelling them
would hand the sweep keys no collection ever issued. The list is
case-sensitive, and the respelling keeps that policy rather than inventing a
looser one, so a lowercased `atcc 14990` folds as any other form and gains
nothing. A hyphen needs no key of its own: it is already a word boundary, so
`ATCC-14990` keys as the spaced form and only the joined spelling is added.

The grammar lives here rather than beside the evaluation's reader of it
(`d3text.datasets.culture_numbers`, which imports it) because importing any
module of that package runs its `__init__` and reaches the BRENDA data layer,
which writes an `lpsn.log` into the working directory. Building an index has to
stay free of that, so the dependency runs one way only.

One consequence to know about: a joined key is a single word, so it joins the
buckets `fuzzy_ids` scores against, and a deposit number one digit from a known
one now abstains where it used to train as a negative. The largest bucket
roughly doubles and none comes near `FUZZY_CANDIDATE_MAX_TERMS`. It also makes
266 pre-existing exact keys ambiguous (0.24%), 263 of them strain–strain, where
BRENDA holds one deposit on two strain records and the added spelling is what
brings the pair under a single key.

## Fingerprinting an index

`index_digest` reads the two lookup tables — their keys sorted, and the entity
IDs sorted inside each — rather than the forms they were built from, which is
what makes it move with the extractors and with `index_keys`'s filters as well
as with the inputs. That is what lets an artifact derived from an index refuse
a later run whose index would differ; the distant-supervision page describes
the store that does it.

::: d3text.surface_forms
