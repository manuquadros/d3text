# Distant supervision: token-level targets

BRENDA has no span annotation. What it has is per-document entity links and a
table of surface forms per entity, which is enough to place token labels by
matching — but only if the labels admit what matching cannot know.

The document-level objective localizes badly: the best operating point measured
anywhere over six arms is 29.5% precision at 29.5% recall, from a head firing on
42% of all tokens. A token-level tagger needs token-level targets.

## Three outcomes, not two

| Target | When |
| --- | --- |
| an **entity type** | the token matches a surface form of an entity in *this* document's gold set, and that entity's type is the target |
| `IGNORE_INDEX` | it matches a surface form of some *other* entity |
| `OUTSIDE` | it matches nothing |

**The middle value is a target, not a class.** The tagger's output space is one
column per entity type plus `OUTSIDE` — the `O` of an ordinary tagger — and
`IGNORE_INDEX` marks tokens the loss does not read, which is why it is spelled
as torch's own `ignore_index` default rather than as an extra label.

**Two-way labelling is the trap.** Over 300 validation fulltexts a document
matches a median of 87 distinct entities against a median of 3 gold ones, so 97%
of what matches is not gold-linked; calling all of it negative teaches BRENDA's
notion of *salience* rather than entity-hood, and suppresses hardest exactly
where a novel entity resembles an uncurated one. Abstaining costs ~2.8% of
tokens and keeps ~96% of the negative signal.

`NEGATIVE` is an alias of `OUTSIDE` because "negative" is what the target *is* —
an assertion that the token names no entity — while `OUTSIDE` is what the
tagger's column is called. They have to be the same integer.

## The label space is recorded inside the artifact

`LabelSpace` reads the type set and its order off `d3text.schema.BRENDA_SCHEMA`,
and `write_label_space` stamps that order onto the store's root attributes.
`OUTSIDE` is always 0 and the types take 1, 2, 3, … in declaration order.

Nothing in an array of small integers says which column is which type, so a
store written under one order and read under another would score every type
against another type's target without a shape ever disagreeing — the same trap
`d3text.checkpoint` records a vocabulary against, for the same reason. A width
change would at least fail loudly; a re-permutation does not. `load_token_labels`
therefore takes the space it is being read under and refuses a store that
records a different one, rather than leaving the comparison to a reader's good
intentions.

The space is built from a `Schema` rather than declared, so the type set has one
definition: `d3text.datasets.brenda` derives the class head's columns from the
same object.

Codes are `int8`, which holds −128..127, so they fit until a schema declares 127
entity types; `IGNORE_INDEX` is −100 and so cannot collide with a code, which is
what keeps "the loss skips this token" orthogonal to "this token is of type t".

## Matching

`find_mentions` takes the longest match first, and matches do not overlap:
*Streptomyces griseocarneus* is one mention of one bacterium rather than that
plus a mention of the genus *Streptomyces*. The consequence worth knowing is
that a long non-gold form covering a short gold one yields `IGNORE_INDEX` where
a type was available — abstention, which is the direction this whole scheme errs
in.

`MAX_MENTION_GAP` bounds the characters allowed between two words of one
multi-word mention. The words of a form are matched against the words of the
text, so whatever punctuation separates them is not compared — which is the
point, since `3beta-hydroxysteroid: oxygen oxidoreductase` is one BRENDA synonym
written with three different separators. Bounding the gap is what stops that
indifference from joining words across a paragraph.

A `Mention`'s `entity_ids` is a set because a surface form is not owned by one
entity: `AS-A` names four separate enzymes, and a species nested inside a strain
designation yields both.

### Fuzzy mentions may only withhold a label

A word the exact index found nothing for is tried once against
`SurfaceFormIndex.fuzzy_ids` before it is left negative. That call is only ever
reached for a word already known to match no surface form outright, which keeps
the fuzzy layer's cost proportional to the exact index's misses rather than to
the whole document.

A hit is recorded as a `fuzzy` mention, which is forced to `IGNORE_INDEX`
regardless of `entity_ids`, gold or not. An uncalibrated cutoff may recover a
real variant of the *wrong* entity, so a near-miss may only withhold a label,
never assert one — a near-miss of the right entity is exactly as unverified as
one of the wrong one. An exact hit still carries the sharper claim that this
string, unmodified, is a known form, which is what still lets it become
positive.

### Resolving a mention to a type

`character_labels` carries the *type* of a gold entity a mention could be naming
— any such entity rather than all, because an ambiguous form that includes the
curated entity is evidence for it, and demanding the form be unambiguous would
throw away every acronym BRENDA shares between enzymes. Three resolutions, two
of which abstain rather than guess:

- Several candidates of the **same** type is that type. `AS-A` names four
  separate enzymes and every one of them makes the token an enzyme, so ambiguity
  about *which* entity is not ambiguity about the target.
- Gold candidates of **different** types — a species nested in a strain
  designation names both — is `IGNORE_INDEX`. A flat scheme has one code per
  token, so choosing either type would assert that the other is wrong here, and
  the token is genuinely evidence for both.
- A **gold** candidate of one type beside a **non-gold** candidate of another
  resolves to the gold one's type. The non-gold match is exactly what
  `IGNORE_INDEX` exists not to assert, and the gold link is curated fact.

## Spans and codes: one artifact, two views

Read as tokens, the targets are "per token, an entity type or `O`", so two
mentions of the *same* type with no token between them read as one span: the
separator normally supplies that token — mentions are word-aligned and BRENDA's
forms are separated by punctuation or whitespace, and punctuation is its own
token — but a space produces no token at all. The boundary is not lost in the
labeller, only in the projection, so the store keeps `mention_spans` as well:
one row per mention, `(start, end, type_code, gold)`. Flat, `BIO`, `BIOES` and a
span objective are then all derivable from one artifact, and choosing between
them stops being a property of the dataset.

**The two cannot disagree, because they are not computed twice**:
`character_labels` paints the spans and `project_onto_tokens` reads the painting
off, so the codes are downstream of the spans rather than parallel to them.
`character_labels` *is* a call to `character_labels_from_spans`.

The last two span columns are not a restatement of each other. `gold` is whether
the loss may read the mention's type at all; `type_code` is the single entity
type its *candidates* point at, `OUTSIDE` when they point at more than one. So a
mention of an entity this document was not annotated with — the case
`IGNORE_INDEX` collapses to a bare "do not look" — keeps the type it would have
been given, which is exactly what a consumer needs to weight an abstention or to
propose a candidate span. That is what makes `mentioned_types` reconstructable
from the store at all.

### Character coordinates, not token coordinates

A mention's span is a fact about the text, while a token index is a fact about a
tokenizer, a window size and a stride. A mention lying in the 20-token overlap
has two token spans and one that straddles a window boundary has none that
contains it, so token coordinates would have to choose a duplication convention
and would still truncate exactly the boundaries this record exists to keep.
Character spans also make a re-tokenization cheap — re-project and the matcher,
which is the expensive half, need not run again. The cost is that a consumer
wanting token indices must have the offset mapping, which means re-tokenizing
the document text.

`DocumentLabels.text_length` is the third thing neither view holds: a consumer
painting the spans back onto a character array — a span objective, a `BIO`
derivation — would otherwise have to guess it as the last mention's `end`,
silently shortening every document whose text outruns its last match. The codes
cannot catch that, since they come out identical under either length.

## Projection onto tokens

**Matching runs once per document, not once per window.** The 512-token windows
overlap by a 20-token stride, so a mention near a boundary lives in two of them
and one split across a boundary lives whole in neither. Labels are therefore
placed on the document's *characters* and projected onto every window's offset
map, which makes the two windows agree by construction and costs one pass over
the text rather than one per window.

A token covering characters of **one** type and any number of ignored or outside
characters takes that type: a subword straddling a mention boundary is never
asserted `OUTSIDE` on the strength of the half of it that fell outside. A token
covering **two** types — two adjacent mentions, one subword spanning both — is
`IGNORE_INDEX`, for the reason a mention naming two types is.

Special and padding tokens carry an empty `(0, 0)` span and are ignored outright
— a `[PAD]` contributing to the loss would be a divisor bug of exactly the kind
this module exists to avoid.

**The text has to be the text the encodings were built from**, which is
`d3text.corpus.document_text(abstract, fulltext)` — abstract and body joined
with a newline and *then* stripped of JATS tags. It is not `encode_split`'s
`fulltext` column, which strips tags from the body alone and never sees the
abstract; offsets taken against that string do not address the stored
`input_ids`.

## Document-level false negatives

`mentioned_types` reports every entity-type code appearing anywhere in the
spans, gold or not. A document-level negative for a type whose code shows up
there matched a dictionary form of that type without BRENDA linking it — the
false negative a document-level class loss would otherwise assert against.
`OUTSIDE` rows are dropped: they are a mention whose gold candidates disagreed
on type, so there is no type there to assert either.

Its `min_chars` gate drops a mention shorter than that many characters before
its type is counted — a short match is far likelier to be incidental than a long
one, and an "≥ 8 chars" filter was measured to cut the false-negative rate the
function otherwise reports raw. A uniform 8-character gate rescues `strains` and
`other_organisms` but not `bacteria`, whose lower prevalence means the same
residual over-abstention costs it far more precision — which is why a single
number is not always enough, and why a `code -> cutoff` mapping is accepted
beside a bare `int`. The default, 0, keeps every mention.

## The store

A **parallel HDF5 artifact keyed by pubmed id**, mirroring the encodings file
rather than riding on the split frames, for three reasons. The targets are
produced offline against a tokenizer and the BRENDA entity tables, and a frame
column would recompute both on every run. The frames carry no token geometry, so
a column could only hold character spans and would have to be projected at load
time anyway. And `BrendaDataset` narrows its frame to four columns and emits six
keys, so a new column dies at that narrowing unless both are widened — a reader
keyed on pubmed id needs neither change, since that is already how the encodings
are addressed.

One group per pubmed id, holding the per-token `codes` and the character `spans`
they were projected from. `store_token_labels` takes a `DocumentLabels` rather
than the two arrays so that a store of codes with no spans cannot be written at
all.

Each dataset is Zstd-compressed unless it is empty: a filter needs chunks and a
chunk cannot be zero-sized, so a document that matched nothing would fail to
store under Zstd — and there is nothing to compress in that case anyway.

`write_label_space` is called once, when the store is created.
`store_token_labels` refuses to write into a store that has not got it, because
targets whose meaning lives only in the code that produced them are the failure
`LabelSpace` exists to prevent — and a store already full of them cannot be
repaired, only regenerated.

`TOKEN_LABELS_FORMAT` was bumped from 1 to 2 when the mention spans joined the
per-token codes: a format-1 store keys each document to a bare code array, so it
can neither be read as a format-2 document nor be completed without re-running
the matcher. A store stamped with no version at all is either one from before
they were recorded or a file that is not one of these; the distinction does not
help, since both have to be regenerated.

::: d3text.token_labels
