# How the splits are drawn

The three splits exist to answer one question the aggregate scores cannot:
does the model find an entity it was never trained on? Everything below follows
from what "never trained on" has to mean for this model.

## What an unseen entity is here

No parameter belongs to an entity. The span tagger is one linear layer over the
trunk's hidden states, scoring each token for an entity *type*; entity IDs
arrive only afterwards, when a proposed span is grounded against the
[surface-form dictionary](surface-forms.md). The dictionary covers all of
BRENDA, so it can ground an entity training never named, but only a span the
tagger has already proposed. What the tagger learns is which strings are
entities, from [distant-supervision targets](distant-supervision.md) that mark a
token positive where it matches a surface form of an entity the document is
linked to.

An entity is therefore unseen in the sense that matters when **none of its
surface forms was ever a positive target in training**. An ID training never
named is not enough: BRENDA registers many strains under their species' name or
a shared collection number, and a strain training never saw, written with a
form training labelled for another entity, tests memorisation, not
generalisation.

`detection_by_novelty` buckets gold mentions by ID
([evaluation](evaluation.md#detection-recall-split-by-novelty)). The splits are
drawn so that the ID test and the surface-form test give the same answer: **no
entity missing from training shares a surface form with any entity in it.**
The converse does not hold: an entity training did name can appear in
validation or test under a synonym no training paper wrote, which tests the
tagger on a new form of a known entity, and the ID buckets count that mention
as `seen`.

## Why not a random or a stratified split

A random split leaves rare entities out of training by chance, and many of them
share a form with an entity that is in training, so the unseen bucket mixes
both kinds. The count of unseen mentions per type also depends on the draw, and
validation and test drift apart with it.

Iterative multi-label stratification (Sechidis et al., 2011) spreads every label
across the splits in proportion, rarest first. That is what makes it cover the
vocabulary well, and also what makes it wrong here: a label occurring in one or
two papers goes to training, so the unseen bucket all but empties.

The splits published before this generator existed were drawn by greedy
maximum-entropy sampling, one split after another from a shrinking pool. Validation, drawn second, maximised entropy over
the rare entities training had left behind, so it measured something different
from test, and neither was built to contain unseen entities at all.

## Holding entities out

`brenda_references.sampling.entity_holdout_splits` holds out whole groups of
entities and stratifies the rest.

**Groups are closed under sharing a form.** Two entities share a form when one
dictionary lookup could return both, which `surface_forms.collision_keys` turns
into intersecting keys. Holding one entity out puts everything sharing a form
with it outside training too, and then everything sharing a form with those, so
the unit of holdout is the transitive closure. Most groups are one entity.
Collection numbers and generic enzyme symbols (`ATPase`) join most of the
corpus into a single group far too large to hold out, and it stays in
training.

**A group is eligible** when every member has a surface form (an entity with
none can never be matched, so holding it out measures nothing) and all its
members together occur in at most `max_held_documents` papers. Rare entities
are the ones that are new in practice, too: an entity the literature names
often is one the corpus has already seen.

**Holding a group out can strand another.** A held-out paper also names other
entities. One whose every paper is now held out cannot reach training, so its
group must be held out with it, or the candidate is refused. An entity with no
surface form strands nothing.

Groups are drawn at random until their papers fill `held_share` of validation
and test. The held-out papers are split evenly between the two by
stratification, and the remaining papers are stratified into all three splits.
A last pass trades places: a rare entity that stratification left outside
training although training has its form moves in, in exchange for a training
paper whose every entity occurs in training at least twice, so nothing is
stranded and the sizes hold.

**Validation and test measure the same thing.** Held-out papers are split
between them before anything else, and stratification counts two splits within
half a document of each other's desired share as tied. Compared exactly, a
split one paper larger wins every tie on a rare entity, and in a measurement it
took about twice test's unseen entities.

## Papers, not rows

BRENDA curates one reference per enzyme a paper documents, so one paper can be
several corpus rows sharing a `pubmed_id`, each carrying part of its gold set.
The generator splits papers: it unions each paper's entities, the way the
loader's `merge_duplicate_documents` does, and writes every row of a paper into
that paper's split. Split row by row, one full text could land in training and
test at once.

## What the guarantee does not cover

- **Whole forms, not words.** A held-out strain `E. coli B` shares no key with
  the bacterium `Escherichia coli`, but its words overlap with that entity's
  abbreviated form. Excluding word overlap would exclude almost every strain.
- **Lowercased keys.** `collision_keys` lowercases every key, so two symbols
  differing only in case count as one form. The error runs toward holding out
  less, never toward calling a seen form unseen.
- **Entities the dictionary cannot find** never reach the unseen bucket as gold
  mentions, whatever split they are in.

## The pool

Every full-text paper in the corpus enters the pool, except papers linked to a
bacterium but to no strain. Papers without relations are split like the rest;
they used to be appended to training after sampling.
