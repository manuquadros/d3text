# Schema, vocabulary, checkpoints and linking

## The schema

A `Schema` is meant to be the single place that answers which entity types
exist, which prefix their IDs carry, and which relation types hold between them.
Those facts used to be spelled out once per call site and kept in step by hand.

`d3text.datasets.brenda` reads them off a schema, and so do the model
constructors: `BrendaClassificationModel` and `NERClassificationModel` derive
`self.classes` from `schema.class_names`, and `ETEBrendaModel` derives its
relation set from `schema.relation_names` / `schema.none_relation_index` instead
of a hardcoded tuple.

`BRENDA_SCHEMA` lives in the schema module rather than beside its loader because
the leaf modules need it. `d3text.corpus`, `d3text.surface_forms` and
`d3text.token_labels` all have to know which entity types exist and which prefix
their IDs wear, and none of them may import `d3text.datasets.brenda`, which
reaches the BRENDA data layer. Declared where only the loader could see it,
every leaf grew a copy of the same four names. Every name is a column of the
split frames as well as a class label, which is what lets `d3text.corpus` read a
document's gold entity set off a row without being told the column names.

The module is a leaf: it imports nothing from `d3text`, so `d3text/__init__.py`
can export it without dragging in the BRENDA data layer.

A `Schema` is frozen and built from tuples, hence hashable: a schema is
identity, not state — two runs over the same schema must be comparable, and a
mutable one could drift out of step with a model's already-sized output layers.

Three of `BRENDA_SCHEMA`'s entity types carry a `vocab_path` pointing at a
plain wordlist under `data/` (`strains.txt`, `bacteria.txt`, `enzymes.txt`) —
the dictionary each type's surface-form matcher reads.

### Column conventions the schema does and does not own

`class_names` is the order of the class head's target columns. The extra column
the head scores on top — `OOS` — is deliberately absent: it is a property of the
head, not of the data, and the models append and locate it by name themselves.

`relation_names` is the opposite case. Unlike the class head's `OOS`, the null
relation class *is* part of the schema: it is one
of the relation head's ordinary softmax columns, and the loss targets index it.
`none_relation_index` finds it by the `is_none` flag rather than by name or by
position, so a schema that names its null class something else, or declares it
first, still lands on the right column.

A `RelationType`'s `subject_types` is a tuple rather than a single name because
a relation's subject is not always one type: BRENDA's `HasEnzyme` holds between
an enzyme and whichever of a bacterium, a strain or an other-organism names it,
and a single `subject_type` cannot express that union. The null class has no
arguments at all, which is why the argument types are optional; `validate`
requires them of every other relation.

`Schema.validate` is called from `__post_init__`, so an invalid schema cannot be
built and no consumer has to remember to ask; it is public so that a schema
assembled elsewhere — parsed from a config, read back from a checkpoint — can be
re-checked at the boundary.

## The vocabulary

The class head has one column per entity type and it is **positional**: nothing
in a `state_dict` records which class owns which column.

`train` used to save the weights alone, so `evaluate` had to rebuild that order
from the corpus and land on it by luck. A *width* change fails loudly on
`load_state_dict`; a same-width repermutation does not, and scores every class
against another class's logits, reading as a mediocre model rather than a
broken one.

`Vocabulary` is that order made explicit, so it can be written into the
checkpoint beside the weights and *read back* at evaluation instead of
re-derived. It is the whole of what a checkpoint needs to be interpreted: the
class columns, in order, with their members. The head's trailing `OOS` column
is deliberately absent, exactly as `Schema.class_names` omits it. A class with
no groundable instances still holds its key: `check_fits` requires the
recorded class names to equal the schema's, and `dataset/classes` counts the
class-map keys.

The members are not decoration. `entity_ids` — their union — is the entity
vocabulary the *training split* named, which is what splits the span tagger's
detection recall into `test/detection_novelty_seen_recall` and its `unseen`
twin. Nothing about it is positional: no head has a column per entity.

**Sorting is not cosmetic.** `from_class_map` walks the types in the schema's
declaration order and *sorts* each type's IDs before laying them down, so one
training split yields one payload in every process: a `set` of strings iterates
in an order that depends on `PYTHONHASHSEED`, which CPython randomizes per
process.

`check_fits` exists because the class head's targets are built in *schema*
order (`encode_split`) while its columns are built in *vocabulary* order, so
the two orders being equal is what keeps a class scored against its own column.
Equal sets in a different order is the dangerous case and is rejected with the
rest.

`validate` runs from `__post_init__` and is public for the same reason
`Schema.validate` is: a repeated class name, or a class repeating an entity ID,
is what a truncated or hand-edited payload looks like.

`_reject_duplicates` counts rather than calling `names.count(name)` per element
as the schema module does: a class's member list runs to thousands of IDs on
the full corpus, and it is on the path of every `Vocabulary` construction.

The module is a leaf — `d3text.schema` only. `d3text.checkpoint` and the
dataset adapters sit above it.

## The checkpoint file

The keys `save` writes are listed in [the checkpoint
reference](../reference/checkpoint.md); what follows is why each is there.

The vocabulary goes in as plain builtins rather than as a pickled `Vocabulary`,
so the file stays loadable under `weights_only=True` — torch's default since
2.6, and what `load` relies on to read a checkpoint without executing anything
it contains. The digest is a string for the same reason.

`token_labels_digest` is the [surface-form index
stamp](distant-supervision.md) of the label store the run's token-level targets
came from, `None` for a run that read none. It answers the half of the
provenance question the vocabulary does not: the columns say which class owns
which logit, and nothing says which strings the store's dictionary counted as
mentions. That count *is* the detection metrics' denominator, so a checkpoint
scored against a store rebuilt from another index scores a different number of
gold spans while both existing guards stay silent — the store's own because
each store is self-consistent, the vocabulary's because the columns never
moved. `evaluate` compares the two and warns, where the store's own
`check_index` refuses: that one is about to *extend* a file whose halves would
then disagree, while this one only makes two scores incomparable, and a stale
digest must cost an evaluation its silence rather than its hours.

`encodings_digest` is the [content digest](data.md) of the encodings store the
run's inputs were read from, `None` for a store that carries none. It answers
the last of the three questions the weights cannot: the vocabulary says which
class owns which logit, the label digest says which strings the targets were
matched against, and this one says which token ids the heads were ever shown. A
store rebuilt under a newer tokenizer revision, or after `document_text`
changed what it feeds the tokenizer, holds different ids for the same documents
at the same base model, window and stride — so the store's own geometry stamp
is unchanged and the columns never moved. `evaluate` warns and scores, on the
same argument as the label digest.

Both digests are optional within the format rather than a format of their own.
Bumping for them would refuse every checkpoint already on disk, and gain
nothing: a reader that does not know a key reads exactly the checkpoint it read
before, since these fields qualify a comparison rather than interpreting a
weight.

`surface_form_index` is `train`'s own copy of the surface-form index
`DictionaryLinker` queries (see *Linking*, below), as plain builtins, like
the vocabulary (`d3text.surface_forms.index_to_payload`/
`index_from_payload`). `infer` used to rebuild this index at every run from
BRENDA's entity dump and split files, so running a trained model needed the
training-data package and about 1.8 GB of files, and linked against whatever
those files held at inference time rather than what the model trained
alongside. Shipping the index takes both out of linking: `infer` reads it back
off the checkpoint, so linking no longer needs `brenda_references` or those
files, and links against what `train` built; importing `infer` pulls in neither.
`train` builds it before loading its own data or training, skipped under
`-prof`, which writes no checkpoint. It is optional within the format on the
same argument as the two digests above — it qualifies what `infer` can do with
the weights, not how to interpret them — and is `None` for a checkpoint
written before this was recorded, or for a training run that could not
build one (`linking_corpora.brenda_index`'s warning names why); `infer`
then warns and links no span rather than falling back to a BRENDA read of
its own.

`state_dict` is stored exactly as `torch.save` received it, including the
`_orig_mod.` prefixes a checkpoint written while `train` wrapped the model in
`torch.compile` carries, which `factory.fix_keys_hook` strips on the way into an
uncompiled model. The same hook drops `_neg_inf`, a constant older checkpoints
stored as a buffer; the padding fill is now taken from the logits' dtype.

**Format 2 refuses everything older.** Format 1 and the bare `state_dict` that
predates the format key both carry the entity-linking head, whose parameters no
model this code can build has a place for, and whose `vocabulary` payload is a
different shape. Loading them for the part that still fits would put a class
head's weights into a run whose other half is missing, so `load` raises and
says to retrain.

A checkpoint whose recorded format this code does not know — a file from a
*newer* d3text — raises for the same reason. Silently reading its `state_dict`
and ignoring the rest is how a format change becomes a wrong-numbers bug rather
than an error.

## Linking

The tagger proposes typed spans; something has to turn a span into entity IDs,
and that something is deliberately **not part of the model**. It holds no
learned parameters, so it can be swapped — a dictionary today, a bi-encoder
retriever later that catches the variation edit distance misses — without
touching the model's weights. `Linker` is that seam.

Two facts of the contract are load-bearing:

- **The answer is a set, not an ID.** A surface form is not owned by one entity
  — `AS-A` names four separate enzymes — and a species nested inside a strain
  designation is meant to yield both entities rather than force a choice at link
  time. Whoever consumes the set (the relation head, an evaluation) is the one
  with the context to narrow it.
- **The empty set is an answer**, not a failure: a typed span the dictionary
  cannot resolve is a NIL mention, emitted with no ID and scored as *correct*
  exactly when the mention has no BRENDA entity.

`DictionaryLinker` matches only what the tagger proposed — a handful of lookups
per document, each against one type's slice of the index, instead of one query
per n-gram window over the whole vocabulary. That ordering is what makes linking
cheap; the index itself is [the exact, case-aware
one](surface-forms.md).

Where the index comes from differs by command. `train` builds it once, off
BRENDA's entity dump and all three data splits, verified against the
`SHA256SUMS` manifest `brenda_references` ships — the entity dump, the test
split and the manifest are inputs its own dataset never reads — and records it
in the checkpoint (`surface_form_index`, above); `infer` reads that recording
back instead of rebuilding one, so linking against a checkpoint needs neither
`brenda_references` nor the entity dump and split files the build reads. A
checkpoint carrying none — written before this was recorded, or from a
training run that could not build one (`linking_corpora.brenda_index`'s
warning names why) — leaves `infer` linking no span rather than falling back
to a BRENDA read of its own. `evaluate` keeps building its own index directly
from the data (`linking_corpora.brenda_index`), since its own test-split read
already needs `brenda_references` on that machine.

Longest-first is its disambiguation rule: over `Streptomyces griseocarneus` the
species wins and the bare genus is never emitted, because a window that long
matched and every shorter window lies inside it. Between equally long matches
nothing can choose, so their IDs are unioned. The type conditions the *filter*,
not the sweep, so nested entities of another type stay reachable from the same
span: linking `Escherichia coli K-12` as a strain yields the designation's ID,
and linking the same span as a bacterium yields the nested species.
