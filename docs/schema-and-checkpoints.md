# Schema, vocabulary, checkpoints and linking

## The schema

A `Schema` is meant to be the single place that answers which entity types
exist, which prefix their IDs carry, and which relation types hold between them.
Those facts used to be spelled out once per call site and kept in step by hand.

`d3text.datasets.brenda` reads them off a schema, and so do the model
constructors: `BrendaClassificationModel` and `NERClassificationModel` derive
`self.classes` from `schema.class_names`, and `ETEBrendaModel` derives its
relation set from `schema.relation_names` / `schema.none_relation_index` instead
of a hardcoded tuple. `DictTagger.from_schema` builds a tagger's label → vocab
mapping from the entity types' `vocab_path`s the same way.

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

### Column conventions the schema does and does not own

`class_names` is the order of the class head's target columns. The extra column
the head scores on top — `OOS` — is deliberately absent: it is a property of the
head, not of the data, and the models append and locate it by name themselves.

`relation_names` is the opposite case. Unlike the entity head's `UNK` and the
class head's `OOS`, the null relation class *is* part of the schema: it is one
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

An entity head has one column per training-split entity ID and a class head one
per entity type, and both are **positional**: nothing in a `state_dict` records
which ID owns which column.

`train` used to save the weights alone, so `evaluate` had to rebuild that order
from the corpus and land on it by luck. Anything that moves the training split
moves the columns with it — a different `--limit`, a changed `noise=`, a
`brenda_references` refresh. A *width* change fails loudly on `load_state_dict`;
a same-width repermutation does not, and scores every entity against another
entity's logits, reading as a mediocre model rather than a broken one.

`Vocabulary` is that order made explicit, so it can be written into the
checkpoint beside the weights and *read back* at evaluation instead of
re-derived. It is the whole of what a checkpoint needs to be interpreted: the
entity column order, and the class columns with their members — enough to
rebuild `entity_index` and `class_matrix` without consulting the corpus at all.
The head's trailing `UNK` column is deliberately absent, exactly as
`Schema.class_names` omits `OOS`. A class with no groundable instances still
holds its key, because the class head is sized from the mapping.

**Sorting is not cosmetic.** `from_class_map` walks the types in the schema's
declaration order and *sorts* each type's IDs before laying them down, so one
training split yields one column order in every process: a `set` of strings
iterates in an order that depends on `PYTHONHASHSEED`, which CPython randomizes
per process. `from_index` treats `entity_index` as authoritative for the column
order — it is what the labels were encoded against — while `class_map`
contributes only membership, so its sets are sorted there and their iteration
order never reaches the checkpoint.

`class_matrix` is built by walking the classes rather than by inverting them
into an entity → class dict, so an ID declared under two types lights both
columns instead of whichever the inversion happened to write last.

`check_fits` exists because the class head's targets are built in *schema* order
(`encode_split`) while its columns are built in *vocabulary* order
(`class_matrix`), so the two orders being equal is what keeps a class scored
against its own column. Equal sets in a different order is the dangerous case
and is rejected with the rest.

`disagreement_with` returns a one-line report rather than a bool: the two ways a
corpus can drift away from a checkpoint — resized and repermuted — call for
different responses from the operator, and only the first is visible in the
shapes.

`validate` runs from `__post_init__` and is public for the same reason
`Schema.validate` is. A class naming an entity that owns no column is what a
truncated or hand-edited payload looks like, and it would otherwise surface as a
`KeyError` deep inside `class_matrix`.

`_reject_duplicates` counts rather than calling `names.count(name)` per element
as the schema module does: the entity list runs to thousands of IDs on the full
corpus, and it is on the path of every `Vocabulary` construction.

The module is a leaf — torch and `d3text.schema` only. `d3text.checkpoint` and
the dataset adapters sit above it.

## The checkpoint file

`save` writes:

```python
{"d3text_checkpoint_format": 1, "state_dict": {...}, "vocabulary": {...}}
```

The vocabulary goes in as plain builtins rather than as a pickled `Vocabulary`,
so the file stays loadable under `weights_only=True` — torch's default since
2.6, and what `load` relies on to read a checkpoint without executing anything
it contains.

`state_dict` is stored exactly as `torch.save` received it, including the
`_orig_mod.` prefixes a checkpoint written while `train` wrapped the model in
`torch.compile` carries, which `factory.fix_keys_hook` strips on the way into an
uncompiled model.

**Checkpoints written before this existed still load.** `load` reports them as
`vocabulary=None` rather than refusing them, and the caller decides — the
alternative declares every existing `.pt` file dead, and the guess those
checkpoints force is at least a *loud* guess now, warned about at the point it
is made.

A checkpoint whose recorded format this code does not know — a file from a
*newer* d3text — raises instead. Silently reading its `state_dict` and ignoring
the rest is how a format change becomes a wrong-numbers bug rather than an
error.

## Linking

The tagger proposes typed spans; something has to turn a span into entity IDs,
and that something is deliberately **not part of the model**. It holds no
learned parameters, so it can be swapped — a dictionary today, a bi-encoder
retriever later that catches the variation edit distance misses — without
touching a checkpoint. `Linker` is that seam.

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

Longest-first is its disambiguation rule: over `Streptomyces griseocarneus` the
species wins and the bare genus is never emitted, because a window that long
matched and every shorter window lies inside it. Between equally long matches
nothing can choose, so their IDs are unioned. The type conditions the *filter*,
not the sweep, so nested entities of another type stay reachable from the same
span: linking `Escherichia coli K-12` as a strain yields the designation's ID,
and linking the same span as a bacterium yields the nested species.

::: d3text.schema

::: d3text.vocabulary

::: d3text.checkpoint

::: d3text.linking
