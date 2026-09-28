# Why the dataset adapter and the model factory exist

## The BRENDA adapter

`d3text.schema.BRENDA_SCHEMA` is the single place that says which entity types
the corpus carries and which prefix their database IDs wear; `brenda_dataset`
derives from it everything the loader used to spell out inline — the column
list, the ID prefixes, the class column order and the per-document class
labels. Adding a fifth entity type is a line in the schema rather than four
edits that have to agree.

Reaching `brenda_dataset` pulls in the BRENDA data layer (`brenda_references`
→ `lpsn_interface`, and their database and API dependencies), so
`d3text.datasets` resolves it lazily and it is reached only where the dataset is
actually wanted; `d3text.schema` itself stays a leaf.

The schema is declared in `d3text.schema` rather than beside the adapter because
`d3text.corpus`, `d3text.surface_forms` and `d3text.token_labels` all need the
entity types and their prefixes, and none of them may import the adapter, which
reaches the BRENDA data layer.

`encode_split` assigns each split's class targets as a plain list, positionally.
A `Series` would be aligned on the split's index, and the splits do not carry a
`RangeIndex` — the corpus loaders boolean-filter them without resetting — so
under alignment every row after the first dropped one would take some other
row's labels, and the rows whose label runs past the filtered length would get
`NaN`. A split filtered down to no row at all is legal (`limit` interacting with
the loaders' `dropna` reaches it), and its `classes` column is built directly,
since pandas would type an empty list `float64` where every other label column
is `object`.

### `--limit` truncates every split, noise included

`limit` is the number of documents kept from *each* split, and `None` and `0`
both mean all of it — `None` is taken directly because that is what an unset
`--limit` is, and translating it is a step every caller would otherwise repeat.

It reaches validation and test as well as training, because a validation pass
costs more than the training pass it follows and runs every epoch: a limit the
validation loader never saw left the expensive half of a short run at full
length. Each split appends synthetic noise documents after its real ones, and
those counts are scaled by the same fraction the truncation keeps, so a slice
holds the proportion of synthetic documents the whole split holds instead of
being mostly noise. Rows carrying no text are dropped before the truncation, so
`N` is the number of documents actually trained on rather than the number read.

### Deriving versus pinning the columns

Without a `vocabulary`, the class columns and their members are derived from
the **training** split alone.

With one, that recorded order is used instead — for *every* split, labels
included. **Pinning only the model's geometry would be worse than not pinning
it at all**: `encode_split` builds each document's class targets in schema
order, so a model built on the checkpoint's columns and targets built on the
corpus's would disagree silently, which is the failure this exists to prevent.

`split_names` exists because loading a split costs a pass over its CSV, so an
evaluation — which needs no training documents once the vocabulary is recorded —
should ask only for the split it scores.

`entity_ids_by_class` gives every type a key, including one that declares
`has_ids=False`: `Vocabulary.check_fits` requires the recorded class names to
equal the schema's, and `dataset/classes` counts the class-map keys, so a type
with no groundable instances must still hold its column. The column order
itself lives in `Vocabulary.from_class_map`, which is also what a checkpoint
records.

### Relation ID prefixes

The relation pairs are keyed by IDs that `brenda_references` prefixes itself,
while the set of IDs the corpus's own classes name is built from the schema's
prefixes. Let the two disagree and no gold relation argument can be matched
against the label store's IDs either — the run trains on nothing the proposer
covers and reports it as a clean loss. `check_relation_ids` fails loudly
instead. It checks each entity type that carries IDs on its own, not the split
as a whole: a split-wide check is satisfied by one correct type's pairs even
while every pair of another type is silently unmatched.

`_reference_split` reads that spelling off the training split when there is one,
since that is the one whose relations a training run would otherwise silently
drop. An evaluation build has no training split and needs the check just as
much: a recorded vocabulary written under different prefixes than the corpus now
carries fails the same way, and scores a relation head on nothing at all.

`filter_relations` drops a pair whose argument types no relation type admits:
its label is fixed `none` by its arguments alone, so keeping it only spends the
relation loss on a constraint the schema already guarantees. It also drops a
pair the schema cannot type at all. Membership of the training split's entity
set is deliberately *not* a condition: a relation argument is a candidate entity
ID out of the label store, which that set does not bound, so culling gold by it
would leave a pair the proposer covers supervised toward `none`.

It drops empty dicts along with the pairs: an empty dict is not
the same as no relations, and the relation head would be handed a candidate list
with a hole in it. Each element is judged on its own, so a document whose first
dict loses every pair keeps whatever the later ones still hold.

## The model factory

`d3text.factory` is the seam between a `ModelConfig` and a ready-to-train
`Model`. `train`, `tune` and `evaluate` each used to spell this out themselves,
and the three copies had already drifted apart.

It lives **above** `d3text.models` rather than inside it because
`dataset_metrics` reads an `EntityRelationDataset`, and importing that pulls in
`d3text.data`. Keeping that out of `d3text.models` keeps the model classes
importable — in tests, in notebooks — without the data layer coming along.

`build_model` resolves `config.model_class` from an **explicit registry** rather
than `getattr(models, name)`, which resolved *any* attribute of the package — an
import, a helper — and so failed late or not at all.

The registry is typed by `ConfigurableModel`, a union of the concrete classes,
rather than by the `Model` base class. `Model` declares neither
`compute_batch_losses` nor `evaluate_model`, though every concrete model
implements both, so a caller holding a `Model` could not train or evaluate it
without the type checker objecting, correctly. `ETEBrendaModel` composes a
`BrendaClassificationModel` rather than subclassing it, so it is a member of
the union in its own right.

`fix_keys_hook` strips the `_orig_mod.` prefix `torch.compile` prepends to every
key. `train` now compiles only the trainable top of the trunk, in place, so
the checkpoints it writes are keyed against the model itself and the hook is a
no-op on them; it stays for the ones written while `train` wrapped the model
instead. It also drops every key ending in `_neg_inf`: older checkpoints
carried that padding-fill constant as a buffer on every `Model`, nested ones
included (`two_head._neg_inf`), and strict loading rejects it as unexpected
now that it is no longer a buffer. It must edit `state_dict`
**in place**: torch slices each child module's state dict out of that very
object after the hook returns, so a fresh dict would be built and dropped on the
floor.

### What gets logged about the build

`model_metrics` reports the trainable count because that is the one that moves
between configurations: the base transformer is frozen apart from the
`unfrozen_top_layers` a configuration opts into, so the head geometry and that
setting are all that change it. A run whose trainable count is the *whole*
model has silently trained the encoder, which is visible there and nowhere else
short of reading the checkpoint.

`dataset_metrics` logs split sizes as metrics rather than params so the run
table sorts on them numerically: the first question asked of a surprising loss
curve is whether that run saw the whole corpus or a `--limit` slice of it, and a
param sorts as a string. The document counts are what each split *planned* to
hold — it runs at setup, before anything has been read, so it cannot know how
many documents the encodings file actually backs; `coverage_metrics` logs that
from the pass that does know, under the same `dataset/` prefix.

Batch counts are deliberately absent. `TokenBudgetBatchSampler` declares no
`__len__`, so `len(loader)` raises for exactly the configuration whose batch
count would be most worth knowing. `run_epoch` counts batches as it goes and the
per-epoch rate metrics carry the total instead.

`dataset/<split>_<type>_unseen_rate` is the share of a split's distinct
entities of that type absent from `dataset.class_map` — the training split's
vocabulary, whether derived from a loaded training split or read back off a
checkpoint. A type with no entity in the split is omitted rather than divided
by zero; a type whose `class_map` entry is empty or missing still reports, at
1.0, since every one of its split entities is then unseen.
