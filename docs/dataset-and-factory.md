# Building a dataset and a model

## The BRENDA adapter

`d3text.schema.BRENDA_SCHEMA` is the single place that says which entity types
the corpus carries and which prefix their database IDs wear; `brenda_dataset`
derives from it everything the loader used to spell out inline — the column
list, the ID prefixes, the class-matrix column order and the per-document class
labels. Adding a fifth entity type is a line in the schema rather than four
edits that have to agree.

Importing `d3text.datasets` pulls in the corpus loaders and therefore the BRENDA
data layer (`brenda_references` → `lpsn_interface`, which writes `lpsn.log` into
the cwd at import time), so import it where the dataset is actually wanted;
`d3text.schema` itself stays a leaf.

### `--limit` selects the vocabulary, not just the documents

`limit` truncates the training split, and `None` and `0` both mean all of it —
`None` is taken directly because that is what an unset `--limit` is, and
translating it is a step every caller would otherwise repeat. It selects the
entity vocabulary along with the documents, so it is a property of a *training*
run and of any run that must reproduce one, which is why passing a recorded
`vocabulary` makes it irrelevant.

### Deriving versus pinning the columns

Without a `vocabulary`, the entity columns are derived from the **training**
split alone: an entity seen only in validation or test has no column of its own
and is scored as `UNK`, which is the point of the `UNK` column.

With one, that recorded order is used instead — for *every* split, labels
included. **Pinning only the model's geometry would be worse than not pinning it
at all**: `encode_split` multi-hot-encodes each document's entities against the
index it is handed, so a model built on the checkpoint's columns and targets
built on the corpus's would disagree silently, which is the failure this exists
to prevent.

`split_names` exists because loading a split costs a pass over its CSV, so an
evaluation — which needs no training documents once the vocabulary is recorded —
should ask only for the split it scores.

`entity_ids_by_class` gives every type a key, including one that declares
`has_ids=False`: the class head is sized from that mapping, so a type with no
groundable instances must still hold its column. `build_entity_index` keeps the
name the corpus-side callers use, but the ordering itself lives in
`Vocabulary.from_class_map`, which is also what a checkpoint records.

### Relation ID prefixes

The relation pairs are keyed by IDs that `brenda_references` prefixes itself,
while the known-entity set is built from the schema's prefixes. Let the two
disagree and every pair fails the `filter_relations` membership test — the run
trains on zero relations and reports it as a clean loss. `check_relation_ids`
fails loudly instead, returning as soon as one pair lands so the healthy case
pays for a single lookup.

`_reference_split` reads that spelling off the training split when there is one,
since that is the one whose relations a training run would otherwise silently
drop. An evaluation build has no training split and needs the check just as
much: a recorded vocabulary written under different prefixes than the corpus now
carries fails the same way, and scores a relation head on nothing at all.

`filter_relations` drops empty dicts along with the pairs: an empty dict is not
the same as no relations, and the relation head would be handed a candidate list
with a hole in it. Each element is judged on its own, so a document whose first
dict loses every pair keeps whatever the later ones still hold.

## The model factory

`d3text.factory` is the seam between a `ModelConfig` plus a dataset and a
ready-to-train `Model`. `train`, `tune` and `evaluate` each used to spell this
out themselves, and the three copies had already drifted apart.

It lives **above** `d3text.models` rather than inside it: resolving a dataset
into constructor arguments needs `d3text.data`, and importing that pulls in
`brenda_references` → `lpsn_interface`. Keeping that out of `d3text.models`
keeps the model classes importable — in tests, in notebooks — without the BRENDA
data layer coming along.

`build_model` resolves `config.model_class` from an **explicit registry** rather
than `getattr(models, name)`, which was wrong twice over: a name naming no model
at all surfaced as an `AttributeError` only once the ~300 MB dataset had
finished loading, and a name matching *any* attribute of the package — an
import, a helper — resolved to it and failed later still.

`fix_keys_hook` strips the `_orig_mod.` prefix `torch.compile` prepends to every
key. `train` now compiles the model in place, so the checkpoints it writes are
keyed against the model itself and the hook is a no-op on them; it stays for the
ones written while `train` wrapped the model instead. It must edit `state_dict`
**in place**: torch slices each child module's state dict out of that very
object after the hook returns, so a fresh dict would be built and dropped on the
floor.

### What gets logged about the build

`model_metrics` reports the trainable count because that is the one that moves
between configurations: the base transformer is frozen, so the head geometry is
all that changes it. A run whose trainable count is the *whole* model has
silently trained the encoder, which is visible there and nowhere else short of
reading the checkpoint.

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

## The dictionary tagger

`DictTagger` matches token spans against per-type wordlists with a fuzzy scorer.
It is a different job from [the surface-form
index](surface-forms.md), which is exact and case-aware.

`_normalize` replaces punctuation with spaces rather than deleting it: `MMP-3`
and `MMP 3` are the same enzyme written two ways, and a scorer comparing them
raw puts them at 80. Replacing keeps the words on either side separate words;
the length is left untouched as a side effect, but `Vocab` buckets terms by
their processed length rather than resting on that.

`_Population` keys by the *processed* length for the same reason: the
cutoff-derived band bounds `len(term)` against `len(query)` as `QRatio` sees
them, so a bucket keyed by a length the scorer never sees would prune terms that
clear the cutoff. Its `scored` and `surface` lists are parallel per bucket — same
length, same order — so the search space stays the lazy chain of tuples
rapidfuzz iterates fastest, and the surface form is recovered afterwards from
the winner's position alone. Zipping them into pairs up front costs about 2.5×
per window on a full wordlist, and `match` runs once per prefix window.

`_length_band_ratios` derives the pruning band. `fuzz.QRatio` scores `200 * M /
(len(a) + len(b))`, where `M` is the length of the longest common subsequence and
so is at most `min(len(a), len(b))`. A term of length `t` therefore cannot score
above `200 * min(t, q) / (t + q)` against a query of length `q`, and reaches
exactly that when one string's characters are a subsequence of the other's.
Requiring that ceiling to reach the cutoff gives the inclusive band `q * cutoff
/ (200 - cutoff) <= t <= q * (200 - cutoff) / cutoff`.

`None` asks for no pruning at all, which is what a degenerate cutoff gets: at or
below 0 every term clears it, at or above 200 no term can, and neither has a
finite band to divide out. Scoring a term that cannot win only costs time, so
declining to prune is always the safe answer — which is also why
`_candidate_lengths` rounds its bounds *outwards*: skipping a term that could
clear the cutoff is a silent miss no score can explain.

`match` returns `None` rather than a zero score, because 0.0 is a score
rapidfuzz really returns and a caller could not otherwise tell "no candidate"
from "scored 0.0". The query is punctuation-normalized before scoring and
case-folded against the descriptive half of the wordlist — `Catalase` scores
87.5 against `catalase` raw and so misses at any usable cutoff — while the
symbol half is scored with case intact.

`AMBIGUOUS` is distinct from `"O"`: no wordlist matching at all and several
matching equally well are different facts, and a consumer that has to exclude
ambiguous spans from its targets can only do so if a match that happened is
still recorded as one. Picking one by the order the vocabularies were
constructed would only make the arbitrary answer reproducible, so every tied
label is kept in `SpanMatch.matches` and the span is marked ambiguous.

`DictTagger.from_schema` silently skips an entity type with no wordlist
(BRENDA's `other_organisms`): it is a detectable class with nothing to match it
against, and skipping it beats replacing the mapping's labels with a hard-coded
skip list.

::: d3text.datasets.brenda

::: d3text.factory

::: d3text.models.dict_tagger
