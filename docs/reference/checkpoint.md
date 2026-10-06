# Checkpoint format

`train` writes a checkpoint with `d3text.checkpoint.save`; `evaluate` reads it
with `d3text.checkpoint.load`. The file is a `torch.save` of one `dict`
holding plain builtins only, so it loads under `torch.load(...,
weights_only=True)`.

## Keys

| Key | Type | Meaning |
| --- | --- | --- |
| `d3text_checkpoint_format` | `int` | Layout version; the current value is `checkpoint.FORMAT` |
| `state_dict` | `dict[str, Tensor]` | The model's `state_dict()`, exactly as `torch.save` received it |
| `vocabulary` | `dict` | `d3text.vocabulary.Vocabulary` in plain-builtin form: the class head's column order and each class's entity IDs |
| `token_labels_digest` | `str \| None` | Surface-form index digest of the token-label store the run trained against; `None` for a run that read none |
| `labelling_rules_digest` | `str \| None` | Digest of the labelling rules recorded in that same store; `None` for a run that read none |
| `encodings_digest` | `str \| None` | Content digest of the encodings store the run read; `None` for a store that carries none |
| `surface_form_index` | `dict \| None` | `d3text.surface_forms.SurfaceFormIndex` in plain-builtin form (`exact`, `folded`, `excluded_words`), the `train`-time index `infer` links spans against and grounds relation arguments in; `None` for a run that could not build one (`linking_corpora.brenda_index`'s warning names why) |
| `relation_thresholds` | `dict[str, float] \| None` | Per typed relation, the probability at which a candidate pair takes that label (see [deciding a relation](../explanation/models.md#deciding-a-relation)); `None` for a model with no relation head and for one never calibrated, whose relations are then decided by argmax |

The three digests are optional: a checkpoint without them loads, and
`evaluate` skips the comparison it would have made, and `infer`, when the
model has a relation head and the checkpoint carries a surface-form index,
warns that nothing ties its relation grounding to the training targets.
`surface_form_index` is optional the same way: `infer` links no span and
writes no relation rather than refusing to load the checkpoint.
`relation_thresholds` too: without them relations are decided by argmax, as
before the key existed.

## What `load` refuses

| File | Result |
| --- | --- |
| A bare `state_dict` with no format key | Refused: predates format 1 and holds the entity-linking head |
| Format 1 | Refused: holds the entity-linking head; retrain |
| A format newer than `checkpoint.FORMAT` | Refused |
| No `vocabulary` key | Refused; `evaluate` does not reconstruct one |
| `relation_thresholds` not a mapping of relation names to probabilities in [0, 1] | Refused |

## Related

- Why the vocabulary and the digests are in the file:
  [Schema, vocabulary, checkpoints](../explanation/schema-and-checkpoints.md#the-checkpoint-file).
- API: [`d3text.checkpoint`, `d3text.vocabulary`](api/schema-and-checkpoints.md).
