# Two rules that recur

Most of the design decisions on the other pages are instances of one of
these two.

## A store must say what produced it

A mismatched tokenizer, base model or window produces artifacts of exactly
the right shape and dtype over the wrong vocabulary or representation space,
so every store stamps its provenance and every reader checks it before
reading a row. The same argument puts the label vocabulary inside the
checkpoint: the class head is positional, and nothing in a `state_dict`
says which class owns which column.

Where it applies: [the encodings and embeddings
stores](data.md#provenance-what-a-store-cannot-tell-you-from-its-shapes),
[the token-label store](distant-supervision.md#the-label-space-is-recorded-inside-the-artifact),
[the checkpoint](schema-and-checkpoints.md#the-checkpoint-file), [the
identifier bridge](evaluation.md#scoring-linking-against-outside-identifiers).

## The divisor is the weight sum, not the element count

Every masked loss divides by what it actually read. Dividing by the whole
population instead scales each real element's loss by the share of the
batch that happened to be masked — the dilution the mask exists to remove,
reintroduced by the reduction.

Where it applies: [the three masked
losses](models.md#the-divisor-is-the-weight-sum-not-the-element-count),
[padding in the token
targets](distant-supervision.md#projection-onto-tokens), [pooling by real
token count](models.md#document-level-pooling).
