# Precomputed store formats

The three `precompute-*` commands each write one store, and
`precompute-embeddings` optionally a second. Every store records
what produced it, and every reader checks that record before reading a row.
Names in `code` are the exact attribute, dataset or key names on disk.

## Encodings (`precompute-encodings`, HDF5)

One group per document, keyed by PubMed id. A group written for an external
corpus (`--s800`, `--enzymener`) is keyed by `encodings_store.external_key`.

| Root attribute | Meaning |
| --- | --- |
| `d3text_encodings_format` | Provenance layout version |
| `base_model` | Model id whose tokenizer produced the ids |
| `max_length` | Tokens per window |
| `stride` | Tokens of overlap between windows |
| `content_digest` | SHA-256 over every group's key, shape and ids, in sorted key order. Absent while a pass is writing and on a store no pass finished |

| Group dataset | dtype | Shape |
| --- | --- | --- |
| `input_ids` | `uint32` | `[windows, max_length]` |
| `attention_mask` | `uint8` | `[windows, max_length]` |
| `offset_mapping` | `uint32` | `[windows, max_length, 2]`, character offsets into the document text |

A group also carries `d3text_encoding_complete = True` once every dataset
has landed; a resume rewrites a group without it.

An unstamped store that already holds documents is stamped with a warning
and used. A store stamped with another `base_model`, `max_length` or
`stride` is refused. A store stamped with an older layout version is read
and re-stamped: the one version before the current one carries an extra
per-group `overflow_to_sample_mapping` dataset, written all zeros and opened
by no reader. A version this build does not know is refused outright.

## Embeddings (`precompute-embeddings`, LMDB)

One value per document, keyed by PubMed id (as bytes). Opened read-only and
without a lock by the training loop.

A value is a 13-byte header followed by a blosc2 frame: magic `D3EB`, a
format version, the row count and the column count, then the matrix as
bfloat16 bit patterns compressed with zstd level 5 behind a byte shuffle.
`embeddings_store.bytes_to_tensor` refuses a value with another magic.

The key `\x00provenance` holds a JSON record — `format`, base model, window
and stride, plus the precision the precompute's forward ran in. That last
field is optional: a record written before it existed records none, and is
read as such rather than refused. A store already holding documents but no
record at all is refused, as is one recording another geometry; a differing
forward precision refuses nothing, being diagnostic only.

## Layer-boundary embeddings (`precompute-embeddings --layer_boundary_store`, LMDB)

Written beside the embeddings store for a run that leaves its top
`unfrozen_top_layers` encoder layers trainable. One value per document, keyed
by PubMed id (as bytes): the hidden states each window leaves the last frozen
layer with, one row per window rather than one aggregated row per document,
since the trainable layers resumed from them attend only within a window.

A value is a header followed by a blosc2 frame: magic `D3WL`, a format
version, the window, token and feature counts, then the tensor as bfloat16
bit patterns compressed as in the embeddings store.
`embeddings_store.bytes_to_windowed_tensor` refuses a value with another
magic or version.

The key `\x00layer_provenance` holds a JSON record — the embeddings store's
fields plus `frozen_layers`, the number of leading encoder layers the rows
were computed through. `precompute-embeddings` refuses to append to a store
recording another geometry or boundary, or holding documents but no record;
`-f` does not lift that refusal. The training run refuses a store recorded for
another base model or another boundary, and embeds live any document whose
stored window count disagrees with its encodings.

## Token labels (`precompute-token-labels`, HDF5)

One group per document, keyed by PubMed id, shaped like the encodings.

| Root attribute | Meaning |
| --- | --- |
| `d3text_token_labels_format` | Layout version; current value `token_labels.TOKEN_LABELS_FORMAT` |
| `label_types`, `label_prefixes`, `label_codes` | The label space: entity types in code order, their ID prefixes, and the code each type is written as |
| `ignore_index`, `outside_index` | The two reserved codes (`IGNORE_INDEX`, `OUTSIDE`) |
| `surface_form_index_digest` | `surface_forms.index_digest` of the index the targets were placed by |
| `surface_form_index_sources` | The corpus files the index pooled organism names from |
| `labelling_rules` | One `name=fingerprint` line per function the sweep runs |
| `tokenizer_base_model` | Model id whose tokenizer the codes were projected through; kept so a refusal can name it |
| `tokenizer_digest` | `token_labels.tokenizer_digest` of that tokenizer, the identity a mismatch is judged on |
| `window_length`, `window_stride` | Tokens per window and tokens of overlap the codes were projected at |

| Group member | Meaning |
| --- | --- |
| `codes` | Per-window, per-token `int8` type codes, `IGNORE_INDEX` where the loss must not read |
| `ambiguous` | Per-token mask of comma-joined multi-word matches |
| `spans` | One row per mention: `(start, end, type_code, gold)` in character coordinates |
| `entity_ids`, `entity_masks` | Each gold entity and its per-token mask |
| `candidate_counts`, `candidate_ids` | Every exact mention's candidate IDs, counts row-for-row with `spans` |
| `anchors` | `(span_row, window, start, end)` for each window an exact mention reaches |
| attribute `document_fingerprint` | Digest of this document's text and sorted gold entity IDs; absent on a group written before it existed |
| attribute `text_length` | Length of the document text the spans address; written last |

A resume skips a group holding every member and whose `document_fingerprint`
still matches the text and gold set the corpus gives that document now, and
relabels any other — including a group with no fingerprint at all, which
reads the same as a mismatch. A store recording a different label space,
index digest, labelling rules, tokenizer or window geometry is refused rather
than extended; so is one of an older format. `-f` replaces such a store with
a fresh one instead of refusing it: none of its groups is readable under the
current stamps.

## Related

- Why the stores are stamped, and what each stamp can and cannot detect:
  [Provenance](../explanation/data.md#provenance-what-a-store-cannot-tell-you-from-its-shapes),
  [the label space](../explanation/distant-supervision.md#the-label-space-is-recorded-inside-the-artifact).
- API: [`d3text.encodings_store`, `d3text.embeddings_store`](api/data.md),
  [`d3text.token_labels`](api/distant-supervision.md).
