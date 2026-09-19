# Precomputed store formats

The three `precompute-*` commands each write one store. Every store records
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
| `overflow_to_sample_mapping` | `uint8` | `[windows]` |
| `offset_mapping` | `uint32` | `[windows, max_length, 2]`, character offsets into the document text |

A group also carries `d3text_encoding_complete = True` once every dataset
has landed; a resume rewrites a group without it.

An unstamped store that already holds documents is stamped with a warning
and used. A store stamped with another `base_model`, `max_length` or
`stride` is refused.

## Embeddings (`precompute-embeddings`, LMDB)

One value per document, keyed by PubMed id (as bytes). Opened read-only and
without a lock by the training loop.

A value is a 13-byte header followed by a blosc2 frame: magic `D3EB`, a
format version, the row count and the column count, then the matrix as
bfloat16 bit patterns compressed with zstd level 5 behind a byte shuffle.
`embeddings_store.bytes_to_tensor` refuses a value with another magic.

The key `\x00provenance` holds a JSON record — `format`, base model, window
and stride. A store already holding documents but no such record is refused,
as is one recording another geometry.

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

| Group member | Meaning |
| --- | --- |
| `codes` | Per-window, per-token `int8` type codes, `IGNORE_INDEX` where the loss must not read |
| `ambiguous` | Per-token mask of comma-joined multi-word matches |
| `spans` | One row per mention: `(start, end, type_code, gold)` in character coordinates |
| `entity_ids`, `entity_masks` | Each gold entity and its per-token mask |
| `candidate_counts`, `candidate_ids` | Every exact mention's candidate IDs, counts row-for-row with `spans` |
| `anchors` | `(span_row, window, start, end)` for each window an exact mention reaches |
| attribute `text_length` | Length of the document text the spans address; written last |

A resume skips a group holding every member and relabels any other. A store
recording a different label space, index digest or labelling rules is
refused rather than extended; so is one of an older format.

## Related

- Why the stores are stamped, and what each stamp can and cannot detect:
  [Provenance](../explanation/data.md#provenance-what-a-store-cannot-tell-you-from-its-shapes),
  [the label space](../explanation/distant-supervision.md#the-label-space-is-recorded-inside-the-artifact).
- API: [`d3text.encodings_store`, `d3text.embeddings_store`](api/data.md),
  [`d3text.token_labels`](api/distant-supervision.md).
