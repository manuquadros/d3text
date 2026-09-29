# Precomputed store formats

The three `precompute-*` commands each write one store, and
`precompute-embeddings` optionally a second. Every store records
what produced it, and every reader checks that record before reading a row.
Names in `code` are the exact key or field names on disk.

## Encodings (`precompute-encodings`, LMDB)

One LMDB env, one value per document, keyed by PubMed id (as bytes). A
document of an external corpus (`--s800`, `--enzymener`) is keyed by
`encodings_store.external_key`. A document is written in one transaction,
so it is in the store whole or not at all.

A value is a 17-byte header followed by a blosc2 frame: magic `D3EN`, a
format version, then the plane, window and token counts. The frame holds
four `uint32` planes of shape `[windows, max_length]`, compressed as the
embeddings values are (below), and `encodings_store.bytes_to_encoding`
returns them as:

| Array | dtype | Shape |
| --- | --- | --- |
| `input_ids` | `uint32` | `[windows, max_length]` |
| `attention_mask` | `uint8` | `[windows, max_length]` |
| `offset_mapping` | `uint32` | `[windows, max_length, 2]`, character offsets into the document text |

A value with another magic or version is refused.

Two keys starting with a NUL byte are stamps, not documents:

| Key | Holds |
| --- | --- |
| `\x00provenance` | JSON: `format` (`_PROVENANCE_FORMAT` in `d3text.encodings_store`), `base_model`, `max_length`, `stride` |
| `\x00content_digest` | JSON string: SHA-256 over every document's key, shape and ids, in sorted key order. Absent while a pass is writing and on a store no pass finished |

A store recording another `base_model`, `max_length` or `stride` is refused
by the writer, and so is one holding documents but no provenance. A reader
refuses a store recording no provenance, another base model or another
stride. A provenance format this build does not know is refused outright.

Stores written before this layout are single HDF5 files. They are refused,
by readers and by `precompute-encodings` alike, with a message saying to
build a new store under another path or delete the old one first; nothing
migrates them. `inspect-encodings` prints what a store records.

## Embeddings (`precompute-embeddings`, LMDB)

One LMDB env per base model. Every cut of the trunk it caches is a named
sub-database of that env, keyed by PubMed id (as bytes):

| Sub-database | Holds |
| --- | --- |
| `aggregated` | One matrix per document: the last layer's hidden states, the windows aggregated into one row per token |
| `unfrozen_<n>` | One tensor per document: the hidden states each window leaves the last frozen layer with, for a run whose `unfrozen_top_layers` is `n` |

A boundary is named by its unfrozen count, the number a training config
sets, and holds one row per window rather than one per token, since the
trainable layers resumed from it attend only within a window. Both kinds are
optional: `precompute-embeddings` writes the ones its flags ask for. A
training run adds its own boundary if the env lacks it; a run with the whole
trunk frozen adds `aggregated` only to an env holding no boundary, and
otherwise derives its rows from the boundary with the fewest unfrozen layers
(see [the models page](../explanation/models.md#a-partially-trainable-trunk)).

An `aggregated` value is a 13-byte header followed by a blosc2 frame: magic
`D3EB`, a format version, the row count and the column count, then the
matrix as bfloat16 bit patterns compressed with zstd level 1 behind a byte
shuffle. An `unfrozen_<n>` value has magic `D3WL` and the window, token and
feature counts, then the tensor compressed the same way.
`embeddings_store.bytes_to_tensor` and
`embeddings_store.bytes_to_windowed_tensor` refuse a value with another magic
or version.

The main database's key `\x00provenance` holds a JSON record for the whole env
— `format` (`_PROVENANCE_FORMAT` in `d3text.embeddings_store`), base model,
window and stride, plus the precision the forward ran in. That last field is
optional: a record without it is read as recording none rather than refused. An
env already holding documents but no record at all is refused, as is one
recording another geometry; a differing forward precision refuses nothing,
being diagnostic only.

An env in the older one-cut-per-env layout — a format-1 record, or the
layer-boundary store's own `\x00layer_provenance` record — is refused with a
message saying to rebuild it; its rows are never read. The training loop
opens an env read-only and without a lock unless it has a sub-database to
add.

## Token labels (`precompute-token-labels`, LMDB)

One LMDB env, one value per document, keyed by PubMed id (as bytes), shaped
like the encodings. A document is written in one transaction, so it is in the
store whole or not at all.

A value is a 9-byte header followed by a blosc2 frame: magic `D3TL`, a
format version, then the byte length of the packed document, compressed as
the embeddings values are (above). The packed document is a 4-byte length, a
JSON record of that length, then six little-endian arrays at the shapes the
record lists:

| Field | Meaning |
| --- | --- |
| `codes` (`int8`) | Per-window, per-token type codes, `IGNORE_INDEX` where the loss must not read |
| `ambiguous` (`int8`) | Per-token mask of comma-joined multi-word matches |
| `spans` (`int32`) | One row per mention: `(start, end, type_code, gold)` in character coordinates |
| `entity_masks` (`int8`) | Each gold entity's per-token mask, in the order of the record's `entity_ids` |
| `candidate_counts` (`int32`) | How many of the record's `candidate_ids` belong to each `spans` row |
| `anchors` (`int32`) | `(span_row, window, start, end)` for each window an exact mention reaches |
| record `text_length` | Length of the document text the spans address |
| record `fingerprint` | Digest of this document's text and sorted gold entity IDs, or `null` |
| record `entity_ids`, `candidate_ids` | The gold entity IDs, and every exact mention's candidate IDs concatenated in `spans` order |

A value with another magic or version is refused.

The key `\x00provenance` holds the stamps, a JSON object:

| Field | Meaning |
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

A resume skips a document whose `fingerprint` still matches the text and gold
set the corpus gives that document now, and relabels any other — including
one with no fingerprint at all, which reads the same as a mismatch. A store
recording a different label space, index digest, labelling rules, tokenizer
or window geometry is refused rather than extended; so is one of an older
format. `-f` replaces such a store with a fresh one instead of refusing it:
none of its documents is readable under the current stamps. A run that
finishes compacts the store, so the pages its relabelled documents left
behind do not accumulate.

Stores written before this layout are single HDF5 files. They are refused,
by readers and by `precompute-token-labels` alike, with a message naming the
`-f` rerun that replaces them; nothing migrates them.

## Related

- Why the stores are stamped, and what each stamp can and cannot detect:
  [Provenance](../explanation/data.md#provenance-what-a-store-cannot-tell-you-from-its-shapes),
  [the label space](../explanation/distant-supervision.md#the-label-space-is-recorded-inside-the-artifact).
- API: [`d3text.encodings_store`, `d3text.embeddings_store`](api/data.md),
  [`d3text.token_labels`](api/distant-supervision.md).
