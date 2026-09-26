"""The label-store reader: space checked at open, codes carried across the
window merge by the same arithmetic that merges the embeddings."""

import gc
import types

import h5py
import numpy
import pytest
import torch
from d3text import token_labels
from d3text.models import token_supervision
from d3text.models.token_supervision import (
    TokenLabelReader,
    document_lengths,
    padded_targets,
)
from d3text.token_labels import BRENDA_LABELS, IGNORE_INDEX, DocumentLabels
from d3text.utils import WINDOW_LENGTH, WINDOW_STRIDE

NO_SPANS = numpy.zeros((0, token_labels.SPAN_COLUMNS), dtype=numpy.int32)
_STAMP = token_labels.IndexStamp(digest="test-index")
# Every store this file builds is stamped with this, except where a test
# needs another window geometry; the base-model mismatch test reads this
# same stamp back under another name.
_TOKENIZER_STAMP = token_labels.TokenizerStamp(
    base_model="model-a",
    digest="digest-a",
    window_length=WINDOW_LENGTH,
    window_stride=WINDOW_STRIDE,
)


def write_store(path, documents, space=BRENDA_LABELS):
    """A label store holding `documents` (pubmed id -> [windows, T] codes)."""
    with h5py.File(path, "w") as store:
        token_labels.write_label_space(
            store, space, stamp=_STAMP, tokenizer=_TOKENIZER_STAMP
        )
        for pubmed_id, codes in documents.items():
            codes = numpy.asarray(codes, dtype=numpy.int8)
            token_labels.store_token_labels(
                store,
                pubmed_id,
                DocumentLabels(
                    codes=codes,
                    ambiguous=numpy.zeros_like(codes),
                    spans=NO_SPANS,
                    text_length=0,
                ),
            )
    return path


def test_a_store_of_another_space_is_refused_at_open(tmp_path) -> None:
    """A permuted space re-means every integer; the reader must not serve it."""
    permuted = token_labels.LabelSpace(
        types=tuple(reversed(BRENDA_LABELS.types)),
        prefixes=tuple(reversed(BRENDA_LABELS.prefixes)),
    )
    path = write_store(tmp_path / "labels.hdf5", {}, space=permuted)

    with pytest.raises(ValueError, match="label space"):
        TokenLabelReader(path, base_model="model-a")


def test_a_store_stamped_for_another_base_model_is_refused_when_asked(
    tmp_path,
) -> None:
    """`document_codes`'s own shape check cannot catch this: a store built
    under one base model and read under another can project to the exact
    same [windows, tokens] shape, which is why the reader takes the base
    model to check against rather than relying on shape alone."""
    path = tmp_path / "labels.hdf5"
    with h5py.File(path, "w") as store:
        token_labels.write_label_space(
            store, BRENDA_LABELS, stamp=_STAMP, tokenizer=_TOKENIZER_STAMP
        )

    with pytest.raises(ValueError, match="model-a"):
        TokenLabelReader(path, base_model="model-b")


def test_a_store_stamped_at_another_window_stride_is_refused_when_asked(
    tmp_path,
) -> None:
    """`split_and_tokenize` pads every window to `max_length`, so a
    stride-only mismatch can still tile a document into the same window
    count and slip past the shape check `document_codes` runs; the reader's
    own geometry check is what catches it instead."""
    path = tmp_path / "labels.hdf5"
    with h5py.File(path, "w") as store:
        token_labels.write_label_space(
            store,
            BRENDA_LABELS,
            stamp=_STAMP,
            tokenizer=token_labels.TokenizerStamp(
                base_model="model-a",
                digest="digest-a",
                window_length=WINDOW_LENGTH,
                window_stride=WINDOW_STRIDE + 1,
            ),
        )

    with pytest.raises(ValueError, match="stride"):
        TokenLabelReader(path, base_model="model-a")


def test_a_store_stamped_for_the_same_base_model_is_accepted(
    tmp_path,
) -> None:
    path = tmp_path / "labels.hdf5"
    with h5py.File(path, "w") as store:
        token_labels.write_label_space(
            store, BRENDA_LABELS, stamp=_STAMP, tokenizer=_TOKENIZER_STAMP
        )

    reader = TokenLabelReader(path, base_model="model-a")

    assert reader.space == BRENDA_LABELS


def test_codes_ride_the_same_window_merge_as_the_embeddings(tmp_path) -> None:
    """Two 32-token windows under the 20-token stride: the first window
    keeps its half of the overlap, the second supplies the rest — element
    for element what `aggregate_embeddings` selects for the embeddings."""
    codes = numpy.zeros((2, 32), dtype=numpy.int8)
    codes[0] = numpy.arange(32)
    codes[1] = 64 + numpy.arange(32)
    reader = TokenLabelReader(
        write_store(tmp_path / "labels.hdf5", {"77": codes}),
        base_model="model-a",
    )

    aggregated = reader.document_codes("77", numpy.ones((2, 32)))

    assert aggregated is not None
    assert aggregated.dtype == torch.int64
    assert aggregated.tolist() == list(range(1, 21)) + list(range(75, 95))


def test_a_collated_mask_is_flattened_before_the_merge(tmp_path) -> None:
    """The DataLoader path hands the mask as [1, windows, T]; the reader must
    read it as the [windows, T] it masks."""
    codes = numpy.zeros((1, 32), dtype=numpy.int8)
    codes[0, 5] = 2
    reader = TokenLabelReader(
        write_store(tmp_path / "labels.hdf5", {"77": codes}),
        base_model="model-a",
    )

    aggregated = reader.document_codes("77", torch.ones((1, 1, 32)))

    assert aggregated is not None
    assert aggregated.shape[0] == 30  # 32 minus [CLS] and [SEP]
    assert aggregated[4] == 2


def test_a_document_the_store_lacks_is_none(tmp_path) -> None:
    reader = TokenLabelReader(
        write_store(tmp_path / "labels.hdf5", {}), base_model="model-a"
    )

    assert reader.document_codes("404", numpy.ones((1, 32))) is None


def write_store_with_spans(path, spans_by_document, space=BRENDA_LABELS):
    """A label store holding one row of `spans` per document, no codes."""
    with h5py.File(path, "w") as store:
        token_labels.write_label_space(
            store, space, stamp=_STAMP, tokenizer=_TOKENIZER_STAMP
        )
        for pubmed_id, spans in spans_by_document.items():
            rows = numpy.asarray(spans, dtype=numpy.int32).reshape(
                -1, token_labels.SPAN_COLUMNS
            )
            token_labels.store_token_labels(
                store,
                pubmed_id,
                DocumentLabels(
                    codes=numpy.zeros((0,), dtype=numpy.int8),
                    ambiguous=numpy.zeros((0,), dtype=numpy.int8),
                    spans=rows,
                    text_length=0,
                    candidate_ids=(frozenset(),) * rows.shape[0],
                ),
            )
    return path


def test_mentioned_types_reads_the_spans_regardless_of_gold(tmp_path) -> None:
    enzyme = BRENDA_LABELS.code_of("enz1")
    bacterium = BRENDA_LABELS.code_of("bac3")
    reader = TokenLabelReader(
        write_store_with_spans(
            tmp_path / "labels.hdf5",
            {"77": [(0, 8, enzyme, 1), (9, 20, bacterium, 0)]},
        ),
        base_model="model-a",
    )

    assert reader.mentioned_types("77") == {enzyme, bacterium}


def test_mentioned_types_of_a_document_the_store_lacks_is_none(
    tmp_path,
) -> None:
    reader = TokenLabelReader(
        write_store_with_spans(tmp_path / "labels.hdf5", {}),
        base_model="model-a",
    )

    assert reader.mentioned_types("404") is None


def test_mentioned_types_min_chars_gates_per_type(tmp_path) -> None:
    enzyme = BRENDA_LABELS.code_of("enz1")
    bacterium = BRENDA_LABELS.code_of("bac3")
    reader = TokenLabelReader(
        write_store_with_spans(
            tmp_path / "labels.hdf5",
            # enzyme: 3 chars; bacterium: 10 chars
            {"77": [(0, 3, enzyme, 1), (10, 20, bacterium, 0)]},
        ),
        base_model="model-a",
    )

    assert reader.mentioned_types(
        "77", min_chars={enzyme: 8, bacterium: 8}
    ) == {bacterium}


def test_mentioned_types_min_chars_drops_a_short_span(tmp_path) -> None:
    """A short match should not, on its own, carry a type through the gate
    that feeds the class-negative abstention mask."""
    enzyme = BRENDA_LABELS.code_of("enz1")
    bacterium = BRENDA_LABELS.code_of("bac3")
    reader = TokenLabelReader(
        write_store_with_spans(
            tmp_path / "labels.hdf5",
            # enzyme span is 3 chars long, bacterium span is 11
            {"77": [(0, 3, enzyme, 1), (9, 20, bacterium, 0)]},
        ),
        base_model="model-a",
    )

    assert reader.mentioned_types("77") == {enzyme, bacterium}
    assert reader.mentioned_types("77", min_chars=8) == {bacterium}


def test_mismatched_window_geometry_raises(tmp_path) -> None:
    """Codes stored against other encodings would land on the wrong tokens."""
    reader = TokenLabelReader(
        write_store(
            tmp_path / "labels.hdf5",
            {"77": numpy.zeros((2, 32), dtype=numpy.int8)},
        ),
        base_model="model-a",
    )

    with pytest.raises(ValueError, match="different encodings"):
        reader.document_codes("77", numpy.ones((3, 32)))


def test_entity_positions_reads_the_entitys_own_mask(tmp_path) -> None:
    """The aggregated-axis positions come back sorted, keyed by entity ID
    rather than by type code, and a document or entity the store lacks reads
    back as None."""
    mask_a = numpy.zeros((1, 32), dtype=numpy.int8)
    mask_a[0, 5] = 1
    mask_b = numpy.zeros((1, 32), dtype=numpy.int8)
    mask_b[0, 20] = 1
    path = tmp_path / "labels.hdf5"
    with h5py.File(path, "w") as store:
        token_labels.write_label_space(
            store, BRENDA_LABELS, stamp=_STAMP, tokenizer=_TOKENIZER_STAMP
        )
        token_labels.store_token_labels(
            store,
            "77",
            DocumentLabels(
                codes=numpy.zeros((1, 32), dtype=numpy.int8),
                ambiguous=numpy.zeros((1, 32), dtype=numpy.int8),
                spans=NO_SPANS,
                text_length=0,
                entity_token_masks={"enz1": mask_a, "enz2": mask_b},
            ),
        )
    reader = TokenLabelReader(path, base_model="model-a")

    positions = reader.entity_positions("77", "enz1", numpy.ones((1, 32)))

    assert positions is not None
    assert positions.tolist() == [4]  # aggregated position of token 5
    assert reader.entity_positions("77", "oth99", numpy.ones((1, 32))) is None
    assert reader.entity_positions("404", "enz1", numpy.ones((1, 32))) is None


def test_entity_positions_loads_a_documents_label_group_once(
    tmp_path, monkeypatch
) -> None:
    """Two gold entities read off the same document must cost one HDF5 group
    read, not one per entity: `_gold_entity_positions` calls
    `entity_positions` once per gold entity per document in a batch, so an
    uncached `_load` would reread the same document's codes, spans and
    entity masks once per entity."""
    mask_a = numpy.zeros((1, 32), dtype=numpy.int8)
    mask_a[0, 5] = 1
    mask_b = numpy.zeros((1, 32), dtype=numpy.int8)
    mask_b[0, 20] = 1
    path = tmp_path / "labels.hdf5"
    with h5py.File(path, "w") as store:
        token_labels.write_label_space(
            store, BRENDA_LABELS, stamp=_STAMP, tokenizer=_TOKENIZER_STAMP
        )
        token_labels.store_token_labels(
            store,
            "77",
            DocumentLabels(
                codes=numpy.zeros((1, 32), dtype=numpy.int8),
                ambiguous=numpy.zeros((1, 32), dtype=numpy.int8),
                spans=NO_SPANS,
                text_length=0,
                entity_token_masks={"enz1": mask_a, "enz2": mask_b},
            ),
        )
    reader = TokenLabelReader(path, base_model="model-a")

    real_load = token_labels.load_token_labels
    calls: list[str] = []

    def spy(store, key, space):
        calls.append(key)
        return real_load(store, key, space)

    monkeypatch.setattr(token_labels, "load_token_labels", spy)

    mask = numpy.ones((1, 32))
    first = reader.entity_positions("77", "enz1", mask)
    second = reader.entity_positions("77", "enz2", mask)

    assert first is not None and second is not None
    assert calls == ["77"]


def test_entity_positions_aggregates_the_window_geometry_once(
    tmp_path, monkeypatch
) -> None:
    """Two gold entities of one document must share a single
    `aggregate_embeddings` call for the window geometry: the merge depends
    only on the mask, identical for both entities, so the second lookup
    gathers the cached selection instead of re-running it."""
    mask_a = numpy.zeros((1, 32), dtype=numpy.int8)
    mask_a[0, 5] = 1
    mask_b = numpy.zeros((1, 32), dtype=numpy.int8)
    mask_b[0, 20] = 1
    path = tmp_path / "labels.hdf5"
    with h5py.File(path, "w") as store:
        token_labels.write_label_space(
            store, BRENDA_LABELS, stamp=_STAMP, tokenizer=_TOKENIZER_STAMP
        )
        token_labels.store_token_labels(
            store,
            "77",
            DocumentLabels(
                codes=numpy.zeros((1, 32), dtype=numpy.int8),
                ambiguous=numpy.zeros((1, 32), dtype=numpy.int8),
                spans=NO_SPANS,
                text_length=0,
                entity_token_masks={"enz1": mask_a, "enz2": mask_b},
            ),
        )
    reader = TokenLabelReader(path, base_model="model-a")

    real_aggregate = token_supervision.aggregate_embeddings
    calls: list[int] = []

    def spy(*args, **kwargs):
        calls.append(1)
        return real_aggregate(*args, **kwargs)

    monkeypatch.setattr(token_supervision, "aggregate_embeddings", spy)

    mask = numpy.ones((1, 32))
    first = reader.entity_positions("77", "enz1", mask)
    second = reader.entity_positions("77", "enz2", mask)

    assert first is not None and second is not None
    assert len(calls) == 1


def _document_with_heavy_candidate_ids(prefix: str) -> DocumentLabels:
    """A document whose `candidate_ids` dwarfs its array fields.

    200 mentions of 5 candidate IDs each cost ~190 KB in `candidate_ids`
    against ~3 KB of `codes`/`spans` -- realistic mention density, sized so a
    cost function that drops `candidate_ids` and one that doesn't disagree by
    two orders of magnitude, not a rounding difference a loose budget could
    paper over.
    """
    n_mentions = 200
    candidate_ids = tuple(
        frozenset(f"{prefix}{row}_{member}" for member in range(5))
        for row in range(n_mentions)
    )
    spans = numpy.zeros(
        (n_mentions, token_labels.SPAN_COLUMNS), dtype=numpy.int32
    )
    return DocumentLabels(
        codes=numpy.zeros((1, 8), dtype=numpy.int8),
        ambiguous=numpy.zeros((1, 8), dtype=numpy.int8),
        spans=spans,
        text_length=0,
        candidate_ids=candidate_ids,
    )


@pytest.fixture(scope="module")
def store_past_the_old_budget(tmp_path_factory) -> tuple[str, list[str]]:
    """A store whose decoded label groups total well over 64 MB -- the byte
    budget the reader's cache used to decline everything past -- plus one
    more document, `gold`, carrying three gold entities. Sixty documents of
    600 512-token windows with two entity masks each: 1.2 MB decoded, 73 MB
    in all, and `gold` is 1.5 MB, larger than whatever such a budget had
    left once they filled it."""
    windows, tokens = 600, 512
    path = tmp_path_factory.mktemp("labels") / "labels.hdf5"
    mask = numpy.zeros((windows, tokens), dtype=numpy.int8)
    mask[0, 5] = 1
    codes = numpy.zeros((windows, tokens), dtype=numpy.int8)
    big = DocumentLabels(
        codes=codes,
        ambiguous=codes,
        spans=NO_SPANS,
        text_length=0,
        entity_token_masks={"enz1": mask, "bac1": mask},
    )
    gold = DocumentLabels(
        codes=codes,
        ambiguous=codes,
        spans=NO_SPANS,
        text_length=0,
        entity_token_masks={"enz1": mask, "enz2": mask, "bac1": mask},
    )
    keys = [str(1000 + index) for index in range(60)]
    with h5py.File(path, "w") as store:
        token_labels.write_label_space(
            store, BRENDA_LABELS, stamp=_STAMP, tokenizer=_TOKENIZER_STAMP
        )
        for key in keys:
            token_labels.store_token_labels(store, key, big)
        token_labels.store_token_labels(store, "gold", gold)
    return str(path), keys


def test_every_document_still_hits_past_the_old_64_mb_budget(
    store_past_the_old_budget,
) -> None:
    """The cache is bounded by the dataset, not a byte budget: with far more
    label bytes resident than the 64 MB the old budget allowed, a second pass
    over the same documents must miss on none of them. A budgeted cache
    declined every document past its fill and re-read each one from HDF5 on
    every later lookup for the rest of the run."""
    path, keys = store_past_the_old_budget
    reader = TokenLabelReader(path, base_model="model-a")

    for _ in range(2):
        for key in keys:
            assert reader.mentioned_types(key) == set()

    cache = reader._label_cache
    assert cache._used > 64_000_000
    assert cache.misses == len(keys)
    assert cache.hits == len(keys)


def test_a_documents_gold_entities_cost_one_group_read_past_the_old_budget(
    store_past_the_old_budget, monkeypatch
) -> None:
    """`_gold_entity_positions` reads a document's group exactly once however
    many gold entities it carries and however much the cache already holds:
    under the old budget, a document looked up once the cache was full was
    declined, and `entity_positions`' per-entity `_load` then re-read the
    whole group once per entity, on every batch the document appeared in."""
    path, keys = store_past_the_old_budget
    reader = TokenLabelReader(path, base_model="model-a")
    for key in keys:
        reader.mentioned_types(key)

    real_load = token_labels.load_token_labels
    calls: list[str] = []

    def spy(store, key, space):
        calls.append(key)
        return real_load(store, key, space)

    monkeypatch.setattr(token_labels, "load_token_labels", spy)

    mask = numpy.ones((600, 512))
    first = reader._gold_entity_positions("gold", mask)
    second = reader._gold_entity_positions("gold", mask)

    assert set(first) == set(second) == {"enz1", "enz2", "bac1"}
    assert calls == ["gold"]


def test_repeated_reads_of_a_cached_document_retain_no_objects(
    tmp_path,
) -> None:
    """Reading a document's fields again and again must leave nothing behind
    once the cache holds it. The package is beartyped at import, and the
    hook decorates a nested `def` every time it runs and memoises the result
    by function object, so a per-call closure in the read path was held for
    the life of the process together with what it closed over -- an int64
    copy of the document's window mask, per field per lookup -- which is
    what drove a training run's host memory past the OOM killer while every
    cache counter read as healthy."""
    mask_a = numpy.zeros((4, 32), dtype=numpy.int8)
    mask_a[0, 5] = 1
    path = tmp_path / "labels.hdf5"
    with h5py.File(path, "w") as store:
        token_labels.write_label_space(
            store, BRENDA_LABELS, stamp=_STAMP, tokenizer=_TOKENIZER_STAMP
        )
        token_labels.store_token_labels(
            store,
            "77",
            DocumentLabels(
                codes=numpy.zeros((4, 32), dtype=numpy.int8),
                ambiguous=numpy.zeros((4, 32), dtype=numpy.int8),
                spans=NO_SPANS,
                text_length=0,
                entity_token_masks={"enz1": mask_a},
            ),
        )
    reader = TokenLabelReader(path, base_model="model-a")
    mask = numpy.ones((4, 32))

    def read() -> None:
        reader.document_codes("77", mask)
        reader.document_ambiguous("77", mask)
        reader.entity_positions("77", "enz1", mask)
        reader.exact_mentions("77", mask)

    read()
    gc.collect()
    before = {id(o) for o in gc.get_objects()}
    for _ in range(20):
        read()
    gc.collect()
    retained = [o for o in gc.get_objects() if id(o) not in before]

    assert not [o for o in retained if isinstance(o, types.FunctionType)]
    assert not [o for o in retained if isinstance(o, types.CellType)]


def test_loaded_candidate_ids_cost_far_less_than_one_frozenset_per_mention(
    tmp_path,
) -> None:
    """`load_token_labels`' packed representation must charge close to the
    flat ID strings' own size, not the ~190 KB a `frozenset` per mention
    (`_document_with_heavy_candidate_ids`) actually costs -- and must still
    decode back to exactly the same sets, row for row."""
    doc = _document_with_heavy_candidate_ids("enz")
    eager_cost = token_supervision._document_labels_bytes(doc)

    path = tmp_path / "labels.hdf5"
    with h5py.File(path, "w") as store:
        token_labels.write_label_space(
            store, BRENDA_LABELS, stamp=_STAMP, tokenizer=_TOKENIZER_STAMP
        )
        token_labels.store_token_labels(store, "77", doc)
    with h5py.File(path, "r") as store:
        loaded = token_labels.load_token_labels(store, "77", BRENDA_LABELS)
    packed_cost = token_supervision._document_labels_bytes(loaded)

    assert loaded.candidate_ids == doc.candidate_ids
    assert eager_cost > 150_000
    assert packed_cost < 70_000
    assert packed_cost < eager_cost / 2


def test_exact_mentions_carry_the_anchors_across_the_window_merge(
    tmp_path,
) -> None:
    """Two 32-token windows under the 20-token stride, as in the codes test:
    window 0 keeps its tokens 1-20 and window 1 supplies 11-30, window 1's
    token p being window 0's p + 10. A mention anchored in both windows of the
    overlap comes back once, a token two mentions share is placed in both, and
    a fuzzy row comes back not at all."""
    candidate_ids = (
        frozenset({"enz1"}),
        frozenset(),
        frozenset({"enz1", "enz5"}),
        frozenset({"bac3"}),
        frozenset({"str4"}),
    )
    anchors = numpy.array(
        [
            [0, 0, 3, 5],
            [2, 0, 18, 24],
            [2, 1, 8, 14],
            [3, 1, 25, 27],
            [4, 1, 26, 28],
        ],
        dtype=numpy.int32,
    )
    path = tmp_path / "labels.hdf5"
    with h5py.File(path, "w") as store:
        token_labels.write_label_space(
            store, BRENDA_LABELS, stamp=_STAMP, tokenizer=_TOKENIZER_STAMP
        )
        token_labels.store_token_labels(
            store,
            "77",
            DocumentLabels(
                codes=numpy.zeros((2, 32), dtype=numpy.int8),
                ambiguous=numpy.zeros((2, 32), dtype=numpy.int8),
                spans=numpy.zeros(
                    (len(candidate_ids), token_labels.SPAN_COLUMNS),
                    dtype=numpy.int32,
                ),
                text_length=0,
                candidate_ids=candidate_ids,
                anchors=anchors,
            ),
        )
    reader = TokenLabelReader(path, base_model="model-a")

    mentions = reader.exact_mentions("77", numpy.ones((2, 32)))

    assert mentions is not None
    assert [
        (mention.entity_ids, mention.positions.tolist()) for mention in mentions
    ] == [
        (frozenset({"enz1"}), [2, 3]),
        (frozenset({"enz1", "enz5"}), [17, 18, 19, 20, 21, 22]),
        (frozenset({"bac3"}), [34, 35]),
        (frozenset({"str4"}), [35, 36]),
    ]
    assert reader.exact_mentions("404", numpy.ones((2, 32))) is None
    with pytest.raises(ValueError, match="different encodings"):
        reader.exact_mentions("77", numpy.ones((3, 32)))


def test_exact_mentions_aggregates_the_window_geometry_once(
    tmp_path, monkeypatch
) -> None:
    """A second call for the same document must cost no extra
    `aggregate_embeddings` call: both the aggregated-axis source index and
    the mention tuple itself are cached per document, not re-derived from
    the raw anchors on every call."""
    candidate_ids = (frozenset({"enz1"}),)
    anchors = numpy.array([[0, 0, 3, 5]], dtype=numpy.int32)
    path = tmp_path / "labels.hdf5"
    with h5py.File(path, "w") as store:
        token_labels.write_label_space(
            store, BRENDA_LABELS, stamp=_STAMP, tokenizer=_TOKENIZER_STAMP
        )
        token_labels.store_token_labels(
            store,
            "77",
            DocumentLabels(
                codes=numpy.zeros((1, 32), dtype=numpy.int8),
                ambiguous=numpy.zeros((1, 32), dtype=numpy.int8),
                spans=numpy.zeros(
                    (len(candidate_ids), token_labels.SPAN_COLUMNS),
                    dtype=numpy.int32,
                ),
                text_length=0,
                candidate_ids=candidate_ids,
                anchors=anchors,
            ),
        )
    reader = TokenLabelReader(path, base_model="model-a")

    real_aggregate = token_supervision.aggregate_embeddings
    calls: list[int] = []

    def spy(*args, **kwargs):
        calls.append(1)
        return real_aggregate(*args, **kwargs)

    monkeypatch.setattr(token_supervision, "aggregate_embeddings", spy)

    mask = numpy.ones((1, 32))
    first = reader.exact_mentions("77", mask)
    second = reader.exact_mentions("77", mask)

    assert first is not None and second is not None
    as_lists = [
        [(mention.entity_ids, mention.positions.tolist()) for mention in call]
        for call in (first, second)
    ]
    assert as_lists[0] == as_lists[1] == [(frozenset({"enz1"}), [2, 3])]
    assert len(calls) == 1


def test_load_counts_a_hit_then_a_miss_and_log_cache_stats_resets(
    tmp_path, caplog
) -> None:
    """The first `_load` of a document is a miss, a repeat is a hit; logging
    the pass's stats must reset the counters so the next pass starts clean."""
    path = write_store(tmp_path / "labels.hdf5", {"77": [[0] * 32]})
    reader = TokenLabelReader(path, base_model="model-a")

    reader._load("77")
    reader._load("77")

    assert reader._label_cache.misses == 1
    assert reader._label_cache.hits == 1

    with caplog.at_level("INFO", logger=token_supervision.__name__):
        reader.log_cache_stats("training")

    assert "1/2 hits" in caplog.text
    assert reader._label_cache.hits == 0
    assert reader._label_cache.misses == 0


def test_padded_targets_pad_with_the_ignore_index() -> None:
    padded = padded_targets([torch.tensor([1, 2]), torch.tensor([3])], length=4)

    assert padded.tolist() == [
        [1, 2, IGNORE_INDEX, IGNORE_INDEX],
        [3, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX],
    ]


def test_document_lengths_count_unpadded_tokens() -> None:
    mask = torch.tensor([[True, True, False], [True, False, False]])

    assert document_lengths(mask) == [2, 1]
