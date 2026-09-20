"""Reading a tagger's own spans out of an encodings-store group: the join
that feeds the linker a checkpoint's own detections rather than gold
mentions, from a corpus-prefixed group's windowed token ids to `TaggedSpan`s
grounded in that document's own text."""

import pathlib

import h5py
import numpy
import torch
from d3text.encodings_store import external_key, mark_group_complete
from d3text.models.token_supervision import predicted_spans_from_store
from d3text.token_labels import BRENDA_LABELS

ENZYMES = BRENDA_LABELS.by_prefix["enz"]

_TEXT = "ABCDEFGHIJKL"


def _offsets_and_mask() -> tuple[numpy.ndarray, numpy.ndarray]:
    """One 12-token window: CLS and SEP at positions 0 and 11, ten real
    tokens between them, each covering one character of `_TEXT`."""
    offset_mapping = numpy.zeros((1, 12, 2), dtype=numpy.uint32)
    for token in range(1, 11):
        offset_mapping[0, token] = (token - 1, token)
    attention_mask = numpy.ones((1, 12), dtype=numpy.int64)
    return offset_mapping, attention_mask


def _write_group(store: h5py.File, key: str, finished: bool = True) -> None:
    offset_mapping, attention_mask = _offsets_and_mask()
    group = store.create_group(key)
    group.create_dataset(
        "input_ids", data=numpy.zeros((1, 12), dtype=numpy.uint32)
    )
    group.create_dataset("attention_mask", data=attention_mask)
    group.create_dataset("offset_mapping", data=offset_mapping)
    group.create_dataset(
        "overflow_to_sample_mapping", data=numpy.zeros(1, dtype=numpy.uint8)
    )
    if finished:
        mark_group_complete(group)


def _codes_logits(codes: list[int]) -> torch.Tensor:
    """One-hot logits whose argmax reproduces `codes` exactly."""
    width = max(codes) + 2
    logits = torch.full((1, len(codes), width), -10.0)
    for position, code in enumerate(codes):
        logits[0, position, code] = 10.0
    return logits


def _stub_tagger(codes: list[int]):
    """A model stand-in: ignores every input, tags the aggregated axis
    exactly as `codes` says regardless of what `get_token_embeddings` read."""

    def get_token_embeddings(batch):
        return torch.zeros(1, len(codes), 1), torch.ones(1, len(codes))

    def hidden(embeddings):
        return embeddings

    def token_tagger(_hidden_output):
        return _codes_logits(codes)

    return get_token_embeddings, hidden, token_tagger


def test_a_finished_group_is_tagged_and_grounded(
    tmp_path: pathlib.Path,
) -> None:
    codes = [0, 0, ENZYMES, ENZYMES, ENZYMES, 0, 0, 0, 0, 0]
    get_token_embeddings, hidden, token_tagger = _stub_tagger(codes)

    with h5py.File(tmp_path / "store.hdf5", "w") as store:
        _write_group(store, external_key("s800", "doc1"))
        (span,) = predicted_spans_from_store(
            store,
            "s800",
            {"doc1": _TEXT},
            get_token_embeddings,
            hidden,
            token_tagger,
        )

    assert (span.document, span.start, span.end) == ("doc1", 2, 5)
    assert span.surface == "CDE"
    assert span.entity_type == BRENDA_LABELS.type_of(ENZYMES)


def test_an_unfinished_group_is_skipped(tmp_path: pathlib.Path) -> None:
    """A group a torn precompute pass left without its completion marker
    carries no reliable `offset_mapping` and is not read."""
    get_token_embeddings, hidden, token_tagger = _stub_tagger([0] * 10)

    with h5py.File(tmp_path / "store.hdf5", "w") as store:
        _write_group(store, external_key("s800", "doc1"), finished=False)
        spans = predicted_spans_from_store(
            store,
            "s800",
            {"doc1": _TEXT},
            get_token_embeddings,
            hidden,
            token_tagger,
        )

    assert spans == []


def test_a_document_the_caller_holds_no_text_for_is_skipped(
    tmp_path: pathlib.Path,
) -> None:
    get_token_embeddings, hidden, token_tagger = _stub_tagger([0] * 10)

    with h5py.File(tmp_path / "store.hdf5", "w") as store:
        _write_group(store, external_key("s800", "doc1"))
        spans = predicted_spans_from_store(
            store, "s800", {}, get_token_embeddings, hidden, token_tagger
        )

    assert spans == []


def test_a_brenda_document_is_read_under_its_bare_pubmed_key(
    tmp_path: pathlib.Path,
) -> None:
    """The same join for an article, whose group key carries no corpus prefix
    — what lets a trained tagger be run over the corpus itself rather than
    over the two external gold sets alone."""
    codes = [0, 0, ENZYMES, ENZYMES, ENZYMES, 0, 0, 0, 0, 0]
    get_token_embeddings, hidden, token_tagger = _stub_tagger(codes)

    with h5py.File(tmp_path / "store.hdf5", "w") as store:
        _write_group(store, "12345")
        (span,) = predicted_spans_from_store(
            store,
            None,
            {"12345": _TEXT},
            get_token_embeddings,
            hidden,
            token_tagger,
        )

    assert (span.document, span.start, span.end, span.surface) == (
        "12345",
        2,
        5,
        "CDE",
    )


def test_an_external_group_is_not_read_as_a_brenda_document(
    tmp_path: pathlib.Path,
) -> None:
    """The prefixed key is the store's whole defence against S800's own
    document ids colliding with a pubmed id, so `None` must not fall back to
    scanning for one."""
    get_token_embeddings, hidden, token_tagger = _stub_tagger([0] * 10)

    with h5py.File(tmp_path / "store.hdf5", "w") as store:
        _write_group(store, external_key("s800", "12345"))
        spans = predicted_spans_from_store(
            store,
            None,
            {"12345": _TEXT},
            get_token_embeddings,
            hidden,
            token_tagger,
        )

    assert spans == []


def test_a_group_of_another_corpus_is_skipped(tmp_path: pathlib.Path) -> None:
    """`enzymener:doc1` is not read as an `s800` group, and a bare pubmed
    key carries no corpus prefix at all -- both are outside `corpus`."""
    get_token_embeddings, hidden, token_tagger = _stub_tagger([0] * 10)

    with h5py.File(tmp_path / "store.hdf5", "w") as store:
        _write_group(store, external_key("enzymener", "doc1"))
        _write_group(store, "12345")
        spans = predicted_spans_from_store(
            store,
            "s800",
            {"doc1": _TEXT, "12345": _TEXT},
            get_token_embeddings,
            hidden,
            token_tagger,
        )

    assert spans == []


def test_two_corpora_of_one_store_get_distinct_document_ids(
    tmp_path: pathlib.Path,
) -> None:
    """`evaluate.report_predicted_linking` runs both external corpora against
    one model in one process, and `get_token_embeddings` keys a cache on the
    id it is handed without checking the token count behind it -- so an id
    counted off each call's own `texts` hands the second corpus the first
    corpus' cached activations."""
    seen: list[int] = []
    embeddings, hidden, token_tagger = _stub_tagger([0] * 10)

    def recording(batch):
        seen.append(int(batch[0]["id"].item()))
        return embeddings(batch)

    with h5py.File(tmp_path / "store.hdf5", "w") as store:
        _write_group(store, external_key("s800", "doc1"))
        _write_group(store, external_key("enzymener", "doc1"))
        for corpus in ("s800", "enzymener"):
            predicted_spans_from_store(
                store, corpus, {"doc1": _TEXT}, recording, hidden, token_tagger
            )

    assert len(seen) == 2
    assert len(set(seen)) == 2
