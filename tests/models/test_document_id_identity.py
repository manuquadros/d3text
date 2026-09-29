"""The document id is what identifies a document to the embedding caches.

The cache checks only row count behind an id, so two same-length documents
sharing one are silently served each other's activations. No two external
documents share an id, and a non-pubmed key cannot enter the article half.
"""

import contextlib
import pathlib

import numpy
import pytest
import torch
from d3text.encodings_store import EncodingsStore, external_key
from d3text.models.token_supervision import predicted_spans_from_store

_TEXT = "ABCDEFGHIJKL"


def _write_group(store: EncodingsStore, key: str) -> None:
    """A document of one 12-token window, the same geometry for every key,
    so nothing but the id distinguishes two of these documents."""
    offset_mapping = numpy.zeros((1, 12, 2), dtype=numpy.uint32)
    for token in range(1, 11):
        offset_mapping[0, token] = (token - 1, token)
    store.put(
        key,
        {
            "input_ids": numpy.zeros((1, 12), dtype=numpy.uint32),
            "attention_mask": numpy.ones((1, 12), dtype=numpy.uint8),
            "offset_mapping": offset_mapping,
        },
    )


def _recorder(seen: list[int]):
    """The four model calls the join takes, recording the id each item it is
    handed carries and tagging every token as no mention."""

    def get_token_embeddings(batch):
        seen.append(int(batch[0]["id"].item()))
        return torch.zeros(1, 10, 1), torch.ones(1, 10)

    def hidden(embeddings, _mask):
        return embeddings

    def token_tagger(_hidden_output):
        return torch.zeros(1, 10, 2)

    return get_token_embeddings, hidden, token_tagger, contextlib.nullcontext


def test_two_stores_of_one_process_do_not_share_a_document_id(
    tmp_path: pathlib.Path,
) -> None:
    """An id counted off one store's key order names a different document in
    the next store, while the cache it keys outlives both: each of these two
    documents is the first key of its own store."""
    seen: list[int] = []
    calls = _recorder(seen)

    for name, document in (("a", "docA"), ("b", "docB")):
        with EncodingsStore(tmp_path / name, writable=True) as store:
            _write_group(store, external_key("s800", document))
            predicted_spans_from_store(store, "s800", {document: _TEXT}, *calls)

    assert len(seen) == 2
    assert seen[0] != seen[1]
    # Below zero, so neither can be read as an article's pubmed id either.
    assert all(document_id < 0 for document_id in seen)


def test_one_document_keeps_its_id_when_the_store_around_it_changes(
    tmp_path: pathlib.Path,
) -> None:
    """The other half of the same property: one document read from two stores
    is one document, so it must not be handed two ids and cached twice. Its
    position moves because the keys before it do."""
    seen: list[int] = []
    calls = _recorder(seen)

    for name, before in (("small", ()), ("large", ("111", "222"))):
        with EncodingsStore(tmp_path / name, writable=True) as store:
            for key in before:
                _write_group(store, key)
            _write_group(store, external_key("s800", "docC"))
            predicted_spans_from_store(store, "s800", {"docC": _TEXT}, *calls)

    assert len(seen) == 2
    assert seen[0] == seen[1]


def test_a_key_that_is_not_a_pubmed_id_is_refused(
    tmp_path: pathlib.Path,
) -> None:
    """A caller's own mapping keys enter the positive half of the id space
    unvalidated, and one reading as an external document's id would be served
    that document's activations rather than raising."""
    seen: list[int] = []
    calls = _recorder(seen)

    with EncodingsStore(tmp_path / "store", writable=True) as store:
        _write_group(store, "-1")
        with pytest.raises(ValueError, match="not a pubmed id"):
            predicted_spans_from_store(store, None, {"-1": _TEXT}, *calls)
