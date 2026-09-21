"""The document id is what identifies a document to the embedding caches.

`get_token_embeddings` keys a cache that lives as long as the process on a
batch item's id, and checks nothing behind that id but the entry's row count,
so two documents of one token count handed one id are served each other's
activations with nothing raising and the output plausible. These pin the two
halves of that id space: no two external documents share an id, whichever
store or call each was read from, and a key that is not a pubmed id cannot
enter the half reserved for articles.
"""

import pathlib

import h5py
import numpy
import pytest
import torch
from d3text.encodings_store import external_key, mark_group_complete
from d3text.models.token_supervision import predicted_spans_from_store

_TEXT = "ABCDEFGHIJKL"


def _write_group(store: h5py.File, key: str) -> None:
    """A finished group of one 12-token window, the same geometry for every
    key, so nothing but the id distinguishes two of these documents."""
    offset_mapping = numpy.zeros((1, 12, 2), dtype=numpy.uint32)
    for token in range(1, 11):
        offset_mapping[0, token] = (token - 1, token)
    group = store.create_group(key)
    group.create_dataset(
        "input_ids", data=numpy.zeros((1, 12), dtype=numpy.uint32)
    )
    group.create_dataset(
        "attention_mask", data=numpy.ones((1, 12), dtype=numpy.int64)
    )
    group.create_dataset("offset_mapping", data=offset_mapping)
    mark_group_complete(group)


def _recorder(seen: list[int]):
    """The three model calls the join takes, recording the id each item it is
    handed carries and tagging every token as no mention."""

    def get_token_embeddings(batch):
        seen.append(int(batch[0]["id"].item()))
        return torch.zeros(1, 10, 1), torch.ones(1, 10)

    def hidden(embeddings):
        return embeddings

    def token_tagger(_hidden_output):
        return torch.zeros(1, 10, 2)

    return get_token_embeddings, hidden, token_tagger


def test_two_stores_of_one_process_do_not_share_a_document_id(
    tmp_path: pathlib.Path,
) -> None:
    """An id counted off one store's key order names a different document in
    the next store, while the cache it keys outlives both: each of these two
    documents is the first key of its own store."""
    seen: list[int] = []
    calls = _recorder(seen)

    for name, document in (("a.hdf5", "docA"), ("b.hdf5", "docB")):
        with h5py.File(tmp_path / name, "w") as store:
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

    for name, before in (("small.hdf5", ()), ("large.hdf5", ("111", "222"))):
        with h5py.File(tmp_path / name, "w") as store:
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

    with h5py.File(tmp_path / "store.hdf5", "w") as store:
        _write_group(store, "-1")
        with pytest.raises(ValueError, match="not a pubmed id"):
            predicted_spans_from_store(store, None, {"-1": _TEXT}, *calls)
