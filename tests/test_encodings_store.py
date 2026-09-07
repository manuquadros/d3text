"""The encodings HDF5's own provenance stamp.

`precompute-encodings` writes token ids into an HDF5 keyed by pubmed id; the
ids alone say nothing about which model, window or stride tokenized them, and
`d3text.embeddings_store` already showed the aggregated row count comes to the
same value under any window or stride. `record_provenance` is the guard that
keeps two geometries out of the same file, mirroring
`d3text.cli.precompute_embeddings.record_provenance`; `content_digest` is the
one that separates two files the geometry describes identically.
"""

import h5py
import numpy
import pytest

from d3text.encodings_store import (
    EncodingsProvenance,
    content_digest,
    read_content_digest,
    read_provenance,
    record_provenance,
    stamp_content_digest,
    store_content_digest,
    write_provenance,
    writing_pass,
)

BASE_MODEL = "michiyasunaga/BioLinkBERT-base"
PROVENANCE = EncodingsProvenance(
    base_model=BASE_MODEL, max_length=512, stride=20
)


def _write_document(store, key, windows):
    """One document's windows of token ids, as `precompute-encodings` stores
    them: the group is replaced outright, which is what `-f` does."""
    if key in store:
        del store[key]
    store.create_group(key).create_dataset(
        name="input_ids", data=numpy.asarray(windows, dtype="uint32")
    )


def _store_with(path, documents):
    """An encodings file over `documents`, stamped with `PROVENANCE`.

    Keyed pubmed id -> the windows of token ids.
    """
    with h5py.File(path, "w") as f:
        write_provenance(f, PROVENANCE)
        for key, windows in documents.items():
            _write_document(f, key, windows)


def test_a_store_reports_the_model_window_and_stride_it_was_written_with(
    tmp_path,
):
    with h5py.File(tmp_path / "store.hdf5", "w") as f:
        write_provenance(f, PROVENANCE)
        assert read_provenance(f) == PROVENANCE


def test_a_store_from_before_provenance_was_recorded_reports_none(tmp_path):
    with h5py.File(tmp_path / "store.hdf5", "w") as f:
        assert read_provenance(f) is None


def test_a_provenance_record_from_a_future_format_is_refused(tmp_path):
    with h5py.File(tmp_path / "store.hdf5", "w") as f:
        f.attrs["d3text_encodings_format"] = 99
        f.attrs["base_model"] = BASE_MODEL
        f.attrs["max_length"] = 512
        f.attrs["stride"] = 20

        with pytest.raises(ValueError, match="format"):
            read_provenance(f)


def test_a_fresh_store_is_stamped_with_this_runs_geometry(tmp_path):
    with h5py.File(tmp_path / "store.hdf5", "w") as f:
        record_provenance(f, PROVENANCE)
        assert read_provenance(f) == PROVENANCE


def test_a_second_run_at_the_same_geometry_is_accepted(tmp_path):
    """The resume path this command is built around must not cost anything."""
    with h5py.File(tmp_path / "store.hdf5", "w") as f:
        record_provenance(f, PROVENANCE)
        f.create_group("1").create_dataset("input_ids", data=[1, 2, 3])
        record_provenance(f, PROVENANCE)
        assert read_provenance(f) == PROVENANCE
        assert "1" in f


def test_a_store_written_at_one_window_is_refused_at_another(tmp_path):
    """The window and stride are the two fields nothing else compares —
    the row count is the same for any of them, so this is the only place a
    drift is ever caught."""
    with h5py.File(tmp_path / "store.hdf5", "w") as f:
        record_provenance(f, PROVENANCE)
        f.create_group("1").create_dataset("input_ids", data=[1, 2, 3])

        other = EncodingsProvenance(
            base_model=BASE_MODEL, max_length=256, stride=20
        )
        with pytest.raises(ValueError, match="window 512, stride 20"):
            record_provenance(f, other)

        # refused before anything is overwritten
        assert read_provenance(f) == PROVENANCE
        assert "1" in f


def test_a_store_written_by_another_model_is_refused(tmp_path):
    with h5py.File(tmp_path / "store.hdf5", "w") as f:
        record_provenance(f, PROVENANCE)

        other = EncodingsProvenance(
            base_model="google-bert/bert-base-cased", max_length=512, stride=20
        )
        with pytest.raises(ValueError, match="was written by"):
            record_provenance(f, other)


def test_writing_into_an_unstamped_but_nonempty_store_stamps_it(
    tmp_path, caplog
):
    """Every encodings file `precompute-encodings` had ever written predates
    this stamp, so refusing them outright would make this build's first run
    against any of them fail; warning and stamping keeps the resume working
    and attributes everything written from here on."""
    with h5py.File(tmp_path / "store.hdf5", "w") as f:
        f.create_group("1").create_dataset("input_ids", data=[1, 2, 3])

        with caplog.at_level("WARNING"):
            record_provenance(f, PROVENANCE)

        assert "does not record" in caplog.text
        assert read_provenance(f) == PROVENANCE
        assert "1" in f


def test_writing_into_an_unstamped_empty_store_stamps_it(tmp_path):
    """An empty store is indistinguishable from a fresh one; refusing it
    would make a bare `h5py.File(..., 'w-')` unusable as the first write."""
    with h5py.File(tmp_path / "store.hdf5", "w") as f:
        record_provenance(f, PROVENANCE)
        assert read_provenance(f) == PROVENANCE


def test_two_stores_of_one_geometry_over_different_ids_digest_apart(tmp_path):
    """The mistake the geometry stamp cannot catch. A corpus re-tokenized
    under a newer tokenizer revision, or under a corrected `document_text`,
    yields the same documents at the same window and stride over different
    ids, and both files carry a stamp that agrees to the character."""
    first, second = tmp_path / "first.hdf5", tmp_path / "second.hdf5"
    _store_with(first, {"10": [[1, 2, 3, 4]]})
    _store_with(second, {"10": [[1, 2, 3, 5]]})

    with h5py.File(first, "r") as a, h5py.File(second, "r") as b:
        assert read_provenance(a) == read_provenance(b)
        assert content_digest(a) != content_digest(b)


def test_the_same_documents_digest_the_same_in_any_write_order(tmp_path):
    """`precompute-encodings` writes in corpus order and resumes in whatever
    order the remaining rows arrive, so a digest that followed the file's own
    layout would call two identical stores different."""
    first, second = tmp_path / "first.hdf5", tmp_path / "second.hdf5"
    _store_with(first, {"10": [[1, 2]], "20": [[3, 4]]})
    _store_with(second, {"20": [[3, 4]], "10": [[1, 2]]})

    with h5py.File(first, "r") as a, h5py.File(second, "r") as b:
        assert content_digest(a) == content_digest(b)


def test_the_same_ids_cut_into_different_windows_digest_apart(tmp_path):
    """`sum(L_i)` comes to the document's token count under any window, so
    hashing the ids alone would let a document split at one window and resumed
    at another digest identically to the one it replaced."""
    first, second = tmp_path / "first.hdf5", tmp_path / "second.hdf5"
    _store_with(first, {"10": [[1, 2, 3, 4]]})
    _store_with(second, {"10": [[1, 2], [3, 4]]})

    with h5py.File(first, "r") as a, h5py.File(second, "r") as b:
        assert content_digest(a) != content_digest(b)


def test_a_store_from_before_the_digest_existed_reports_none(tmp_path):
    """Every encodings file on disk is one. A reader has to report the absence
    rather than refuse the file, exactly as the geometry stamp does."""
    path = tmp_path / "store.hdf5"
    _store_with(path, {"10": [[1, 2, 3, 4]]})

    with h5py.File(path, "r") as f:
        assert read_content_digest(f) is None
    assert store_content_digest(path) is None


def test_a_stamped_store_hands_its_digest_over_without_its_ids(tmp_path):
    """Where the cost is paid: the writer decompresses the store once, and
    every reader after it answers from one root attribute."""
    path = tmp_path / "store.hdf5"
    _store_with(path, {"10": [[1, 2, 3, 4]]})

    with h5py.File(path, "r+") as f:
        stamped = stamp_content_digest(f)
        assert content_digest(f) == stamped

    assert store_content_digest(path) == stamped


def test_no_store_to_read_is_no_digest(tmp_path):
    """`train` records the digest of a file the data layer already treats as
    optional, so reading it must not be the call that reports its absence."""
    assert store_content_digest(None) is None
    assert store_content_digest(tmp_path / "absent.hdf5") is None


def test_a_completed_writing_pass_stamps_what_it_wrote(tmp_path):
    """The digest has to describe the file as the pass left it, not as it
    found it: a resume that adds or replaces documents re-fingerprints all of
    them."""
    path = tmp_path / "store.hdf5"
    _store_with(path, {"10": [[1, 2, 3, 4]]})

    with h5py.File(path, "r+") as f:
        before = stamp_content_digest(f)
        with writing_pass(f):
            _write_document(f, "20", [[5, 6, 7, 8]])

    with h5py.File(path, "r") as f:
        assert read_content_digest(f) == content_digest(f)
        assert read_content_digest(f) != before


def test_the_stamp_is_off_the_file_for_the_length_of_the_pass(tmp_path):
    """Dropping it on the way *in* is the whole mechanism. Declining to
    restate it at the end would leave the window between the first group
    written and the last one covered by a digest that is already false."""
    path = tmp_path / "store.hdf5"
    _store_with(path, {"10": [[1, 2, 3, 4]]})

    with h5py.File(path, "r+") as f:
        stamp_content_digest(f)
        with writing_pass(f):
            assert read_content_digest(f) is None


def test_an_interrupted_pass_leaves_the_store_unstamped(tmp_path):
    """A Ctrl-C out of a re-tokenization propagates through the enclosing
    `with h5py.File(...)`, which closes the file cleanly — so a store whose
    ids have changed would keep the previous pass's fingerprint and read as
    agreeing with a checkpoint it no longer matches. Unstamped is the honest
    report, and the one `evaluate` already warns about."""
    path = tmp_path / "store.hdf5"
    _store_with(path, {"10": [[1, 2, 3, 4]]})
    with h5py.File(path, "r+") as f:
        stale = stamp_content_digest(f)

    with pytest.raises(KeyboardInterrupt):
        with h5py.File(path, "r+") as f:
            with writing_pass(f):
                _write_document(f, "10", [[5, 6, 7, 8]])
                raise KeyboardInterrupt

    assert store_content_digest(path) is None
    with h5py.File(path, "r") as f:
        assert content_digest(f) != stale


def test_a_pass_that_writes_nothing_leaves_the_digest_it_found(tmp_path):
    """A resume with nothing left to do restates the same value, so the file
    it hands back is the file it was given."""
    path = tmp_path / "store.hdf5"
    _store_with(path, {"10": [[1, 2, 3, 4]]})

    with h5py.File(path, "r+") as f:
        before = stamp_content_digest(f)
        with writing_pass(f):
            pass

    assert store_content_digest(path) == before
