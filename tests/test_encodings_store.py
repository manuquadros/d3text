"""The encodings LMDB: its document codec and its provenance stamps.

Token ids do not say which model, window or stride produced them, so
`record_provenance` keeps two geometries out of one store and
`content_digest` tells apart two stores of the same geometry.
"""

import os

import h5py
import numpy
import pytest

from d3text import encodings_store
from d3text.encodings_store import (
    EncodingsProvenance,
    check_provenance,
    content_digest,
    encodings_provenance,
    external_document,
    external_key,
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


def _encoding(windows):
    """One document's windows of token ids, with a full mask and offsets."""
    ids = numpy.asarray(windows, dtype="uint32")
    return {
        "input_ids": ids,
        "attention_mask": numpy.ones(ids.shape, dtype="uint8"),
        "offset_mapping": numpy.zeros((*ids.shape, 2), dtype="uint32"),
    }


def _open(path, writable=False):
    return encodings_store.EncodingsStore(path, writable=writable)


def _write_document(store, key, windows):
    """Store `windows` under `key`, replacing it, as `-f` does."""
    store.put(key, _encoding(windows))


def _store_with(path, documents):
    """An encodings store over `documents`, stamped with `PROVENANCE`.

    Keyed pubmed id -> the windows of token ids.
    """
    with _open(path, writable=True) as store:
        write_provenance(store, PROVENANCE)
        for key, windows in documents.items():
            _write_document(store, key, windows)


def test_an_hdf5_store_is_refused_with_a_rebuild_hint(tmp_path):
    """Stores written before the move to LMDB are HDF5 files. There is no
    migration, since the store is rebuilt from the corpus, so the refusal
    has to say how to get one this build reads."""
    path = tmp_path / "encodings.hdf5"
    with h5py.File(path, "w") as f:
        f.attrs["d3text_encodings_format"] = 2

    with pytest.raises(ValueError, match="HDF5.*precompute-encodings"):
        _open(path)
    with pytest.raises(ValueError, match="HDF5.*precompute-encodings"):
        store_content_digest(path)


def test_a_document_round_trips_through_the_store(tmp_path):
    """Ids, mask and offsets come back at the dtypes and shapes the old
    per-document datasets had, which every reader indexes by name."""
    ids = numpy.arange(12, dtype="uint32").reshape(2, 6) + 70_000
    mask = numpy.array([[1] * 6, [1, 1, 1, 0, 0, 0]], dtype="uint8")
    offsets = numpy.arange(24, dtype="uint32").reshape(2, 6, 2)
    path = tmp_path / "store"
    with _open(path, writable=True) as store:
        store.put(
            "10",
            {
                "input_ids": ids,
                "attention_mask": mask,
                "offset_mapping": offsets,
            },
        )

    with _open(path) as store:
        stored = store.get("10")
        assert store.get("20") is None
        assert store.windows("10") == 2
        assert store.windows("20") is None
        assert store.keys() == ["10"]

    assert stored is not None
    for name, expected in (
        ("input_ids", ids),
        ("attention_mask", mask),
        ("offset_mapping", offsets),
    ):
        assert stored[name].dtype == expected.dtype
        numpy.testing.assert_array_equal(stored[name], expected)


def test_a_put_that_fails_leaves_the_document_absent(tmp_path):
    """A document is one value written in one transaction, so a write that
    fails leaves the key absent rather than holding part of a document; a
    mask that disagrees with the ids is refused before anything lands."""
    path = tmp_path / "store"
    with _open(path, writable=True) as store:
        _write_document(store, "10", [[1, 2, 3, 4]])
        broken = _encoding([[5, 6, 7, 8], [9, 10, 11, 12]])
        broken["attention_mask"] = numpy.ones((1, 4), dtype="uint8")

        with pytest.raises(ValueError, match="attention_mask"):
            store.put("20", broken)

    with _open(path) as store:
        assert "20" not in store
        assert store.keys() == ["10"]


def test_a_store_reports_the_model_window_and_stride_it_was_written_with(
    tmp_path,
):
    with _open(tmp_path / "store", writable=True) as store:
        write_provenance(store, PROVENANCE)
        assert read_provenance(store) == PROVENANCE


def test_a_store_no_writer_has_stamped_reports_none(tmp_path):
    with _open(tmp_path / "store", writable=True) as store:
        assert read_provenance(store) is None


def test_a_provenance_record_from_a_future_format_is_refused(
    tmp_path, monkeypatch
):
    path = tmp_path / "store"
    with _open(path, writable=True) as store:
        monkeypatch.setattr(encodings_store, "_PROVENANCE_FORMAT", 99)
        write_provenance(store, PROVENANCE)
        monkeypatch.undo()

        with pytest.raises(ValueError, match="format"):
            read_provenance(store)


def test_a_fresh_store_is_stamped_with_this_runs_geometry(tmp_path):
    with _open(tmp_path / "store", writable=True) as store:
        record_provenance(store, PROVENANCE)
        assert read_provenance(store) == PROVENANCE


def test_a_second_run_at_the_same_geometry_is_accepted(tmp_path):
    """The resume path this command is built around must not cost anything."""
    with _open(tmp_path / "store", writable=True) as store:
        record_provenance(store, PROVENANCE)
        _write_document(store, "1", [[1, 2, 3]])
        record_provenance(store, PROVENANCE)
        assert read_provenance(store) == PROVENANCE
        assert "1" in store


def test_check_provenance_accepts_a_matching_stamp(tmp_path):
    """A store stamped with this run's own base model and stride is
    accepted without raising — the check
    `BrendaDataset._check_encodings_provenance` delegates to."""
    path = tmp_path / "store"
    _store_with(path, {})

    check_provenance(path, PROVENANCE.base_model, PROVENANCE.stride)


def test_check_provenance_refuses_another_base_model(tmp_path):
    path = tmp_path / "store"
    _store_with(path, {})

    with pytest.raises(ValueError, match="was tokenized by"):
        check_provenance(path, "other-model", PROVENANCE.stride)


def test_check_provenance_refuses_another_stride(tmp_path):
    path = tmp_path / "store"
    _store_with(path, {})

    with pytest.raises(ValueError, match="stride of"):
        check_provenance(path, PROVENANCE.base_model, PROVENANCE.stride + 1)


def test_check_provenance_refuses_an_unstamped_store(tmp_path):
    """Every writer stamps before its first document, so an unstamped store
    is one nothing here wrote: its ids cannot be attributed to any model."""
    path = tmp_path / "store"
    with _open(path, writable=True):
        pass

    with pytest.raises(ValueError, match="does not record"):
        check_provenance(path, "any-model", 20)


def test_a_store_written_at_one_window_is_refused_at_another(tmp_path):
    """The window and stride are the two fields nothing else compares —
    the row count is the same for any of them, so this is the only place a
    drift is ever caught."""
    with _open(tmp_path / "store", writable=True) as store:
        record_provenance(store, PROVENANCE)
        _write_document(store, "1", [[1, 2, 3]])

        other = EncodingsProvenance(
            base_model=BASE_MODEL, max_length=256, stride=20
        )
        with pytest.raises(ValueError, match="window 512, stride 20"):
            record_provenance(store, other)

        # refused before anything is overwritten
        assert read_provenance(store) == PROVENANCE
        assert "1" in store


def test_a_store_written_by_another_model_is_refused(tmp_path):
    with _open(tmp_path / "store", writable=True) as store:
        record_provenance(store, PROVENANCE)

        other = EncodingsProvenance(
            base_model="google-bert/bert-base-cased", max_length=512, stride=20
        )
        with pytest.raises(ValueError, match="was written by"):
            record_provenance(store, other)


def test_writing_into_an_unstamped_but_nonempty_store_is_refused(tmp_path):
    """Stamping it would attribute documents nothing here wrote to this
    run's geometry."""
    with _open(tmp_path / "store", writable=True) as store:
        _write_document(store, "1", [[1, 2, 3]])

        with pytest.raises(ValueError, match="does not record"):
            record_provenance(store, PROVENANCE)

        assert read_provenance(store) is None


def test_writing_into_an_unstamped_empty_store_stamps_it(tmp_path):
    """An empty store is indistinguishable from a fresh one; refusing it
    would make the first write impossible."""
    with _open(tmp_path / "store", writable=True) as store:
        record_provenance(store, PROVENANCE)
        assert read_provenance(store) == PROVENANCE


def test_two_stores_of_one_geometry_over_different_ids_digest_apart(tmp_path):
    """The mistake the geometry stamp cannot catch. A corpus re-tokenized
    under a newer tokenizer revision, or under a corrected `document_text`,
    yields the same documents at the same window and stride over different
    ids, and both files carry a stamp that agrees to the character."""
    first, second = tmp_path / "first", tmp_path / "second"
    _store_with(first, {"10": [[1, 2, 3, 4]]})
    _store_with(second, {"10": [[1, 2, 3, 5]]})

    with _open(first) as a, _open(second) as b:
        assert read_provenance(a) == read_provenance(b)
        assert content_digest(a) != content_digest(b)


def test_the_same_documents_digest_the_same_in_any_write_order(tmp_path):
    """`precompute-encodings` writes in corpus order and resumes in whatever
    order the remaining rows arrive, so a digest that followed the file's own
    layout would call two identical stores different."""
    first, second = tmp_path / "first", tmp_path / "second"
    _store_with(first, {"10": [[1, 2]], "20": [[3, 4]]})
    _store_with(second, {"20": [[3, 4]], "10": [[1, 2]]})

    with _open(first) as a, _open(second) as b:
        assert content_digest(a) == content_digest(b)


def test_the_same_ids_cut_into_different_windows_digest_apart(tmp_path):
    """`sum(L_i)` comes to the document's token count under any window, so
    hashing the ids alone would let a document split at one window and resumed
    at another digest identically to the one it replaced."""
    first, second = tmp_path / "first", tmp_path / "second"
    _store_with(first, {"10": [[1, 2, 3, 4]]})
    _store_with(second, {"10": [[1, 2], [3, 4]]})

    with _open(first) as a, _open(second) as b:
        assert content_digest(a) != content_digest(b)


def test_a_store_no_pass_finished_reports_no_digest(tmp_path):
    """A pass killed before its end leaves one. A reader has to report the
    absence rather than refuse the store."""
    path = tmp_path / "store"
    _store_with(path, {"10": [[1, 2, 3, 4]]})

    with _open(path) as f:
        assert read_content_digest(f) is None
    assert store_content_digest(path) is None


def test_a_stamped_store_hands_its_digest_over_without_its_ids(tmp_path):
    """Where the cost is paid: the writer decompresses the store once, and
    every reader after it answers from one stamp."""
    path = tmp_path / "store"
    _store_with(path, {"10": [[1, 2, 3, 4]]})

    with _open(path, writable=True) as f:
        stamped = stamp_content_digest(f)
        assert content_digest(f) == stamped

    assert store_content_digest(path) == stamped


def test_no_store_to_read_is_no_digest(tmp_path):
    """`train` records the digest of a file the data layer already treats as
    optional, so reading it must not be the call that reports its absence."""
    assert store_content_digest(None) is None
    assert store_content_digest(tmp_path / "absent") is None


# Two content digests, hex sha256s of a store, as `content_digest` gives them.
TOKENIZED = "c" * 64
RETOKENIZED = "d" * 64


def test_the_encodings_the_checkpoint_trained_on_are_recognised():
    assert encodings_provenance(TOKENIZED, TOKENIZED) == "matched"


def test_a_retokenized_corpus_warns_and_is_still_scored():
    """The failure this exists to catch. A store rebuilt under a newer
    tokenizer revision, or after `document_text` changed what it feeds the
    tokenizer, holds different ids for the same documents at the same window
    and stride — so the model is scored on inputs it never trained on, with
    the geometry stamp silent because it did not move and the vocabulary
    silent because the columns did not either."""
    with pytest.warns(RuntimeWarning, match="different token ids"):
        tag = encodings_provenance(TOKENIZED, RETOKENIZED)

    assert tag == "mismatched"


def test_a_checkpoint_recording_no_tokenization_warns():
    with pytest.warns(RuntimeWarning, match="records no encodings digest"):
        tag = encodings_provenance(None, TOKENIZED)

    assert tag == "unrecorded"


def test_two_absent_digests_are_not_reported_as_a_match():
    """`None == None` is not agreement. Reporting it as `matched` would be the
    stamp asserting something no file on disk says, which is worse than the
    silence it replaced."""
    with pytest.warns(RuntimeWarning, match="records no encodings digest"):
        assert encodings_provenance(None, None) == "unrecorded"


def test_an_unstamped_store_cannot_confirm_a_checkpoints_inputs():
    """Every encodings file written before the digest existed is this case, so
    it warns and scores rather than refusing: rebuilding the store is hours,
    and the numbers are still the numbers."""
    with pytest.warns(RuntimeWarning, match="carries no digest of its own"):
        tag = encodings_provenance(TOKENIZED, None)

    assert tag == "unstamped"


@pytest.mark.parametrize(
    "recorded,current",
    [(None, TOKENIZED), (TOKENIZED, RETOKENIZED)],
)
def test_the_warning_makes_no_claim_about_scores(recorded, current):
    """`infer` calls `encodings_provenance` too, and produces predictions, not
    scores, so an `infer` run must not be told about an evaluation that never
    happened."""
    with pytest.warns(RuntimeWarning) as caught:
        encodings_provenance(recorded, current)

    message = str(caught[0].message).lower()
    assert "score" not in message
    assert "evaluat" not in message


def test_a_completed_writing_pass_stamps_what_it_wrote(tmp_path):
    """The digest has to describe the file as the pass left it, not as it
    found it: a resume that adds or replaces documents re-fingerprints all of
    them."""
    path = tmp_path / "store"
    _store_with(path, {"10": [[1, 2, 3, 4]]})

    with _open(path, writable=True) as f:
        before = stamp_content_digest(f)
        with writing_pass(f):
            _write_document(f, "20", [[5, 6, 7, 8]])

    with _open(path) as f:
        assert read_content_digest(f) == content_digest(f)
        assert read_content_digest(f) != before


def test_the_stamp_is_off_the_file_for_the_length_of_the_pass(tmp_path):
    """Dropping it on the way *in* is the whole mechanism. Declining to
    restate it at the end would leave the window between the first group
    written and the last one covered by a digest that is already false."""
    path = tmp_path / "store"
    _store_with(path, {"10": [[1, 2, 3, 4]]})

    with _open(path, writable=True) as f:
        stamp_content_digest(f)
        with writing_pass(f):
            assert read_content_digest(f) is None


def test_an_interrupted_pass_leaves_the_store_unstamped(tmp_path):
    """A Ctrl-C out of a re-tokenization propagates through the enclosing
    `with` block holding the store, which closes it cleanly — so a store whose
    ids have changed would keep the previous pass's fingerprint and read as
    agreeing with a checkpoint it no longer matches. Unstamped is the honest
    report, and the one `evaluate` already warns about."""
    path = tmp_path / "store"
    _store_with(path, {"10": [[1, 2, 3, 4]]})
    with _open(path, writable=True) as f:
        stale = stamp_content_digest(f)

    with pytest.raises(KeyboardInterrupt):
        with _open(path, writable=True) as f:
            with writing_pass(f):
                _write_document(f, "10", [[5, 6, 7, 8]])
                raise KeyboardInterrupt

    assert store_content_digest(path) is None
    with _open(path) as f:
        assert content_digest(f) != stale


def test_a_pass_that_writes_nothing_leaves_the_digest_it_found(tmp_path):
    """A resume with nothing left to do restates the same value, so the file
    it hands back is the file it was given."""
    path = tmp_path / "store"
    _store_with(path, {"10": [[1, 2, 3, 4]]})

    with _open(path, writable=True) as f:
        before = stamp_content_digest(f)
        with writing_pass(f):
            pass

    assert store_content_digest(path) == before


def test_a_nested_pass_leaves_no_stamp_over_ids_it_does_not_cover(tmp_path):
    """The failure a re-entrant bracket produces is silent: the inner exit
    restates the digest, the outer goes on writing, and a kill after that
    leaves a fingerprint asserting agreement with ids that have moved — read
    as `matched`, warned about nowhere. Whatever the bracket does about
    nesting, what it must never do is that."""
    path = tmp_path / "store"
    _store_with(path, {"10": [[1, 2, 3, 4]]})
    with _open(path, writable=True) as f:
        stamp_content_digest(f)

    with pytest.raises((RuntimeError, KeyboardInterrupt)):
        with _open(path, writable=True) as f:
            with writing_pass(f):
                _write_document(f, "10", [[5, 6, 7, 8]])
                with writing_pass(f):
                    pass
                # Reached only where re-entry is permitted: this is the group
                # the inner pass's stamp would not cover.
                _write_document(f, "20", [[9, 10, 11, 12]])
                raise KeyboardInterrupt

    recorded = store_content_digest(path)
    with _open(path) as f:
        assert recorded is None or recorded == content_digest(f)


def test_re_entering_a_writing_pass_is_refused(tmp_path):
    """Refusal is the whole mechanism, and it has to leave the store in the
    state an interrupt leaves it in: the aborted outer pass stamps nothing,
    so what is on disk reads as unstamped rather than as agreeing."""
    path = tmp_path / "store"
    _store_with(path, {"10": [[1, 2, 3, 4]]})
    with _open(path, writable=True) as f:
        stamp_content_digest(f)

    with pytest.raises(RuntimeError):
        with _open(path, writable=True) as f:
            with writing_pass(f):
                _write_document(f, "10", [[5, 6, 7, 8]])
                with writing_pass(f):
                    pass

    assert store_content_digest(path) is None


def test_a_refused_re_entry_leaves_the_store_open_to_later_passes(tmp_path):
    """The guard is released with the pass that took it, so one refusal does
    not make every later pass on the same store unstampable — which would
    turn a caught programming error into a store nothing can fingerprint."""
    path = tmp_path / "store"
    _store_with(path, {"10": [[1, 2, 3, 4]]})

    with pytest.raises(RuntimeError):
        with _open(path, writable=True) as f:
            with writing_pass(f):
                with writing_pass(f):
                    pass

    with _open(path, writable=True) as f:
        with writing_pass(f):
            _write_document(f, "20", [[9, 10, 11, 12]])

    with _open(path) as f:
        assert read_content_digest(f) == content_digest(f)


def test_two_handles_on_one_store_do_not_nest_either(tmp_path):
    """A second handle onto the same store is a second writer into the same
    ids, so an exit on either one stamps over what the other is still
    writing. Keying the guard to the handle would miss it."""
    path = tmp_path / "store"
    _store_with(path, {"10": [[1, 2, 3, 4]]})

    with pytest.raises(RuntimeError):
        with _open(path, writable=True) as outer:
            with writing_pass(outer):
                with _open(path, writable=True) as inner:
                    with writing_pass(inner):
                        pass

    assert store_content_digest(path) is None


def test_a_symlink_to_the_store_is_refused_as_re_entry(tmp_path):
    """Two names, one directory, one set of ids underneath — the guard has
    to catch a writer opened through either. (A hard link, the case this
    guarded against for a single-file store, cannot name a directory.)"""
    path = tmp_path / "store"
    linked = tmp_path / "store-link"
    _store_with(path, {"10": [[1, 2, 3, 4]]})
    os.symlink(path, linked)

    with pytest.raises(RuntimeError):
        with _open(path, writable=True) as outer:
            with writing_pass(outer):
                with _open(linked, writable=True) as inner:
                    with writing_pass(inner):
                        pass

    assert store_content_digest(path) is None


def test_a_relative_handle_after_a_chdir_is_refused_as_re_entry(
    tmp_path, monkeypatch
):
    """The outer pass is opened by an absolute path before the working
    directory moves; the inner reopens the same store by a bare relative
    name resolved against the new cwd. Different strings, same file — the
    guard must not be fooled by the spelling."""
    path = tmp_path / "store"
    _store_with(path, {"10": [[1, 2, 3, 4]]})

    with pytest.raises(RuntimeError):
        with _open(path, writable=True) as outer:
            with writing_pass(outer):
                monkeypatch.chdir(tmp_path)
                with _open("store", writable=True) as inner:
                    with writing_pass(inner):
                        pass

    assert store_content_digest(path) is None


def test_external_key_and_document_round_trip():
    """`external_document` is the inverse of `external_key`, for the shapes
    both external corpora actually produce."""
    assert external_key("s800", "species001") == "s800:species001"
    assert external_document("s800:species001") == ("s800", "species001")


def test_external_document_splits_on_the_first_colon_only():
    """enzymeNER's own document id is itself `article:sentence`; the corpus
    prefix must not eat part of it."""
    key = external_key("enzymener", "PMC1233920:M01009")
    assert key == "enzymener:PMC1233920:M01009"
    assert external_document(key) == ("enzymener", "PMC1233920:M01009")


def test_external_document_is_none_for_a_bare_pubmed_id():
    """BRENDA's own keys carry no prefix and must not be misread as one."""
    assert external_document("21183147") is None


def test_inspect_encodings_reports_the_stamps_and_a_named_document(
    tmp_path, monkeypatch, capsys
):
    """The command replaces `h5dump`/`h5ls` for this store, so it has to
    show the stamps and a document's arrays without anything else to open
    the LMDB with."""
    from d3text.cli import inspect_encodings

    path = tmp_path / "store"
    _store_with(path, {"10": [[101, 7, 102]]})
    monkeypatch.setattr("sys.argv", ["inspect-encodings", str(path), "10"])

    inspect_encodings.main()

    out = capsys.readouterr().out
    assert BASE_MODEL in out
    assert "content digest: not recorded" in out
    assert "1 documents" in out
    assert "10: 1 windows" in out
    assert "[[101   7 102]]" in out


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_a_forked_process_reads_through_a_store_of_its_own(tmp_path):
    """`lmdb` refuses to open a path its process already holds, and a
    forked child inherits the parent's handle, which LMDB forbids it to use;
    a `DataLoader` worker still has to read the store."""
    path = tmp_path / "store"
    _store_with(path, {"10": [[1, 2, 3, 4]]})

    with _open(path) as parent:
        pid = os.fork()
        if pid == 0:
            try:
                with _open(path) as child:
                    ok = child.windows("10") == 1
            except BaseException:
                ok = False
            os._exit(0 if ok else 1)
        _, status = os.waitpid(pid, 0)
        assert parent.windows("10") == 1

    assert os.waitstatus_to_exitcode(status) == 0
