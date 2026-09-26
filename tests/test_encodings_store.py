"""The encodings HDF5's own provenance stamp.

`precompute-encodings` writes token ids into an HDF5 keyed by pubmed id; the
ids alone say nothing about which model, window or stride tokenized them, and
`d3text.embeddings_store` already showed the aggregated row count comes to the
same value under any window or stride. `record_provenance` is the guard that
keeps two geometries out of the same file, mirroring
`d3text.cli.precompute_embeddings.record_provenance`; `content_digest` is the
one that separates two files the geometry describes identically.
`encodings_provenance` compares two such digests for a checkpoint being
scored against a store.
"""

import os

import h5py
import numpy
import pytest

from d3text.encodings_store import (
    EncodingsProvenance,
    check_provenance,
    content_digest,
    encodings_provenance,
    external_document,
    external_key,
    has_populated_mask,
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


def test_a_store_from_the_layout_before_the_last_bump_is_read(tmp_path):
    """The only layout this build has superseded differs by a per-group
    dataset nothing ever read, so refusing it would make every store on disk
    worthless over a field no number depends on."""
    with h5py.File(tmp_path / "store.hdf5", "w") as f:
        f.attrs["d3text_encodings_format"] = 1
        f.attrs["base_model"] = BASE_MODEL
        f.attrs["max_length"] = 512
        f.attrs["stride"] = 20

        assert read_provenance(f) == PROVENANCE


def test_a_resume_restamps_a_store_written_under_an_older_layout(tmp_path):
    """The groups a resume appends are in this build's layout; a stamp left
    at the older one would describe them as carrying a dataset they do not."""
    with h5py.File(tmp_path / "store.hdf5", "w") as f:
        f.attrs["d3text_encodings_format"] = 1
        f.attrs["base_model"] = BASE_MODEL
        f.attrs["max_length"] = 512
        f.attrs["stride"] = 20
        f.create_group("1").create_dataset("input_ids", data=[1, 2, 3])

        record_provenance(f, PROVENANCE)

        assert int(f.attrs["d3text_encodings_format"]) > 1
        assert read_provenance(f) == PROVENANCE


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


def test_check_provenance_accepts_a_matching_stamp(tmp_path):
    """A store stamped with this run's own base model and stride is
    accepted without raising — the check
    `BrendaDataset._check_encodings_provenance` now delegates to."""
    path = tmp_path / "store.hdf5"
    with h5py.File(path, "w") as f:
        write_provenance(f, PROVENANCE)

    check_provenance(path, PROVENANCE.base_model, PROVENANCE.stride)


def test_check_provenance_refuses_another_base_model(tmp_path):
    path = tmp_path / "store.hdf5"
    with h5py.File(path, "w") as f:
        write_provenance(f, PROVENANCE)

    with pytest.raises(ValueError, match="was tokenized by"):
        check_provenance(path, "other-model", PROVENANCE.stride)


def test_check_provenance_refuses_another_stride(tmp_path):
    path = tmp_path / "store.hdf5"
    with h5py.File(path, "w") as f:
        write_provenance(f, PROVENANCE)

    with pytest.raises(ValueError, match="stride of"):
        check_provenance(path, PROVENANCE.base_model, PROVENANCE.stride + 1)


def test_check_provenance_reads_an_unstamped_store_anyway(tmp_path, caplog):
    """Mirrors `record_provenance`'s own asymmetry: nothing before the stamp
    existed is refused for lacking one, but it is warned about once."""
    path = tmp_path / "store.hdf5"
    with h5py.File(path, "w"):
        pass

    with caplog.at_level("WARNING", logger="d3text.encodings_store"):
        check_provenance(path, "any-model", 20)

    assert "does not record" in caplog.text


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


def test_a_nested_pass_leaves_no_stamp_over_ids_it_does_not_cover(tmp_path):
    """The failure a re-entrant bracket produces is silent: the inner exit
    restates the digest, the outer goes on writing, and a kill after that
    leaves a fingerprint asserting agreement with ids that have moved — read
    as `matched`, warned about nowhere. Whatever the bracket does about
    nesting, what it must never do is that."""
    path = tmp_path / "store.hdf5"
    _store_with(path, {"10": [[1, 2, 3, 4]]})
    with h5py.File(path, "r+") as f:
        stamp_content_digest(f)

    with pytest.raises((RuntimeError, KeyboardInterrupt)):
        with h5py.File(path, "r+") as f:
            with writing_pass(f):
                _write_document(f, "10", [[5, 6, 7, 8]])
                with writing_pass(f):
                    pass
                # Reached only where re-entry is permitted: this is the group
                # the inner pass's stamp would not cover.
                _write_document(f, "20", [[9, 10, 11, 12]])
                raise KeyboardInterrupt

    recorded = store_content_digest(path)
    with h5py.File(path, "r") as f:
        assert recorded is None or recorded == content_digest(f)


def test_re_entering_a_writing_pass_is_refused(tmp_path):
    """Refusal is the whole mechanism, and it has to leave the store in the
    state an interrupt leaves it in: the aborted outer pass stamps nothing,
    so what is on disk reads as unstamped rather than as agreeing."""
    path = tmp_path / "store.hdf5"
    _store_with(path, {"10": [[1, 2, 3, 4]]})
    with h5py.File(path, "r+") as f:
        stamp_content_digest(f)

    with pytest.raises(RuntimeError):
        with h5py.File(path, "r+") as f:
            with writing_pass(f):
                _write_document(f, "10", [[5, 6, 7, 8]])
                with writing_pass(f):
                    pass

    assert store_content_digest(path) is None


def test_a_refused_re_entry_leaves_the_store_open_to_later_passes(tmp_path):
    """The guard is released with the pass that took it, so one refusal does
    not make every later pass on the same store unstampable — which would
    turn a caught programming error into a store nothing can fingerprint."""
    path = tmp_path / "store.hdf5"
    _store_with(path, {"10": [[1, 2, 3, 4]]})

    with pytest.raises(RuntimeError):
        with h5py.File(path, "r+") as f:
            with writing_pass(f):
                with writing_pass(f):
                    pass

    with h5py.File(path, "r+") as f:
        with writing_pass(f):
            _write_document(f, "20", [[9, 10, 11, 12]])

    with h5py.File(path, "r") as f:
        assert read_content_digest(f) == content_digest(f)


def test_two_handles_on_one_store_do_not_nest_either(tmp_path):
    """A second handle onto the same file is a second writer into the same
    ids, so an exit on either one stamps over what the other is still
    writing. Keying the guard to the handle would miss it."""
    path = tmp_path / "store.hdf5"
    _store_with(path, {"10": [[1, 2, 3, 4]]})

    with pytest.raises(RuntimeError):
        with h5py.File(path, "r+") as outer:
            with writing_pass(outer):
                with h5py.File(path, "r+") as inner:
                    with writing_pass(inner):
                        pass

    assert store_content_digest(path) is None


def test_a_hard_link_to_the_store_is_refused_as_re_entry(tmp_path):
    """A hard link shares the original's `(st_dev, st_ino)` but is not a
    symlink to it, so `os.path.realpath` reports it as a different file. Two
    names, one inode, one set of ids underneath — the guard has to catch a
    pass opened through either."""
    path = tmp_path / "store.hdf5"
    linked = tmp_path / "store-link.hdf5"
    _store_with(path, {"10": [[1, 2, 3, 4]]})
    os.link(path, linked)

    with pytest.raises(RuntimeError):
        with h5py.File(path, "r+") as outer:
            with writing_pass(outer):
                with h5py.File(linked, "r+") as inner:
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
    path = tmp_path / "store.hdf5"
    _store_with(path, {"10": [[1, 2, 3, 4]]})

    with pytest.raises(RuntimeError):
        with h5py.File(path, "r+") as outer:
            with writing_pass(outer):
                monkeypatch.chdir(tmp_path)
                with h5py.File("store.hdf5", "r+") as inner:
                    with writing_pass(inner):
                        pass

    assert store_content_digest(path) is None


def test_a_group_holding_no_ids_digests_as_though_it_were_not_there(tmp_path):
    """A pass killed between `create_group` and the `create_dataset` that
    follows it leaves one, and a resume skips a key already present, so it
    stays for good. No reader can serve that document, so the file has to
    digest as the file without it does — while still separating one set of
    ids from another, which is the whole point of the digest."""
    partial = tmp_path / "partial.hdf5"
    complete = tmp_path / "complete.hdf5"
    other = tmp_path / "other.hdf5"
    _store_with(partial, {"10": [[1, 2, 3, 4]]})
    with h5py.File(partial, "r+") as f:
        f.create_group("20")
    _store_with(complete, {"10": [[1, 2, 3, 4]]})
    _store_with(other, {"10": [[1, 2, 3, 5]]})

    with (
        h5py.File(partial, "r") as a,
        h5py.File(complete, "r") as b,
        h5py.File(other, "r") as c,
    ):
        assert content_digest(a) == content_digest(b)
        assert content_digest(a) != content_digest(c)


def test_a_store_left_holding_such_a_group_can_still_be_stamped(tmp_path):
    """What this costs is not a wrong number but an unrecoverable store: the
    stamp is taken at the end of every pass, so a digest that died on the
    leftover group left the file unstampable — and so permanently unattributed
    — until someone deleted that group by hand."""
    path = tmp_path / "store.hdf5"
    _store_with(path, {"10": [[1, 2, 3, 4]]})

    with pytest.raises(KeyboardInterrupt):
        with h5py.File(path, "r+") as f:
            with writing_pass(f):
                f.create_group("20")
                raise KeyboardInterrupt

    assert store_content_digest(path) is None

    with h5py.File(path, "r+") as f:
        with writing_pass(f):
            _write_document(f, "30", [[5, 6, 7, 8]])

    with h5py.File(path, "r") as f:
        assert read_content_digest(f) == content_digest(f)


def test_a_group_holding_only_ids_has_no_populated_mask(tmp_path):
    """A pass killed between `create_group` and the `attention_mask` write
    leaves a group `stored_ids` accepts but with no mask at all — the same
    tear `is_finished_group` catches, without needing its completion
    marker."""
    path = tmp_path / "ids_only.hdf5"
    with h5py.File(path, "w") as f:
        f.create_group("10").create_dataset(
            "input_ids", data=numpy.zeros((2, 8), dtype="uint32")
        )

    with h5py.File(path, "r") as f:
        assert has_populated_mask(f["10"]) is False


def test_a_zero_filled_mask_window_has_no_populated_mask(tmp_path):
    """A pass killed after `create_dataset("attention_mask", ...)` but before
    it is filled leaves the array at h5py's own zero fill: one window with no
    set position at all, which a real tokenization never produces since it
    always sets at least the special tokens. Checked across every window, not
    only a single-window group's."""
    path = tmp_path / "zero_mask.hdf5"
    with h5py.File(path, "w") as f:
        group = f.create_group("10")
        group.create_dataset(
            "input_ids", data=numpy.zeros((3, 8), dtype="uint32")
        )
        group.create_dataset(
            "attention_mask", data=numpy.zeros((3, 8), dtype="uint8")
        )

    with h5py.File(path, "r") as f:
        assert has_populated_mask(f["10"]) is False


def test_a_mask_set_in_every_window_is_populated(tmp_path):
    """The ordinary case: ids and a mask of the same shape, each window
    carrying at least one set position — a real tokenization, whitespace
    document or not."""
    path = tmp_path / "sound.hdf5"
    with h5py.File(path, "w") as f:
        group = f.create_group("10")
        group.create_dataset(
            "input_ids", data=numpy.zeros((2, 8), dtype="uint32")
        )
        mask = numpy.zeros((2, 8), dtype="uint8")
        mask[:, :2] = 1
        group.create_dataset("attention_mask", data=mask)

    with h5py.File(path, "r") as f:
        assert has_populated_mask(f["10"]) is True


def test_a_mask_of_a_different_shape_than_the_ids_has_no_populated_mask(
    tmp_path,
):
    """Ids and mask disagreeing on shape are not something any writer here
    produces intentionally, so it is refused the same as a torn write rather
    than read against the wrong axis."""
    path = tmp_path / "mismatched.hdf5"
    with h5py.File(path, "w") as f:
        group = f.create_group("10")
        group.create_dataset(
            "input_ids", data=numpy.zeros((2, 8), dtype="uint32")
        )
        group.create_dataset(
            "attention_mask", data=numpy.ones((1, 8), dtype="uint8")
        )

    with h5py.File(path, "r") as f:
        assert has_populated_mask(f["10"]) is False


def test_a_missing_key_has_no_populated_mask():
    """`h5py.File.get` returns `None` for an absent key; the predicate reads
    that the same way `stored_ids` and `is_finished_group` do."""
    assert has_populated_mask(None) is False


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
