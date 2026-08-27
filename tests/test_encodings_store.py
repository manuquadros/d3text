"""The encodings HDF5's own provenance stamp.

`precompute-encodings` writes token ids into an HDF5 keyed by pubmed id; the
ids alone say nothing about which model, window or stride tokenized them, and
`d3text.embeddings_store` already showed the aggregated row count comes to the
same value under any window or stride. `record_provenance` is the guard that
keeps two geometries out of the same file, mirroring
`d3text.cli.precompute_embeddings.record_provenance`.
"""

import h5py
import pytest

from d3text.encodings_store import (
    EncodingsProvenance,
    read_provenance,
    record_provenance,
    write_provenance,
)

BASE_MODEL = "michiyasunaga/BioLinkBERT-base"
PROVENANCE = EncodingsProvenance(
    base_model=BASE_MODEL, max_length=512, stride=20
)


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
