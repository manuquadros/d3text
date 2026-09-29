"""`localization_probe.encoded_ids` reads the LMDB encodings store."""

import importlib.util
import pathlib

import numpy
from d3text.encodings_store import EncodingsStore

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts/dec02_probe/localization_probe.py"
)


def _probe():
    spec = importlib.util.spec_from_file_location("localization_probe", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_encoded_ids_reads_a_stored_document_and_none_for_an_absent_one(
    tmp_path,
):
    """The probe compares these ids with a fresh tokenization; the store is
    LMDB, so a reader written for HDF5 cannot even index it."""
    ids = numpy.asarray([[101, 7, 8, 102]], dtype="uint32")
    path = tmp_path / "encodings"
    with EncodingsStore(path, writable=True) as store:
        store.put(
            "1",
            {
                "input_ids": ids,
                "attention_mask": numpy.ones(ids.shape, dtype="uint8"),
                "offset_mapping": numpy.zeros((*ids.shape, 2), dtype="uint32"),
            },
        )
    probe = _probe()
    with EncodingsStore(path) as store:
        assert probe.encoded_ids(store, "1").tolist() == [7, 8]
        assert probe.encoded_ids(store, "2") is None
    assert probe.encoded_ids(None, "1") is None
