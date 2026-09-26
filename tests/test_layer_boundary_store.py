"""Opening a `LayerBoundaryStore` enables blosc2's process-global GIL release.

`_resolve_layer_boundary_cached` reads a batch's hits from a single
background thread while the main thread replays an earlier hit, so
`blosc2.decompress2` must not hold the GIL during that read -- otherwise the
background thread just queues up behind the main thread's own work and
nothing overlaps. `blosc2.set_releasegil` defaults to off and is a process
setting, not a per-call one, so this pins that opening a store is what turns
it on.
"""

import json

import blosc2
import lmdb
import pytest

from d3text.embeddings_store import (
    LayerBoundaryProvenance,
    LayerBoundaryStore,
    ProvenanceError,
    read_layer_provenance,
    write_layer_provenance,
)

BASE_MODEL = "michiyasunaga/BioLinkBERT-base"
PROVENANCE = LayerBoundaryProvenance(
    base_model=BASE_MODEL, max_length=512, stride=20, frozen_layers=6
)


def test_opening_a_store_enables_blosc2s_gil_release(tmp_path):
    path = tmp_path / "store"
    with lmdb.open(str(path), map_size=2**20) as env:
        write_layer_provenance(env, PROVENANCE)

    blosc2.set_releasegil(False)
    store = LayerBoundaryStore(
        path, BASE_MODEL, frozen_layers=6, max_length=512
    )
    try:
        previously_enabled = blosc2.set_releasegil(False)
    finally:
        store.close()

    assert previously_enabled is True


def test_an_unknown_key_in_the_layer_record_does_not_refuse_it(tmp_path):
    """A future build may add a diagnostic field without bumping the format
    number; reading it back must not mistake that extra field for one of
    this build's own missing fields."""
    record = {
        "format": 1,
        "base_model": BASE_MODEL,
        "max_length": 512,
        "stride": 20,
        "frozen_layers": 6,
        "future_diagnostic_field": "whatever",
    }
    with lmdb.open(str(tmp_path / "store"), map_size=2**20) as env:
        with env.begin(write=True) as transaction:
            transaction.put(
                b"\x00layer_provenance", json.dumps(record).encode()
            )

        assert read_layer_provenance(env) == PROVENANCE


def test_frozen_layers_written_as_a_string_still_reads_as_an_int(tmp_path):
    """The reader casts a numeric string to int rather than passing it
    straight to the dataclass, where `NonNegative` would refuse it as the
    wrong type."""
    record = {
        "format": 1,
        "base_model": BASE_MODEL,
        "max_length": 512,
        "stride": 20,
        "frozen_layers": "6",
    }
    with lmdb.open(str(tmp_path / "store"), map_size=2**20) as env:
        with env.begin(write=True) as transaction:
            transaction.put(
                b"\x00layer_provenance", json.dumps(record).encode()
            )

        recorded = read_layer_provenance(env)

    assert recorded == PROVENANCE
    assert isinstance(recorded.frozen_layers, int)


def test_an_uncastable_layer_field_raises_provenance_error(tmp_path):
    """A value neither this build nor an older one could have written is a
    record it cannot read, not a raw `int()` failure escaping past it."""
    record = {
        "format": 1,
        "base_model": BASE_MODEL,
        "max_length": 512,
        "stride": 20,
        "frozen_layers": "six",
    }
    with lmdb.open(str(tmp_path / "store"), map_size=2**20) as env:
        with env.begin(write=True) as transaction:
            transaction.put(
                b"\x00layer_provenance", json.dumps(record).encode()
            )

        with pytest.raises(ProvenanceError):
            read_layer_provenance(env)
