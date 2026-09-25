"""Opening a `LayerBoundaryStore` enables blosc2's process-global GIL release.

`_resolve_layer_boundary_cached` reads a batch's hits from a single
background thread while the main thread replays an earlier hit, so
`blosc2.decompress2` must not hold the GIL during that read -- otherwise the
background thread just queues up behind the main thread's own work and
nothing overlaps. `blosc2.set_releasegil` defaults to off and is a process
setting, not a per-call one, so this pins that opening a store is what turns
it on.
"""

import blosc2
import lmdb

from d3text.embeddings_store import LayerBoundaryProvenance, LayerBoundaryStore
from d3text.embeddings_store import write_layer_provenance

BASE_MODEL = "michiyasunaga/BioLinkBERT-base"
PROVENANCE = LayerBoundaryProvenance(
    base_model=BASE_MODEL, max_length=512, stride=20, frozen_layers=6
)


def test_opening_a_store_enables_blosc2s_gil_release(tmp_path):
    path = tmp_path / "store"
    with lmdb.open(str(path), map_size=2**20) as env:
        write_layer_provenance(env, PROVENANCE)

    blosc2.set_releasegil(False)
    store = LayerBoundaryStore(path, BASE_MODEL, frozen_layers=6)
    try:
        previously_enabled = blosc2.set_releasegil(False)
    finally:
        store.close()

    assert previously_enabled is True
