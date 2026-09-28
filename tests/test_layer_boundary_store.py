"""Opening a `LayerBoundaryStore` enables blosc2's process-global GIL release.

Its hits are decompressed on a background thread that overlaps the main
thread only if blosc2 drops the GIL, which it does not by default.
"""

import blosc2

from d3text.embeddings_store import LayerBoundaryStore, StoreProvenance

BASE_MODEL = "michiyasunaga/BioLinkBERT-base"
PROVENANCE = StoreProvenance(base_model=BASE_MODEL, max_length=512, stride=20)


def test_opening_a_store_enables_blosc2s_gil_release(tmp_path):
    path = tmp_path / "store"
    LayerBoundaryStore.create(path, PROVENANCE, unfrozen_top_layers=6).close()

    blosc2.set_releasegil(False)
    store = LayerBoundaryStore(
        path, BASE_MODEL, unfrozen_top_layers=6, max_length=512
    )
    try:
        previously_enabled = blosc2.set_releasegil(False)
    finally:
        store.close()

    assert previously_enabled is True
