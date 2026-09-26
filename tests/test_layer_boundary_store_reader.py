"""`LayerBoundaryStore` reading back what `precompute-embeddings` wrote.

The windowed codec's round trip and header layout are pinned in
`test_embeddings_store.py`. This file mirrors
`test_embeddings_store_reader.py` for this store — refusing a store
stamped at another `frozen_layers` or not stamped at all, and turning a
window-count disagreement into `None` plus one warning — and adds what
that file has no counterpart for: the constructor closing its env when it
raises, and the `D3WL` codec's own magic and version refusals.
"""

import pathlib
import struct

import lmdb
import pytest
import torch
from d3text import embeddings_store
from d3text.embeddings_store import (
    LayerBoundaryProvenance,
    LayerBoundaryStore,
    ProvenanceError,
    bytes_to_windowed_tensor,
    windowed_tensor_to_bytes,
    write_layer_provenance,
)

BASE_MODEL = "michiyasunaga/BioLinkBERT-base"
FROZEN_LAYERS = 6
MAX_LENGTH = 512
PROVENANCE = LayerBoundaryProvenance(
    base_model=BASE_MODEL,
    max_length=MAX_LENGTH,
    stride=20,
    frozen_layers=FROZEN_LAYERS,
)


def _write_store(
    path: pathlib.Path,
    provenance: LayerBoundaryProvenance | None = PROVENANCE,
    documents: dict[int, torch.Tensor] | None = None,
) -> pathlib.Path:
    """An LMDB stamped with `provenance` and holding windowed `documents`."""
    env = lmdb.open(str(path), map_size=8 * 1024**2)
    if provenance is not None:
        write_layer_provenance(env, provenance)
    with env.begin(write=True) as transaction:
        for pubmed_id, tensor in (documents or {}).items():
            transaction.put(
                str(pubmed_id).encode(), windowed_tensor_to_bytes(tensor)
            )
    env.close()

    return path


@pytest.fixture
def store_path(tmp_path):
    """An LMDB holding one 3-window document under pubmed id 100."""
    return _write_store(
        tmp_path / "layer-boundary",
        documents={100: torch.rand(3, 12, 8)},
    )


def test_a_store_stamped_at_another_frozen_layers_is_refused(tmp_path):
    """The failure this store exists for on top of `EmbeddingsStore`: a
    prefix cached at another boundary is a valid tensor of the right shape
    for the wrong layer, so if `frozen_layers` were dropped from the
    comparison nothing downstream would fail loudly."""
    path = _write_store(
        tmp_path / "other-boundary",
        provenance=LayerBoundaryProvenance(
            base_model=BASE_MODEL,
            max_length=512,
            stride=20,
            frozen_layers=3,
        ),
        documents={100: torch.rand(3, 12, 8)},
    )

    with pytest.raises(ProvenanceError, match="frozen through layer 3"):
        LayerBoundaryStore(
            path, BASE_MODEL, frozen_layers=FROZEN_LAYERS, max_length=MAX_LENGTH
        )


def test_a_store_that_does_not_say_who_wrote_it_is_refused(tmp_path):
    """An unstamped store is not a store known to be right; it is a store
    nothing attributes to anything."""
    path = _write_store(
        tmp_path / "unstamped",
        provenance=None,
        documents={100: torch.rand(3, 12, 8)},
    )

    with pytest.raises(ProvenanceError, match="does not record which model"):
        LayerBoundaryStore(
            path, BASE_MODEL, frozen_layers=FROZEN_LAYERS, max_length=MAX_LENGTH
        )


def test_a_store_stamped_at_another_window_is_refused(tmp_path):
    """A store built at a window other than the caller's own opened without
    complaint before this check. The window-count check `get` runs is no
    substitute for a one-window document: it compares how many windows were
    stored against how many the encodings imply, which agrees whatever width
    each window was actually embedded at, so a wrong-window store would
    still pass it there.

    The caller here asks for a window that is neither the stamped one nor
    `MAX_LENGTH`, so the refusal cannot be coming from a constant the store
    hardcodes instead of the argument it was actually given.
    """
    path = _write_store(
        tmp_path / "other-window",
        provenance=LayerBoundaryProvenance(
            base_model=BASE_MODEL,
            max_length=128,
            stride=20,
            frozen_layers=FROZEN_LAYERS,
        ),
        documents={100: torch.rand(1, 128, 8)},
    )

    with pytest.raises(ProvenanceError, match="window 128"):
        LayerBoundaryStore(
            path, BASE_MODEL, frozen_layers=FROZEN_LAYERS, max_length=256
        )


def test_a_window_count_that_disagrees_with_the_encodings_is_refused(
    store_path,
):
    """The stored document has 3 windows; a document whose encodings imply
    4 was built from different text than the store, and must be run live
    rather than handed the wrong prefix."""
    store = LayerBoundaryStore(
        store_path, BASE_MODEL, FROZEN_LAYERS, MAX_LENGTH
    )

    assert store.get(100, expected_windows=4) is None
    assert (store.hits, store.mismatches) == (0, 1)


def test_the_window_mismatch_is_warned_about_once(store_path, caplog):
    """Once, not once per document, for the same reason `EmbeddingsStore`
    limits its own mismatch warning to one line."""
    store = LayerBoundaryStore(
        store_path, BASE_MODEL, FROZEN_LAYERS, MAX_LENGTH
    )

    with caplog.at_level("WARNING"):
        for _ in range(3):
            store.get(100, expected_windows=4)

    warnings = [
        record for record in caplog.records if record.levelname == "WARNING"
    ]
    assert len(warnings) == 1
    assert store.mismatches == 3


def test_the_env_is_closed_when_provenance_error_is_raised_in_init(
    tmp_path, monkeypatch
):
    """`__init__` opens the environment before it knows the store is
    attributable; a `ProvenanceError` must not leak that handle. A closed
    `lmdb.Environment` raises on any further use, so that is the probe."""
    path = _write_store(
        tmp_path / "unstamped",
        provenance=None,
        documents={100: torch.rand(3, 12, 8)},
    )
    opened: dict[str, lmdb.Environment] = {}
    real_open = lmdb.open

    def spy_open(*args: object, **kwargs: object) -> lmdb.Environment:
        env = real_open(*args, **kwargs)
        opened["env"] = env
        return env

    monkeypatch.setattr(embeddings_store.lmdb, "open", spy_open)

    with pytest.raises(ProvenanceError):
        LayerBoundaryStore(
            path, BASE_MODEL, frozen_layers=FROZEN_LAYERS, max_length=MAX_LENGTH
        )

    with pytest.raises(lmdb.Error):
        opened["env"].stat()


def test_bytes_to_windowed_tensor_refuses_the_wrong_magic():
    """A blob with the `D3WL` header's exact length and a valid compressed
    body, but someone else's magic. Without the check, `_unpack` reads the
    header and shape as written and `_decompress` succeeds on the
    untouched body, so this would decode silently into the tensor it was
    written as rather than raise; the magic is what turns that into an
    error instead."""
    packed = windowed_tensor_to_bytes(torch.rand(3, 4, 5))
    wrong_magic = b"XXXX" + packed[4:]

    with pytest.raises(ValueError, match="not a layer-boundary store blob"):
        bytes_to_windowed_tensor(wrong_magic)


def test_a_future_windowed_format_version_is_refused():
    packed = windowed_tensor_to_bytes(torch.ones(2, 3, 4))
    bumped = struct.pack("<4sB", b"D3WL", 2) + packed[5:]

    with pytest.raises(ValueError, match="version 2 is not readable"):
        bytes_to_windowed_tensor(bumped)
