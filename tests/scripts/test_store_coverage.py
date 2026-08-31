"""The reserved provenance key must not be counted as a document."""

import json
import sys
from pathlib import Path

import lmdb
import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "scripts/dec03_full/vm")
)
import store_coverage  # noqa: E402

from d3text.embeddings_store import StoreProvenance, write_provenance  # noqa: E402


@pytest.fixture
def stamped_store(tmp_path: Path) -> Path:
    store_path = tmp_path / "store.lmdb"
    env = lmdb.open(str(store_path), map_size=10 * 1024 * 1024)
    with env.begin(write=True) as txn:
        for pmid in ("111", "222", "333"):
            txn.put(pmid.encode(), b"\x00")
    write_provenance(
        env,
        StoreProvenance(base_model="test-model", max_length=512, stride=20),
    )
    env.close()
    return store_path


def test_keys_equals_document_count_for_a_stamped_store(
    stamped_store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "training_data.csv").write_text("pubmed_id\n111\n222\n")

    monkeypatch.setattr(store_coverage, "CORPUS", corpus_dir)
    monkeypatch.setattr(store_coverage, "SOURCES", ("training_data.csv",))

    out_path = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "store_coverage.py",
            str(stamped_store),
            "--out",
            str(out_path),
        ],
    )

    store_coverage.main()

    report = json.loads(out_path.read_text())
    assert report["keys"] == 3
