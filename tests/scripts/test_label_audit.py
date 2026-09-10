"""``label_audit.py``: the guard-check script over the index it audits.

`build_index` used to be a hand-copy of
`d3text.cli.precompute_token_labels.build_index`; the identity check below is
what would have caught that. The digest checks exercise `check_store_index`
directly rather than `main()`'s stdout, per the same reasoning `conftest.py`
gives for `empty_token_label_store`: a bare `IndexStamp` is enough to name a
"different index" without building a second real one.
"""

import importlib.util
import json
import pathlib

import h5py
import polars as pl
import pytest
from d3text import token_labels
from d3text.cli import precompute_token_labels

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts/dec04_full/label_audit.py"
)


def _load_label_audit():
    """`label_audit.py` as a module, without putting `scripts/` on the path.

    Same reasoning as `tests/scripts/test_build_organism_taxid_bridge.py`:
    every name under `scripts/` is a top-level one.
    """
    spec = importlib.util.spec_from_file_location(_SCRIPT.stem, _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


label_audit = _load_label_audit()

_TABLES = {
    "enzymes": {
        "3494": {
            "recommended_name": "cholesterol oxidase",
            "ec_class": "1.1.3.6",
            "synonyms": ["COD"],
        },
    },
    "bacteria": {
        "42": {"organism": "Streptomyces griseocarneus", "synonyms": []}
    },
    "strains": {},
}

_ROWS = [
    {
        "pubmed_id": 10822008,
        "abstract": "cholesterol oxidase from Streptomyces griseocarneus",
        "fulltext": "and some catalase besides",
        "enzymes": "[3494]",
        "bacteria": "{'42': 'Streptomyces griseocarneus'}",
        "strains": "[]",
        "other_organisms": "{'7': 'Jaculus orientalis'}",
    },
]

_CORPUS_SCHEMA = {
    "pubmed_id": pl.Int64,
    "abstract": pl.Utf8,
    "fulltext": pl.Utf8,
    "enzymes": pl.Utf8,
    "bacteria": pl.Utf8,
    "strains": pl.Utf8,
    "other_organisms": pl.Utf8,
}


@pytest.fixture
def entity_tables(tmp_path) -> pathlib.Path:
    path = tmp_path / "documents.json"
    path.write_text(json.dumps(_TABLES), encoding="utf8")
    return path


@pytest.fixture
def corpus_csv(tmp_path) -> pathlib.Path:
    path = tmp_path / "split.csv"
    pl.DataFrame(_ROWS, schema=_CORPUS_SCHEMA).write_csv(path)
    return path


def test_build_index_is_the_precompute_command_s_not_a_copy() -> None:
    """Proves the import replaced the local duplicate rather than merely
    matching its behaviour — a hand-copy that happened to agree would pass
    every other test in this file too."""
    assert label_audit.build_index is precompute_token_labels.build_index


def test_a_store_stamped_from_the_same_index_reports_no_mismatch(
    entity_tables, corpus_csv, tmp_path
) -> None:
    index = label_audit.build_index(entity_tables, [corpus_csv])
    stamp = token_labels.IndexStamp.from_index(
        index, sources=[str(entity_tables), str(corpus_csv)]
    )

    store_path = tmp_path / "labels.hdf5"
    with h5py.File(store_path, "w", libver="latest") as store:
        token_labels.write_label_space(store, stamp=stamp)

    with h5py.File(store_path, "r") as store:
        assert label_audit.check_store_index(store, stamp) is None


def test_a_store_stamped_from_another_index_is_reported_as_a_mismatch(
    entity_tables, corpus_csv, tmp_path
) -> None:
    index = label_audit.build_index(entity_tables, [corpus_csv])
    stamp = token_labels.IndexStamp.from_index(
        index, sources=[str(entity_tables), str(corpus_csv)]
    )
    other_stamp = token_labels.IndexStamp(digest="deliberately-wrong")

    store_path = tmp_path / "labels.hdf5"
    with h5py.File(store_path, "w", libver="latest") as store:
        token_labels.write_label_space(store, stamp=other_stamp)

    with h5py.File(store_path, "r") as store:
        report = label_audit.check_store_index(store, stamp)

    assert report is not None
    assert stamp.digest[:12] in report
    assert other_stamp.digest[:12] in report
