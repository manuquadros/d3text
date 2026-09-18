"""Regression test for `entity_stats`'s dict/list entity-shape branching."""

import pathlib

from tinydb import TinyDB
from tinydb.storages import JSONStorage

from scripts.statistics import entity_stats

TESTDB_PATH = pathlib.Path(__file__).parent / "test_files/testdb.json"


def test_entity_stats_reads_dict_and_list_shaped_entities() -> None:
    """Bacteria are stored as a `{id: name}` dict, strains and enzymes as
    plain id lists; `entity_stats` must recognize both shapes (previously
    `type(x) == dict`/`== list`, now `isinstance`) and still tally the
    `HasEnzyme` relations.
    """
    with TinyDB(TESTDB_PATH, storage=JSONStorage) as db:
        docs = db.table("documents").all()
        refcounts = entity_stats(docs, db)

    assert refcounts["bacteria"]
    assert refcounts["strains"]
    assert refcounts["enzymes"]
    assert refcounts["has_enzyme"]
