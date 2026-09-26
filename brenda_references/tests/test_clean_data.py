"""`clean_data`'s per-document strip must actually drop removed strain ids.

Before this fix the loop in `main()` fetched a document's `strains` list and
did `pass`, so a document kept naming a strain id `clean_data` had just moved
out of the `strains` table. `strip_removed_strains` is that loop's body,
pulled out so it is callable without a real BRENDA doc db.
"""

from tinydb import TinyDB
from tinydb.storages import MemoryStorage

from scripts.clean_data import strip_removed_strains


def test_strip_removed_strains_drops_ids_the_table_no_longer_holds() -> None:
    with TinyDB(storage=MemoryStorage) as db:
        documents = db.table("documents")
        doc_id = documents.insert({"strains": [1, 2, 3]})

        strip_removed_strains(documents, removed_ids=[2])

        assert documents.get(doc_id=doc_id)["strains"] == [1, 3]


def test_strip_removed_strains_leaves_documents_without_strains_alone() -> None:
    with TinyDB(storage=MemoryStorage) as db:
        documents = db.table("documents")
        doc_id = documents.insert({"title": "no strains column"})

        strip_removed_strains(documents, removed_ids=[2])

        assert documents.get(doc_id=doc_id) == {"title": "no strains column"}
