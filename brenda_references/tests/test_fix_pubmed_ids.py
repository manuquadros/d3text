"""Regression tests for `scripts/fix_pubmed_ids.py`."""

import re
import runpy
from pathlib import Path

import pytest
from brenda_references import docdb
from brenda_references.docdb import BrendaDocDB

SCRIPT = Path(__file__).parents[1] / "scripts" / "fix_pubmed_ids.py"


def test_script_strips_whitespace_padded_pubmed_ids(monkeypatch) -> None:
    """Every stored `pubmed_id` ends up all digits, padding included.

    `int()` accepts surrounding whitespace, so a padded id passed the
    script's old "is it numeric" test and was never rewritten.
    """

    class _Memory(BrendaDocDB):
        def __init__(self) -> None:
            super().__init__(storage="memory")

        def __exit__(self, *exc: object) -> None:
            return None

    db = _Memory()
    for pmid in (" 17546672", "23246158 ", "99", None):
        db.documents.insert({"pubmed_id": pmid})
    monkeypatch.setattr(docdb, "BrendaDocDB", lambda: db)

    runpy.run_path(str(SCRIPT), run_name="__main__")

    assert [d["pubmed_id"] for d in db.documents] == [
        "17546672",
        "23246158",
        "99",
        None,
    ]


@pytest.mark.integration
def test_doc_db_pubmed_ids_are_all_digits() -> None:
    """No non-empty stored `pubmed_id` carries anything but digits.

    Empty strings are how the db records a missing id; the script skips them.
    """
    with BrendaDocDB() as db:
        bad = [
            d["pubmed_id"]
            for d in db.references
            if d["pubmed_id"] and not re.fullmatch(r"\d+", d["pubmed_id"])
        ]
    assert bad == []
