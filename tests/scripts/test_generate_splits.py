"""A paper curated under several BRENDA references is split as one paper."""

import importlib.util
import pathlib

from brenda_references.sampling import SPLITS, entity_holdout_splits

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2] / "scripts/generate_splits.py"
)


def _load_generator():
    """The generator script as a module, keeping `scripts/` off the path.

    Every name under `scripts/` is a top-level one, so importing by path keeps
    the whole directory from shadowing installed packages for the rest of the
    session.
    """
    spec = importlib.util.spec_from_file_location(_SCRIPT.stem, _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generator = _load_generator()


def _row(pubmed_id: object, enzymes: list[int]) -> dict[str, object]:
    return {
        "pubmed_id": pubmed_id,
        "enzymes": enzymes,
        "bacteria": {},
        "strains": [],
        "other_organisms": {"7": "Homo sapiens"},
    }


def test_rows_sharing_a_pubmed_id_are_one_paper() -> None:
    """Each row carries part of the paper's gold set; the paper carries all.

    Split row by row, one paper's full text could sit in training and test at
    once, and an entity's paper count would count one paper several times.
    The ID is spelled as a string on one row and an int on another, as the
    TinyDB dump and a CSV round trip respectively spell it.
    """
    papers = generator.papers([_row("11", [1]), _row(11, [2]), _row("12", [1])])

    assert papers == {
        11: frozenset({"enz1", "enz2", "oth7"}),
        12: frozenset({"enz1", "oth7"}),
    }


def test_no_paper_lands_in_two_splits() -> None:
    rows = [_row(pubmed_id % 40, [pubmed_id % 9]) for pubmed_id in range(120)]
    documents = generator.papers(rows)

    splits = entity_holdout_splits(documents, {}, evaluation_share=0.2)

    homes = [
        next(name for name in SPLITS if int(row["pubmed_id"]) in splits[name])
        for row in rows
    ]
    by_paper: dict[object, set[str]] = {}
    for row, home in zip(rows, homes, strict=True):
        by_paper.setdefault(row["pubmed_id"], set()).add(home)
    assert all(len(found) == 1 for found in by_paper.values())
