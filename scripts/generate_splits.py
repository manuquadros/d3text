#!/usr/bin/env python
"""Draw BRENDA's training, validation and test splits and write them as CSV.

Rare entities that share no surface form with any other are held out of
training, so the unseen bucket of detection recall measures surface forms the
tagger was never trained on. The splits page of the documentation says why;
`brenda_references.sampling.entity_holdout_splits` is the algorithm::

    pdm run python scripts/generate_splits.py <output_dir>

A paper BRENDA curates under several references is several rows sharing one
`pubmed_id`. The rows are split as one paper, with their entities unioned the
way `merge_duplicate_documents` unions them on load, and written unmerged.
"""

import argparse
import collections
import pathlib
from collections.abc import Iterable, Mapping, Sequence

import pandas as pd
from brenda_references.data_paths import documents_path, split_path
from brenda_references.docdb import BrendaDocDB
from brenda_references.sampling import SPLITS, entity_holdout_splits
from tinydb.table import Document

from d3text.surface_forms import (
    BRENDA_PREFIXES,
    brenda_surface_forms,
    collision_keys,
    load_entity_tables,
)


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="generate_splits.py",
        description=(
            "Split BRENDA's full-text papers into training, validation and "
            "test, holding rare entities out of training."
        ),
    )
    parser.add_argument(
        "output_dir",
        type=pathlib.Path,
        help="directory to write the three split CSVs into",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--evaluation-share",
        type=float,
        default=0.15,
        help="share of papers in validation, and again in test",
    )
    parser.add_argument(
        "--held-share",
        type=float,
        default=0.3,
        help="share of validation and test papers naming a held-out entity",
    )
    parser.add_argument(
        "--max-held-documents",
        type=int,
        default=3,
        help="most papers a held-out group of entities may occur in",
    )
    return parser.parse_args()


def entity_ids(row: Mapping[str, object]) -> frozenset[str]:
    """The prefixed IDs of every entity a corpus row is linked to."""
    ids: set[str] = set()
    for field, prefix in BRENDA_PREFIXES.items():
        linked = row.get(field) or ()
        if not isinstance(linked, Iterable):
            msg = f"row {row.get('pubmed_id')}: {field} is {linked!r}"
            raise TypeError(msg)
        ids.update(prefix + str(entity_id) for entity_id in linked)
    return frozenset(ids)


def papers(rows: Iterable[Mapping[str, object]]) -> dict[int, frozenset[str]]:
    """pubmed ID -> the entities of every row carrying it, unioned."""
    linked: dict[int, set[str]] = collections.defaultdict(set)
    for row in rows:
        linked[int(str(row["pubmed_id"]))] |= entity_ids(row)
    return {pubmed_id: frozenset(ids) for pubmed_id, ids in linked.items()}


def report(
    splits: Mapping[str, Sequence[int]],
    documents: Mapping[int, frozenset[str]],
    form_keys: Mapping[str, frozenset[str]],
) -> None:
    """Print each split's size and what validation and test hold unseen."""
    training = set().union(*(documents[doc] for doc in splits["training"]))
    training_keys = set().union(
        *(form_keys.get(entity, frozenset()) for entity in training)
    )
    for name in SPLITS:
        entities = set().union(*(documents[doc] for doc in splits[name]))
        unseen = entities - training
        by_type = collections.Counter(
            prefix
            for entity in unseen
            for prefix in BRENDA_PREFIXES.values()
            if entity.startswith(prefix)
        )
        seen_forms = sum(
            1
            for entity in unseen
            if form_keys.get(entity, set()) & training_keys
        )
        print(
            f"{name}: {len(splits[name])} papers, {len(entities)} entities, "
            f"{len(unseen)} absent from training "
            f"({', '.join(f'{p}={n}' for p, n in sorted(by_type.items()))}), "
            f"{seen_forms} of them with a surface form training has"
        )


def main() -> None:
    args = read_args()

    with BrendaDocDB() as docdb:
        rows: list[Document] = [
            row
            for row in docdb.fulltext_articles()
            if row["strains"] or not row["bacteria"]
        ]
    keyed = [row for row in rows if row.get("pubmed_id")]
    print(
        f"{len(rows)} full-text rows, {len(rows) - len(keyed)} skipped "
        "for carrying no pubmed_id"
    )

    forms = brenda_surface_forms(
        load_entity_tables(documents_path()),
        [row.get("other_organisms") or {} for row in keyed],
    )
    form_keys = {entity: collision_keys(f) for entity, f in forms.items()}
    documents = papers(keyed)

    splits = entity_holdout_splits(
        documents,
        form_keys,
        evaluation_share=args.evaluation_share,
        held_share=args.held_share,
        max_held_documents=args.max_held_documents,
        seed=args.seed,
    )
    report(splits, documents, form_keys)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLITS:
        members = set(splits[name])
        split_rows = [
            row for row in keyed if int(str(row["pubmed_id"])) in members
        ]
        path = args.output_dir / split_path(name).name
        pd.DataFrame(split_rows).to_csv(path)
        print(f"wrote {len(split_rows)} rows to {path}")


if __name__ == "__main__":
    main()
