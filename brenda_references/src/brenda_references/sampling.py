"""Splits that hold rare entities, and their surface forms, out of training.

Why the splits are drawn this way is on the splits page of the documentation.
"""

import collections
from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt

SPLITS = ("training", "validation", "test")

_TRAINING = 0


def entity_holdout_splits(
    documents: Mapping[int, frozenset[str]],
    form_keys: Mapping[str, frozenset[str]],
    evaluation_share: float = 0.15,
    held_share: float = 0.3,
    max_held_documents: int = 3,
    seed: int = 0,
) -> dict[str, list[int]]:
    """Split `documents` so no unseen entity has a surface form training saw.

    Groups of entities that share no surface form with any other entity, and
    occur in few documents, are held out whole: every document naming one
    goes to validation or test. The remaining documents are split by
    iterative stratification over their entity sets. Validation and test are
    the same size and are filled the same way, so they measure the same
    thing.

    :param documents: pubmed ID -> the prefixed IDs of the entities the
        document is linked to.
    :param form_keys: prefixed entity ID -> the keys its surface forms are
        looked up by. Two entities share a surface form exactly when their
        keys intersect. An entity missing here has no surface form, so no
        dictionary match can ever find it.
    :param evaluation_share: the share of documents validation gets, and
        test gets the same; training gets the rest.
    :param held_share: the share of validation and test documents to fill
        with documents of held-out entities, reached as nearly as whole
        groups of entities allow.
    :param max_held_documents: the most documents a held-out group of
        entities may occur in, all together.
    :param seed: seeds every random choice, so one seed and one input give
        one split.
    :return: split name -> the pubmed IDs in it, sorted.
    :raises ValueError: if `evaluation_share` is outside `[0, 0.5]`,
        `held_share` outside `[0, 1]`, or `max_held_documents` below 1.
    :raises RuntimeError: if no training document can make room for one
        that has to move into training.
    """
    if not 0 <= evaluation_share <= 0.5:
        msg = (
            f"evaluation_share must be within [0, 0.5], got {evaluation_share}"
        )
        raise ValueError(msg)
    if not 0 <= held_share <= 1:
        msg = f"held_share must be within [0, 1], got {held_share}"
        raise ValueError(msg)
    if max_held_documents < 1:
        msg = f"max_held_documents must be at least 1, got {max_held_documents}"
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    pubmed_ids = sorted(documents)
    labels = [documents[pubmed_id] for pubmed_id in pubmed_ids]
    # Capped because rounding half a pool of odd size up would ask validation
    # and test for one document more than the pool holds.
    evaluation_size = min(
        round(len(labels) * evaluation_share), len(labels) // 2
    )

    docs_of: dict[str, set[int]] = collections.defaultdict(set)
    for doc, entities in enumerate(labels):
        for entity in entities:
            docs_of[entity].add(doc)

    held_docs = _held_documents(
        labels,
        docs_of,
        form_keys,
        target=held_share * 2 * evaluation_size,
        capacity=2 * evaluation_size,
        max_held_documents=max_held_documents,
        rng=rng,
    )

    held = sorted(held_docs)
    rest = [doc for doc in range(len(labels)) if doc not in held_docs]
    to_test = len(held) // 2
    to_validation = len(held) - to_test

    assign = np.empty(len(labels), dtype=np.int64)
    assign[held] = 1 + _stratify(
        [labels[doc] for doc in held], [to_validation, to_test], rng
    )
    assign[rest] = _stratify(
        [labels[doc] for doc in rest],
        [
            len(labels) - 2 * evaluation_size,
            evaluation_size - to_validation,
            evaluation_size - to_test,
        ],
        rng,
    )
    assign = _repair(assign, labels, docs_of, form_keys, held_docs, rng)

    return {
        name: [pubmed_ids[doc] for doc in np.flatnonzero(assign == split)]
        for split, name in enumerate(SPLITS)
    }


def _form_groups(
    entities: Sequence[str], form_keys: Mapping[str, frozenset[str]]
) -> dict[str, str]:
    """Entity -> a representative of its group of entities sharing forms.

    Transitively, because holding out one entity puts every entity sharing a
    form with it out of training too, and then everything sharing a form with
    those.
    """
    parent = {entity: entity for entity in entities}

    def find(entity: str) -> str:
        while parent[entity] != entity:
            parent[entity] = parent[parent[entity]]
            entity = parent[entity]
        return entity

    first_with: dict[str, str] = {}
    for entity in entities:
        for key in form_keys.get(entity, frozenset()):
            other = first_with.setdefault(key, entity)
            parent[find(entity)] = find(other)

    return {entity: find(entity) for entity in entities}


def _held_documents(
    labels: Sequence[frozenset[str]],
    docs_of: Mapping[str, set[int]],
    form_keys: Mapping[str, frozenset[str]],
    target: float,
    capacity: int,
    max_held_documents: int,
    rng: np.random.Generator,
) -> set[int]:
    """The documents of entity groups drawn at random until `target` is met.

    A group is taken together with every group it strands: an entity with a
    surface form whose every document is already held can reach no training
    document, so its own group has to be held with it, or the candidate is
    refused. An entity with no surface form strands nothing, since no
    dictionary match can find it in either split.
    """
    group_of = _form_groups(sorted(docs_of), form_keys)
    members: dict[str, list[str]] = collections.defaultdict(list)
    for entity, group in group_of.items():
        members[group].append(entity)

    def group_docs(group: str) -> set[int]:
        return set().union(*(docs_of[entity] for entity in members[group]))

    eligible = {
        group
        for group, entities in members.items()
        if all(form_keys.get(entity) for entity in entities)
        and len(group_docs(group)) <= max_held_documents
    }

    held_groups: set[str] = set()
    held_docs: set[int] = set()

    def closure(first: str) -> tuple[set[str], set[int]] | None:
        """`first` and every group it strands, with their papers; or None."""
        pending = [first]
        taken = {first}
        docs: set[int] = set()
        while pending:
            group = pending.pop()
            if group not in eligible:
                return None
            docs |= group_docs(group)
            for doc in docs:
                for entity in labels[doc]:
                    other = group_of[entity]
                    stranded = (
                        other not in taken
                        and other not in held_groups
                        and bool(form_keys.get(entity))
                        and docs_of[entity] <= docs | held_docs
                    )
                    if stranded:
                        taken.add(other)
                        pending.append(other)
        return taken, docs

    groups = sorted(eligible)
    for index in rng.permutation(len(groups)):
        if len(held_docs) >= target:
            break
        if groups[index] in held_groups:
            continue
        grown = closure(groups[index])
        if grown is not None and len(held_docs | grown[1]) <= capacity:
            held_groups |= grown[0]
            held_docs |= grown[1]

    return held_docs


def _stratify(
    label_sets: Sequence[frozenset[str]],
    sizes: Sequence[int],
    rng: np.random.Generator,
) -> npt.NDArray[np.int64]:
    """Each document's split, by iterative stratification (Sechidis 2011).

    The label with the fewest unassigned documents is placed first, so a rare
    label is spread in proportion before common ones fill the splits. A split
    already at its size takes no more documents, which is what keeps the
    sizes exact.
    """
    want = np.array(sizes, dtype=np.float64)
    share = want / want.sum() if want.sum() else want
    docs_with: dict[str, set[int]] = collections.defaultdict(set)
    for doc, label_set in enumerate(label_sets):
        for label in label_set:
            docs_with[label].add(doc)
    names = sorted(docs_with)
    column = {label: index for index, label in enumerate(names)}
    remaining = np.array([len(docs_with[label]) for label in names], float)
    want_label = np.outer(remaining, share)
    assign = np.full(len(label_sets), -1, dtype=np.int64)

    def put(doc: int, split: int) -> None:
        assign[doc] = split
        want[split] -= 1
        for label in label_sets[doc]:
            index = column[label]
            want_label[index, split] -= 1
            remaining[index] -= 1
            docs_with[label].discard(doc)

    def best(desire: npt.NDArray[np.float64]) -> int:
        # Two splits whose sizes differ by one document want every label in
        # slightly different amounts; an exact comparison would hand the
        # larger every tie, and with it every rare label. Under half a
        # document apart is rounding, so the split with more room left wins
        # instead, and room alternates between them.
        splits = np.flatnonzero(desire >= desire.max() - 0.5)
        splits = splits[want[splits] == want[splits].max()]
        return int(rng.choice(splits))

    while (live := np.flatnonzero(remaining > 0)).size:
        rarest = live[remaining[live] == remaining[live].min()]
        index = int(rng.choice(rarest))
        for doc in rng.permutation(sorted(docs_with[names[index]])):
            put(int(doc), best(np.where(want > 0, want_label[index], -np.inf)))

    for doc in np.flatnonzero(assign < 0):
        put(int(doc), best(want))

    return assign


def _repair(
    assign: npt.NDArray[np.int64],
    labels: Sequence[frozenset[str]],
    docs_of: Mapping[str, set[int]],
    form_keys: Mapping[str, frozenset[str]],
    held_docs: set[int],
    rng: np.random.Generator,
) -> npt.NDArray[np.int64]:
    """Move in every entity absent from training that shares a form with it.

    Stratification leaves some rare entity outside training whose surface
    form a training entity also has. One of its documents trades places with
    a training document whose every entity occurs in training at least twice,
    so the trade strands nothing and keeps the split sizes. Such an entity
    always has a document outside `held_docs`: one with none there would
    have been held with its group.
    """
    assign = assign.copy()
    in_training = collections.Counter(
        entity
        for doc in np.flatnonzero(assign == _TRAINING)
        for entity in labels[doc]
    )
    while True:
        training_keys = set().union(
            *(
                form_keys.get(entity, frozenset())
                for entity, count in in_training.items()
                if count
            )
        )
        strays = sorted(
            {
                entity
                for doc in np.flatnonzero(assign != _TRAINING)
                for entity in labels[doc]
                if not in_training[entity]
                and form_keys.get(entity, frozenset()) & training_keys
            }
        )
        if not strays:
            return assign

        incoming = min(docs_of[strays[0]] - held_docs)
        movable = [
            int(doc)
            for doc in np.flatnonzero(assign == _TRAINING)
            if all(in_training[entity] >= 2 for entity in labels[doc])
        ]
        if not movable:
            msg = (
                f"no training document can leave to make room for "
                f"{strays[0]!r}, whose surface form training already has"
            )
            raise RuntimeError(msg)
        outgoing = int(rng.choice(movable))

        assign[outgoing] = assign[incoming]
        assign[incoming] = _TRAINING
        in_training.subtract(labels[outgoing])
        in_training.update(labels[incoming])
