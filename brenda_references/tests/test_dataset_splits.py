"""Invariants of `entity_holdout_splits`."""

import pytest
from brenda_references.sampling import SPLITS, entity_holdout_splits
from hypothesis import assume, given
from hypothesis import strategies as st


def _corpus(
    common: int = 10, isolated: int = 40, lookalikes: int = 0, docs: int = 200
) -> tuple[dict[int, frozenset[str]], dict[str, frozenset[str]]]:
    """A pool of common entities plus rare ones, one document each.

    `isolated` rare entities have a surface form of their own; `lookalikes`
    have the form of common entity 0, like a strain registered under its
    species' name.
    """
    forms = {f"c{i}": frozenset({f"common {i}"}) for i in range(common)}
    documents = {
        pubmed_id: frozenset({f"c{pubmed_id % common}"})
        for pubmed_id in range(docs)
    }
    for i in range(isolated):
        forms[f"r{i}"] = frozenset({f"rare {i}"})
        documents[i] |= {f"r{i}"}
    for i in range(lookalikes):
        forms[f"l{i}"] = frozenset({"common 0"})
        documents[isolated + i] |= {f"l{i}"}
    return documents, forms


def _unseen(
    splits: dict[str, list[int]], documents: dict[int, frozenset[str]]
) -> dict[str, set[str]]:
    training = set().union(*(documents[doc] for doc in splits["training"]))
    return {
        name: set().union(*(documents[doc] for doc in splits[name])) - training
        for name in SPLITS[1:]
    }


def _assert_form_novel(
    splits: dict[str, list[int]],
    documents: dict[int, frozenset[str]],
    forms: dict[str, frozenset[str]],
) -> None:
    training = set().union(*(documents[doc] for doc in splits["training"]))
    training_keys = set().union(*(forms.get(e, set()) for e in training))
    for name, entities in _unseen(splits, documents).items():
        for entity in entities:
            assert not forms.get(entity, set()) & training_keys, (name, entity)


def _assert_partition(
    splits: dict[str, list[int]], documents: dict[int, frozenset[str]]
) -> None:
    drawn = [doc for name in SPLITS for doc in splits[name]]
    assert sorted(drawn) == sorted(documents)
    assert len(splits["validation"]) == len(splits["test"])


def test_holds_isolated_rare_entities_out_of_training() -> None:
    """Held-out entities land in both evaluation splits and never in training.

    Validation and test are filled the same way, so each must carry some;
    one carrying all would measure novelty the other does not.
    """
    documents, forms = _corpus()

    splits = entity_holdout_splits(documents, forms, held_share=0.5)

    unseen = _unseen(splits, documents)
    assert unseen["validation"]
    assert unseen["test"]
    assert all(entity.startswith("r") for e in unseen.values() for entity in e)
    _assert_form_novel(splits, documents, forms)


def test_validation_and_test_hold_as_many_unseen_entities() -> None:
    """Held-out papers dense in rare entities go to both evaluation splits.

    An odd number of held-out papers makes validation's share one paper
    larger than test's. Were that enough to win every tie on a rare entity,
    the papers naming the most rare entities, being reached first, would
    fill validation and leave test the sparse ones: the two would then
    measure novelty at different rates. Summed over seeds so that one draw's
    luck cannot pass or fail it.
    """
    documents, forms = _corpus(isolated=0, docs=300)
    for doc in range(90):
        dense = 6 if doc < 30 else 1
        rare = {f"r{doc}.{i}" for i in range(dense)}
        documents[doc] |= rare
        forms.update({entity: frozenset({entity}) for entity in rare})

    totals = dict.fromkeys(SPLITS[1:], 0)
    for seed in range(20):
        splits = entity_holdout_splits(
            documents, forms, held_share=0.3, max_held_documents=1, seed=seed
        )
        for name, unseen in _unseen(splits, documents).items():
            totals[name] += len(unseen)

    assert abs(totals["validation"] - totals["test"]) <= 0.1 * max(
        totals.values()
    ), totals


def test_an_entity_sharing_a_training_form_is_trained_on() -> None:
    """A rare entity whose form training already has is moved into training.

    Stratification alone sends a one-document entity to whichever split
    wants the largest share, here validation or test; there it would count
    as unseen although the tagger was trained on its surface form.
    """
    documents, forms = _corpus(isolated=0, lookalikes=30)

    splits = entity_holdout_splits(
        documents, forms, evaluation_share=0.4, held_share=0.0
    )

    assert not any(_unseen(splits, documents).values())
    _assert_partition(splits, documents)


def test_a_stranded_lookalike_blocks_its_holdout() -> None:
    """An isolated entity sharing its only document with a lookalike stays.

    Holding the isolated entity out would put the lookalike's only document
    outside training too, and the lookalike's form is common entity 0's, so
    training would already know it. The group cannot be held out.
    """
    documents, forms = _corpus(isolated=0)
    forms["r"] = frozenset({"rare"})
    forms["l"] = frozenset({"common 0"})
    documents[0] |= {"r", "l"}

    splits = entity_holdout_splits(documents, forms, held_share=1.0)

    assert 0 in splits["training"]
    _assert_form_novel(splits, documents, forms)


def test_one_seed_gives_one_split() -> None:
    documents, forms = _corpus(lookalikes=20)

    assert entity_holdout_splits(
        documents, forms, seed=3
    ) == entity_holdout_splits(dict(reversed(documents.items())), forms, seed=3)


@pytest.mark.parametrize(
    ("evaluation_share", "held_share", "max_held_documents"),
    [(0.6, 0.3, 3), (-0.1, 0.3, 3), (0.15, 1.5, 3), (0.15, 0.3, 0)],
)
def test_rejects_out_of_range_arguments(
    evaluation_share: float, held_share: float, max_held_documents: int
) -> None:
    documents, forms = _corpus()

    with pytest.raises(ValueError):
        entity_holdout_splits(
            documents,
            forms,
            evaluation_share=evaluation_share,
            held_share=held_share,
            max_held_documents=max_held_documents,
        )


@given(
    st.lists(
        st.frozensets(st.integers(0, 29), max_size=4), min_size=20, max_size=80
    ),
    st.dictionaries(
        st.integers(0, 29), st.frozensets(st.integers(0, 12), max_size=2)
    ),
    st.floats(0, 0.5),
    st.floats(0, 1),
    st.integers(0, 2**16),
)
def test_every_split_is_a_partition_and_form_novel(
    entity_sets: list[frozenset[int]],
    keys: dict[int, frozenset[int]],
    evaluation_share: float,
    held_share: float,
    seed: int,
) -> None:
    """The two invariants hold over arbitrary pools and form collisions.

    A pool can make them unsatisfiable -- every training document the only
    one naming some entity, so nothing can leave to make room -- and the
    splitter says so rather than returning a split that breaks them.
    """
    documents = {
        doc: frozenset(f"e{entity}" for entity in entities)
        for doc, entities in enumerate(entity_sets)
    }
    forms = {
        f"e{entity}": frozenset(f"k{key}" for key in entity_keys)
        for entity, entity_keys in keys.items()
    }
    try:
        splits = entity_holdout_splits(
            documents,
            forms,
            evaluation_share=evaluation_share,
            held_share=held_share,
            seed=seed,
        )
    except RuntimeError:
        assume(False)
        raise

    _assert_partition(splits, documents)
    assert len(splits["test"]) == min(
        round(len(documents) * evaluation_share), len(documents) // 2
    )
    _assert_form_novel(splits, documents, forms)
