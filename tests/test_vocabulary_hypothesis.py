"""Property-based tests for `Vocabulary`'s payload round trip.

`test_vocabulary.py::test_the_payload_round_trips_the_order` pins this at one
hand-built vocabulary. The property it is standing in for — any vocabulary
that can be constructed survives `to_payload`/`from_payload` unchanged, entity
order and class membership included — is generated here instead, over
vocabularies with varying entity counts, class counts, entities shared across
classes, and classes with no members at all.

Marked `slow`: `@given` draws many examples per test.
"""

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from d3text.vocabulary import Vocabulary

pytestmark = pytest.mark.slow

_ENTITY_ID = st.text(
    alphabet=st.characters(min_codepoint=97, max_codepoint=122),
    min_size=1,
    max_size=6,
)
_CLASS_NAME = st.text(
    alphabet=st.characters(min_codepoint=65, max_codepoint=90),
    min_size=1,
    max_size=6,
)


@st.composite
def _vocabulary(draw: st.DrawFn) -> Vocabulary:
    entities = tuple(
        draw(st.lists(_ENTITY_ID, min_size=0, max_size=12, unique=True))
    )
    class_names = draw(
        st.lists(_CLASS_NAME, min_size=0, max_size=4, unique=True)
    )

    # Each class draws an arbitrary subset of `entities` -- possibly none,
    # possibly all of them, possibly overlapping with another class's subset,
    # which is what exercises `class_matrix`'s "an entity in two classes
    # lights both columns" case across generated instances rather than the
    # one hand-built one in test_vocabulary.py.
    members = (
        st.lists(st.sampled_from(entities), max_size=len(entities), unique=True)
        if entities
        else st.just([])
    )
    class_map = {name: tuple(draw(members)) for name in class_names}

    return Vocabulary(entities=entities, class_map=class_map)


@given(vocabulary=_vocabulary())
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_any_constructible_vocabulary_round_trips_through_its_payload(
    vocabulary,
):
    restored = Vocabulary.from_payload(vocabulary.to_payload())

    assert restored == vocabulary
    assert restored.entities == vocabulary.entities
    assert restored.class_map == vocabulary.class_map


@given(vocabulary=_vocabulary())
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_the_payload_is_always_plain_builtins(vocabulary):
    """`torch.load`'s `weights_only=True` default is what a checkpoint's
    vocabulary has to survive -- tensors and builtins only, no matter how the
    vocabulary was shaped."""
    payload = vocabulary.to_payload()

    assert isinstance(payload, dict)
    assert isinstance(payload["entities"], list)
    assert all(isinstance(entity_id, str) for entity_id in payload["entities"])
    assert isinstance(payload["class_map"], dict)
    for name, entity_ids in payload["class_map"].items():
        assert isinstance(name, str)
        assert isinstance(entity_ids, list)
        assert all(isinstance(entity_id, str) for entity_id in entity_ids)


@given(vocabulary=_vocabulary())
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_class_matrix_shape_always_matches_entities_and_classes(vocabulary):
    """The geometry `class_matrix` promises the class head: one row per
    entity, one column per class, regardless of how sparse or how densely
    overlapping the membership is."""
    matrix = vocabulary.class_matrix()

    assert matrix.shape == (len(vocabulary.entities), len(vocabulary.class_map))


@given(vocabulary=_vocabulary())
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_every_classified_entity_owns_a_column_in_its_class(vocabulary):
    """The invariant `validate` enforces at construction time, checked here
    against the *matrix* rather than against `validate` succeeding, so a
    passing test also proves `class_matrix` and `class_map` agree."""
    matrix = vocabulary.class_matrix()
    index = vocabulary.entity_index

    for column, entity_ids in enumerate(vocabulary.class_map.values()):
        for entity_id in entity_ids:
            assert matrix[index[entity_id], column] == 1.0
