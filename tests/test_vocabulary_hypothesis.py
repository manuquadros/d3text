"""Property-based tests for `Vocabulary`'s payload round trip.

`test_vocabulary.py` pins this at one hand-built vocabulary. The property — any
constructible vocabulary survives `to_payload`/`from_payload` unchanged, class
order and membership included — is generated here instead. Marked `slow`.
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
    # which is what exercises the "an entity in two classes" case across
    # generated instances rather than the one hand-built one in
    # test_vocabulary.py.
    members = (
        st.lists(st.sampled_from(entities), max_size=len(entities), unique=True)
        if entities
        else st.just([])
    )
    class_map = {name: tuple(draw(members)) for name in class_names}

    return Vocabulary(class_map=class_map)


@given(vocabulary=_vocabulary())
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_any_constructible_vocabulary_round_trips_through_its_payload(
    vocabulary,
):
    restored = Vocabulary.from_payload(vocabulary.to_payload())

    assert restored == vocabulary
    assert restored.class_map == vocabulary.class_map
    assert restored.class_names == vocabulary.class_names


@given(vocabulary=_vocabulary())
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_the_payload_is_always_plain_builtins(vocabulary):
    """`torch.load`'s `weights_only=True` default is what a checkpoint's
    vocabulary has to survive -- tensors and builtins only, no matter how the
    vocabulary was shaped."""
    payload = vocabulary.to_payload()

    assert isinstance(payload, dict)
    assert isinstance(payload["class_map"], dict)
    for name, entity_ids in payload["class_map"].items():
        assert isinstance(name, str)
        assert isinstance(entity_ids, list)
        assert all(isinstance(entity_id, str) for entity_id in entity_ids)


@given(vocabulary=_vocabulary())
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_entity_ids_is_exactly_the_union_of_the_class_members(vocabulary):
    """What `evaluate` hands the detection accumulator as the training
    split's vocabulary, however sparse or overlapping the membership is."""
    expected = set()
    for entity_ids in vocabulary.class_map.values():
        expected.update(entity_ids)

    assert vocabulary.entity_ids == expected
