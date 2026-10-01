"""A dropped model is freed by refcount, not kept by beartype's memo cache.

`beartype_this_package()` decorates every `def` including closures, and
beartype's memoised blacklist check keeps each decorated function for the
life of the process. A per-instance closure over `self` therefore pinned
the whole model; `hidden` must be an ordinary method.
"""

import weakref

import pytest

from d3text.models.config import ModelConfig
from d3text.models.entity_linking import BrendaClassificationModel
from d3text.schema import EntityType, Schema

pytestmark = pytest.mark.slow


@pytest.fixture(autouse=True)
def _offline(patch_base_model):
    """Inject the tiny random BERT instead of downloading one."""


@pytest.mark.parametrize("checkpointing", [False, True])
def test_dropped_model_is_freed_without_gc(checkpointing):
    schema = Schema(entity_types=(EntityType(name="x", prefix="e0"),))
    model = BrendaClassificationModel(
        schema=schema,
        config=ModelConfig(
            model_class="BrendaClassificationModel",
            base_model="prajjwal1/bert-mini",
            hidden_layers=[8],
            gradient_checkpointing=checkpointing,
        ),
        device="cpu",
    )
    ref = weakref.ref(model)
    del model
    assert ref() is None
