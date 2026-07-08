"""Shared test fixtures.

The `stub` factory is what makes the model methods in models.py unit-testable
without constructing a full model (no base-model download, no GPU): it builds a
bare instance carrying only the handful of attributes the method under test
reads.
"""

import pytest


@pytest.fixture
def stub():
    """Return a factory that builds a bare `cls` instance with `attrs` set.

    Bypasses ``__init__`` (via ``cls.__new__``) and ``nn.Module.__setattr__``
    (via ``object.__setattr__``), so tensors, sub-modules, and plain values can
    be attached directly. Methods that aren't overridden resolve normally off
    the class, so a method can call its real collaborators (e.g.
    ``self._pool_logsumexp``) while reading only the stubbed attributes.
    """

    def _make(cls, **attrs):
        obj = cls.__new__(cls)
        for key, value in attrs.items():
            object.__setattr__(obj, key, value)
        return obj

    return _make
