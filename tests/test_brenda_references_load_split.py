"""`load_split` must refuse a negative `limit`, not truncate to nothing.

`DataFrame.truncate(after=limit - 1)` on a `RangeIndex` keeps zero rows for
any negative `after`, so a negative `limit` used to build a silently empty
split instead of raising. The check has to fire before the split's CSV is
even opened, so this needs no on-disk fixture.
"""

import pytest
from brenda_references.brenda_references import load_split


def test_a_negative_limit_is_refused_rather_than_emptying_the_split():
    with pytest.raises(ValueError, match="non-negative"):
        load_split("training", limit=-1)
