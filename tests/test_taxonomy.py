"""Regression test for the shared NCBI merged-taxid forwarding helper."""

import pandas as pd
import pytest
from taxonomy.ncbitax import ncbitax

from d3text.taxonomy import merged_taxids


def test_merged_taxids_maps_a_retired_id_to_its_current_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller importing `d3text.taxonomy` gets `merged.dmp`'s table.

    Before this helper was extracted from the organism bridge script, no
    module outside it could reach this mapping without reimplementing it.
    """
    fake = pd.DataFrame({"old_tax_id": ["2254"], "new_tax_id": ["2246"]})
    monkeypatch.setattr(ncbitax, "load_df", lambda table: fake)

    assert merged_taxids() == {2254: 2246}
