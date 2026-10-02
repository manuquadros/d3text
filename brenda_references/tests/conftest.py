"""Fixtures shared by the `brenda_references` tests."""

from collections.abc import Collection, Mapping

import pytest
from apiadapters.straininfo import StrainRecord
from scripts import fix_taxonomy


class FakeStrainInfoAdapter:
    """Stands in for `StrainInfoAdapter`: resolves no designation.

    The real adapter resolves designations against the StrainInfo network
    API and returns only the keys it resolved; resolving none makes
    `update_doc_strain` insert the placeholder it builds locally.
    """

    def __enter__(self) -> "FakeStrainInfoAdapter":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def retrieve_strain_models(
        self, designations: Mapping[int, Collection[str]]
    ) -> dict[int, StrainRecord]:
        return {}


@pytest.fixture
def fake_strain_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep `fix_taxonomy` from reaching the StrainInfo network API."""
    monkeypatch.setattr(
        fix_taxonomy, "StrainInfoAdapter", FakeStrainInfoAdapter
    )
