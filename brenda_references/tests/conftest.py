"""Fixtures shared by the `brenda_references` tests."""

import pytest
from d3types import Strain
from scripts import fix_taxonomy


class FakeStrainInfoAdapter:
    """Stands in for `StrainInfoAdapter`: echoes the models it is given.

    The real adapter resolves designations against the StrainInfo network
    API; these tests only need `retrieve_strain_models` to be a no-op so
    `update_doc_strain` can insert the model it built locally.
    """

    def __enter__(self) -> "FakeStrainInfoAdapter":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def retrieve_strain_models(
        self, strains: dict[int, Strain]
    ) -> dict[int, Strain]:
        return strains


@pytest.fixture
def fake_strain_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep `fix_taxonomy` from reaching the StrainInfo network API."""
    monkeypatch.setattr(
        fix_taxonomy, "StrainInfoAdapter", FakeStrainInfoAdapter
    )
