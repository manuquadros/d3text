"""`batch_progress`: a bar for loaders that cannot report a batch count."""

from typing import Any

import pytest
import torch
from d3text import progress
from d3text.data.data import get_batch_loader


class FakeBar:
    """Records what a bar was told, instead of drawing one."""

    instances: list["FakeBar"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.updates: list[int] = []
        self.postfixes: list[dict[str, Any]] = []
        FakeBar.instances.append(self)

    def __enter__(self) -> "FakeBar":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def update(self, n: int) -> None:
        self.updates.append(n)

    def set_postfix(self, **kwargs: Any) -> None:
        kwargs.pop("refresh", None)
        self.postfixes.append(kwargs)


class Documents(torch.utils.data.Dataset):
    """Chunk counts chosen so the budget cannot split them evenly."""

    sequence_lengths = {ix: 1 + ix % 4 for ix in range(10)}

    def __len__(self) -> int:
        return 10

    def __getitem__(self, ix: int | list[int]) -> Any:
        if isinstance(ix, list):
            return [{"id": i} for i in ix]
        return {"id": ix}


@pytest.fixture
def bar(monkeypatch: pytest.MonkeyPatch) -> type[FakeBar]:
    FakeBar.instances = []
    monkeypatch.setattr(progress, "tqdm", FakeBar)
    return FakeBar


def test_totals_the_documents_of_a_loader_with_no_len(
    bar: type[FakeBar],
) -> None:
    """The token-budget sampler has no `__len__`; the dataset does."""
    loader = get_batch_loader(Documents(), batch_size=4, max_chunks=6)

    with pytest.raises(TypeError):
        len(loader)

    list(progress.batch_progress(loader))

    assert bar.instances[0].kwargs["total"] == 10


def test_yields_every_batch_and_advances_by_document(
    bar: type[FakeBar],
) -> None:
    loader = get_batch_loader(Documents(), batch_size=4, max_chunks=6)

    batches = list(progress.batch_progress(loader))
    drawn = bar.instances[0]

    assert drawn.updates == [len(batch) for batch in batches]
    assert sum(drawn.updates) == 10
    assert drawn.postfixes[-1] == {"batches": len(batches)}


def test_keeps_the_total_when_the_loader_does_report_a_length(
    bar: type[FakeBar],
) -> None:
    """The fixed-batch-size path is unchanged: still one bar over documents."""
    loader = get_batch_loader(Documents(), batch_size=4)

    assert len(loader) == 3
    list(progress.batch_progress(loader))

    assert bar.instances[0].kwargs["total"] == 10
    assert sum(bar.instances[0].updates) == 10
