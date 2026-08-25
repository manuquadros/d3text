"""`batch_progress`: a bar for loaders that cannot report a batch count."""

import logging
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


class Unavailable(torch.utils.data.Dataset):
    """`missing` rows are absent from the encodings, as in `_getitems`.

    A batch drawn entirely from them collates to `[]`, which is what the
    epoch and evaluation loops cannot consume.
    """

    def __init__(self, size: int, missing: set[int]) -> None:
        self.size = size
        self.missing = missing
        self.sequence_lengths = {ix: 1 for ix in range(size)}

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, ix: int | list[int]) -> Any:
        if isinstance(ix, list):
            return [{"id": i} for i in ix if i not in self.missing]
        return {"id": ix}


class Unsized(torch.utils.data.Dataset):
    """A split that declares no `__len__`, so it has no stateable size."""

    def __init__(self, documents: Unavailable) -> None:
        self.documents = documents

    def __getitem__(self, ix: int | list[int]) -> Any:
        return self.documents[ix]


def test_never_yields_a_batch_whose_documents_were_all_missing(
    bar: type[FakeBar],
) -> None:
    """`evaluate` loads with `batch_size=1`, so one missing pmid is one
    empty batch, and `ground_truth`'s `torch.concat(())` raises on it."""
    loader = get_batch_loader(
        Unavailable(4, missing={1, 2}),
        batch_size=1,
        sampler=torch.utils.data.SequentialSampler(range(4)),
    )

    assert [] in list(loader)  # the loader still produces it

    batches = list(progress.batch_progress(loader))

    assert all(batch for batch in batches)
    assert [int(doc["id"]) for batch in batches for doc in batch] == [0, 3]
    assert sum(bar.instances[0].updates) == 2


def test_a_partly_missing_batch_still_yields_its_survivors(
    bar: type[FakeBar],
) -> None:
    """Only an *entirely* missing batch is dropped; the per-row skip in
    `_getitems` is unchanged."""
    loader = get_batch_loader(
        Unavailable(4, missing={1}),
        batch_size=4,
        sampler=torch.utils.data.SequentialSampler(range(4)),
    )

    batches = list(progress.batch_progress(loader))

    assert [int(item["id"]) for batch in batches for item in batch] == [
        0,
        2,
        3,
    ]


def warnings_of(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING
        and record.name == "d3text.progress"
    ]


def test_reports_the_skipped_batches_once_per_pass(
    bar: type[FakeBar], caplog: pytest.LogCaptureFixture
) -> None:
    """A shrunk denominator has to be stated — but once, not per batch: the
    condition is a stale encodings file, and every batch reports it."""
    loader = get_batch_loader(
        Unavailable(4, missing={1, 2}),
        batch_size=1,
        sampler=torch.utils.data.SequentialSampler(range(4)),
    )

    with caplog.at_level(logging.WARNING, logger="d3text.progress"):
        list(progress.batch_progress(loader))

    messages = warnings_of(caplog)

    assert len(messages) == 2
    assert "2 of 4 documents" in messages[0]
    assert "2 batch(es)" in messages[1]


def test_reports_a_shortfall_with_no_batch_lost_entirely(
    bar: type[FakeBar], caplog: pytest.LogCaptureFixture
) -> None:
    """The common shape of a stale encodings file, and the one that used to
    pass unremarked: `_getitems` drops its missing rows one at a time, so a
    batch shrinks rather than emptying and no batch is ever skipped."""
    loader = get_batch_loader(
        Unavailable(4, missing={1}),
        batch_size=4,
        sampler=torch.utils.data.SequentialSampler(range(4)),
    )

    with caplog.at_level(logging.WARNING, logger="d3text.progress"):
        batches = list(progress.batch_progress(loader))

    assert all(batch for batch in batches)

    assert warnings_of(caplog) == [
        "1 of 4 documents in this split never reached the model, so nothing "
        "in them was trained on or scored."
    ]


def test_reports_the_skipped_batches_of_a_split_of_unknown_size(
    bar: type[FakeBar], caplog: pytest.LogCaptureFixture
) -> None:
    """No `__len__` means no denominator, so only the batches can be named."""
    loader = get_batch_loader(
        Unsized(Unavailable(4, missing={1, 2})),
        batch_size=1,
        sampler=torch.utils.data.SequentialSampler(range(4)),
    )

    assert progress.split_documents(loader) is None

    with caplog.at_level(logging.WARNING, logger="d3text.progress"):
        list(progress.batch_progress(loader))

    assert warnings_of(caplog) == [
        "Skipped 2 batch(es) in which every document was missing from the "
        "encodings file."
    ]


def test_says_nothing_when_every_document_arrived(
    bar: type[FakeBar], caplog: pytest.LogCaptureFixture
) -> None:
    """A healthy split must not gain a warning."""
    loader = get_batch_loader(Documents(), batch_size=4)

    with caplog.at_level(logging.WARNING, logger="d3text.progress"):
        list(progress.batch_progress(loader))

    assert [
        record for record in caplog.records if record.name == "d3text.progress"
    ] == []


def test_a_missing_pmid_alone_in_its_batch_reaches_no_loop(tiny_brenda) -> None:
    """The end-to-end shape: a split frame naming a pmid the HDF5 does not
    hold, drawn the way `evaluate` draws it.

    The assertion is the crash site itself — every yielded batch must survive
    the `torch.concat` in `ground_truth`.
    """
    loader = get_batch_loader(
        tiny_brenda.full,
        batch_size=1,
        sampler=torch.utils.data.SequentialSampler(
            range(len(tiny_brenda.full))
        ),
    )

    ids = []
    for batch in progress.batch_progress(loader):
        torch.concat(tuple(torch.as_tensor(doc["entities"]) for doc in batch))
        ids.extend(doc["id"] for doc in batch)

    assert [int(pmid) for pmid in ids] == [10, 20, 30]
