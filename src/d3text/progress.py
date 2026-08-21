"""Progress bars for loaders whose batch count is not known in advance."""

from collections.abc import Iterator, Sized
from typing import Any, cast

from torch.utils.data import DataLoader
from tqdm import tqdm


def batch_progress(
    data: DataLoader,
    desc: str = "Batches",
    position: int = 1,
    leave: bool = False,
) -> Iterator[Any]:
    """Iterate `data` behind a bar measured in documents, not in batches.

    `TokenBudgetBatchSampler` deliberately has no `__len__` — how many batches
    an epoch takes depends on the order the inner sampler draws — so
    `len(loader)` raises, tqdm gets no total, and the bar degrades to a bare
    counter. The *document* count of a split is fixed whatever the batching,
    so the bar counts documents and carries the batch count as a postfix.

    The bar can stop short of its total: a document whose pmid is missing from
    the HDF5 file is dropped by `BrendaDataset._getitems` and never reaches a
    batch.
    """
    try:
        total = len(cast(Sized, data.dataset))
    except TypeError:
        total = None

    with tqdm(
        total=total,
        desc=desc,
        unit="doc",
        dynamic_ncols=True,
        position=position,
        leave=leave,
    ) as bar:
        for batches, batch in enumerate(data, start=1):
            yield batch
            bar.set_postfix(batches=batches, refresh=False)
            bar.update(len(batch) if isinstance(batch, Sized) else 1)
