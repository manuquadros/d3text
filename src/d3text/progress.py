"""Progress bars for loaders whose batch count is not known in advance."""

import logging
from collections.abc import Iterator, Sized
from typing import Any, cast

from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def split_documents(data: DataLoader) -> int | None:
    """How many documents `data`'s split holds, or None if it cannot say.

    `TokenBudgetBatchSampler` declares no `__len__`, so this asks the *dataset*
    rather than the loader. It is the denominator both the progress bar and the
    coverage metrics measure a pass against, and it is defined once here so the
    bar's shortfall warning and the logged counts cannot disagree.
    """
    try:
        return len(cast(Sized, data.dataset))
    except TypeError:
        return None


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
    batch. When *every* document a batch was drawn for is missing, the batch
    collates to `[]`; that batch is dropped here rather than yielded, because
    each of the six epoch and evaluation loops would otherwise hand it to
    `ground_truth`, whose `torch.concat(())` raises. `evaluate` loads with
    `batch_size=1`, so there one missing pmid is one empty batch.

    Dropping it is a skip, not a raise: a stale encodings file is exactly the
    condition that produces this, and it must not cost a multi-hour run its
    remaining hours. It is also not silent — the split's documents did not all
    reach the model, so the shortfall is logged once when the pass ends,
    instead of once per batch or not at all.
    """
    total = split_documents(data)

    delivered_batches = 0
    delivered_documents = 0
    empty_batches = 0

    with tqdm(
        total=total,
        desc=desc,
        unit="doc",
        dynamic_ncols=True,
        position=position,
        leave=leave,
    ) as bar:
        for batch in data:
            size = len(batch) if isinstance(batch, Sized) else 1
            if size == 0:
                empty_batches += 1
                continue

            yield batch

            delivered_batches += 1
            delivered_documents += size
            bar.set_postfix(batches=delivered_batches, refresh=False)
            bar.update(size)

    if empty_batches:
        shortfall = (
            ""
            if total is None
            else " %d of %d documents in this split never reached the model"
            % (total - delivered_documents, total)
        )
        logger.warning(
            "Skipped %d batch(es) in which every document was missing from "
            "the encodings file, so nothing in them was trained on or "
            "scored;%s.",
            empty_batches,
            shortfall or " the split's document count is unknown",
        )
