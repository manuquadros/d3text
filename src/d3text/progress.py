"""Progress bars for loaders whose batch count is not known in advance."""

import logging
from collections.abc import Iterator, Sized
from typing import Any, cast

from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def split_documents(data: DataLoader) -> int | None:
    """How many documents `data`'s split holds, or None if it cannot say.

    Asks the *dataset* rather than the loader, since `TokenBudgetBatchSampler`
    declares no `__len__`, and is defined once so the bar's shortfall warning
    and the logged coverage counts cannot disagree.

    :param data: the loader to measure.
    :return: the split's document count, or None.
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

    The batch count of an epoch is not known in advance, but a split's document
    count is fixed whatever the batching. A batch whose every document was
    missing from the encodings collates to `[]` and is dropped rather than
    yielded, since `ground_truth`'s `torch.concat(())` would raise on it — a
    skip, because a stale encodings file must not cost a multi-hour run its
    remaining hours. The shortfall and the dropped batches are counted
    independently and reported separately at the end of the pass.

    :param data: the loader to iterate.
    :param desc: the bar's label.
    :param position: the bar's row, below the epoch bar.
    :param leave: whether the bar survives the pass.
    :return: the non-empty batches.
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

    if total is not None and delivered_documents < total:
        logger.warning(
            "%d of %d documents in this split never reached the model, so "
            "nothing in them was trained on or scored.",
            total - delivered_documents,
            total,
        )

    if empty_batches:
        logger.warning(
            "Skipped %d batch(es) in which every document was missing from "
            "the encodings file.",
            empty_batches,
        )
