"""Reading the precomputed token targets in the geometry the model scores.

`precompute-token-labels` writes per-window codes shaped like the stored
encodings; the model scores the *aggregated* document — the 512-token windows
merged along their 20-token overlaps by `aggregate_embeddings`. The reader
here carries the codes across that same merge, and it does so by running the
codes through `aggregate_embeddings` itself rather than by restating its
overlap arithmetic: the targets exist to sit element-for-element beside the
embeddings, so a second, drifting copy of the selection rule would be a hole
in exactly the alignment being provided (`document_token_count` makes the same
argument for the same function). The int8 codes ride through the float pass
losslessly — every value, `IGNORE_INDEX` included, is a small integer float32
represents exactly.

The label space is verified at open, not assumed: a store written under a
permuted schema holds codes whose integers mean different types, and nothing
in the arrays says so. `load_token_labels` returns codes without checking
their meaning, so the first reader — this one — refuses at the door instead:
the store's recorded space must equal the space the tagger head was sized to,
or nothing is read at all.
"""

import logging
import os

import h5py
import numpy
import torch
from jaxtyping import Int64
from torch import Tensor

from d3text import token_labels
from d3text.utils import aggregate_embeddings

logger = logging.getLogger(__name__)


class TokenLabelReader:
    """One run's handle on a token-label store, space-checked once at open."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        space: token_labels.LabelSpace = token_labels.BRENDA_LABELS,
    ) -> None:
        self._store = h5py.File(path, "r")
        recorded = token_labels.read_label_space(self._store)
        if recorded != space:
            self._store.close()
            msg = (
                f"{os.fspath(path)} records the label space {recorded}, but "
                f"this model's tagger head is sized to {space}; its codes "
                "would be scored against the wrong columns — regenerate the "
                "store, or build the model over the space it records"
            )
            raise ValueError(msg)
        self.space = space

    def close(self) -> None:
        self._store.close()

    def document_codes(
        self,
        pubmed_id: int | str,
        window_attention_mask: object,
    ) -> Int64[Tensor, " token"] | None:
        """One document's targets on the aggregated token axis, or None.

        None means the store holds no targets for `pubmed_id` — the caller's
        to skip or to mask, since only it knows whether that is a truncated
        split or a stale store.

        `window_attention_mask` is the document's own mask as the batch item
        carries it — any leading collation axes are flattened away, exactly
        as `batch_input_tensors` flattens the encodings they mask.

        :raises ValueError: if the stored codes and the mask disagree in
            window geometry, which means the store was built against different
            encodings and its every row would land on the wrong token.
        """
        key = str(pubmed_id)
        try:
            labels = token_labels.load_token_labels(self._store, key)
        except KeyError:
            return None

        mask = numpy.asarray(window_attention_mask)
        mask = mask.reshape(-1, mask.shape[-1]).astype(numpy.int64)
        if labels.codes.shape != mask.shape:
            msg = (
                f"document {key} stores codes of shape {labels.codes.shape} "
                f"against encodings of shape {mask.shape}; the label store "
                "was built from different encodings — regenerate it"
            )
            raise ValueError(msg)

        aggregated = aggregate_embeddings(
            torch.as_tensor(labels.codes, dtype=torch.float32).unsqueeze(-1),
            torch.as_tensor(mask),
        )
        return aggregated.squeeze(-1).to(torch.int64)


def padded_targets(
    rows: list[Int64[Tensor, " token"]],
    length: int,
    ignore_index: int = token_labels.IGNORE_INDEX,
) -> Int64[Tensor, "document token"]:
    """Stack per-document target rows to `length`, padding with the mask.

    Padding is `ignore_index` rather than a class: the padded positions have
    no token under them, and a pad contributing to the loss would be the
    divisor bug `masked_token_cross_entropy` exists to avoid.
    """
    padded = torch.full((len(rows), length), ignore_index, dtype=torch.int64)
    for row_index, row in enumerate(rows):
        padded[row_index, : row.shape[0]] = row
    return padded


def document_lengths(attention_mask: Tensor) -> list[int]:
    """Unpadded token count per document of a batch-level attention mask."""
    return [int(count) for count in attention_mask.sum(dim=1).tolist()]


__all__ = ["TokenLabelReader", "document_lengths", "padded_targets"]
