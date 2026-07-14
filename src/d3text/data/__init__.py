# `brenda_dataset` is deliberately absent: it lives in `d3text.datasets.brenda`,
# which imports this package. Re-exporting it here would close that cycle, and
# the loader is corpus-specific besides — this package is not.
from .data import (
    DatasetConfig,
    EntityRelationDataset,
    compute_frequencies,
    get_batch_loader,
    index_tensor,
    multi_hot_encode_series,
)

__all__ = [
    "DatasetConfig",
    "EntityRelationDataset",
    "compute_frequencies",
    "get_batch_loader",
    "index_tensor",
    "multi_hot_encode_series",
]
