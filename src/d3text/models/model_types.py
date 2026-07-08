from collections.abc import Mapping
from typing import NamedTuple, TypedDict

from jaxtyping import Float, Integer
from torch import Tensor

type BatchedLogits = Float[Tensor, "sequence logits"]


class BatchItem(TypedDict, total=False):
    """One document's inputs as consumed by the model methods.

    Mirrors the dict ``BrendaDataset`` yields. ``doc_id`` (the batch-position
    index) is present only on the list-index DataLoader path, hence
    ``total=False``.
    """

    id: Tensor
    # Per-sequence tensor; its last-dim size counts an item's HDF5 sequences.
    doc_id: Tensor
    sequence: Mapping[str, Tensor]
    entities: Tensor
    classes: Tensor
    relations: list[dict[tuple[str, str], Tensor]]


class IndexedRelation(NamedTuple):
    """Represents a relation triple indexed to a document.

    :param docix: Document identifier
    :param subject: Subject of the triple
    :param object: Object of the triple
    :param label: Identifier of the predicate of the triple, identified as
        label because it is the target of classification in the model.
    """

    docix: int
    subject: str
    object: str
    label: Integer[Tensor, ""]


class RelationIndex(NamedTuple):
    """Specifies the location of the arguments of a relation in a batch.

    sequence - the index of the sequence in the batch
    arg_positions - the index of each argument in the sequence
    arg_predictions - the index of each argument in the entity index
    """

    sequence: int
    arg_positions: tuple[int, int]
    arg_predictions: tuple[int, int]
