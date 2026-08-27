from collections.abc import Mapping
from typing import NamedTuple, TypedDict

from jaxtyping import Float, Integer
from torch import Tensor

type BatchedLogits = Float[Tensor, "sequence logits"]


class BatchItem(TypedDict, total=False):
    """One document's inputs as consumed by the model methods.

    Every field is that **one document's** tensor, with no batch dimension: a
    batch is the ``Sequence[BatchItem]`` that ``data.collate_documents`` builds,
    not a stack. Nothing here could be stacked anyway — documents differ in how
    many chunks they hold — so a model wanting a ``[batch, …]`` target builds it
    itself out of the per-document rows below.

    ``total=False`` because the model methods are also called with hand-built
    items carrying only the fields the method under test reads.
    """

    # 0-dim: the document's pmid.
    id: Tensor
    # Per-chunk tensor of the document's batch position; its size counts the
    # document's HDF5 sequences, which is what slices the base model's output
    # back into documents.
    doc_id: Tensor
    # ``input_ids`` / ``attention_mask``, each ``[n_chunks, token]``.
    sequence: Mapping[str, Tensor]
    # Multi-hot over the entity index / the class columns: ``[n_labels]``.
    entities: Tensor
    classes: Tensor
    # The corpus stores a document's relation dict wrapped in a one-element
    # list; the models read ``relations[0]``.
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
