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


class GroundTruth(NamedTuple):
    """What `Model.ground_truth` reads off a batch.

    One shape for every model that carries an entity and a class head:
    `relations` is `None` for a model with no relation head
    (`BrendaClassificationModel`) and populated for one that has it
    (`ETEBrendaModel`). Composition rather than inheritance means both
    return exactly this type instead of two different tuple arities, so a
    caller no longer has to know which model it holds before it can unpack
    the result.
    """

    entities: Float[Tensor, "batch entities"]
    classes: Float[Tensor, "batch classes"]
    relations: list[IndexedRelation] | None = None


type RelationCandidates = tuple[
    dict[str, Tensor], Float[Tensor, "pairs logits"]
]


class BatchLogits(NamedTuple):
    """What `Model.forward` / `Model.get_batch_logits` return.

    `relations` mirrors `GroundTruth.relations`: absent for a two-head model,
    the pooled candidate-pair metadata and relation logits for a three-head
    one.
    """

    entities: BatchedLogits
    classes: BatchedLogits
    relations: RelationCandidates | None = None


class BatchLosses(NamedTuple):
    """What `Model.compute_batch_losses` returns, one field per objective.

    `relation` is `None` for a model with no relation head; `token` is
    `None` for a model with no configured token-label store. Both are
    trailing so a caller reading only the tail (`*_, token = ...`) still
    gets the token loss regardless of which model produced the tuple.
    """

    entity: Float[Tensor, ""]
    class_: Float[Tensor, ""]
    relation: Float[Tensor, ""] | None = None
    token: Float[Tensor, ""] | None = None


class RelationIndex(NamedTuple):
    """Specifies the location of the arguments of a relation in a batch.

    sequence - the index of the sequence in the batch
    arg_positions - the index of each argument in the sequence
    arg_predictions - the index of each argument in the entity index
    """

    sequence: int
    arg_positions: tuple[int, int]
    arg_predictions: tuple[int, int]
