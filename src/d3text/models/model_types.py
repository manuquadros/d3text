from collections.abc import Mapping
from typing import NamedTuple, TypedDict

from jaxtyping import Float, Integer
from torch import Tensor

type BatchedLogits = Float[Tensor, "sequence logits"]


class BatchItem(TypedDict, total=False):
    """One document's inputs as consumed by the model methods.

    Every field is that one document's tensor, with no batch dimension: a batch
    is the `Sequence[BatchItem]` `data.collate_documents` builds, and nothing
    here could be stacked anyway since documents differ in chunk count.
    `total=False` because the model methods are also called with hand-built
    items carrying only the fields the method under test reads.
    """

    # 0-dim: the document's pubmed id, or for a document of an external
    # corpus, which has none, the id `encodings_store.external_document_id`
    # mints.
    id: Integer[Tensor, ""]
    # Per-chunk tensor of the document's batch position; its size counts the
    # document's HDF5 sequences, which is what slices the base model's output
    # back into documents.
    doc_id: Tensor
    # ``input_ids`` / ``attention_mask``, each ``[n_chunks, token]``.
    sequence: Mapping[str, Tensor]
    # Multi-hot over the class columns: ``[n_classes]``.
    classes: Tensor
    # The document's relation dicts, as the corpus stores them;
    # `ETEBrendaModel.ground_truth` reads every one.
    relations: list[dict[tuple[str, str], Tensor]]


class IndexedRelation(NamedTuple):
    """A relation triple indexed to a document.

    `docix` identifies the document, `subject` and `object` are the triple's
    two arguments, and `label` is its predicate — named for being the model's
    classification target.
    """

    docix: int
    subject: str
    object: str
    label: Integer[Tensor, ""]


class GroundTruth(NamedTuple):
    """What `BrendaClassificationModel.ground_truth` reads off a batch.

    `ETEBrendaModel.ground_truth` returns the same type: `relations` is `None`
    for the model with no relation head. Both return exactly this type
    instead of two tuple arities, so a caller need not know which of the two
    it holds before unpacking. `NERClassificationModel` returns the bare class
    tensor instead.
    """

    classes: Float[Tensor, "batch classes"]
    relations: list[IndexedRelation] | None = None


type RelationCandidates = tuple[
    dict[str, Tensor], Float[Tensor, "pairs logits"]
]


class BatchLogits(NamedTuple):
    """What `forward` / `get_batch_logits` return on the two BRENDA models.

    Those are `BrendaClassificationModel` and `ETEBrendaModel`;
    `NERClassificationModel` returns the bare class logits instead.

    `relations` mirrors `GroundTruth.relations`: absent for a model with no
    relation head, the pooled candidate-pair metadata and relation logits for
    one that has it.
    """

    classes: BatchedLogits
    relations: RelationCandidates | None = None


class BatchLosses(NamedTuple):
    """What the BRENDA models' `compute_batch_losses` return, per objective.

    `NERClassificationModel` returns its class loss bare instead. `relation`
    is `None` without a relation head and `token` without a configured label
    store. Both are trailing so a caller reading only the tail still gets the
    token loss regardless of which model produced the tuple.
    """

    class_: Float[Tensor, ""]
    relation: Float[Tensor, ""] | None = None
    token: Float[Tensor, ""] | None = None
