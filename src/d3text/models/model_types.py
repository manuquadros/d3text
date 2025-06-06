from typing import NamedTuple

from jaxtyping import Float
from torch import Tensor

type BatchedLogits = Float[Tensor, "sequence logits"]


class RelationIndex(NamedTuple):
    """Specifies the location of the arguments of a relation in a batch.

    sequence - the index of the sequence in the batch
    args - the index of each argument in the sequence
    """

    sequence: int
    args: tuple[int, int]
