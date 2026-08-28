"""Pure unit tests for `d3text.models.ner.NERClassificationModel`.

Every test here runs on CPU with tiny synthetic tensors and no data, network,
or GPU. Methods are exercised through the `stub` fixture (see
`tests/conftest.py`), which supplies only the attributes each method reads.
"""

import torch

from d3text.models.ner import NERClassificationModel


# --------------------------------------------------------------------------- #
# NERClassificationModel.ground_truth (batch dimension)                        #
# --------------------------------------------------------------------------- #
def test_ner_ground_truth_keeps_a_batch_dimension_across_documents(stub):
    m = stub(NERClassificationModel, device="cpu")
    batch = [
        {"classes": torch.tensor([1.0, 0.0])},
        {"classes": torch.tensor([0.0, 1.0])},
    ]
    class_targets = m.ground_truth(batch)

    # `torch.concat` would flatten these into a 1-D vector of length B*C; the
    # class head and loss expect one row per document instead.
    assert tuple(class_targets.shape) == (2, 2)
    assert class_targets.tolist() == [[1.0, 0.0], [0.0, 1.0]]
