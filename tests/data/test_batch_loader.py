"""The batch contract: what `get_batch_loader` hands the model.

Nothing used to drive the loader, so the collate that sits between the dataset
and the model was never pinned — and the models were written against the shape
torch's `default_collate` happened to produce (a phantom leading dim on every
field) rather than the one they document. These tests pin the shape the model
methods actually read, with documents of *differing chunk counts* in one batch,
which is where the accidental shape stopped working at all.
"""

import numpy as np
import torch
from torch.utils.data import SequentialSampler

from d3text.data.data import collate_documents, get_batch_loader


def batches(dataset, batch_size):
    """Every batch the loader yields, in order, over the whole dataset."""
    return list(
        get_batch_loader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=SequentialSampler(dataset),
        )
    )


def test_loader_yields_one_item_per_document_with_no_batch_dim(tiny_brenda):
    """A batch is a list of documents, each holding its own tensors: the
    sequences are 2-D `[n_chunks, token]`, and the labels are the 1-D multi-hot
    rows the corpus stores. A batch dimension would be a fiction — the three
    documents hold 2, 5 and 1 chunks, and do not stack."""
    (batch,) = batches(tiny_brenda.present, batch_size=3)

    assert len(batch) == 3
    assert [doc["sequence"]["input_ids"].shape[0] for doc in batch] == [2, 5, 1]
    for doc in batch:
        assert doc["sequence"]["input_ids"].ndim == 2
        assert doc["sequence"]["attention_mask"].ndim == 2
        assert doc["entities"].shape == (3,)
        assert doc["classes"].shape == (2,)


def test_loader_keeps_doc_id_counting_each_document_s_chunks(tiny_brenda):
    """`get_token_embeddings` slices the base model's output back into documents
    by `doc_id.shape[-1]`, so `doc_id` has to stay one entry per chunk."""
    (batch,) = batches(tiny_brenda.present, batch_size=3)

    assert [doc["doc_id"].shape[-1] for doc in batch] == [2, 5, 1]
    assert [int(doc["id"]) for doc in batch] == [10, 20, 30]
    assert batch[0]["id"].ndim == 0  # `.item()` in get_token_embeddings


def test_loader_batches_the_whole_dataset(tiny_brenda):
    """The sampler is the loader's *batch* sampler, so each batch is one call
    into the dataset — and every document is dealt exactly once."""
    seen = [
        int(doc["id"])
        for batch in batches(tiny_brenda.present, 2)
        for doc in batch
    ]

    assert seen == [10, 20, 30]
    assert [len(batch) for batch in batches(tiny_brenda.present, 2)] == [2, 1]


def test_collate_converts_the_corpus_arrays_to_tensors():
    """The dataset reads numpy out of the HDF5 file and the DataFrame; the model
    reads tensors. The relation labels are tensors too — the models take their
    `argmax` as the target column."""
    doc = {
        "id": np.int64(10),
        "doc_id": torch.zeros(2, dtype=torch.uint8),
        "sequence": {
            "input_ids": np.zeros((2, 8), dtype=np.int64),
            "attention_mask": np.ones((2, 8), dtype=np.int64),
        },
        "entities": np.array([1, 0, 1], dtype=np.uint8),
        "classes": np.array([1, 0], dtype=np.float32),
        "relations": [
            {("bac1", "enz2"): np.array([0, 1, 0], dtype=np.float16)}
        ],
    }

    (item,) = collate_documents([doc])

    assert torch.is_tensor(item["id"]) and int(item["id"]) == 10
    assert torch.is_tensor(item["sequence"]["input_ids"])
    assert item["sequence"]["input_ids"].shape == (2, 8)
    assert torch.is_tensor(item["entities"])
    assert item["entities"].tolist() == [1, 0, 1]
    label = item["relations"][0][("bac1", "enz2")]
    assert torch.is_tensor(label) and int(label.argmax()) == 1
