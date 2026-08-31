"""The HDF5 resume loop must not re-append documents already in the store.

``generate_embeddings.py`` keys its resume set off ``pubmed_ids[:]``, which
h5py hands back as ``bytes`` for a ``string_dtype`` dataset, and compares it
against ``str(row_id)``. A ``bytes``/``str`` set membership test is always
``False``, so an unfixed script re-embeds and re-appends every document on
every rerun, silently duplicating rows at new offsets rather than skipping
them.

The heavy dependencies (the base model, the tokenizer) are mocked out; the
corpus reader and the HDF5 I/O are exercised for real, since those are the
code the fix touches.
"""

import pathlib
import runpy
import sys
from unittest import mock

import h5py
import torch

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "generate_embeddings.py"

_HIDDEN_SIZE = 4


def _write_corpus(path: pathlib.Path) -> None:
    path.write_text(
        ",pubmed_id,abstract,fulltext\n"
        "0,111,<p>first abstract</p>,<p>first body</p>\n"
        "1,222,<p>second abstract</p>,<p>second body</p>\n"
    )


def _run_script(output_path: pathlib.Path, corpus_path: pathlib.Path) -> None:
    fake_model = mock.MagicMock()
    fake_model.cuda.return_value = fake_model
    fake_model.eval.return_value = fake_model
    fake_model.config.hidden_size = _HIDDEN_SIZE

    with (
        mock.patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=mock.MagicMock(),
        ),
        mock.patch(
            "transformers.AutoModel.from_pretrained",
            return_value=fake_model,
        ),
        mock.patch(
            "d3text.utils.embed_document",
            return_value=torch.zeros((2, _HIDDEN_SIZE)),
        ),
    ):
        argv = sys.argv
        sys.argv = [
            "generate_embeddings.py",
            "fake-base-model",
            str(output_path),
            str(corpus_path),
        ]
        try:
            runpy.run_path(str(_SCRIPT), run_name="__main__")
        finally:
            sys.argv = argv


def test_resume_does_not_duplicate_already_embedded_documents(
    tmp_path: pathlib.Path,
) -> None:
    corpus_path = tmp_path / "corpus.csv"
    _write_corpus(corpus_path)
    output_path = tmp_path / "embeddings.hdf5"

    _run_script(output_path, corpus_path)

    with h5py.File(output_path, mode="r") as f:
        first_run_pubmed_ids = sorted(f["pubmed_ids"][:])
        first_run_doc_count = f["offsets"].shape[0]
        first_run_token_count = f["embeddings"].shape[0]

    assert first_run_pubmed_ids == [b"111", b"222"]
    assert first_run_doc_count == 2

    _run_script(output_path, corpus_path)

    with h5py.File(output_path, mode="r") as f:
        second_run_pubmed_ids = sorted(f["pubmed_ids"][:])
        second_run_doc_count = f["offsets"].shape[0]
        second_run_token_count = f["embeddings"].shape[0]

    assert second_run_pubmed_ids == first_run_pubmed_ids
    assert second_run_doc_count == first_run_doc_count
    assert second_run_token_count == first_run_token_count
