"""``precompute-encodings`` must not store a document that has no text.

A document whose abstract and body are both missing — or are markup wrapping
nothing but whitespace — tokenizes to a single window holding ``[CLS]`` and
``[SEP]`` and no token of its own. The command warned about exactly that case
and then wrote the group anyway, so the artifact carried content-free documents
that the data layer had to detect and drop at read time.

The reader keys on pubmed id throughout (`BrendaDataset._getitems`,
`sequence_lengths`, `_drop_empty_documents`), never on group position, and a
pmid with no group is an already-supported condition, so a skipped document is
invisible to it.
"""

import io
import logging
import pathlib

import h5py
import numpy as np
import polars as pl
import pytest
from d3text import logs
from d3text.cli import precompute_encodings

# Markup wrapping only whitespace: the tags strip away and what is left is
# blank, which is what `corpus.document_text` reports as an empty document.
_BLANK_BODY = "<p>   </p>"

_WINDOW = 8


def _encoding_stub(doc: str, tokenizer: object) -> dict[str, np.ndarray]:
    """Stands in for `encode_document`, which would download a tokenizer.

    Shaped like the real `BatchEncoding` the command stores: one window per
    document, and the three arrays it writes as datasets.
    """
    return {
        "input_ids": np.ones((1, _WINDOW), dtype=np.uint32),
        "attention_mask": np.ones((1, _WINDOW), dtype=np.uint8),
        "overflow_to_sample_mapping": np.zeros(1, dtype=np.uint8),
    }


def _write_corpus(path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    pl.DataFrame(
        rows,
        schema={
            "pubmed_id": pl.Int64,
            "abstract": pl.Utf8,
            "fulltext": pl.Utf8,
        },
    ).write_csv(path)


@pytest.fixture
def run_command(monkeypatch, tmp_path):
    """Run `main` over a corpus, returning the console output it produced.

    The package's own handler is installed with a readable stream rather than
    left to pytest's capture: `logs.configure` sets ``propagate = False`` on
    the `d3text` logger, so nothing the command logs reaches `caplog`.
    """
    configure = logs.configure

    def run(dataset: pathlib.Path, output: pathlib.Path, *flags: str) -> str:
        stream = io.StringIO()
        monkeypatch.setattr(
            precompute_encodings.logs,
            "configure",
            lambda: configure(logging.WARNING, stream=stream),
        )
        monkeypatch.setattr(
            precompute_encodings.utils,
            "load_fast_tokenizer",
            lambda base_model: object(),
        )
        monkeypatch.setattr(
            precompute_encodings, "encode_document", _encoding_stub
        )
        monkeypatch.setattr(
            "sys.argv",
            [
                "precompute-encodings",
                "a-base-model",
                str(output),
                str(dataset),
                *flags,
            ],
        )

        precompute_encodings.main()
        return stream.getvalue()

    yield run

    logs.configure()


def test_an_empty_document_gets_no_group(run_command, tmp_path):
    """The warning names the problem; storing the group anyway created it."""
    dataset = tmp_path / "corpus.csv"
    _write_corpus(
        dataset,
        [
            {"pubmed_id": 1, "abstract": "an abstract", "fulltext": None},
            {"pubmed_id": 2, "abstract": None, "fulltext": _BLANK_BODY},
        ],
    )
    output = tmp_path / "encodings.hdf5"

    logged = run_command(dataset, output)

    with h5py.File(output, "r") as f:
        assert "1" in f
        assert "2" not in f

    assert "2" in logged


def test_force_regenerate_removes_a_stored_empty_document(
    run_command, tmp_path
):
    """`-f` makes the file agree with the corpus, in both directions.

    A document that had text when it was encoded and has none now must lose
    its group, or the one flag that exists to refresh the artifact can never
    clear what the corpus has stopped supplying.
    """
    output = tmp_path / "encodings.hdf5"
    with h5py.File(output, "w-") as f:
        f.create_group("2").create_dataset(
            name="input_ids", data=np.ones((1, _WINDOW), dtype=np.uint32)
        )

    dataset = tmp_path / "corpus.csv"
    _write_corpus(
        dataset,
        [{"pubmed_id": 2, "abstract": None, "fulltext": _BLANK_BODY}],
    )

    run_command(dataset, output, "-f")

    with h5py.File(output, "r") as f:
        assert "2" not in f


def test_a_stored_empty_document_survives_a_run_without_force(
    run_command, tmp_path
):
    """Without `-f` a stored pmid is not read at all, let alone rewritten."""
    output = tmp_path / "encodings.hdf5"
    with h5py.File(output, "w-") as f:
        f.create_group("2").create_dataset(
            name="input_ids", data=np.ones((1, _WINDOW), dtype=np.uint32)
        )

    dataset = tmp_path / "corpus.csv"
    _write_corpus(
        dataset,
        [{"pubmed_id": 2, "abstract": None, "fulltext": _BLANK_BODY}],
    )

    run_command(dataset, output)

    with h5py.File(output, "r") as f:
        assert "2" in f
