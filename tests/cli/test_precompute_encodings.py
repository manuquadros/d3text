"""`precompute-encodings` must not store a document that has no text.

A document whose halves are missing, or are markup wrapping whitespace,
tokenizes to one window of `[CLS]` and `[SEP]`. The command warned about
exactly that and wrote the group anyway, leaving the data layer to detect and
drop it at read time. The reader keys on pubmed id throughout and a pmid with
no group is already supported, so a skipped document is invisible to it.
"""

import io
import logging
import pathlib
import re

import h5py
import numpy as np
import polars as pl
import pytest
from d3text import logs
from d3text.cli import precompute_encodings
from d3text.datasets import enzymener, s800
from d3text.encodings_store import (
    EncodingsProvenance,
    content_digest,
    mark_group_complete,
    read_content_digest,
    read_provenance,
)

# Markup wrapping only whitespace: the tags strip away and what is left is
# blank, which is what `corpus.document_text` reports as an empty document.
_BLANK_BODY = "<p>   </p>"

_WINDOW = 8


def _encoding_stub(doc: str, tokenizer: object) -> dict[str, np.ndarray]:
    """Stands in for `encode_document`, which would download a tokenizer.

    Shaped like the real `BatchEncoding` the command stores: one window per
    document, and the four arrays it writes as datasets.
    """
    return {
        "input_ids": np.ones((1, _WINDOW), dtype=np.uint32),
        "attention_mask": np.ones((1, _WINDOW), dtype=np.uint8),
        "overflow_to_sample_mapping": np.zeros(1, dtype=np.uint8),
        "offset_mapping": np.zeros((1, _WINDOW, 2), dtype=np.uint32),
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

    The package's own handler is installed with a readable stream:
    `logs.configure` sets `propagate = False`, so nothing the command logs
    reaches `caplog`.
    """
    configure = logs.configure

    def run(
        dataset: pathlib.Path,
        output: pathlib.Path,
        *flags: str,
        base_model: str = "a-base-model",
        encode=_encoding_stub,
    ) -> str:
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
        monkeypatch.setattr(precompute_encodings, "encode_document", encode)
        monkeypatch.setattr(
            "sys.argv",
            [
                "precompute-encodings",
                base_model,
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

    A document that had text when encoded and has none now must lose its group,
    or the one flag that refreshes the artifact can never clear what the corpus
    has stopped supplying.
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


def test_the_store_records_the_model_window_and_stride_that_wrote_it(
    run_command, tmp_path
):
    """None of the three is recoverable from a bare array of token ids, and
    the row count is the same for any window or stride — this stamp is the
    only place a later drift can be caught."""
    dataset = tmp_path / "corpus.csv"
    _write_corpus(
        dataset, [{"pubmed_id": 1, "abstract": "x", "fulltext": None}]
    )
    output = tmp_path / "encodings.hdf5"

    run_command(dataset, output)

    with h5py.File(output, "r") as f:
        assert read_provenance(f) == EncodingsProvenance(
            base_model="a-base-model", max_length=512, stride=20
        )


def test_resuming_under_a_different_base_model_is_refused(
    run_command, tmp_path
):
    """A resume that disagrees with the store's own stamp is the way two
    tokenizers' ids end up in one file with nothing to tell them apart."""
    dataset = tmp_path / "corpus.csv"
    _write_corpus(
        dataset, [{"pubmed_id": 1, "abstract": "x", "fulltext": None}]
    )
    output = tmp_path / "encodings.hdf5"
    run_command(dataset, output)

    other_dataset = tmp_path / "second.csv"
    _write_corpus(
        other_dataset, [{"pubmed_id": 2, "abstract": "y", "fulltext": None}]
    )

    with pytest.raises(ValueError, match="was written by a-base-model"):
        run_command(other_dataset, output, base_model="another-base-model")

    with h5py.File(output, "r") as f:
        assert "2" not in f
        assert read_provenance(f) == EncodingsProvenance(
            base_model="a-base-model", max_length=512, stride=20
        )


def _write_finished_group(f: h5py.File, key: str, fill: int) -> None:
    """A group shaped exactly as a completed write leaves it."""
    group = f.create_group(key)
    group.create_dataset(
        name="input_ids", data=np.full((1, _WINDOW), fill, dtype=np.uint32)
    )
    group.create_dataset(
        name="attention_mask", data=np.ones((1, _WINDOW), dtype=np.uint8)
    )
    group.create_dataset(
        name="overflow_to_sample_mapping", data=np.zeros(1, dtype=np.uint8)
    )
    mark_group_complete(group)


def test_a_stored_empty_document_survives_a_run_without_force(
    run_command, tmp_path
):
    """Without `-f` a stored, finished pmid is not read at all, let alone
    rewritten."""
    output = tmp_path / "encodings.hdf5"
    with h5py.File(output, "w-") as f:
        _write_finished_group(f, "2", fill=1)

    dataset = tmp_path / "corpus.csv"
    _write_corpus(
        dataset,
        [{"pubmed_id": 2, "abstract": None, "fulltext": _BLANK_BODY}],
    )

    run_command(dataset, output)

    with h5py.File(output, "r") as f:
        assert "2" in f
        assert np.array_equal(f["2"]["input_ids"][:], np.ones((1, _WINDOW)))


def test_an_empty_group_is_rewritten_on_resume(run_command, tmp_path):
    """A kill right after `create_group`, before any dataset, leaves a group
    that `stored_ids` already treats as holding no ids -- but that makes the
    document invisible to a reader forever, not merely once, since a plain
    `key in f` resume guard also treats the group as already done."""
    output = tmp_path / "encodings.hdf5"
    with h5py.File(output, "w-") as f:
        f.create_group("2")

    dataset = tmp_path / "corpus.csv"
    _write_corpus(
        dataset, [{"pubmed_id": 2, "abstract": "some text", "fulltext": None}]
    )

    run_command(dataset, output)

    with h5py.File(output, "r") as f:
        assert "attention_mask" in f["2"]
        assert "overflow_to_sample_mapping" in f["2"]


def test_a_group_missing_mask_and_mapping_is_rewritten_on_resume(
    run_command, tmp_path
):
    """A kill between the first and second `create_dataset` call leaves
    `input_ids` alone in the group; `stored_ids` accepts that as a document
    with ids, but the reader that pairs it with a mask and mapping never
    gets one."""
    output = tmp_path / "encodings.hdf5"
    with h5py.File(output, "w-") as f:
        f.create_group("2").create_dataset(
            name="input_ids", data=np.ones((1, _WINDOW), dtype=np.uint32)
        )

    dataset = tmp_path / "corpus.csv"
    _write_corpus(
        dataset, [{"pubmed_id": 2, "abstract": "some text", "fulltext": None}]
    )

    run_command(dataset, output)

    with h5py.File(output, "r") as f:
        assert "attention_mask" in f["2"]
        assert "overflow_to_sample_mapping" in f["2"]


def test_a_zero_filled_input_ids_is_rewritten_on_resume(run_command, tmp_path):
    """h5py names a dataset before it is populated, so a kill during the
    very first `create_dataset` call -- not only between calls -- can leave
    `input_ids` present and correctly shaped but still zero-filled: a
    plausible-looking document of padding tokens rather than a crash."""
    output = tmp_path / "encodings.hdf5"
    with h5py.File(output, "w-") as f:
        f.create_group("2").create_dataset(
            name="input_ids", data=np.zeros((1, _WINDOW), dtype=np.uint32)
        )

    dataset = tmp_path / "corpus.csv"
    _write_corpus(
        dataset, [{"pubmed_id": 2, "abstract": "some text", "fulltext": None}]
    )

    run_command(dataset, output)

    with h5py.File(output, "r") as f:
        assert not np.array_equal(
            f["2"]["input_ids"][:], np.zeros((1, _WINDOW))
        )


def test_the_store_records_a_digest_of_the_ids_it_holds(run_command, tmp_path):
    """The model, window and stride describe two tokenizations of one corpus
    identically; the digest is the only thing that separates them."""
    dataset = tmp_path / "corpus.csv"
    _write_corpus(
        dataset, [{"pubmed_id": 1, "abstract": "x", "fulltext": None}]
    )
    output = tmp_path / "encodings.hdf5"

    run_command(dataset, output)

    with h5py.File(output, "r") as f:
        assert read_content_digest(f) == content_digest(f)


def test_a_resume_restamps_the_digest_over_the_whole_file(
    run_command, tmp_path
):
    """The digest is a property of the file, not of the pass that wrote it.
    Stamped once at creation it would keep attributing the store to the first
    pass's documents through every resume that added more."""
    output = tmp_path / "encodings.hdf5"
    first = tmp_path / "first.csv"
    _write_corpus(first, [{"pubmed_id": 1, "abstract": "x", "fulltext": None}])
    run_command(first, output)
    with h5py.File(output, "r") as f:
        after_first = read_content_digest(f)

    second = tmp_path / "second.csv"
    _write_corpus(second, [{"pubmed_id": 2, "abstract": "y", "fulltext": None}])
    run_command(second, output)

    with h5py.File(output, "r") as f:
        assert read_content_digest(f) == content_digest(f)
        assert read_content_digest(f) != after_first


def test_an_interrupted_retokenization_leaves_the_store_unstamped(
    run_command, tmp_path
):
    """A killed `-f` pass has already replaced some of the ids the digest was
    taken over, and the enclosing `with h5py.File(...)` closes the file
    cleanly on the way out — so a stamp only ever restated at the end would
    survive as a fingerprint of ids that are no longer there, and `evaluate`
    would report the store and a checkpoint as agreeing."""
    dataset = tmp_path / "corpus.csv"
    _write_corpus(
        dataset,
        [
            {"pubmed_id": 1, "abstract": "one", "fulltext": None},
            {"pubmed_id": 2, "abstract": "two", "fulltext": None},
        ],
    )
    output = tmp_path / "encodings.hdf5"
    run_command(dataset, output)
    with h5py.File(output, "r") as f:
        stale = read_content_digest(f)

    def retokenize_then_die(doc, tokenizer):
        if doc == "two":
            raise KeyboardInterrupt
        return dict(
            _encoding_stub(doc, tokenizer),
            input_ids=np.full((1, _WINDOW), 7, dtype=np.uint32),
        )

    with pytest.raises(KeyboardInterrupt):
        run_command(dataset, output, "-f", encode=retokenize_then_die)

    with h5py.File(output, "r") as f:
        assert content_digest(f) != stale
        assert read_content_digest(f) is None


@pytest.fixture
def run_main(monkeypatch):
    """Run `main` over an arbitrary argv tail, returning its console output.

    Generalizes `run_command` for a run with no positional dataset at all --
    `--s800`/`--enzymener` alone -- which `run_command` cannot express since
    it always appends one.
    """
    configure = logs.configure

    def run(
        *argv_tail: str,
        base_model: str = "a-base-model",
        encode=_encoding_stub,
    ) -> str:
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
        monkeypatch.setattr(precompute_encodings, "encode_document", encode)
        monkeypatch.setattr(
            "sys.argv",
            ["precompute-encodings", base_model, *argv_tail],
        )
        precompute_encodings.main()
        return stream.getvalue()

    yield run

    logs.configure()


def _word_offset_stub(doc: str, tokenizer: object) -> dict[str, np.ndarray]:
    """A tokenizer stand-in whose `offset_mapping` is exact word spans.

    A real subword tokenizer would split unpredictably; splitting on
    whitespace instead gives one `(start, end)` per token that a test can
    compute independently of the command, without downloading a real
    tokenizer.
    """
    words = list(re.finditer(r"\S+", doc))[:_WINDOW]
    offsets = np.zeros((1, _WINDOW, 2), dtype=np.uint32)
    mask = np.zeros((1, _WINDOW), dtype=np.uint8)
    for index, word in enumerate(words):
        offsets[0, index] = (word.start(), word.end())
        mask[0, index] = 1
    return {
        "input_ids": np.ones((1, _WINDOW), dtype=np.uint32),
        "attention_mask": mask,
        "overflow_to_sample_mapping": np.zeros(1, dtype=np.uint8),
        "offset_mapping": offsets,
    }


def _resolve_span(
    offset_mapping: np.ndarray, start: int, end: int
) -> tuple[int, int]:
    """The one stored token offset exactly matching `(start, end)`.

    Pins the actual round trip the ticket asks for: not that a group exists,
    but that a mention's own offsets can be found again among what got
    stored.
    """
    for row in offset_mapping.reshape(-1, 2):
        if (int(row[0]), int(row[1])) == (start, end):
            return int(row[0]), int(row[1])
    raise AssertionError(f"no stored token offset covers [{start}, {end})")


def test_s800_offset_mapping_round_trips_through_the_prefixed_key(
    run_main, tmp_path
):
    """S800's `end` is inclusive on disk and half-open once loaded; the
    `s800:`-prefixed group's stored `offset_mapping` must resolve back to
    that same half-open span and the surface it addresses."""
    root = tmp_path / "s800corpus"
    (root / s800.ABSTRACTS).mkdir(parents=True)
    (root / s800.ANNOTATIONS).write_text(
        "999\tspecies001:12345\t9\t18\tSalmonella\n", encoding="utf8"
    )
    (root / s800.ABSTRACTS / "species001.txt").write_text(
        "Study of Salmonella today.", encoding="utf8"
    )
    mention = s800.load_s800(root).mentions[0]
    assert (mention.start, mention.end) == (9, 19)

    output = tmp_path / "encodings.hdf5"
    run_main(str(output), "--s800", str(root), encode=_word_offset_stub)

    with h5py.File(output, "r") as f:
        key = f"s800:{mention.document}"
        assert key in f
        offset_mapping = f[key]["offset_mapping"][:]

    start, end = _resolve_span(offset_mapping, mention.start, mention.end)
    text = (root / s800.ABSTRACTS / "species001.txt").read_text(encoding="utf8")
    assert text[start:end] == mention.surface


def test_enzymener_offset_mapping_round_trips_through_the_prefixed_key(
    run_main, tmp_path
):
    """enzymeNER's offsets are read half-open as written; the `enzymener:`
    -prefixed group's stored `offset_mapping` must resolve back to that same
    span and the surface it addresses."""
    root = tmp_path / "enzymenercorpus"
    root.mkdir()
    (root / enzymener.SENTENCES).write_text(
        "﻿PMC9\tS01\tExpression of catalase was measured.\n",
        encoding="utf8",
    )
    (root / enzymener.ANNOTATIONS).write_text(
        "﻿PMC9\tS01\t14\t22\tcatalase\n", encoding="utf8"
    )
    corpus_data = enzymener.load_enzymener(root)
    mention = corpus_data.mentions[0]
    assert (mention.start, mention.end) == (14, 22)

    output = tmp_path / "encodings.hdf5"
    run_main(str(output), "--enzymener", str(root), encode=_word_offset_stub)

    with h5py.File(output, "r") as f:
        key = f"enzymener:{mention.document}"
        assert key in f
        offset_mapping = f[key]["offset_mapping"][:]

    start, end = _resolve_span(offset_mapping, mention.start, mention.end)
    assert corpus_data.texts[mention.document][start:end] == mention.surface


def test_encoding_with_nothing_to_encode_is_a_clear_argument_error(
    run_main, tmp_path, capsys
):
    """No positional dataset and neither flag must fail loudly, not encode
    nothing silently -- `argparse.error` exits non-zero with a message
    naming the problem."""
    output = tmp_path / "encodings.hdf5"

    with pytest.raises(SystemExit):
        run_main(str(output))

    assert "nothing to encode" in capsys.readouterr().err
