"""`precompute-encodings` must not store a document that has no text.

A document whose halves are missing, or are markup wrapping whitespace,
tokenizes to one window of `[CLS]` and `[SEP]`. The command warned about
that and wrote the document anyway, leaving the data layer to drop it at
read time. The reader keys on pubmed id and already supports a pmid the
store lacks, so a skipped document is invisible to it.
"""

import io
import logging
import pathlib
import re
import string
import sys

import brenda_references
import numpy as np
import polars as pl
import pytest
from d3text import logs
from d3text.cli import precompute_encodings
from d3text.datasets import enzymener, s800
from d3text.encodings_store import (
    EncodingsProvenance,
    EncodingsStore,
    content_digest,
    read_content_digest,
    read_provenance,
    write_provenance,
)
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

# Markup wrapping only whitespace: the tags strip away and what is left is
# blank, which is what `corpus.document_text` reports as an empty document.
_BLANK_BODY = "<p>   </p>"

_WINDOW = 8

# What `run_command` writes under by default.
_PROVENANCE = EncodingsProvenance(
    base_model="a-base-model", max_length=512, stride=20
)


def _prefill(
    path: pathlib.Path,
    documents: dict[str, np.ndarray],
    provenance: EncodingsProvenance = _PROVENANCE,
) -> None:
    """A store stamped with `provenance`, holding `documents`' ids as a
    completed write leaves them. Keyed pmid -> the ids, `[windows, tokens]`.
    """
    with EncodingsStore(path, writable=True) as store:
        write_provenance(store, provenance)
        for key, ids in documents.items():
            store.put(
                key,
                {
                    "input_ids": ids,
                    "attention_mask": np.ones(ids.shape, dtype=np.uint8),
                    "offset_mapping": np.zeros((*ids.shape, 2), np.uint32),
                },
            )


def _encoding_stub(docs: list[str], tokenizer: object) -> dict[str, np.ndarray]:
    """Stands in for `encode_documents`, which would download a tokenizer.

    Shaped like the real `BatchEncoding` the command reads: one window per
    document, stacked in call order. `overflow_to_sample_mapping` is what
    the writer selects each document's rows with; the other three are what
    it stores.
    """
    n = len(docs)
    return {
        "input_ids": np.ones((n, _WINDOW), dtype=np.uint32),
        "attention_mask": np.ones((n, _WINDOW), dtype=np.uint8),
        "overflow_to_sample_mapping": np.arange(n, dtype=np.uint8),
        "offset_mapping": np.zeros((n, _WINDOW, 2), dtype=np.uint32),
    }


def _build_offline_fast_tokenizer() -> PreTrainedTokenizerFast:
    """A real WordPiece/BertPreTokenizer tokenizer built in-process.

    Genuine tokenization logic -- offsets, overflow windows, stride -- over a
    tiny inline vocabulary, so any word made of ASCII lowercase letters
    decomposes one token per character. No download, no network. Duplicated
    from the twin in `tests/test_utils.py` rather than imported, so this file
    stays free of a cross-test-module dependency for one small helper.
    """
    specials = ("[PAD]", "[UNK]", "[CLS]", "[SEP]")
    vocabulary = {token: index for index, token in enumerate(specials)}
    for character in string.ascii_lowercase:
        vocabulary.setdefault(character, len(vocabulary))
        vocabulary.setdefault("##" + character, len(vocabulary))

    backend = Tokenizer(models.WordPiece(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[
            ("[CLS]", vocabulary["[CLS]"]),
            ("[SEP]", vocabulary["[SEP]"]),
        ],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )


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
        monkeypatch.setattr(precompute_encodings, "encode_documents", encode)
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


def test_an_empty_document_is_not_stored(run_command, tmp_path):
    """The warning names the problem; storing the document anyway created
    it."""
    dataset = tmp_path / "corpus.csv"
    _write_corpus(
        dataset,
        [
            {"pubmed_id": 1, "abstract": "an abstract", "fulltext": None},
            {"pubmed_id": 2, "abstract": None, "fulltext": _BLANK_BODY},
        ],
    )
    output = tmp_path / "encodings"

    logged = run_command(dataset, output)

    with EncodingsStore(output) as f:
        assert "1" in f
        assert "2" not in f

    assert "2" in logged


def test_force_regenerate_removes_a_stored_empty_document(
    run_command, tmp_path
):
    """`-f` makes the store agree with the corpus, in both directions.

    A document that had text when encoded and has none now must be removed,
    or the one flag that refreshes the artifact can never clear what the corpus
    has stopped supplying.
    """
    output = tmp_path / "encodings"
    _prefill(output, {"2": np.ones((1, _WINDOW), dtype=np.uint32)})

    dataset = tmp_path / "corpus.csv"
    _write_corpus(
        dataset,
        [{"pubmed_id": 2, "abstract": None, "fulltext": _BLANK_BODY}],
    )

    run_command(dataset, output, "-f")

    with EncodingsStore(output) as f:
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
    output = tmp_path / "encodings"

    run_command(dataset, output)

    with EncodingsStore(output) as f:
        assert read_provenance(f) == EncodingsProvenance(
            base_model="a-base-model", max_length=512, stride=20
        )


def test_a_written_document_stores_no_sample_mapping(run_command, tmp_path):
    """The tokenizer's sample index selects rows; it is not stored."""
    dataset = tmp_path / "corpus.csv"
    _write_corpus(
        dataset, [{"pubmed_id": 1, "abstract": "an abstract", "fulltext": None}]
    )
    output = tmp_path / "encodings"

    run_command(dataset, output)

    with EncodingsStore(output) as f:
        stored = f.get("1")
    assert stored is not None
    assert set(stored) == {"input_ids", "attention_mask", "offset_mapping"}


def test_resuming_under_a_different_base_model_is_refused(
    run_command, tmp_path
):
    """A resume that disagrees with the store's own stamp is the way two
    tokenizers' ids end up in one store with nothing to tell them apart."""
    dataset = tmp_path / "corpus.csv"
    _write_corpus(
        dataset, [{"pubmed_id": 1, "abstract": "x", "fulltext": None}]
    )
    output = tmp_path / "encodings"
    run_command(dataset, output)

    other_dataset = tmp_path / "second.csv"
    _write_corpus(
        other_dataset, [{"pubmed_id": 2, "abstract": "y", "fulltext": None}]
    )

    with pytest.raises(ValueError, match="was written by a-base-model"):
        run_command(other_dataset, output, base_model="another-base-model")

    with EncodingsStore(output) as f:
        assert "2" not in f
        assert read_provenance(f) == EncodingsProvenance(
            base_model="a-base-model", max_length=512, stride=20
        )


def test_a_stored_empty_document_survives_a_run_without_force(
    run_command, tmp_path
):
    """Without `-f` a stored pmid is not read at all, let alone rewritten."""
    output = tmp_path / "encodings"
    _prefill(output, {"2": np.ones((1, _WINDOW), dtype=np.uint32)})

    dataset = tmp_path / "corpus.csv"
    _write_corpus(
        dataset,
        [{"pubmed_id": 2, "abstract": None, "fulltext": _BLANK_BODY}],
    )

    run_command(dataset, output)

    with EncodingsStore(output) as f:
        stored = f.get("2")
    assert stored is not None
    assert np.array_equal(stored["input_ids"], np.ones((1, _WINDOW)))


def test_resuming_onto_an_hdf5_store_is_refused_untouched(
    run_command, tmp_path
):
    """A store of the older HDF5 layout is no resume target: nothing here
    reads one, and writing an LMDB beside or over it would lose the old
    store with no data migrated."""
    output = tmp_path / "encodings.hdf5"
    output.write_bytes(b"\x89HDF\r\n\x1a\n" + bytes(504))
    before = output.read_bytes()
    dataset = tmp_path / "corpus.csv"
    _write_corpus(
        dataset, [{"pubmed_id": 1, "abstract": "x", "fulltext": None}]
    )

    with pytest.raises(ValueError, match="HDF5.*precompute-encodings"):
        run_command(dataset, output)

    assert output.read_bytes() == before


def test_the_store_records_a_digest_of_the_ids_it_holds(run_command, tmp_path):
    """The model, window and stride describe two tokenizations of one corpus
    identically; the digest is the only thing that separates them."""
    dataset = tmp_path / "corpus.csv"
    _write_corpus(
        dataset, [{"pubmed_id": 1, "abstract": "x", "fulltext": None}]
    )
    output = tmp_path / "encodings"

    run_command(dataset, output)

    with EncodingsStore(output) as f:
        assert read_content_digest(f) == content_digest(f)


def test_a_resume_restamps_the_digest_over_the_whole_store(
    run_command, tmp_path
):
    """The digest is a property of the store, not of the pass that wrote it.
    Stamped once at creation it would keep attributing the store to the first
    pass's documents through every resume that added more."""
    output = tmp_path / "encodings"
    first = tmp_path / "first.csv"
    _write_corpus(first, [{"pubmed_id": 1, "abstract": "x", "fulltext": None}])
    run_command(first, output)
    with EncodingsStore(output) as f:
        after_first = read_content_digest(f)

    second = tmp_path / "second.csv"
    _write_corpus(second, [{"pubmed_id": 2, "abstract": "y", "fulltext": None}])
    run_command(second, output)

    with EncodingsStore(output) as f:
        assert read_content_digest(f) == content_digest(f)
        assert read_content_digest(f) != after_first


def test_an_interrupted_retokenization_leaves_the_store_unstamped(
    run_command, tmp_path
):
    """A killed `-f` pass may not have restored any of the ids the digest
    was taken over, and the enclosing `with` block closes the store cleanly
    on the way out — so a stamp only ever restated at the end would
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
    output = tmp_path / "encodings"
    run_command(dataset, output)
    with EncodingsStore(output) as f:
        stale = read_content_digest(f)

    def retokenize_then_die(docs, tokenizer):
        if "two" in docs:
            raise KeyboardInterrupt
        return dict(
            _encoding_stub(docs, tokenizer),
            input_ids=np.full((len(docs), _WINDOW), 7, dtype=np.uint32),
        )

    with pytest.raises(KeyboardInterrupt):
        run_command(dataset, output, "-f", encode=retokenize_then_die)

    with EncodingsStore(output) as f:
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
        monkeypatch.setattr(precompute_encodings, "encode_documents", encode)
        monkeypatch.setattr(
            "sys.argv",
            ["precompute-encodings", base_model, *argv_tail],
        )
        precompute_encodings.main()
        return stream.getvalue()

    yield run

    logs.configure()


def _word_offset_stub(
    docs: list[str], tokenizer: object
) -> dict[str, np.ndarray]:
    """A tokenizer stand-in whose `offset_mapping` is exact word spans.

    A real subword tokenizer would split unpredictably; splitting on
    whitespace instead gives one `(start, end)` per token that a test can
    compute independently of the command, without downloading a real
    tokenizer. One window per document, stacked in call order, like
    `_encoding_stub`.
    """
    n = len(docs)
    offsets = np.zeros((n, _WINDOW, 2), dtype=np.uint32)
    mask = np.zeros((n, _WINDOW), dtype=np.uint8)
    for row, doc in enumerate(docs):
        words = list(re.finditer(r"\S+", doc))[:_WINDOW]
        for index, word in enumerate(words):
            offsets[row, index] = (word.start(), word.end())
            mask[row, index] = 1
    return {
        "input_ids": np.ones((n, _WINDOW), dtype=np.uint32),
        "attention_mask": mask,
        "overflow_to_sample_mapping": np.arange(n, dtype=np.uint8),
        "offset_mapping": offsets,
    }


def _resolve_span(
    offset_mapping: np.ndarray, start: int, end: int
) -> tuple[int, int]:
    """The one stored token offset exactly matching `(start, end)`.

    Pins the actual round trip that matters: not that a document exists,
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
    `s800:`-prefixed document's stored `offset_mapping` must resolve back to
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

    output = tmp_path / "encodings"
    run_main(str(output), "--s800", str(root), encode=_word_offset_stub)

    with EncodingsStore(output) as f:
        key = f"s800:{mention.document}"
        assert key in f
        offset_mapping = f.get(key)["offset_mapping"]

    start, end = _resolve_span(offset_mapping, mention.start, mention.end)
    text = (root / s800.ABSTRACTS / "species001.txt").read_text(encoding="utf8")
    assert text[start:end] == mention.surface


def test_enzymener_offset_mapping_round_trips_through_the_prefixed_key(
    run_main, tmp_path
):
    """enzymeNER's offsets are read half-open as written; the `enzymener:`
    -prefixed document's stored `offset_mapping` must resolve back to that same
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

    output = tmp_path / "encodings"
    run_main(str(output), "--enzymener", str(root), encode=_word_offset_stub)

    with EncodingsStore(output) as f:
        key = f"enzymener:{mention.document}"
        assert key in f
        offset_mapping = f.get(key)["offset_mapping"]

    start, end = _resolve_span(offset_mapping, mention.start, mention.end)
    assert corpus_data.texts[mention.document][start:end] == mention.surface


def test_naming_no_dataset_encodes_the_configured_corpus(monkeypatch, tmp_path):
    """No positional dataset and neither flag encodes what a run reads.

    The list used to be required, and one retyped per invocation is what
    left both noise pools out of every store while each split appended a
    block of each. Points `brenda_references` at a stub `tmp_path` corpus
    so the check needs no real BRENDA data.
    """
    for name in (path.name for path in brenda_references.corpus_files()):
        (tmp_path / name).touch()
    monkeypatch.setattr(brenda_references.data_paths, "DATA_DIR", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "precompute-encodings",
            "base-model",
            str(tmp_path / "encodings"),
        ],
    )

    assert precompute_encodings.read_args().datasets == list(
        brenda_references.corpus_files()
    )


def test_an_external_corpus_alone_still_encodes_only_itself(
    monkeypatch, tmp_path
):
    """`--s800` on its own is a request for that corpus, not for both."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "precompute-encodings",
            "base-model",
            str(tmp_path / "encodings"),
            "--s800",
            str(tmp_path),
        ],
    )

    assert precompute_encodings.read_args().datasets == []


def test_an_absent_configured_corpus_is_named_rather_than_skipped(
    monkeypatch, tmp_path, capsys
):
    """A machine without the data gets the missing paths, not an empty pass.

    Guards `d3text.cli.args.resolve_datasets`, which every precompute
    command defaults through: an unreadable default that silently encoded
    nothing would look exactly like a finished run.
    """
    monkeypatch.setattr(
        brenda_references,
        "corpus_files",
        lambda: (tmp_path / "never_fetched.csv",),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "precompute-encodings",
            "base-model",
            str(tmp_path / "encodings"),
        ],
    )

    with pytest.raises(SystemExit):
        precompute_encodings.read_args()

    assert "never_fetched.csv" in capsys.readouterr().err


def test_batched_tokenization_is_byte_identical_to_one_document_at_a_time(
    monkeypatch, tmp_path
):
    """A mid-batch already-stored document must not corrupt its neighbours.

    The finished middle document must leave the batch before the tokenizer
    call. The other four are compared byte for byte against a solo
    `encode_documents` call each, with the real offline tokenizer.
    """
    tokenizer = _build_offline_fast_tokenizer()
    monkeypatch.setattr(
        precompute_encodings.utils,
        "load_fast_tokenizer",
        lambda base_model: tokenizer,
    )
    monkeypatch.setattr(precompute_encodings, "MAX_LENGTH", 8)
    monkeypatch.setattr(precompute_encodings, "STRIDE", 2)

    texts = {
        "1": "ab",
        "2": "abcdefgh",  # 8 content tokens: overflows one 8-token window.
        "3": "untouched",  # already stored; sits in the middle of the batch.
        "4": "cd",
        "5": "ef",
    }
    output = tmp_path / "encodings"
    sentinel_ids = np.full((1, _WINDOW), 99, dtype=np.uint32)
    _prefill(
        output,
        {"3": sentinel_ids},
        EncodingsProvenance(base_model="a-base-model", max_length=8, stride=2),
    )

    dataset = tmp_path / "corpus.csv"
    _write_corpus(
        dataset,
        [
            {"pubmed_id": int(key), "abstract": text, "fulltext": None}
            for key, text in texts.items()
        ],
    )

    real_configure = logs.configure
    stream = io.StringIO()
    monkeypatch.setattr(
        precompute_encodings.logs,
        "configure",
        lambda: real_configure(logging.WARNING, stream=stream),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["precompute-encodings", "a-base-model", str(output), str(dataset)],
    )

    precompute_encodings.main()

    with EncodingsStore(output) as f:
        stored = {key: f.get(key) for key in texts}

    # Untouched: the resume-skip caught it before the batch was built.
    assert np.array_equal(stored["3"]["input_ids"], sentinel_ids)

    for key, text in texts.items():
        if key == "3":
            continue
        solo = precompute_encodings.encode_documents([text], tokenizer)
        assert np.array_equal(
            stored[key]["input_ids"], solo["input_ids"].numpy()
        )
        assert np.array_equal(
            stored[key]["attention_mask"], solo["attention_mask"].numpy()
        )
        assert np.array_equal(
            stored[key]["offset_mapping"], solo["offset_mapping"].numpy()
        )

    # "2" is longer than one window, so the split really overflowed --
    # not every document in the batch collapsed to a single row.
    assert stored["2"]["input_ids"].shape[0] > 1


def test_a_document_listed_twice_in_one_window_is_encoded_once(
    run_command, tmp_path
):
    """A pubmed_id the corpus repeats reaches the store once.

    `corpus.stream_rows` merges the repeat before any window sees it;
    unmerged, a batched write of the repeat died on `name already exists`.
    """
    dataset = tmp_path / "corpus.csv"
    _write_corpus(
        dataset,
        [
            {"pubmed_id": 1, "abstract": "an abstract", "fulltext": None},
            {"pubmed_id": 1, "abstract": "an abstract", "fulltext": None},
            {"pubmed_id": 2, "abstract": "another", "fulltext": None},
        ],
    )
    output = tmp_path / "encodings"

    run_command(dataset, output)

    with EncodingsStore(output) as f:
        assert f.keys() == ["1", "2"]
        assert f.windows("1") == 1
