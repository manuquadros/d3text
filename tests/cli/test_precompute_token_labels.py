"""``precompute-token-labels``: the producer of the distant-supervision store.

Everything here runs off inline fixtures and a tokenizer built in-process — no
``documents.json``, no split csv, no download. The entity tables are a
hand-built dump of the same shape ``surface_forms.load_entity_tables`` reads,
which is all the command asks of them.
"""

import functools
import json
import pathlib
import string
import subprocess
import sys

import h5py
import numpy
import polars as pl
import pytest
from d3text import corpus, token_labels
from d3text.cli import precompute_token_labels
from d3text.utils import split_and_tokenize
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

_SPECIALS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]")

_TABLES = {
    "enzymes": {
        "3494": {
            "recommended_name": "cholesterol oxidase",
            "ec_class": "1.1.3.6",
            "synonyms": ["COD"],
        },
        "9999": {"recommended_name": "catalase", "synonyms": []},
    },
    "bacteria": {
        "42": {"organism": "Streptomyces griseocarneus", "synonyms": []}
    },
    "strains": {},
}

_ROWS = [
    {
        "pubmed_id": 10822008,
        "abstract": "cholesterol oxidase from Streptomyces griseocarneus",
        "fulltext": "and some catalase besides",
        "enzymes": "[3494]",
        "bacteria": "{'42': 'Streptomyces griseocarneus'}",
        "strains": "[]",
        "other_organisms": "{'7': 'Jaculus orientalis'}",
    },
    {
        "pubmed_id": 287675,
        "abstract": "Jaculus orientalis was studied",
        "fulltext": "at length",
        "enzymes": "[]",
        "bacteria": "{}",
        "strains": "[]",
        "other_organisms": "{'7': 'Jaculus orientalis'}",
    },
]


@functools.cache
def _tokenizer() -> PreTrainedTokenizerFast:
    """One token per character, built in-process. See `tests/test_token_labels`."""
    vocabulary = {token: index for index, token in enumerate(_SPECIALS)}
    for character in string.ascii_letters + string.digits:
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


_CORPUS_SCHEMA = {
    "pubmed_id": pl.Int64,
    "abstract": pl.Utf8,
    "fulltext": pl.Utf8,
    "enzymes": pl.Utf8,
    "bacteria": pl.Utf8,
    "strains": pl.Utf8,
    "other_organisms": pl.Utf8,
}


def _write_corpus(path: pathlib.Path, rows: list[dict]) -> pathlib.Path:
    pl.DataFrame(rows, schema=_CORPUS_SCHEMA).write_csv(path)
    return path


@pytest.fixture
def corpus_csv(tmp_path) -> pathlib.Path:
    return _write_corpus(tmp_path / "split.csv", _ROWS)


@pytest.fixture
def entity_tables(tmp_path) -> pathlib.Path:
    path = tmp_path / "documents.json"
    path.write_text(json.dumps(_TABLES), encoding="utf8")
    return path


@pytest.fixture
def run_command(monkeypatch, tmp_path):
    """Run `main` over the fixtures, with the in-process tokenizer."""

    def run(
        entity_tables: pathlib.Path,
        corpus_csv: pathlib.Path,
        output: pathlib.Path,
        *flags: str,
    ) -> None:
        monkeypatch.setattr(
            precompute_token_labels.utils,
            "load_fast_tokenizer",
            lambda base_model: _tokenizer(),
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "precompute-token-labels",
                "base-model",
                str(entity_tables),
                str(output),
                str(corpus_csv),
                *flags,
            ],
        )
        precompute_token_labels.main()

    return run


def test_it_writes_one_target_array_per_document(
    run_command, entity_tables, corpus_csv, tmp_path
) -> None:
    output = tmp_path / "labels.hdf5"

    run_command(entity_tables, corpus_csv, output)

    with h5py.File(output, "r") as store:
        assert set(store) == {"10822008", "287675"}


def test_the_store_says_what_its_codes_mean(
    run_command, entity_tables, corpus_csv, tmp_path
) -> None:
    """Free to record now and impossible to recover later: the codes' meaning
    has to be in the artifact before the artifact exists."""
    output = tmp_path / "labels.hdf5"

    run_command(entity_tables, corpus_csv, output)

    with h5py.File(output, "r") as store:
        assert (
            token_labels.read_label_space(store) == token_labels.BRENDA_LABELS
        )


def test_a_gold_mention_gets_its_own_type(
    run_command, entity_tables, corpus_csv, tmp_path
) -> None:
    """End to end: the enzyme and the bacterium of one document, apart.

    Both are gold for 10822008, both are in the tables, and the two targets
    have to differ — under the binary predecessor they did not.
    """
    output = tmp_path / "labels.hdf5"

    run_command(entity_tables, corpus_csv, output)

    with h5py.File(output, "r") as store:
        labels = token_labels.load_token_labels(store, "10822008").codes

    present = set(numpy.unique(labels).tolist())

    assert token_labels.BRENDA_LABELS.code_of("enz3494") in present
    assert token_labels.BRENDA_LABELS.code_of("bac42") in present


def test_an_other_organism_is_labelled_from_another_documents_naming(
    run_command, entity_tables, corpus_csv, tmp_path
) -> None:
    """The namespace with no table in the dump.

    `Jaculus orientalis` exists only in the corpus's own `other_organisms`
    column, so a command that built its index from the entity tables alone
    would label this document's only gold mention as matching nothing.
    """
    output = tmp_path / "labels.hdf5"

    run_command(entity_tables, corpus_csv, output)

    with h5py.File(output, "r") as store:
        labels = token_labels.load_token_labels(store, "287675").codes

    assert token_labels.BRENDA_LABELS.code_of("oth7") in set(
        numpy.unique(labels).tolist()
    )


def test_a_second_run_resumes_rather_than_relabelling(
    run_command, entity_tables, corpus_csv, tmp_path
) -> None:
    """Same contract as `precompute-embeddings`: a stored key is left alone.

    The run is a tokenizer pass and a matcher pass over every document of a
    ~560 MB split, so an interrupted one must not start over.
    """
    output = tmp_path / "labels.hdf5"
    run_command(entity_tables, corpus_csv, output)

    with h5py.File(output, "r+") as store:
        del store["10822008"]
        store.create_group("10822008")

    run_command(entity_tables, corpus_csv, output)

    with h5py.File(output, "r") as store:
        assert set(store["10822008"]) == set()


def test_force_relabels_what_the_store_already_holds(
    run_command, entity_tables, corpus_csv, tmp_path
) -> None:
    output = tmp_path / "labels.hdf5"
    run_command(entity_tables, corpus_csv, output)

    with h5py.File(output, "r+") as store:
        del store["10822008"]
        store.create_group("10822008")

    run_command(entity_tables, corpus_csv, output, "-f")

    with h5py.File(output, "r") as store:
        assert set(store["10822008"]) == {"codes", "spans"}


def test_force_deletes_the_stale_targets_of_a_document_now_without_text(
    run_command, entity_tables, corpus_csv, tmp_path
) -> None:
    """A document the corpus no longer gives any text loses its targets.

    The store was produced when 10822008 had text; the corpus now says it has
    neither an abstract nor a fulltext. Its stored targets address a string
    that no longer exists, so a `-f` run must delete them rather than keep
    them — and must not try to label the empty string instead.
    """
    output = tmp_path / "labels.hdf5"
    run_command(entity_tables, corpus_csv, output)
    with h5py.File(output, "r") as store:
        assert "10822008" in store

    emptied_rows = [dict(row) for row in _ROWS]
    emptied_rows[0]["abstract"] = ""
    emptied_rows[0]["fulltext"] = ""
    emptied = _write_corpus(tmp_path / "emptied.csv", emptied_rows)

    run_command(entity_tables, emptied, output, "-f")

    with h5py.File(output, "r") as store:
        assert "10822008" not in store
        assert "287675" in store


def test_resuming_a_store_of_another_label_space_is_refused(
    run_command, entity_tables, corpus_csv, tmp_path
) -> None:
    """The two halves of such a file would mean different things.

    Nothing about the arrays would object — the widths are identical — so the
    recorded order is the only thing that can refuse, and it must.
    """
    output = tmp_path / "labels.hdf5"
    with h5py.File(output, "w-", libver="latest") as store:
        token_labels.write_label_space(
            store,
            token_labels.LabelSpace(
                types=token_labels.BRENDA_LABELS.types[::-1],
                prefixes=token_labels.BRENDA_LABELS.prefixes[::-1],
            ),
        )

    with pytest.raises(ValueError, match="regenerate it"):
        run_command(entity_tables, corpus_csv, output)


def test_the_run_writes_the_mention_spans_beside_the_codes(
    run_command, entity_tables, corpus_csv, tmp_path
) -> None:
    """A store of codes with no spans must not be creatable by any run.

    The window for adding them closes the moment a real artifact exists, so the
    producer writes both from the start or the whole file has to be rebuilt.
    """
    output = tmp_path / "labels.hdf5"

    run_command(entity_tables, corpus_csv, output)

    with h5py.File(output, "r") as store:
        for key in store:
            assert set(store[key]) == {"codes", "spans"}
        labels = token_labels.load_token_labels(store, "10822008")

    assert labels.spans.shape[1] == token_labels.SPAN_COLUMNS
    assert labels.spans.shape[0] > 0
    assert labels.text_length > 0


def test_the_spans_a_run_writes_reconstruct_the_codes_it_wrote(
    run_command, entity_tables, corpus_csv, tmp_path
) -> None:
    """End to end, over the artifact the command actually produced.

    Painting the stored spans back over the document and projecting them onto
    the tokenizer the run used has to give the stored codes exactly; anything
    less means the two halves of the artifact can drift apart.
    """
    output = tmp_path / "labels.hdf5"
    run_command(entity_tables, corpus_csv, output)
    row = _ROWS[0]
    text = corpus.document_text(row["abstract"], row["fulltext"])

    with h5py.File(output, "r") as store:
        labels = token_labels.load_token_labels(store, "10822008")

    assert labels.text_length == len(text)
    encoding = split_and_tokenize(_tokenizer(), text)
    rebuilt = token_labels.project_onto_tokens(
        token_labels.character_labels_from_spans(
            labels.text_length, labels.spans
        ),
        encoding["offset_mapping"],
    )

    assert numpy.array_equal(rebuilt, labels.codes)


def test_a_missing_corpus_file_is_rejected_before_anything_is_read(
    run_command, entity_tables, tmp_path
) -> None:
    """The tables are 1.1 GB and the index build scans every corpus file, so a
    mistyped path must not be discovered after all of that."""
    with pytest.raises(SystemExit):
        run_command(entity_tables, tmp_path / "absent.csv", tmp_path / "l.hdf5")


def test_an_unwritable_output_directory_is_rejected(
    run_command, entity_tables, corpus_csv, tmp_path
) -> None:
    with pytest.raises(SystemExit):
        run_command(
            entity_tables, corpus_csv, tmp_path / "absent" / "labels.hdf5"
        )


def test_the_command_does_not_import_the_data_layer(tmp_path) -> None:
    """It needs each document's gold entity set, which the BRENDA data layer
    also produces — and importing that layer used to drop an `lpsn.log` into
    whatever directory the command was invoked from. The gold set comes off the
    split frame's own columns instead, through `d3text.corpus`.

    Checked in a subprocess: the suite as a whole imports `d3text.data`, so an
    in-process check would pass no matter what this module pulls in.
    """
    probe = (
        "import sys; import d3text.cli.precompute_token_labels; "
        "print(any(m.startswith(('d3text.data', 'brenda_references', "
        "'lpsn_interface')) for m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=True,
    )

    assert result.stdout.strip().endswith(
        "False"
    ), "precompute-token-labels pulled in the BRENDA data layer"
    assert list(tmp_path.iterdir()) == [], (
        "importing the command littered its working directory: "
        f"{sorted(path.name for path in tmp_path.iterdir())}"
    )
