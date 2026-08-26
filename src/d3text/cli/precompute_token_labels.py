#!/usr/bin/env python

"""Produce the per-token distant-supervision targets, offline.

One HDF5 store of `d3text.token_labels` targets, keyed by pubmed id and shaped
like the encodings the tagger reads. The targets are placed by matching BRENDA's
surface forms against the document text, so producing them needs the entity
tables, the corpus, and the tokenizer the encodings were built with — and
nothing else. In particular it needs no encodings file: re-tokenizing
`corpus.document_text` reproduces the stored `input_ids` element for element,
which is what makes the offsets addressable against them.

**A leaf, like the other two precompute commands.** It reads the corpus through
`d3text.corpus` and takes each document's gold entity set from the split frame's
own columns, rather than through `brenda_references.preprocess_labels`, which
would drag the BRENDA data layer — and its import-time write of an `lpsn.log`
into the working directory — into a command that only reads files it was
handed. `tests/cli/test_precompute_token_labels.py` pins that in a subprocess.

**Two passes over each corpus file.** The other-organism namespace has no table
in the BRENDA dump; the only place those names exist is inline in each
document's own `other_organisms` column, so the index cannot be built until
every file has been scanned for them. Pooling is the point rather than an
accident: a document naming an organism it was *not* annotated with is exactly
the case the ignore target exists for, and that mention is only recognizable
from some other document's naming of it.
"""

import argparse
import logging
import pathlib

import h5py
import numpy
import transformers
from d3text import corpus, logs, surface_forms, token_labels, utils
from numpy.typing import NDArray
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Rows pulled into memory at a time. Not a flag, for the reason
# `precompute-encodings` gives: it trades nothing a caller cares about, and the
# corpus is streamed precisely so it need not be tuned.
STREAM_BATCH = 1000


def build_index(
    entity_tables: pathlib.Path, datasets: list[pathlib.Path]
) -> surface_forms.SurfaceFormIndex:
    """The surface-form index, over all four ID namespaces."""
    tables = surface_forms.load_entity_tables(entity_tables)
    return surface_forms.build_index(
        surface_forms.brenda_surface_forms(
            tables,
            (
                names
                for dataset in datasets
                for names in corpus.other_organism_names(dataset, STREAM_BATCH)
            ),
        )
    )


def label_document(
    text: str,
    gold_entity_ids: frozenset[str],
    index: surface_forms.SurfaceFormIndex,
    tokenizer: transformers.PreTrainedTokenizerFast,
) -> NDArray[numpy.int8]:
    """One document's targets, in the geometry its encodings have."""
    encoding = utils.split_and_tokenize(tokenizer=tokenizer, inputs=text)
    return token_labels.document_token_labels(
        text, index, gold_entity_ids, encoding["offset_mapping"]
    )


def _readable(path: str) -> pathlib.Path:
    resolved = pathlib.Path(path)
    if not resolved.is_file():
        raise argparse.ArgumentTypeError(f"{path} is not a readable file")
    return resolved


def read_args() -> argparse.Namespace:
    """Parse and validate the command line.

    Every path is checked here, before the entity tables and the tokenizer are
    read: the tables are 1.1 GB and the index build scans every corpus file, so
    a mistyped output directory must not be discovered after all of that.
    """
    parser = argparse.ArgumentParser(
        prog="precompute-token-labels",
        description=(
            "Place per-token distant-supervision targets for the documents of "
            "the provided data frames, by matching BRENDA's surface forms."
        ),
    )
    parser.add_argument(
        "base_model",
        help="the model whose tokenizer the encodings were built with",
    )
    parser.add_argument(
        "entity_tables",
        type=_readable,
        help="BRENDA's TinyDB dump, holding the entity tables",
    )
    parser.add_argument("output_path", help="HDF5 store to write")
    parser.add_argument("datasets", nargs="+", type=_readable)
    parser.add_argument(
        "-f",
        "--force-regenerate",
        action="store_true",
        help="re-label documents the store already holds",
    )

    args = parser.parse_args()

    output = pathlib.Path(args.output_path)
    if not output.parent.is_dir():
        parser.error(f"{output.parent} is not a directory")
    args.output_path = output

    return args


def open_store(path: pathlib.Path) -> h5py.File:
    """The label store, with its label space recorded or checked.

    A resumed store is checked rather than re-stamped: its existing targets
    were written under whatever space it records, and continuing under a
    different one would leave a file whose halves mean different things — the
    silent re-permutation `token_labels.LabelSpace` exists to prevent. The
    answer to a mismatch is a regeneration, so the command refuses instead.
    """
    if not path.exists():
        store = h5py.File(path, "w-", libver="latest")
        token_labels.write_label_space(store, token_labels.BRENDA_LABELS)
        return store

    store = h5py.File(path, "r+", libver="latest")
    try:
        recorded = token_labels.read_label_space(store)
    except (KeyError, ValueError):
        store.close()
        raise
    if recorded != token_labels.BRENDA_LABELS:
        store.close()
        msg = (
            f"{path} holds targets over {recorded.types}, but this build "
            f"labels over {token_labels.BRENDA_LABELS.types}; regenerate it"
        )
        raise ValueError(msg)
    return store


def main() -> None:
    logs.configure()
    args = read_args()

    index = build_index(args.entity_tables, args.datasets)
    logger.info(
        "Indexed %d surface forms over %d entities.",
        len(index),
        len(index.entity_ids),
    )
    tokenizer = utils.load_fast_tokenizer(args.base_model)

    with open_store(args.output_path) as store:
        for dataset in tqdm(args.datasets, position=0, desc="Datasets"):
            total, documents = corpus.stream_documents(dataset, STREAM_BATCH)

            for document in tqdm(
                documents, position=1, desc="Rows", total=total
            ):
                key = str(document.pubmed_id)
                if key in store and not args.force_regenerate:
                    continue

                if not document.text:
                    logger.warning(
                        "%s has neither an abstract nor a fulltext; "
                        "storing no targets for it.",
                        key,
                    )
                    # Only reachable with -f, since a stored key is skipped
                    # above otherwise. The corpus now says this document has
                    # no text, so its stale targets go with it.
                    if key in store:
                        del store[key]
                    continue

                token_labels.store_token_labels(
                    store,
                    key,
                    label_document(
                        document.text,
                        document.entity_ids,
                        index,
                        tokenizer,
                    ),
                )


if __name__ == "__main__":
    main()
