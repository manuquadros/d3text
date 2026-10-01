#!/usr/bin/env python

import argparse
import itertools
import logging
import pathlib
import typing

import transformers
from d3text import corpus, encodings_store, logs, utils
from d3text.cli import args as cli_args
from d3text.datasets import enzymener, s800
from tqdm import tqdm

logger = logging.getLogger(__name__)

# The Rust tokenizer parallelizes across documents, not within one, so a
# batch of one is single-threaded; this fills the cores without the padded
# batch tensor growing much.
TOKENIZE_BATCH = 32

# The shared constants themselves, not copies of their values: the stamp
# `record_provenance` writes must match the geometry readers aggregate under.
MAX_LENGTH = utils.WINDOW_LENGTH
STRIDE = utils.WINDOW_STRIDE


def encode_documents(
    docs: list[str],
    tokenizer: transformers.PreTrainedTokenizerFast,
) -> transformers.BatchEncoding:
    """Tokenize `docs` in one batched call, so the tokenizer parallelizes.

    :param docs: the document texts to tokenize together.
    :param tokenizer: the fast tokenizer to encode with.
    :return: the batch encoding, one row per window across all of `docs`;
        `overflow_to_sample_mapping` gives each row's position in `docs`.
    """
    return utils.split_and_tokenize(
        tokenizer=tokenizer, inputs=docs, max_length=MAX_LENGTH, stride=STRIDE
    )


def read_args() -> argparse.Namespace:
    """Parse the command line.

    :return: the parsed arguments; `datasets` may be empty, but only if
        `--s800` or `--enzymener` names something to encode instead.
    """
    parser = argparse.ArgumentParser(
        prog="precompute-encodings",
        description=(
            "Generate and save encodings for the documents from the provided "
            "data frames."
        ),
    )
    parser.add_argument("base_model")
    parser.add_argument("output_path")
    parser.add_argument(
        "datasets",
        nargs="*",
        type=cli_args.readable_path,
        help=(
            "corpus files to encode; with no --s800 or --enzymener, defaults "
            "to the splits and noise pools `brenda_references` is configured "
            "with"
        ),
    )
    parser.add_argument("-f", "--force-regenerate", action="store_true")
    parser.add_argument(
        "--s800",
        metavar="ROOT",
        help="Root directory of the S800 corpus to encode alongside it.",
    )
    parser.add_argument(
        "--enzymener",
        metavar="ROOT",
        help="Root directory of the enzymeNER corpus to encode alongside it.",
    )

    args = parser.parse_args()
    # An external corpus named on its own is a deliberate request to encode
    # only that into the store, so the configured corpus stands in for an
    # empty list only when neither flag is given.
    if not args.datasets and not args.s800 and not args.enzymener:
        args.datasets = cli_args.resolve_datasets(parser, args.datasets)
    return args


def _prepare_document(
    f: encodings_store.EncodingsStore,
    key: str,
    text: str,
    force_regenerate: bool,
) -> bool:
    """Decide whether `key` still needs tokenizing, clearing stale state.

    Runs over a whole window before the batched tokenizer call: a stored
    document is kept and skipped, or under `force_regenerate` dropped now.

    :param f: the open, writable encodings store.
    :param key: the document key to check.
    :param text: the document's text; a falsy value needs no document.
    :param force_regenerate: whether to overwrite a stored document instead
        of skipping it.
    :return: whether `key` should be tokenized and written this pass.
    """
    if key in f:
        if not force_regenerate:
            return False
        # A document whose text is now empty must not survive -f either.
        f.delete(key)

    if not text:
        logger.warning(
            "%s has no text; storing no encoding for it.",
            key,
        )
        return False

    return True


def _write_window(
    f: encodings_store.EncodingsStore,
    window: list[tuple[str, str]],
    tokenizer: object,
    force_regenerate: bool,
) -> None:
    """Tokenize up to `TOKENIZE_BATCH` documents in a single batched call.

    Filtering runs over the whole window first, so a finished document
    never costs the tokenizer call anything. `tokenizer` is opaque here and
    forwarded to `encode_documents`, which carries the real type (and which
    tests replace wholesale).

    :param f: the open, writable encodings store.
    :param window: up to `TOKENIZE_BATCH` `(key, text)` pairs to consider.
    :param tokenizer: the fast tokenizer to encode with, forwarded as-is.
    :param force_regenerate: whether to overwrite a stored document instead
        of skipping it.
    """
    pending: list[tuple[str, str]] = []
    for key, text in window:
        if _prepare_document(f, key, text, force_regenerate):
            pending.append((key, text))

    if not pending:
        return

    # `tokenizer` is deliberately untyped above (see the docstring); the cast
    # is for mypy's benefit only and checks nothing at runtime, so it does not
    # reintroduce the beartype violation a real annotation here would.
    encoding = encode_documents(
        [text for _, text in pending],
        tokenizer=typing.cast(transformers.PreTrainedTokenizerFast, tokenizer),
    )
    documents = encodings_store.document_encodings(encoding, len(pending))
    for (key, _), document in zip(pending, documents, strict=True):
        # The offsets are the only on-disk link from a token position back
        # to an annotation offset.
        f.put(key, document)


def main() -> None:
    """Tokenize every configured source and write it into the encodings store.

    :raises ValueError: if the store already records a different tokenizer,
        window or stride than this run's, or is of the older HDF5 layout.
    """
    logs.configure()
    args = read_args()
    tokenizer = utils.load_fast_tokenizer(args.base_model)
    out_path = pathlib.Path(args.output_path)

    with encodings_store.EncodingsStore(out_path, writable=True) as f:
        # Before the writing pass rather than inside it: a store that refuses
        # this geometry has had nothing written, so it must keep the stamp it
        # still answers for.
        encodings_store.record_provenance(
            f,
            encodings_store.EncodingsProvenance(
                base_model=args.base_model,
                max_length=MAX_LENGTH,
                stride=STRIDE,
            ),
        )

        with encodings_store.writing_pass(f):
            for dataset in tqdm(args.datasets, position=0, desc="Datasets"):
                total, rows = corpus.stream_rows(
                    pathlib.Path(dataset), corpus.STREAM_BATCH
                )

                for window in itertools.batched(
                    tqdm(
                        rows,
                        position=1,
                        desc="Rows",
                        total=total,
                    ),
                    TOKENIZE_BATCH,
                ):
                    _write_window(
                        f,
                        [(str(pubmed_id), text) for pubmed_id, text in window],
                        tokenizer,
                        args.force_regenerate,
                    )

            if args.s800:
                for window in itertools.batched(
                    tqdm(
                        s800.load_s800(args.s800).texts.items(),
                        position=0,
                        desc="S800",
                    ),
                    TOKENIZE_BATCH,
                ):
                    _write_window(
                        f,
                        [
                            (
                                encodings_store.external_key("s800", document),
                                text,
                            )
                            for document, text in window
                        ],
                        tokenizer,
                        args.force_regenerate,
                    )

            if args.enzymener:
                for window in itertools.batched(
                    tqdm(
                        enzymener.load_enzymener(args.enzymener).texts.items(),
                        position=0,
                        desc="enzymeNER",
                    ),
                    TOKENIZE_BATCH,
                ):
                    _write_window(
                        f,
                        [
                            (
                                encodings_store.external_key(
                                    "enzymener", sentence
                                ),
                                text,
                            )
                            for sentence, text in window
                        ],
                        tokenizer,
                        args.force_regenerate,
                    )


if __name__ == "__main__":
    main()
