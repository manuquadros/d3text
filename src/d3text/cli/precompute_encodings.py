#!/usr/bin/env python

import argparse
import itertools
import logging
import pathlib
import typing
from collections.abc import Mapping

import h5py
import hdf5plugin
import transformers
from d3text import corpus, encodings_store, logs, utils
from d3text.cli import args as cli_args
from d3text.datasets import enzymener, s800
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Rows pulled into memory at a time. Not a flag: it trades nothing a caller
# cares about, and the corpus is streamed precisely so it need not be tuned.
STREAM_BATCH = 1000

# Documents tokenized in one batched call. The Rust tokenizer parallelizes
# across this dimension, not within a sequence, so a batch of one runs
# single-threaded regardless of core count or TOKENIZERS_PARALLELISM; ~32
# is enough to occupy the machine's cores without the padded (batch,
# max_length) tensor the batched call also builds growing much past what a
# single window already costs.
TOKENIZE_BATCH = 32

# `split_and_tokenize`'s own defaults, passed explicitly rather than left
# implicit: `record_provenance` stamps whatever this run writes, and the reader
# refuses a store whose stamp disagrees with the geometry it will aggregate
# under, so what is stamped has to be the shared constant itself and not a
# second copy of its value.
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
            "Generate and save encodings for the documents from the provided"
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
    f: h5py.File,
    key: str,
    text: str,
    force_regenerate: bool,
) -> bool:
    """Decide whether `key` still needs tokenizing, clearing stale state.

    Split out of the write so a whole window of documents can be filtered
    before the batched tokenizer call runs, rather than after: an
    already-finished group is left untouched and this returns False;
    anything else (missing, torn, or `force_regenerate`) has its existing
    group, if any, dropped now, mirroring the resume rule the old
    per-document write applied exactly, just ahead of the tokenizer call
    instead of interleaved with it.

    :param f: the open, writable encodings store.
    :param key: the group name to check.
    :param text: the document's text; a falsy value needs no group.
    :param force_regenerate: whether to overwrite an already-finished group
        instead of skipping it.
    :return: whether `key` should be tokenized and written this pass.
    """
    if key in f:
        if not force_regenerate and encodings_store.is_finished_group(f[key]):
            return False
        # Either -f, or a group a killed pass left torn: either way the stale
        # or incomplete group must not survive underneath what gets written
        # next.
        del f[key]

    if not text:
        logger.warning(
            "%s has no text; storing no encoding for it.",
            key,
        )
        return False

    return True


def _store_encoding(
    f: h5py.File,
    key: str,
    encoding: Mapping[str, object],
    compression: hdf5plugin.Zstd,
) -> None:
    """Write one already-tokenized document's `encoding` into `f`.

    Split out of `_write_window` so the four-dataset write and the
    completion marker stay in one place regardless of whether `encoding`
    came from a batch of one document or of `TOKENIZE_BATCH`. `encoding`'s
    values are whatever h5py's own `data=` accepts -- a `Tensor` slice from
    the real tokenizer, a `list[int]` for the zero-filled sample mapping, or
    (in tests) a bare `numpy.ndarray` -- so it is typed as the one thing
    they all are, rather than narrowed to a union only production ever
    produces.
    """
    group = f.create_group(key)
    group.create_dataset(
        name="input_ids",
        data=encoding["input_ids"],
        compression=compression,
        dtype="uint32",
    )
    group.create_dataset(
        name="attention_mask",
        data=encoding["attention_mask"],
        compression=compression,
        dtype="uint8",
    )
    group.create_dataset(
        name="overflow_to_sample_mapping",
        data=encoding["overflow_to_sample_mapping"],
        compression=compression,
        dtype="uint8",
    )
    # Char-span offsets into the source text, per token, per window --
    # `split_and_tokenize` requests it, and it is the one thing on disk that
    # lets a later reader join a stored token position back to an annotation
    # offset. `uint32` matches `input_ids`: both are non-negative and the
    # documents here are far short of 4 billion characters.
    group.create_dataset(
        name="offset_mapping",
        data=encoding["offset_mapping"],
        compression=compression,
        dtype="uint32",
    )
    encodings_store.mark_group_complete(group)


def _write_window(
    f: h5py.File,
    window: list[tuple[str, str]],
    tokenizer: object,
    compression: hdf5plugin.Zstd,
    force_regenerate: bool,
) -> None:
    """Tokenize up to `TOKENIZE_BATCH` documents in a single batched call.

    Shared by every source `precompute-encodings` reads — BRENDA rows, S800
    documents, enzymeNER sentences. Filtering (`_prepare_document`) runs over
    the whole window first, so an already-finished document is dropped
    before the tokenizer call rather than after, and never costs that call
    anything -- the point of batching in the first place. `tokenizer` is
    opaque here: this function never calls a method on it, only forwards it
    to `encode_documents`, which is what carries the real
    `PreTrainedTokenizerFast` constraint (and what tests replace wholesale).

    :param f: the open, writable encodings store.
    :param window: up to `TOKENIZE_BATCH` `(key, text)` pairs to consider.
    :param tokenizer: the fast tokenizer to encode with, forwarded as-is.
    :param compression: the HDF5 filter each dataset is written with.
    :param force_regenerate: whether to overwrite an already-finished group
        instead of skipping it.
    """
    pending = [
        (key, text)
        for key, text in window
        if _prepare_document(f, key, text, force_regenerate)
    ]
    if not pending:
        return

    # `tokenizer` is deliberately untyped above (see the docstring); the cast
    # is for mypy's benefit only and checks nothing at runtime, so it does not
    # reintroduce the beartype violation a real annotation here would.
    encoding = encode_documents(
        [text for _, text in pending],
        tokenizer=typing.cast(transformers.PreTrainedTokenizerFast, tokenizer),
    )
    sample_mapping = encoding["overflow_to_sample_mapping"]
    for index, (key, _) in enumerate(pending):
        rows = sample_mapping == index
        _store_encoding(
            f,
            key,
            {
                "input_ids": encoding["input_ids"][rows],
                "attention_mask": encoding["attention_mask"][rows],
                "offset_mapping": encoding["offset_mapping"][rows],
                # All zeros, matching what a solo call over this one
                # document would have produced; nothing downstream reads
                # this field for more than presence, so the batch-relative
                # sample index a real tokenizer call assigns is discarded
                # rather than stored.
                "overflow_to_sample_mapping": [0] * int(rows.sum()),
            },
            compression,
        )


def main() -> None:
    """Tokenize every configured source and write it into the encodings HDF5.

    :raises ValueError: if the store already records a different tokenizer,
        window or stride than this run's.
    """
    logs.configure()
    args = read_args()
    tokenizer = utils.load_fast_tokenizer(args.base_model)
    out_path = pathlib.Path(args.output_path)
    mode = "r+" if out_path.exists() else "w-"

    # `libver="latest"` is a size knob here, not a compatibility one: the
    # default format spends ~11.4 kB per document on object headers and B-tree
    # nodes, which on the 12230-document file is 108 MiB — 40% of it — against
    # 159 MiB of actual compressed payload. The latest format writes the same
    # groups in ~3.2 kB. It bounds only what *this* writer emits, so an `r+`
    # resume onto an existing default-format file is legal and its new groups
    # get the compact layout too.
    with h5py.File(out_path, mode, libver="latest") as f:
        # Before the writing pass rather than inside it: a store that refuses
        # this geometry has had no group written, so it must keep the stamp
        # it still answers for.
        encodings_store.record_provenance(
            f,
            encodings_store.EncodingsProvenance(
                base_model=args.base_model,
                max_length=MAX_LENGTH,
                stride=STRIDE,
            ),
        )

        with encodings_store.writing_pass(f):
            compression = hdf5plugin.Zstd(clevel=22)
            for dataset in tqdm(args.datasets, position=0, desc="Datasets"):
                total, rows = corpus.stream_rows(
                    pathlib.Path(dataset), STREAM_BATCH
                )

                for window in itertools.batched(
                    tqdm(
                        rows,
                        position=1,
                        desc="Rows (zstd, clevel=22)",
                        total=total,
                    ),
                    TOKENIZE_BATCH,
                ):
                    _write_window(
                        f,
                        [(str(pubmed_id), text) for pubmed_id, text in window],
                        tokenizer,
                        compression,
                        args.force_regenerate,
                    )

            if args.s800:
                for window in itertools.batched(
                    tqdm(
                        s800.load_s800(args.s800).texts.items(),
                        position=0,
                        desc="S800 (zstd, clevel=22)",
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
                        compression,
                        args.force_regenerate,
                    )

            if args.enzymener:
                for window in itertools.batched(
                    tqdm(
                        enzymener.load_enzymener(args.enzymener).texts.items(),
                        position=0,
                        desc="enzymeNER (zstd, clevel=22)",
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
                        compression,
                        args.force_regenerate,
                    )


if __name__ == "__main__":
    main()
