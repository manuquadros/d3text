#!/usr/bin/env python

import argparse
import logging
import pathlib
import typing

import h5py
import hdf5plugin
import transformers
from d3text import corpus, encodings_store, logs, utils
from d3text.datasets import enzymener, s800
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Rows pulled into memory at a time. Not a flag: it trades nothing a caller
# cares about, and the corpus is streamed precisely so it need not be tuned.
STREAM_BATCH = 1000

# `split_and_tokenize`'s own defaults, passed explicitly rather than left
# implicit: `record_provenance` stamps whatever this run writes, and the reader
# refuses a store whose stamp disagrees with the geometry it will aggregate
# under, so what is stamped has to be the shared constant itself and not a
# second copy of its value.
MAX_LENGTH = utils.WINDOW_LENGTH
STRIDE = utils.WINDOW_STRIDE


def encode_document(
    doc: str,
    tokenizer: transformers.PreTrainedTokenizerFast,
) -> transformers.BatchEncoding:
    return utils.split_and_tokenize(
        tokenizer=tokenizer, inputs=doc, max_length=MAX_LENGTH, stride=STRIDE
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
    parser.add_argument("datasets", nargs="*")
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
    if not args.datasets and not args.s800 and not args.enzymener:
        parser.error(
            "nothing to encode: pass at least one dataset, --s800, or "
            "--enzymener"
        )
    return args


def _write_document(
    f: h5py.File,
    key: str,
    text: str,
    tokenizer: object,
    compression: hdf5plugin.Zstd,
    force_regenerate: bool,
) -> None:
    """Tokenize one document and write it into `f`, honouring resume and `-f`.

    Shared by every source `precompute-encodings` reads — BRENDA rows, S800
    documents, enzymeNER sentences — so the skip-if-finished, drop-if-torn,
    and skip-if-empty rules live in one place rather than once per source.
    `tokenizer` is opaque here: this function never calls a method on it,
    only forwards it to `encode_document`, which is what carries the real
    `PreTrainedTokenizerFast` constraint (and what tests replace wholesale).

    :param f: the open, writable encodings store.
    :param key: the group name to write the document under.
    :param text: the document's text; a falsy value stores no group.
    :param tokenizer: the fast tokenizer to encode with, forwarded as-is.
    :param compression: the HDF5 filter each dataset is written with.
    :param force_regenerate: whether to overwrite an already-finished group
        instead of skipping it.
    """
    if key in f:
        if not force_regenerate and encodings_store.is_finished_group(f[key]):
            return
        # Either -f, or a group a killed pass left torn: either way the stale
        # or incomplete group must not survive underneath what gets written
        # next.
        del f[key]

    if not text:
        logger.warning(
            "%s has no text; storing no encoding for it.",
            key,
        )
        return

    # `tokenizer` is deliberately untyped above (see the docstring); the cast
    # is for mypy's benefit only and checks nothing at runtime, so it does not
    # reintroduce the beartype violation a real annotation here would.
    encoding = encode_document(
        text,
        tokenizer=typing.cast(transformers.PreTrainedTokenizerFast, tokenizer),
    )

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

                for pubmed_id, text in tqdm(
                    rows,
                    position=1,
                    desc="Rows (zstd, clevel=22)",
                    total=total,
                ):
                    _write_document(
                        f,
                        str(pubmed_id),
                        text,
                        tokenizer,
                        compression,
                        args.force_regenerate,
                    )

            if args.s800:
                for document, text in tqdm(
                    s800.load_s800(args.s800).texts.items(),
                    position=0,
                    desc="S800 (zstd, clevel=22)",
                ):
                    _write_document(
                        f,
                        f"s800:{document}",
                        text,
                        tokenizer,
                        compression,
                        args.force_regenerate,
                    )

            if args.enzymener:
                for sentence, text in tqdm(
                    enzymener.load_enzymener(args.enzymener).texts.items(),
                    position=0,
                    desc="enzymeNER (zstd, clevel=22)",
                ):
                    _write_document(
                        f,
                        f"enzymener:{sentence}",
                        text,
                        tokenizer,
                        compression,
                        args.force_regenerate,
                    )


if __name__ == "__main__":
    main()
