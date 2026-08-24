#!/usr/bin/env python

import argparse
import logging
import pathlib

import h5py
import hdf5plugin
import transformers
from d3text import corpus, logs, utils
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Rows pulled into memory at a time. Not a flag: it trades nothing a caller
# cares about, and the corpus is streamed precisely so it need not be tuned.
STREAM_BATCH = 1000


def encode_document(
    doc: str,
    tokenizer: transformers.PreTrainedTokenizerFast,
) -> transformers.BatchEncoding:
    return utils.split_and_tokenize(tokenizer=tokenizer, inputs=doc)


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="precompute-encodings",
        description=(
            "Generate and save encodings for the documents from the provided"
            "data frames."
        ),
    )
    parser.add_argument("base_model")
    parser.add_argument("output_path")
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("-f", "--force-regenerate", action="store_true")

    return parser.parse_args()


def main() -> None:
    logs.configure()
    args = read_args()
    tokenizer = utils.load_fast_tokenizer(args.base_model)
    out_path = pathlib.Path(args.output_path)
    mode = "r+" if out_path.exists() else "w-"

    with h5py.File(out_path, mode) as f:
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
                key = str(pubmed_id)
                if key in f and not args.force_regenerate:
                    continue

                if not text:
                    logger.warning(
                        "%s has neither an abstract nor a fulltext.", key
                    )

                encoding = encode_document(text, tokenizer=tokenizer)

                if key in f:
                    del f[key]
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


if __name__ == "__main__":
    main()
