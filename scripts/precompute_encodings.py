#!/usr/bin/env python

import argparse
import pathlib

import h5py
import hdf5plugin
import pandas as pd
import transformers
import xmlparser
from d3text import utils
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm


def encode_document(
    doc: str,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Float[Tensor, "tokens features"]:
    return utils.split_and_tokenize(tokenizer=tokenizer, inputs=doc)


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="precompute_encodings.py",
        description=(
            "Generate and save encodings for the documents in the provided"
            "data frames."
        ),
    )
    parser.add_argument("base_model")
    parser.add_argument("output_path")
    parser.add_argument("datasets", nargs="+")

    return parser.parse_args()


if __name__ == "__main__":
    args = read_args()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model)
    out_path = pathlib.Path(args.output_path)
    if out_path.exists():
        mode = "r+"
    else:
        mode = "w-"

    with h5py.File(args.output_path, mode) as f:
        compression = hdf5plugin.Zstd(clevel=22)
        for dataset in tqdm(args.datasets, position=0, desc="Datasets"):
            path = pathlib.Path(dataset)

            if path.suffix == ".csv":
                dt = pd.read_csv(path, index_col=0)
            elif path.suffix == ".json":
                dt = pd.read_json(path, lines=True).rename(
                    columns={"body": "fulltext"}
                )
            else:
                msg = f"{dataset} has an unrecognized file format."
                raise ValueError(msg)

            for row in tqdm(
                dt.itertuples(),
                position=1,
                desc="Rows (zstd, clevel=22)",
                total=len(dt),
            ):
                pubmed_id = str(row.pubmed_id)
                if pubmed_id not in f:
                    abstract = str(row.abstract) or ""
                    fulltext = str(row.fulltext) or ""
                    if not abstract and not fulltext:
                        tqdm.write(pubmed_id)

                    encoding = encode_document(
                        xmlparser.remove_tags(abstract + fulltext),
                        tokenizer=tokenizer,
                    )
                    group = f.create_group(pubmed_id)
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
