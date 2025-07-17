#!/usr/bin/env python

import argparse
import pathlib

import h5py
import hdf5plugin
import pandas as pd
import transformers
import xmlparser
from d3text import utils
from tqdm import tqdm


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="generate_embeddings.py",
        description=(
            "Generate and save embeddings for the documents in the provided"
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
    model = (
        transformers.AutoModel.from_pretrained(args.base_model).cuda().eval()
    )
    out_path = pathlib.Path(args.output_path)
    if out_path.exists():
        mode = "r+"
    else:
        mode = "w-"

    ACCURACY = 1e-2

    with h5py.File(args.output_path, mode) as f:
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
                desc=f"Rows (zfp, accuracy={ACCURACY})",
                total=len(dt),
            ):
                pubmed_id = str(row.pubmed_id)
                if pubmed_id not in f:
                    abstract = str(row.abstract) or ""
                    fulltext = str(row.fulltext) or ""
                    if not abstract and not fulltext:
                        tqdm.write(pubmed_id)

                    embedding = utils.embed_document(
                        xmlparser.remove_tags(abstract + fulltext),
                        tokenizer=tokenizer,
                        model=model,
                    )

                    f.create_dataset(
                        pubmed_id,
                        data=embedding,
                        compression=hdf5plugin.Zfp(accuracy=ACCURACY),
                    )
