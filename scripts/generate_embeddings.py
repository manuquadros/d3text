#!/usr/bin/env python

import argparse
import pathlib

import h5py
import hdf5plugin
import numpy as np
import pandas as pd
import transformers
import xmlparser
from d3text import utils
from jaxtyping import Float
from torch import Tensor
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

    ACCURACY = 1e-2
    EMBEDDING_DIM = model.config.hidden_size

    with h5py.File(args.output_path, mode="a") as f:
        if "embeddings" not in f:
            f.create_dataset(
                "embeddings",
                shape=(0, EMBEDDING_DIM),
                maxshape=(None, EMBEDDING_DIM),
                dtype=np.float32,
                chunks=(1000, EMBEDDING_DIM),
                compression=hdf5plugin.Zfp(accuracy=ACCURACY),
            )
            f.create_dataset(
                "offsets",
                shape=(0, 2),
                maxshape=(None, 2),
                dtype=np.int64,
            )
            f.create_dataset(
                "pubmed_ids",
                shape=(0,),
                maxshape=(None,),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )

        # Resume from existing offset if applicable
        token_offset = f["embeddings"].shape[0]
        doc_offset = f["offsets"].shape[0]
        existing_ids = set(f["pubmed_ids"][:])

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
                if pubmed_id in existing_ids:
                    continue

                abstract = str(row.abstract) or ""
                fulltext = str(row.fulltext) or ""
                if not abstract and not fulltext:
                    tqdm.write(pubmed_id)
                    continue

                embedding: Float[Tensor, "token embedding"] = (
                    utils.embed_document(
                        xmlparser.remove_tags(abstract + fulltext),
                        tokenizer=tokenizer,
                        model=model,
                    )
                )

                emb_np = embedding.cpu().numpy()
                n_tokens = emb_np.shape[0]

                f["embeddings"].resize(token_offset + n_tokens, axis=0)
                f["embeddings"][token_offset : token_offset + n_tokens] = emb_np

                f["offsets"].resize(doc_offset + 1, axis=0)
                f["offsets"][doc_offset] = (
                    token_offset,
                    token_offset + n_tokens,
                )

                f["pubmed_ids"].resize(doc_offset + 1, axis=0)
                f["pubmed_ids"][doc_offset] = pubmed_id

                token_offset += n_tokens
                doc_offset += 1

                existing_ids.add(pubmed_id)
