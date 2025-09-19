#!/usr/bin/env python
import argparse
import os
import pathlib
import struct

import lmdb
import numpy as np
import polars as pl
import torch
import tqdm
import transformers
from d3text import utils


# ---- args & model ----
def read_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("base_model")
    p.add_argument("output_path")
    p.add_argument("datasets", nargs="+")
    p.add_argument("-f", "--force-regenerate", action="store_true")
    p.add_argument("--batch_size", type=int, default=8)  # tune for your VRAM
    p.add_argument(
        "--max_length", type=int, default=None
    )  # default: tokenizer max
    return p.parse_args()


def tensor_to_bytes(t: torch.Tensor) -> bytes:
    a = t.detach().to(torch.float16).contiguous().cpu().numpy()
    header = struct.pack("QQ", a.shape[0], a.shape[1])
    return header + a.tobytes(order="C")


if __name__ == "__main__":
    # help CUDA memory fragmentation a bit
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    args = read_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model)
    model = (
        transformers.AutoModel.from_pretrained(args.base_model)
        .to(device)
        .eval()
    )
    hidden_size = model.config.hidden_size
    max_len = args.max_length or getattr(tokenizer, "model_max_length", 512)

    lmdbenv = lmdb.open(args.output_path, map_size=64 * 1024**3)

    for dataset in args.datasets:
        print(f"Processing {dataset}")
        path = pathlib.Path(dataset)
        if path.suffix == ".csv":
            dt = pl.scan_csv(path).drop("")
        elif path.suffix == ".json":
            dt = pl.scan_ndjson(path).rename({"body": "fulltext"})
        else:
            raise ValueError(f"{dataset} has an unrecognized file format.")

        # Build (pubmed_id, text) eagerly; keep only what we need
        df = dt.select(
            pl.col("pubmed_id"),
            pl.concat_str(
                [
                    pl.col("abstract").fill_null(""),
                    pl.col("fulltext").fill_null(""),
                ],
                separator="\n",
            ).alias("text"),
        ).collect()

        pmids = df.get_column("pubmed_id").to_list()
        texts = df.get_column("text").to_list()

        count = 0
        tdb = lmdbenv.begin(write=True)
        for pmid, text in tqdm.tqdm(
            zip(pmids, texts), total=len(pmids), desc="Articles"
        ):
            embedding = utils.embed_document(
                text, tokenizer=tokenizer, model=model, stride=20
            )
            tdb.put(str(pmid).encode(), tensor_to_bytes(embedding))
            count += 1
            if count % 250 == 0:
                tdb.commit()
                tdb = lmdbenv.begin(write=True)
        tdb.commit()
