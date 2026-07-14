import timeit

import blosc2
import lmdb
import polars as pl
import torch
import transformers
from cacheout import Cache
from d3text import utils

TRAINING_DATA = "brenda_references/src/brenda_references/data/training_data.csv"
_ID = 9151668

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = transformers.AutoTokenizer.from_pretrained(
    "michiyasunaga/BioLinkBERT-base"
)
model = (
    transformers.AutoModel.from_pretrained("michiyasunaga/BioLinkBERT-base")
    .to(device)
    .eval()
)
cpu_embeddings_cache = Cache(maxsize=100)


def get_from_lmdb(cursor):
    packed = cursor.get(str(_ID).encode())
    t = torch.tensor(blosc2.unpack_array(packed))
    return t


def compute_embedding(text):
    t = utils.embed_document(text, tokenizer, model)
    return t


if __name__ == "__main__":
    lazy = pl.scan_csv(TRAINING_DATA).drop(["", "volume"])

    row = (
        lazy.filter(pl.col("pubmed_id") == _ID)
        .select(
            pl.col("pubmed_id"),
            pl.concat_str(
                [
                    pl.col("abstract").fill_null(""),
                    pl.col("fulltext").fill_null(""),
                ],
                separator="\n",
            ).alias("text"),
        )
        .collect()
    )
    text = row.item(0, 1)

    lmdbenv = lmdb.open(
        "training_data_partial_zstd_9_bitshuffle_biolinkbert-base.lmdb"
    )
    txn = lmdbenv.begin(write=False)

    lmdb_timer = timeit.Timer("get_from_lmdb(txn)", globals=globals())
    compute_timer = timeit.Timer("compute_embedding(text)", globals=globals())

    print("LMDB: ", lmdb_timer.autorange())
    print("Computed: ", compute_timer.autorange())

    cpu_embeddings_cache.set(1, get_from_lmdb(txn))
    cache_timer = timeit.Timer("cpu_embeddings_cache.get(1)", globals=globals())
    print("Cache: ", cache_timer.autorange())
