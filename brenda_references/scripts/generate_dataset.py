"""Generate dataset splits, recording entropy values across sampling steps."""

import pathlib

import pandas as pd
from brenda_references.docdb import BrendaDocDB
from brenda_references.sampling import GMESampler

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"

if __name__ == "__main__":
    print("Loading articles...")
    with BrendaDocDB() as docdb:
        data = docdb.fulltext_articles()
        data = [doc for doc in data if doc["strains"] or not doc["bacteria"]]

    sampler = GMESampler(data=data)

    dfs = sampler.dataset_splits()

    for split, df in dfs.items():
        df.to_csv(DATA_DIR / f"{split}_entropies.csv")

        data_split = filter(
            lambda doc: int(doc["pubmed_id"]) in df["pubmed_id"].to_numpy(),
            data,
        )
        data_split = pd.DataFrame(data_split)

        data_split.to_csv(DATA_DIR / f"{split}_data.csv")
