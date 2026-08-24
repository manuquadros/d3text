"""Generate dataset splits, recording entropy values across sampling steps."""

from importlib import resources

import pandas as pd
from brenda_references.docdb import BrendaDocDB
from brenda_references.sampling import GMESampler

# Must be the directory `brenda_references.load_split` reads, or a regeneration
# lands somewhere nothing loads and every consumer keeps the previous splits.
DATA_DIR = resources.files("brenda_references") / "data"

if __name__ == "__main__":
    print("Loading articles...")
    with BrendaDocDB() as docdb:
        data = docdb.fulltext_articles()
        data = [doc for doc in data if doc["strains"] or not doc["bacteria"]]

    sampler = GMESampler(data=data)

    dfs = sampler.dataset_splits()

    for split, df in dfs.items():
        with resources.as_file(DATA_DIR / f"{split}_entropies.csv") as path:
            df.to_csv(path)

        data_split = filter(
            lambda doc: int(doc["pubmed_id"]) in df["pubmed_id"].to_numpy(),
            data,
        )
        data_split = pd.DataFrame(data_split)

        with resources.as_file(DATA_DIR / f"{split}_data.csv") as path:
            data_split.to_csv(path)
